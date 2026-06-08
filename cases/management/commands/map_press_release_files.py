"""
Django management command to map CIAA press release web URLs to actual document files.

Maps CIAA press release web page URLs (e.g., https://ciaa.gov.np/pressrelease/3345)
to actual PDF/DOCX file URLs from the NGM bucket. Creates DocumentSource records
for each file and updates case evidence accordingly.

Usage:
    python manage.py map_press_release_files --dry-run  # Test first
    python manage.py map_press_release_files            # Apply changes
    python manage.py map_press_release_files --case-id=case-abc123  # Specific case
    python manage.py map_press_release_files --limit=10  # Process first 10 cases
"""

import logging
import urllib.parse
from typing import Optional

import requests
from django.core.management.base import BaseCommand
from django.db import connection, transaction
from nepali.datetime import nepalidate

from cases.models import Case, DocumentSource, SourceType

logger = logging.getLogger(__name__)

# NGM root index URL (always fetches latest dated index)
NGM_ROOT_INDEX_URL = "https://ngm-store.jawafdehi.org/index-v2.json"


class Command(BaseCommand):
    help = "Map CIAA press release web URLs to actual document files from NGM bucket"

    def __init__(self):
        super().__init__()
        self.press_release_index = {}
        self.stats = {
            "cases_processed": 0,
            "cases_fixed": 0,
            "cases_skipped": 0,
            "sources_created": 0,
            "sources_updated": 0,
            "evidence_updated": 0,
            "errors": 0,
        }

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Run in dry-run mode (no database changes)",
        )
        parser.add_argument(
            "--case-id",
            type=str,
            help="Fix specific case by case_id (optional)",
        )
        parser.add_argument(
            "--limit",
            type=int,
            help="Limit number of cases to process (optional)",
        )

    def handle(self, *args, **options):
        dry_run = options["dry_run"]
        case_id = options.get("case_id")
        limit = options.get("limit")

        self.stdout.write(
            self.style.WARNING(
                f"{'[DRY RUN] ' if dry_run else ''}Starting press release mapping..."
            )
        )

        # Load NGM press release index
        if not self.load_press_release_index():
            self.stdout.write(
                self.style.ERROR("Failed to load press release index. Aborting.")
            )
            return

        # Bulk fetch all DocumentSources to avoid N+1 queries
        # First pass: collect all source_ids from DRAFT cases only
        all_source_ids = set()
        queryset = Case.objects.filter(state="DRAFT")
        if case_id:
            queryset = queryset.filter(case_id=case_id)

        for case in queryset:
            if case.evidence:
                for evidence_entry in case.evidence:
                    source_id = evidence_entry.get("source_id")
                    if source_id:
                        all_source_ids.add(source_id)

        source_lookup = {
            source.source_id: source
            for source in DocumentSource.objects.filter(
                source_id__in=all_source_ids, is_deleted=False
            )
        }

        # Find cases with press release evidence
        cases = self.find_cases_with_press_release_evidence(
            case_id, limit, source_lookup
        )
        self.stdout.write(f"Found {len(cases)} case(s) to process")

        # Process each case
        for case in cases:
            try:
                self.process_case(case, dry_run, source_lookup)
            except Exception as e:
                self.stats["errors"] += 1
                logger.exception(f"Error processing case {case.case_id}: {e}")
                self.stdout.write(
                    self.style.ERROR(f"✗ Error processing {case.case_id}: {e}")
                )

        # Print summary
        self.print_summary(dry_run)

    def load_press_release_index(self) -> bool:
        """Load press release index from NGM bucket with pagination support."""
        try:
            # Get root index to find latest press release index URL
            self.stdout.write(f"Loading root index from {NGM_ROOT_INDEX_URL}...")
            response = requests.get(NGM_ROOT_INDEX_URL, timeout=30)
            response.raise_for_status()
            root_data = response.json()

            # Find ciaa-press-releases child and get its $ref URL
            press_release_url = None
            for child in root_data.get("children", []):
                if child.get("name") == "ciaa-press-releases":
                    press_release_url = child.get("$ref")
                    break

            if not press_release_url:
                self.stdout.write(
                    self.style.ERROR("Could not find ciaa-press-releases in root index")
                )
                return False

            self.stdout.write(f"Found press release index: {press_release_url}")

            # Load all pages of press releases
            current_url = press_release_url
            page_num = 1

            while current_url:
                self.stdout.write(f"  Loading page {page_num}...")
                response = requests.get(current_url, timeout=30)
                response.raise_for_status()
                data = response.json()

                # Build index: press_id -> press release data with files
                for manuscript in data.get("manuscripts", []):
                    metadata = manuscript.get("metadata", {})
                    press_id = metadata.get("press_id")
                    if press_id:
                        if press_id not in self.press_release_index:
                            self.press_release_index[press_id] = {
                                "source_url": metadata.get("source_url"),
                                "title": metadata.get("title"),
                                "publication_date": metadata.get("publication_date"),
                                "files": [],
                            }
                        self.press_release_index[press_id]["files"].append(
                            {
                                "url": manuscript.get("url"),
                                "file_name": manuscript.get("file_name"),
                            }
                        )

                # Check for next page
                current_url = data.get("next")
                if current_url:
                    page_num += 1

            self.stdout.write(
                self.style.SUCCESS(
                    f"✓ Loaded {len(self.press_release_index)} press releases from {page_num} page(s)"
                )
            )
            return True

        except Exception as e:
            logger.exception(f"Failed to load press release index: {e}")
            self.stdout.write(self.style.ERROR(f"Failed to load index: {e}"))
            return False

    def find_cases_with_press_release_evidence(
        self, case_id: Optional[str], limit: Optional[int], source_lookup: dict
    ) -> list:
        """Find cases with evidence pointing to CIAA press release URLs.

        Only processes DRAFT cases to avoid modifying published or in-review cases.
        """
        queryset = Case.objects.filter(state="DRAFT")

        if case_id:
            queryset = queryset.filter(case_id=case_id)

        self.stdout.write(f"Scanning {queryset.count()} DRAFT cases...")

        # Filter cases with evidence containing ciaa.gov.np/pressrelease URLs
        # Uses source_lookup passed from handle() to avoid duplicate DB queries
        cases_with_pr_evidence = []
        skipped_already_mapped = 0
        skipped_no_pr_url = 0

        for case in queryset:
            if not case.evidence:
                continue

            has_pr_evidence = False
            for evidence_entry in case.evidence:
                source_id = evidence_entry.get("source_id")
                if source_id:
                    source = source_lookup.get(source_id)
                    if not source:
                        continue

                    # Skip if source is already file-backed (has NGM file URL)
                    is_file_backed = any(
                        "ngm-store.jawafdehi.org" in url for url in source.url_links
                    )

                    if is_file_backed:
                        skipped_already_mapped += 1
                        continue

                    # Check if source URL contains press release URL
                    has_pr_evidence = any(
                        "ciaa.gov.np/pressrelease/" in url for url in source.url_links
                    )

            if has_pr_evidence:
                cases_with_pr_evidence.append(case)
            elif case.evidence:
                skipped_no_pr_url += 1

            if limit and len(cases_with_pr_evidence) >= limit:
                break

        self.stdout.write(
            f"  - Cases with unmapped press releases: {len(cases_with_pr_evidence)}"
        )
        self.stdout.write(
            f"  - Cases already mapped (skipped): {skipped_already_mapped}"
        )
        self.stdout.write(
            f"  - Cases without press release URLs (skipped): {skipped_no_pr_url}"
        )

        return cases_with_pr_evidence

    def process_case(self, case: Case, dry_run: bool, source_lookup: dict):
        """Process a single case and map its press release evidence to actual files.

        Args:
            case: The Case instance to process
            dry_run: Whether to run in dry-run mode
            source_lookup: Dict mapping source_id to DocumentSource for efficient lookups
        """
        self.stats["cases_processed"] += 1
        self.stdout.write(
            f"\n[{self.stats['cases_processed']}] Processing: {case.case_id} - {case.title[:80]}..."
        )

        if not case.evidence:
            self.stats["cases_skipped"] += 1
            self.stdout.write(self.style.WARNING("  ⊘ Skipped: No evidence"))
            return

        updated_evidence = []
        evidence_changed = False

        for evidence_entry in case.evidence:
            source_id = evidence_entry.get("source_id")
            if not source_id:
                updated_evidence.append(evidence_entry)
                continue

            source = source_lookup.get(source_id)
            if not source:
                # Source doesn't exist, keep evidence as is
                updated_evidence.append(evidence_entry)
                continue

            # Check if this is a press release source
            press_release_url = next(
                (url for url in source.url_links if "ciaa.gov.np/pressrelease/" in url),
                None,
            )

            if not press_release_url:
                # Not a press release source, keep as is
                updated_evidence.append(evidence_entry)
                continue

            # Extract press_id from URL
            press_id = self.extract_press_id(press_release_url)
            if not press_id:
                self.stdout.write(
                    self.style.WARNING(
                        f"  ⊘ Could not extract press_id from {press_release_url}"
                    )
                )
                updated_evidence.append(evidence_entry)
                continue

            # Find press release files in index
            pr_data = self.press_release_index.get(press_id)
            if not pr_data or not pr_data.get("files"):
                self.stdout.write(
                    self.style.WARNING(f"  ⊘ No files found for press_id {press_id}")
                )
                updated_evidence.append(evidence_entry)
                continue

            self.stdout.write(
                f"  → Found {len(pr_data['files'])} file(s) for press release {press_id}"
            )

            # Create ONE DocumentSource containing the web URL + all file URLs
            self.stdout.write(
                f"  → Creating single source with web URL + {len(pr_data['files'])} file URL(s)"
            )

            if not dry_run:
                with transaction.atomic():
                    # Collect all file URLs
                    file_urls = []
                    file_names = []
                    for file_data in pr_data["files"]:
                        file_url = file_data.get("url")
                        file_name = file_data.get("file_name", "")
                        if file_url:
                            file_urls.append(file_url)
                            file_names.append(file_name)

                    if file_urls:
                        # Create or update single DocumentSource with all URLs
                        combined_source, created, updated = (
                            self.get_or_create_press_release_source(
                                press_release_url=press_release_url,
                                file_urls=file_urls,
                                title=pr_data.get("title", "CIAA Press Release"),
                                publication_date=pr_data.get("publication_date"),
                                file_names=file_names,
                            )
                        )
                        if created:
                            self.stats["sources_created"] += 1
                        elif updated:
                            self.stats["sources_updated"] += 1
                        # else: reused unchanged — no counter change

                        # Add single evidence entry
                        file_count = len(file_urls)
                        updated_evidence.append(
                            {
                                "source_id": combined_source.source_id,
                                "description": f"CIAA Press Release ({file_count} document{'s' if file_count > 1 else ''})",
                            }
                        )
                        evidence_changed = True
                        self.stdout.write(
                            self.style.SUCCESS(
                                f"    ✓ Created/updated source {combined_source.source_id} with {file_count} file(s)"
                            )
                        )
                    else:
                        # No valid file URLs found - preserve original evidence
                        self.stdout.write(
                            self.style.WARNING(
                                f"  ⊘ No valid file URLs found for press release {press_id}, preserving original evidence"
                            )
                        )
                        updated_evidence.append(evidence_entry)
            else:
                # Dry-run mode - mirror real-run logic
                # Build file_urls same as real-run
                file_urls = [f.get("url") for f in pr_data["files"] if f.get("url")]
                file_count = len(file_urls)

                # If no valid file URLs, log warning and preserve original evidence
                if not file_urls:
                    self.stdout.write(
                        self.style.WARNING(
                            f"  ⊘ No valid file URLs found for press release {press_id}, preserving original evidence"
                        )
                    )
                    updated_evidence.append(evidence_entry)
                    continue

                # Check if source already exists (same logic as get_or_create_press_release_source)
                if connection.vendor == "postgresql":
                    existing = DocumentSource.objects.filter(
                        url__contains=[press_release_url], is_deleted=False
                    ).first()
                else:
                    existing = None
                    candidates = DocumentSource.objects.filter(
                        url__icontains=press_release_url, is_deleted=False
                    )
                    for source in candidates:
                        if press_release_url in source.url_links:
                            existing = source
                            break

                if existing:
                    # Check if it would need updating
                    existing_urls = existing.url_links
                    needs_update = any(
                        file_url not in existing_urls for file_url in file_urls
                    )

                    if needs_update:
                        self.stats["sources_updated"] += 1
                        self.stdout.write(
                            f"    [DRY RUN] Would update existing source with {file_count} file URL(s)"
                        )
                    else:
                        self.stdout.write(
                            "    [DRY RUN] Would reuse existing source (no changes needed)"
                        )
                else:
                    self.stats["sources_created"] += 1
                    self.stdout.write(
                        f"    [DRY RUN] Would create source with web URL + {file_count} file URL(s)"
                    )

                for file_data in pr_data["files"]:
                    if file_data.get("url"):  # Only show files with valid URLs
                        file_name = file_data.get("file_name", "")
                        self.stdout.write(f"      - {file_name}")

                updated_evidence.append(
                    {
                        "source_id": f"dry-run-pr-{press_id}",
                        "description": f"CIAA Press Release ({file_count} document{'s' if file_count > 1 else ''})",
                    }
                )
                evidence_changed = True

        # Update case evidence if changed
        if evidence_changed:
            if not dry_run:
                case.evidence = updated_evidence
                case.save(update_fields=["evidence"])
                self.stats["evidence_updated"] += 1
                self.stats["cases_fixed"] += 1
                self.stdout.write(
                    self.style.SUCCESS(
                        f"  ✓ Mapped: Updated evidence with {len(updated_evidence)} source(s)"
                    )
                )
            else:
                # Dry-run mode - still track what would happen
                self.stats["evidence_updated"] += 1
                self.stats["cases_fixed"] += 1
                self.stdout.write(
                    self.style.SUCCESS(
                        f"  [DRY RUN] Would update evidence with {len(updated_evidence)} source(s)"
                    )
                )
        else:
            self.stats["cases_skipped"] += 1
            self.stdout.write(self.style.WARNING("  ⊘ Skipped: No changes needed"))

    def extract_press_id(self, url: str) -> Optional[int]:
        """Extract press_id from CIAA press release URL."""
        try:
            # URL format: https://ciaa.gov.np/pressrelease/3345
            parts = url.rstrip("/").split("/")
            return int(parts[-1])
        except (ValueError, IndexError):
            return None

    def get_or_create_press_release_source(
        self,
        press_release_url: str,
        file_urls: list[str],
        title: str,
        publication_date: Optional[str],
        file_names: list[str],
    ) -> tuple[DocumentSource, bool, bool]:
        """Get or create DocumentSource for a press release with all its files.

        Creates a single DocumentSource containing:
        - The original web URL (e.g., https://ciaa.gov.np/pressrelease/3294)
        - All file URLs (e.g., PDF/DOCX files from NGM bucket)

        Args:
            press_release_url: The CIAA press release web page URL
            file_urls: List of file URLs from NGM bucket
            title: Press release title
            publication_date: Publication date in BS format (YYYY-MM-DD)
            file_names: List of file names (for logging)

        Returns:
            tuple: (DocumentSource, created, updated) where:
                - created is True if a new source was created
                - updated is True if an existing source was modified
                - both are False if an existing source was reused unchanged
        """
        # Check if source already exists by press release URL (database-agnostic)
        if connection.vendor == "postgresql":
            existing = DocumentSource.objects.filter(
                url__contains=[press_release_url], is_deleted=False
            ).first()
        else:
            # Fallback for SQLite and other databases
            existing = None
            candidates = DocumentSource.objects.filter(
                url__icontains=press_release_url, is_deleted=False
            )
            for source in candidates:
                if press_release_url in source.url_links:
                    existing = source
                    break

        if existing:
            # Check if existing source needs to be updated with file URLs
            existing_links = existing.url_links
            needs_update = any(file_url not in existing_links for file_url in file_urls)

            if needs_update:
                # Build complete URL list: merge new URLs with existing ones
                # Start with existing URLs to preserve any prior URLs
                url_list = list(existing.url) if isinstance(existing.url, list) else []

                # Encode and add press release web URL first (ensure it's at the beginning)
                if press_release_url and str(press_release_url).strip():
                    pr_url_str = str(press_release_url).strip()
                    parsed = urllib.parse.urlsplit(pr_url_str)
                    encoded_path = urllib.parse.quote(parsed.path, safe="/")
                    encoded_query = urllib.parse.quote(parsed.query, safe="=&")
                    encoded_fragment = urllib.parse.quote(parsed.fragment, safe="")
                    encoded_url = urllib.parse.urlunsplit(
                        (
                            parsed.scheme,
                            parsed.netloc,
                            encoded_path,
                            encoded_query,
                            encoded_fragment,
                        )
                    )
                    # Remove if already exists and prepend to ensure it's first
                    if encoded_url in url_list:
                        url_list.remove(encoded_url)
                    url_list.insert(0, encoded_url)

                # Add all file URLs (skip duplicates)
                for file_url in file_urls:
                    if file_url and str(file_url).strip():
                        file_url_str = str(file_url).strip()
                        parsed = urllib.parse.urlsplit(file_url_str)
                        encoded_path = urllib.parse.quote(parsed.path, safe="/")
                        encoded_query = urllib.parse.quote(parsed.query, safe="=&")
                        encoded_fragment = urllib.parse.quote(parsed.fragment, safe="")
                        encoded_url = urllib.parse.urlunsplit(
                            (
                                parsed.scheme,
                                parsed.netloc,
                                encoded_path,
                                encoded_query,
                                encoded_fragment,
                            )
                        )
                        # Only add if not already present
                        if encoded_url not in url_list:
                            url_list.append(encoded_url)

                # Update existing source with complete URL list
                existing.url = url_list

                # Update description with file count
                file_count = len(file_urls)
                existing.description = f"CIAA Press Release with {file_count} document{'s' if file_count > 1 else ''}: {press_release_url}"[
                    :500
                ]

                existing.save(update_fields=["url", "description"])
                logger.info(
                    f"Updated existing source {existing.source_id} with {len(file_urls)} file URL(s)"
                )
                return existing, False, True  # created=False, updated=True
            else:
                logger.debug(
                    f"Reusing existing source: {existing.source_id} (already has all file URLs)"
                )
                return existing, False, False  # created=False, updated=False

        source_type = SourceType.LEGAL_PROCEDURAL

        # Parse publication date (BS format: YYYY-MM-DD)
        pub_date = None
        if publication_date:
            try:
                parts = publication_date.split("-")
                if len(parts) == 3:
                    year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                    pub_date = nepalidate(year, month, day).to_datetime().date()
            except Exception as e:
                logger.warning(
                    f"Failed to parse publication date {publication_date}: {e}"
                )

        # Build URL list: web URL first, then all file URLs
        url_list = []

        # Add press release web URL first
        if press_release_url and str(press_release_url).strip():
            pr_url_str = str(press_release_url).strip()
            parsed = urllib.parse.urlsplit(pr_url_str)
            encoded_path = urllib.parse.quote(parsed.path, safe="/")
            encoded_query = urllib.parse.quote(parsed.query, safe="=&")
            encoded_fragment = urllib.parse.quote(parsed.fragment, safe="")
            encoded_url = urllib.parse.urlunsplit(
                (
                    parsed.scheme,
                    parsed.netloc,
                    encoded_path,
                    encoded_query,
                    encoded_fragment,
                )
            )
            url_list.append(encoded_url)

        # Add all file URLs
        for file_url in file_urls:
            if file_url and str(file_url).strip():
                file_url_str = str(file_url).strip()
                parsed = urllib.parse.urlsplit(file_url_str)
                encoded_path = urllib.parse.quote(parsed.path, safe="/")
                encoded_query = urllib.parse.quote(parsed.query, safe="=&")
                encoded_fragment = urllib.parse.quote(parsed.fragment, safe="")
                encoded_url = urllib.parse.urlunsplit(
                    (
                        parsed.scheme,
                        parsed.netloc,
                        encoded_path,
                        encoded_query,
                        encoded_fragment,
                    )
                )
                url_list.append(encoded_url)

        if not url_list:
            logger.error(f"No valid URLs for press release {press_release_url}")
            raise ValueError(f"No valid URLs for press release {press_release_url}")

        # Create description with file count
        file_count = len(file_urls)
        description = f"CIAA Press Release with {file_count} document{'s' if file_count > 1 else ''}: {press_release_url}"

        # Create new source (save() will generate source_id and validate)
        source = DocumentSource.objects.create(
            title=title[:300],
            description=description[:500],
            source_type=source_type,
            url=url_list,
            publication_date=pub_date,
        )

        logger.info(
            f"Created new source: {source.source_id} for press release with {file_count} file(s)"
        )
        return source, True, False  # created=True, updated=False

    def print_summary(self, dry_run: bool):
        """Print summary statistics."""
        self.stdout.write("\n" + "=" * 60)
        self.stdout.write(
            self.style.WARNING(f"{'[DRY RUN] ' if dry_run else ''}SUMMARY")
        )
        self.stdout.write("=" * 60)
        self.stdout.write(f"Cases processed:     {self.stats['cases_processed']}")
        self.stdout.write(
            self.style.SUCCESS(f"✓ Cases mapped:      {self.stats['cases_fixed']}")
        )
        self.stdout.write(
            self.style.WARNING(f"⊘ Cases skipped:     {self.stats['cases_skipped']}")
        )
        self.stdout.write(f"Sources created:     {self.stats['sources_created']}")
        self.stdout.write(f"Sources updated:     {self.stats['sources_updated']}")
        self.stdout.write(f"Evidence updated:    {self.stats['evidence_updated']}")
        if self.stats["errors"] > 0:
            self.stdout.write(
                self.style.ERROR(f"✗ Errors:            {self.stats['errors']}")
            )
        self.stdout.write("=" * 60)

        if dry_run:
            self.stdout.write(
                self.style.WARNING(
                    "\nThis was a dry run. No changes were made to the database."
                )
            )
            self.stdout.write("Run without --dry-run to apply changes.")
