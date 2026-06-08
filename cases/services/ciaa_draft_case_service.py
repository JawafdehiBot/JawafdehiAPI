"""Service for creating draft cases from CIAA JSON data with deduplication."""

import logging
import urllib.parse
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from django.db import connection, transaction
from nepali.datetime import nepalidate

from cases.models import (
    Case,
    CaseEntityRelationship,
    CaseState,
    CaseType,
    DocumentSource,
    JawafEntity,
    RelationshipType,
    SourceType,
)

logger = logging.getLogger(__name__)


@dataclass
class ImportResult:
    status: str  # "created" | "skipped" | "failed"
    case_id: Optional[str] = None
    message: str = ""
    errors: list[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class CIAADraftCaseService:
    """Service for creating draft cases from CIAA JSON data with deduplication."""

    def __init__(self):
        """Initialize the service with empty caches."""
        self.entity_cache = {}
        self.stats = {
            "entities_created": 0,
            "entities_reused": 0,
            "sources_created": 0,
            "sources_reused": 0,
        }

    def import_case(self, ciaa_json: dict, dry_run: bool = False) -> ImportResult:
        """Import a single CIAA case from JSON. Returns ImportResult with status."""
        try:
            errors = self.validate_ciaa_json(ciaa_json)
            if errors:
                return ImportResult(
                    status="failed", message="JSON validation failed", errors=errors
                )

            case_data = self.map_json_to_case(ciaa_json)
            existing_case = self.check_case_exists(case_data.get("court_cases", []))

            if existing_case:
                return ImportResult(
                    status="skipped",
                    case_id=existing_case.case_id,
                    message=f"Case already exists: {case_data['court_cases']}",
                )

            if dry_run:
                return ImportResult(
                    status="created",
                    message=f"Would create case: {case_data['title']} (dry-run)",
                )

            with transaction.atomic():
                case = Case.objects.create(
                    case_type=case_data["case_type"],
                    state=case_data["state"],
                    title=case_data["title"][:200],
                    case_start_date=case_data.get("case_start_date"),
                    case_end_date=case_data.get("case_end_date"),
                    court_cases=case_data.get("court_cases"),
                    notes=case_data.get("notes", ""),
                    missing_details=case_data.get("missing_details"),
                )
                logger.debug(f"Created case: {case.case_id} - {case.title}")

                self.create_defendants(
                    ciaa_json.get("court_case", {}).get("defendants", []), case
                )
                self.create_document_sources(ciaa_json, case)

            return ImportResult(
                status="created",
                case_id=case.case_id,
                message=f"Created case: {case.title}",
            )

        except Exception as e:
            case_no = ciaa_json.get("case_no", "Unknown")
            error_msg = f"Import failed: {e!s}"
            logger.exception(f"Import failed for case {case_no}: {e!s}")
            return ImportResult(status="failed", message=error_msg, errors=[f"{e!s}"])

    def validate_ciaa_json(self, json_dict: dict) -> list[str]:
        """Validate CIAA JSON structure. Returns list of error messages."""
        errors = []
        if not json_dict.get("case_no"):
            errors.append("Missing required field: case_no")
        if not json_dict.get("case_title"):
            errors.append("Missing required field: case_title")
        if "court_case" not in json_dict:
            errors.append("Missing required field: court_case")

        meta = json_dict.get("meta", {})
        if "match_status" not in meta:
            errors.append("Missing required field: meta.match_status")
        elif meta["match_status"] not in ["confirmed", "needs_review", "unmatched"]:
            errors.append(f"Invalid match_status '{meta['match_status']}'")

        # Validate that CIAA cases have a special:* court case reference
        court_case = json_dict.get("court_case", {})
        court = court_case.get("court")
        case_no = court_case.get("case_no")
        if court and case_no:
            court_case_ref = f"{court}:{case_no}"
            if not court_case_ref.startswith("special:"):
                errors.append(
                    f"Missing required CIAA idempotency key: court_case must be a special:* reference, got '{court_case_ref}'"
                )

        return errors

    def _primary_ciaa_court_case(self, court_cases: list[str]) -> Optional[str]:
        """Extract the primary CIAA court case reference (special:*) for idempotency checks.

        CIAA cases are expected to have exactly one special:* entry in court_cases.
        This is the unique idempotency key and should be used for duplicate detection.

        Args:
            court_cases: List of court case references

        Returns:
            The special:* reference if found, otherwise None
        """
        for cc in court_cases:
            if cc.startswith("special:"):
                return cc
        return None

    def check_case_exists(self, court_cases: list[str]) -> Optional[Case]:
        """Check if case already exists by court_cases field. Returns existing Case or None."""
        primary = self._primary_ciaa_court_case(court_cases)
        if not primary:
            return None

        if connection.vendor == "postgresql":
            return Case.objects.filter(court_cases__contains=[primary]).first()
        else:
            for case in Case.objects.exclude(court_cases__isnull=True):
                if isinstance(case.court_cases, list) and primary in case.court_cases:
                    return case
        return None

    def map_json_to_case(self, ciaa_json: dict) -> dict:
        """Map CIAA JSON fields to Case model fields. Returns dict with case data."""
        case_data = {}
        case_title = ciaa_json.get("case_title", "")
        case_no = ciaa_json.get("case_no", "")

        title_base = case_title[:180] if len(case_title) > 180 else case_title
        case_data["title"] = (
            f"{title_base} ({case_no})"[:200] if case_no else title_base[:200]
        )
        case_data["case_type"] = CaseType.CORRUPTION
        case_data["state"] = CaseState.DRAFT

        court_case = ciaa_json.get("court_case", {})

        # Parse dates
        if reg_date := court_case.get("registration_date_ad"):
            try:
                case_data["case_start_date"] = datetime.strptime(
                    reg_date, "%Y-%m-%d"
                ).date()
            except (ValueError, TypeError):
                case_data["case_start_date"] = None
        else:
            case_data["case_start_date"] = None

        if faisala_date := court_case.get("faisala_date_ad"):
            try:
                case_data["case_end_date"] = datetime.strptime(
                    faisala_date, "%Y-%m-%d"
                ).date()
            except (ValueError, TypeError):
                case_data["case_end_date"] = self.convert_bs_to_ad(
                    court_case.get("faisala_date_bs")
                )
        else:
            case_data["case_end_date"] = self.convert_bs_to_ad(
                court_case.get("faisala_date_bs")
            )

        # Build court_cases list
        court_cases = []
        if (
            court_case
            and (court := court_case.get("court"))
            and (cn := court_case.get("case_no"))
        ):
            court_cases.append(f"{court}:{cn}")

        if appealed := ciaa_json.get("appealed_case"):
            if (ac := appealed.get("court")) and (acn := appealed.get("case_no")):
                court_cases.append(f"{ac}:{acn}")

        case_data["court_cases"] = court_cases
        case_data["missing_details"] = (
            "This case has match_status='needs_review' and requires verification."
            if ciaa_json.get("meta", {}).get("match_status") == "needs_review"
            else None
        )

        return case_data

    def convert_bs_to_ad(self, bs_date_str: str) -> Optional[datetime]:
        """Convert Bikram Sambat date string to AD date. Returns date or None."""
        if not bs_date_str:
            return None
        try:
            devanagari_to_ascii = str.maketrans("०१२३४५६७८९", "0123456789")
            normalized = bs_date_str.translate(devanagari_to_ascii).replace("/", "-")
            parts = normalized.split("-")
            if len(parts) != 3:
                return None
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            return nepalidate(year, month, day).to_datetime().date()
        except Exception:
            return None

    def create_defendants(
        self, defendants: list[dict], case: Case
    ) -> list[JawafEntity]:
        """Create JawafEntity records for defendants and link to case. Returns list of entities."""
        entities = []
        for defendant in defendants:
            if entity := self.get_or_create_entity(defendant.get("name", "")):
                CaseEntityRelationship.objects.get_or_create(
                    case=case,
                    entity=entity,
                    relationship_type=RelationshipType.ACCUSED,
                    defaults={"notes": ""},
                )
                entities.append(entity)
        return entities

    def get_or_create_entity(self, name: str) -> Optional[JawafEntity]:
        """Get or create JawafEntity with deduplication by name. Returns entity or None."""
        if not name or not name.strip():
            return None

        name = name.strip()
        if name in self.entity_cache:
            return self.entity_cache[name]

        entity = JawafEntity.objects.filter(display_name__iexact=name).first()
        if entity:
            self.stats["entities_reused"] += 1
            logger.debug(f"Reusing entity: {name}")
        else:
            entity = JawafEntity.objects.create(display_name=name)
            self.stats["entities_created"] += 1
            logger.debug(f"Created entity: {name}")

        self.entity_cache[name] = entity
        return entity

    def create_document_sources(
        self, ciaa_json: dict, case: Case
    ) -> list[DocumentSource]:
        """Create DocumentSource records from press releases, charge sheets, and court orders. Returns list of sources."""
        sources = []
        evidence = []

        # Process press releases
        for pr in ciaa_json.get("ciaa", {}).get("press_releases", []):
            source_data = {
                "title": pr.get("title", "CIAA Press Release")[:300],
                "url": pr.get("url", ""),
                "source_type": SourceType.LEGAL_PROCEDURAL,
                "publication_date": self.convert_bs_to_ad(pr.get("date", "")),
            }
            if source := self.get_or_create_source(source_data):
                sources.append(source)
                evidence.append(
                    {
                        "source_id": source.source_id,
                        "description": f"CIAA Press Release (ID: {pr.get('release_id', 'N/A')})",
                    }
                )

        # Process abhiyogPatras (AG charge sheets)
        for ap in ciaa_json.get("ciaa", {}).get("abhiyogPatras", []):
            pdf_url = ap.get("pdf_url", "")
            encoded_url = pdf_url
            if pdf_url and isinstance(pdf_url, str) and pdf_url.strip():
                parsed = urllib.parse.urlsplit(pdf_url.strip())
                encoded_url = urllib.parse.urlunsplit(
                    (
                        parsed.scheme,
                        parsed.netloc,
                        urllib.parse.quote(parsed.path, safe="/"),
                        urllib.parse.quote(parsed.query, safe="=&"),
                        urllib.parse.quote(parsed.fragment, safe=""),
                    )
                )
            source_data = {
                "title": ap.get("title", "AG Charge Sheet")[:300],
                "url": [encoded_url] if encoded_url else [],
                "source_type": SourceType.OFFICIAL_GOVERNMENT,
                "publication_date": self.convert_bs_to_ad(ap.get("filing_date", "")),
            }
            if source := self.get_or_create_source(source_data):
                sources.append(source)
                case_no = ap.get("case_number", "N/A")
                evidence.append(
                    {
                        "source_id": source.source_id,
                        "description": f"AG Charge Sheet - {case_no}",
                    }
                )

        # Process faisala links
        for idx, faisala_url in enumerate(
            ciaa_json.get("court_case", {}).get("faisala_link", []), 1
        ):
            if faisala_url:
                source_data = {
                    "title": f"Court Order - {ciaa_json.get('case_no', 'Unknown')}",
                    "url": faisala_url,
                    "source_type": SourceType.LEGAL_COURT_ORDER,
                }
                if source := self.get_or_create_source(source_data):
                    sources.append(source)
                    evidence.append(
                        {
                            "source_id": source.source_id,
                            "description": f"Court Order/Verdict (Document {idx})",
                        }
                    )

        case.evidence = evidence
        case.save()
        return sources

    def get_or_create_source(self, source_data: dict) -> Optional[DocumentSource]:
        """Get or create DocumentSource with deduplication by URL. Returns source or None."""
        url_raw = source_data.get("url", "")
        url_list = (
            [u.strip() for u in url_raw if isinstance(u, str) and u.strip()]
            if isinstance(url_raw, list)
            else (
                [url_raw.strip()]
                if isinstance(url_raw, str) and url_raw.strip()
                else []
            )
        )

        title = source_data.get("title", "").strip()
        if not title:
            return None

        source_type = source_data.get("source_type", "")

        # Try to find existing source by URL
        if url_list:
            if connection.vendor == "postgresql":
                for url in url_list:
                    if (
                        source := DocumentSource.objects.filter(
                            is_deleted=False, url__contains=[{"link": url}]
                        )
                        .only("source_id", "title")
                        .first()
                    ):
                        self.stats["sources_reused"] += 1
                        logger.debug(f"Reusing source: {title}")
                        return source
            else:
                for url in url_list:
                    for source in DocumentSource.objects.filter(is_deleted=False).only(
                        "source_id", "title", "url"
                    ):
                        if url in source.url_links:
                            self.stats["sources_reused"] += 1
                            logger.debug(f"Reusing source: {title}")
                            return source

        # Try to find by title (only if no URL provided)
        if not url_list:
            if source := DocumentSource.objects.filter(
                title=title, is_deleted=False
            ).first():
                self.stats["sources_reused"] += 1
                logger.debug(f"Reusing source: {title}")
                return source

        # Create new source
        publication_date = source_data.get("publication_date")
        source = DocumentSource.objects.create(
            title=title,
            url=url_list,
            source_type=source_type,
            publication_date=publication_date,
        )
        self.stats["sources_created"] += 1
        logger.debug(f"Created source: {title}")
        return source
