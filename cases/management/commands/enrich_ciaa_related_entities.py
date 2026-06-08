"""
Management command to enrich CIAA DRAFT cases with related and location entities.

Extracts entities via LLM from press releases and court orders and creates JawafEntity
and CaseEntityRelationship records.

Usage::

    python manage.py enrich_ciaa_related_entities --dry-run
    python manage.py enrich_ciaa_related_entities --limit 10
    python manage.py enrich_ciaa_related_entities --llm-model claude-sonnet-4-5 --verbose
"""

import logging
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import requests
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from cases.management.commands._enrich_utils import (
    call_llm,
    convert_to_markdown,
    parse_extraction_response,
    resolve_api_key,
)
from cases.models import (
    Case,
    CaseEntityRelationship,
    CaseState,
    DocumentSource,
    JawafEntity,
    RelationshipType,
    SourceType,
)
from cases.services.priority_case_loader import filter_by_priority, load_priority_cases

logger = logging.getLogger(__name__)


def _parse_accused_notes(response_text: str):
    """Extract the accused_notes array from the LLM response JSON.

    The LLM returns a combined object:
      {"entities": [...], "accused_notes": [...]}

    Returns a list of {"name": ..., "notes": ...} dicts, or empty list if absent.
    """
    entries = parse_extraction_response(response_text, {"accused_notes"})
    if not entries:
        return []
    return [
        item
        for item in entries
        if isinstance(item, dict) and item.get("name") and item.get("notes")
    ]


# ------------------------------------------------------------------
# Slicing constants
# ------------------------------------------------------------------
COURT_ORDER_FULL_THRESHOLD = 8_000
COURT_ORDER_HEAD_CHARS = 4_000  # fallback: entities front-loaded in header
COURT_ORDER_TAIL_CHARS = 2_000  # fallback: tail when no ठहर खण्ड
COURT_ORDER_THAHAR_CHARS = 12_000  # max chars from ठहर खण्ड when found

PRESS_RELEASE_CHARS = 3_000  # with court order: 3k press release + ठहर खण्ड
PRESS_RELEASE_CHARS_NO_COURT = (
    18_000  # no court order: use much more — it's the only source
)

PROMPT_HARD_MAX = 25_000  # no court order case: 18k press release + system prompt

DOCUMENT_FORMAT_PRIORITY = {".docx": 4, ".doc": 3, ".pdf": 2}


SYSTEM_PROMPT = """You are a Nepali legal research assistant helping to build a public transparency database of court cases.
Analyze the provided Nepali legal documents (press release and/or court order excerpts) and extract structured data.

You must extract THREE things in a single response:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 1 — LOCATION ENTITIES (relationship_type="location")
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Extract the district(s), municipality, or province WHERE THE CASE EVENTS occurred
or where the key assets/funds at issue are located.

STRICT RULES:
- Extract ONLY where the case events happened or where the assets are.
- DO NOT extract accused home addresses, birthplaces, or permanent addresses.
- DO NOT extract the location of courts or government inquiry offices.
- Extract 1 location for simple cases. Extract 2-3 only if the case genuinely spans
  multiple districts.
- Leave notes BLANK ("") for all location entities.
- The entity_name should include context in the format: "Organisation/Activity - Location"

Examples of CORRECT location entity names:
- "साझा भण्डार सहकारी - सुर्खेत जिल्ला"
- "स्वास्थ्य उपकरण खरिद - जनकपुरधाम"
- "भरत ताल निर्माण परियोजना - सर्लाही जिल्ला"
- "नापी कार्यालय - खैरहनी नगरपालिका"
- (if no specific activity context, just the location name: "काठमाडौं")

Examples of WRONG location names:
- "तनहुँ जिल्ला" ← accused home address, SKIP
- "काठमाडौं" ← if only reason is court/CIAA office, SKIP

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 2 — RELATED ENTITIES (relationship_type="related")
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Any person or organization connected to the case BEYOND the primary accused.
Extract ALL of these categories that appear in the documents:

  GOVERNMENT BODIES — ministry, department, municipality, office whose funds were
  misused or where the accused worked.
  Examples: "जलश्रोत तथा सिँचाइ विभाग"  notes: "आरोपी कार्यरत रहेको सरकारी निकाय"
            "राष्ट्रिय सूचना प्रविधि केन्द्र"  notes: "खरिद प्रक्रियामा संलग्न सरकारी निकाय"

  COMPANIES/CONTRACTORS — firms, JVs, cooperatives, suppliers, foreign companies.
  Examples: "कल्पवृक्ष-कोहिनूर जे.भी."  notes: "ठेक्का प्राप्त गर्ने संयुक्त उद्यम"
            "UOB Singapore बैंक"  notes: "Singapore स्थित बैंक, रकम हस्तान्तरणमा प्रयोग"

  FAMILY MEMBERS — spouse, children, relatives holding assets.
  Example: "श्रृजना गिरी"  notes: "आरोपितको श्रीमती, सम्पत्ति हस्तान्तरण गरिएको"

  CO-DEFENDANTS/ASSOCIATES — secondary actors, facilitators, middlemen.
  Example: "नानी काजी थापा"  notes: "घुस लेनदेनमा सहयोग"

  INVESTIGATING/PROSECUTING BODIES — DO NOT extract the inquiry commission
  (अख्तियार दुरुपयोग अनुसन्धान आयोग) or special attorney office as standalone
  entities — they are present in every case. DO NOT extract individual prosecutors,
  attorneys, judges, or court staff — they are performing standard professional
  duties, not materially connected to the case events.
  Only extract named CIAA investigation officers if they are specifically named
  and their investigation is directly relevant.
  Example: "रविन्द्र कुमार बुढाप्रिथी"  notes: "अनुसन्धान अधिकृत, CIAA"

  WITNESSES/INVESTIGATORS — named inquiry officers, key witnesses.
  Example: "रविन्द्र कुमार बुढाप्रिथी"  notes: "अनुसन्धान अधिकृत, CIAA"

Notes must never be blank for related entities. Always describe the specific connection.
Only extract entities with CONFIRMED connections — not people who were later acquitted.

PRIORITY ORDER: People and organizations DIRECTLY involved in the case events come first.
Generic legal infrastructure (courts, attorney offices) should be skipped unless a
specific named person from those bodies is materially connected.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 3 — ACCUSED NOTES (accused_notes array)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For each primary accused person named in the documents, extract a SHORT note
describing their job title and role. Format: "job title, employer"
Examples:
  "तत्कालीन प्रबन्ध निर्देशक, नेपाल टेलिकम"
  "तत्कालीन नगरप्रमुख, खैरहनी नगरपालिका"
  "नापी अधिकृत, नापी कार्यालय चाबहिल"

Only include primary accused persons. Keep notes under 80 chars.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Output ONLY this JSON object, no other text:
{
  "entities": [
    {
      "entity_name": "Name exactly as in document",
      "relationship_type": "location" or "related",
      "notes": "specific description"
    }
  ],
  "accused_notes": [
    {
      "name": "Accused person name exactly as in document",
      "notes": "job title, employer"
    }
  ]
}
"""


class Command(BaseCommand):
    help = "Extract related and location entities from CIAA cases using LLM"

    def __init__(self):
        super().__init__()
        self.stats = {
            "cases_processed": 0,
            "cases_skipped": 0,
            "cases_enriched": 0,
            "entities_created": 0,
            "relationships_created": 0,
            "accused_notes_updated": 0,
        }
        self._source_lookup = {}

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Preview without saving to database",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=None,
            help="Process only N cases",
        )
        parser.add_argument(
            "--llm-model",
            type=str,
            default=os.environ.get(
                "JAWAFDEHI_ALLEGATION_MODEL",
                "kr/claude-haiku-4.5,kr/claude-sonnet-4.5",
            ),
            help="Comma-separated list of models to try in order (fallback chain)",
        )
        parser.add_argument(
            "--llm-base-url",
            type=str,
            default=os.environ.get(
                "JAWAFDEHI_LLM_PROXY_URL", "https://llm-proxy.jawafdehi.org/v1"
            ),
        )
        parser.add_argument(
            "--llm-api-key",
            type=str,
            help="Override API key from env",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Re-process cases that already have related entities",
        )
        parser.add_argument(
            "--priority",
            action="store_true",
            help="Only process priority cases from priority_cases.json",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Detailed logging",
        )

    def handle(self, *args, **options):
        limit = options["limit"]
        force = options["force"]
        is_verbose = options["verbose"]

        if limit is not None and limit < 0:
            raise CommandError("--limit must be >= 0")

        if is_verbose:
            logger.setLevel(logging.DEBUG)

        # Always suppress urllib3 connection noise — it's not useful at any verbosity level
        logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)

        api_key = resolve_api_key(options["llm_api_key"])
        if not api_key:
            raise CommandError("LLM API key not found in env or args")

        qs = Case.objects.filter(state=CaseState.DRAFT)

        if options["priority"]:
            priority_list = load_priority_cases()
            logger.info(
                "Priority mode: loaded %d case numbers across all fiscal years",
                len(priority_list),
            )
            qs = filter_by_priority(qs, priority_list)

        if not force:
            already_enriched_ids = CaseEntityRelationship.objects.filter(
                relationship_type=RelationshipType.RELATED
            ).values_list("case_id", flat=True)
            qs = qs.exclude(id__in=already_enriched_ids)

        qs = qs.order_by("case_id")

        cases = list(qs)
        if limit is not None:
            cases = cases[:limit]

        self.stdout.write(f"Found {len(cases)} cases to process")

        self._fetch_source_cache(cases)

        with requests.Session() as session:
            for idx, case in enumerate(cases, 1):
                self.stats["cases_processed"] += 1
                self.stdout.write(
                    f"[{idx}/{len(cases)}] Processing case {case.case_id}..."
                )
                self._process_case(case, options, api_key, session, is_verbose)

        self.stdout.write(self.style.SUCCESS(f"Finished. Stats: {self.stats}"))

    # ------------------------------------------------------------------
    # Source lookup
    # ------------------------------------------------------------------

    def _fetch_source_cache(self, cases):
        """Pre-fetch DocumentSource objects for all evidence references."""
        source_ids = set()
        for case in cases:
            for item in case.evidence or []:
                if isinstance(item, dict) and isinstance(item.get("source_id"), str):
                    if item["source_id"].strip():
                        source_ids.add(item["source_id"])
        self._source_lookup = {
            source.source_id: source
            for source in DocumentSource.objects.filter(
                source_id__in=source_ids, is_deleted=False
            ).prefetch_related("uploaded_files")
        }
        logger.debug("Cached %d DocumentSource records", len(self._source_lookup))

    def _get_evidence_sources(self, case):
        """Return DocumentSource objects referenced in case.evidence."""
        sources = []
        for item in case.evidence or []:
            if isinstance(item, dict) and isinstance(item.get("source_id"), str):
                source = self._source_lookup.get(item["source_id"])
                if source is not None:
                    sources.append(source)
        return sources

    # ------------------------------------------------------------------
    # Press release detection (Problem 1)
    # ------------------------------------------------------------------

    def _is_press_release_source(self, source):
        """Check if a source should be treated as a CIAA press release.

        Matches when:
        - source_type is OFFICIAL_GOVERNMENT, OR
        - description contains "CIAA Press Release", OR
        - any URL contains "ciaa.gov.np/pressrelease"
        """
        if source.source_type == SourceType.OFFICIAL_GOVERNMENT:
            return True

        description = (source.description or "").lower()
        if "ciaa press release" in description:
            return True

        urls = [url.strip() for url in source.url_links if url.strip()]
        for url in urls:
            if "ciaa.gov.np/pressrelease" in url.lower():
                return True

        return False

    def _score_source_for_press_release(self, source):
        """Score a source for suitability as a CIAA press release.

        Higher score = better match. Returns 0 for non-press-release sources.
        """
        if not self._is_press_release_source(source):
            return 0

        corpus_parts = [
            source.title or "",
            source.description or "",
            source.uploaded_filename or "",
        ]
        urls = [url.strip() for url in source.url_links if url.strip()]
        corpus_parts.append(" ".join(urls))

        for uploaded in source.uploaded_files.all():
            corpus_parts.append(uploaded.filename or Path(uploaded.file.name).name)

        corpus = " ".join(corpus_parts).lower()

        score = 0

        # Direct source_type match
        if source.source_type == SourceType.OFFICIAL_GOVERNMENT:
            score += 5

        # Press release keywords
        press_keywords = [
            "press release",
            "pressrelease",
            "press-release",
            "प्रेस विज्ञप्ति",
            "विज्ञप्ति",
        ]
        if any(kw in corpus for kw in press_keywords):
            score += 8

        # CIAA-specific keywords
        ciaa_keywords = ["ciaa", "अख्तियार"]
        if any(kw in corpus for kw in ciaa_keywords):
            score += 3

        # Direct CIAA press release URL
        if any("ciaa.gov.np/pressrelease" in u.lower() for u in urls):
            score += 10

        # Has DOC/DOCX URLs (editable formats)
        if any(u.lower().endswith(".docx") for u in urls):
            score += 4
        elif any(u.lower().endswith(".doc") for u in urls):
            score += 2
        elif any(u.lower().endswith(".pdf") for u in urls):
            score += 1

        return score

    def _get_press_release_source(self, case):
        """Return the best press release source for this case.

        Scores all evidence sources and returns the highest-scoring one.
        """
        sources = self._get_evidence_sources(case)
        if not sources:
            return None

        ranked = sorted(
            ((self._score_source_for_press_release(s), s) for s in sources),
            key=lambda row: row[0],
            reverse=True,
        )
        best_score, best_source = ranked[0]
        return best_source if best_score > 0 else None

    def _get_court_order_source(self, case):
        """Return the best court order source for this case."""
        for source in self._get_evidence_sources(case):
            if source.source_type == SourceType.LEGAL_COURT_ORDER:
                return source
        return None

    # ------------------------------------------------------------------
    # URL deduplication and ranking (Problems 2 & 3)
    # ------------------------------------------------------------------

    @staticmethod
    def _ranked_press_release_urls(source):
        """Return URLs ranked by conversion preference for press releases.

        Priority: DOCX > DOC > PDF > other (e.g., CIAA webpage HTML)
        """
        urls = [url.strip() for url in source.url_links if url.strip()]
        if not urls:
            return []

        scored = []
        for url in urls:
            parsed = urlparse(url)
            suffix = Path(parsed.path).suffix.lower()
            priority = DOCUMENT_FORMAT_PRIORITY.get(suffix, 0)
            # Non-document URLs (like webpage) get lowest priority
            scored.append((priority, url))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Return unique in priority order
        seen = set()
        result = []
        for _priority, url in scored:
            if url not in seen:
                seen.add(url)
                result.append(url)
        return result

    # ------------------------------------------------------------------
    # Document conversion
    # ------------------------------------------------------------------

    def _convert_source_to_markdown(self, source, session, is_press_release=False):
        """Convert a DocumentSource to markdown text.

        For press releases: tries DOCX → DOC → uploaded_file → PDF → webpage
        For court orders: tries uploaded files first, then URLs.
        """
        if is_press_release:
            return self._convert_press_release_to_markdown(source, session)
        return self._convert_court_order_to_markdown(source, session)

    def _convert_press_release_to_markdown(self, source, session):
        """Convert press release source. Priority: DOCX > DOC > uploaded > PDF > HTML."""
        ranked_urls = self._ranked_press_release_urls(source)

        # Try each URL in priority order
        for url in ranked_urls:
            md = convert_to_markdown(url, session)
            if md:
                return md

        # Try uploaded files as fallback
        uploaded_md = self._convert_uploaded_file(source)
        if uploaded_md:
            return uploaded_md

        # Fallback to description
        if source.description and len(source.description.strip()) >= 500:
            return source.description

        return None

    def _convert_court_order_to_markdown(self, source, session):
        """Convert court order source. Uploaded files first, then URLs."""
        uploaded_md = self._convert_uploaded_file(source)
        if uploaded_md:
            return uploaded_md

        urls = [url.strip() for url in source.url_links if url.strip()]
        for url in urls:
            md = convert_to_markdown(url, session)
            if md:
                return md

        if source.description and len(source.description.strip()) >= 500:
            return source.description

        return None

    def _convert_uploaded_file(self, source):
        """Download and convert the best uploaded file for a source via markitdown/likhit."""
        try:
            import likhit  # noqa: F401
            from markitdown import MarkItDown
        except ImportError as exc:
            raise CommandError(
                "markitdown and likhit are required for document conversion."
            ) from exc

        file_field = source.uploaded_file
        if not file_field:
            uploaded_files = list(source.uploaded_files.all())
            if uploaded_files and uploaded_files[0].file:
                file_field = uploaded_files[0].file

        if not file_field:
            return None

        suffix = Path(file_field.name).suffix or ""
        tmp_path = None
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            tmp_path = tmp.name
            with file_field.open("rb") as in_file:
                while True:
                    chunk = in_file.read(8192)
                    if not chunk:
                        break
                    tmp.write(chunk)
            tmp.close()

            converter = MarkItDown(enable_plugins=True)
            result = converter.convert(tmp_path)
            if (
                result
                and result.text_content
                and len(result.text_content.strip()) > 200
            ):
                return result.text_content.strip()
            return None
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Intelligent truncation (Problems 4 & 5)
    # ------------------------------------------------------------------

    @staticmethod
    def _truncate_press_release(text, limit=None):
        """Truncate press release, cutting at the last sentence boundary before limit.

        Cuts at the last Nepali danda (।), newline, or full stop before the limit
        to avoid sending a truncated mid-sentence to the LLM.

        limit defaults to PRESS_RELEASE_CHARS. Pass PRESS_RELEASE_CHARS_NO_COURT
        when no court order is available.
        """
        if not text:
            return text
        if limit is None:
            limit = PRESS_RELEASE_CHARS
        if len(text) <= limit:
            return text

        chunk = text[:limit]

        # Cut at last sentence boundary — prefer Nepali danda, then newline, then period
        for sep in ("।", "\n", ".", "!"):
            idx = chunk.rfind(sep)
            if idx >= limit // 2:  # must use at least half the budget
                return chunk[: idx + 1]

        return chunk

    @staticmethod
    def _truncate_court_order(text):
        """Extract the most entity-rich section from a court order.

        Strategy:
        1. If the document is shorter than COURT_ORDER_FULL_THRESHOLD,
           return it as-is (no truncation needed).
        2. If ठहर खण्ड (verdict section) is present, extract up to
           COURT_ORDER_THAHAR_CHARS from it. This section names all related
           parties, assets, banks, spouses, and contractors explicitly.
        3. If no ठहर खण्ड, fall back to head (COURT_ORDER_HEAD_CHARS)
           + tail (COURT_ORDER_TAIL_CHARS).
        """
        if not text:
            return text

        if len(text) < COURT_ORDER_FULL_THRESHOLD:
            return text

        # Try to find ठहर खण्ड
        thahar_marker = "ठहर खण्ड"
        idx = text.find(thahar_marker)
        if idx != -1:
            thahar_text = text[idx:]
            # Cut at sentence boundary within budget
            limit = COURT_ORDER_THAHAR_CHARS
            if len(thahar_text) <= limit:
                return f"\n\n[...ठहर खण्ड (verdict section)...]\n\n{thahar_text}"
            chunk = thahar_text[:limit]
            for sep in ("।", "\n", ".", "!"):
                sep_idx = chunk.rfind(sep)
                if sep_idx >= limit // 2:
                    chunk = chunk[: sep_idx + 1]
                    break
            return f"\n\n[...ठहर खण्ड (verdict section)...]\n\n{chunk}"

        # Fallback: head + tail
        label_head = "\n\n[...court order header section...]\n\n"
        label_tail = "\n\n[...court order verdict section...]\n\n"
        return (
            label_head
            + text[:COURT_ORDER_HEAD_CHARS]
            + label_tail
            + text[-COURT_ORDER_TAIL_CHARS:]
        )

    @staticmethod
    def _enforce_prompt_budget(parts, is_verbose=False):
        """Ensure combined prompt stays within budget. Truncates largest part if over."""
        combined = "\n\n".join(parts)
        total = len(combined)

        if total <= PROMPT_HARD_MAX:
            return combined

        # Find the largest part and truncate it further
        largest_idx = max(range(len(parts)), key=lambda i: len(parts[i]))
        current_overage = total - PROMPT_HARD_MAX

        original = parts[largest_idx]
        if len(original) > current_overage + 1000:
            parts[largest_idx] = original[: len(original) - current_overage - 100]
            if is_verbose:
                logger.debug(
                    "Prompt over budget (%d chars). Truncated part %d from %d to %d.",
                    total,
                    largest_idx,
                    len(original),
                    len(parts[largest_idx]),
                )

        combined = "\n\n".join(parts)
        return combined[:PROMPT_HARD_MAX]

    def _get_accused_relationships(self, case):
        """Return existing ACCUSED CaseEntityRelationship objects for this case.

        Returns a list of (relationship, display_name) tuples.
        Used both for overlap detection and for updating notes.
        """
        rels = CaseEntityRelationship.objects.filter(
            case=case,
            relationship_type=RelationshipType.ACCUSED,
        ).select_related("entity")
        return [
            (rel, rel.entity.display_name.strip())
            for rel in rels
            if rel.entity and rel.entity.display_name
        ]

    def _get_accused_names(self, case):
        """Return a set of display_name values for existing ACCUSED relationships."""
        return {name for _, name in self._get_accused_relationships(case)}

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize a Nepali/English name for fuzzy comparison.

        - Strip leading/trailing whitespace
        - Collapse internal whitespace (handles "जीवन  बहादुर" → "जीवन बहादुर")
        - Normalize Unicode to NFC (handles composed vs decomposed Devanagari)
        - Lowercase (for English names)
        """
        import unicodedata

        name = unicodedata.normalize("NFC", name.strip())
        name = " ".join(name.split())  # collapse all internal whitespace
        return name.lower()

    @staticmethod
    def _names_match(a: str, b: str) -> bool:
        """Return True if two names likely refer to the same entity.

        Strategy (in order):
        1. Exact match after normalization
        2. Space-stripped match — catches "जयकुमार" vs "जय कुमार" (compound word split)
        3. One name is fully contained in the other (handles shortened forms)
        4. Token overlap >= 2 tokens and overlap_ratio >= 0.75
           — catches spelling variants, dropped particles, etc.
        """
        na = Command._normalize_name(a)
        nb = Command._normalize_name(b)

        # 1. Exact after normalization
        if na == nb:
            return True

        # 2. Space-stripped — handles compound Nepali words written with/without space
        # e.g. "जय कुमार खड्का" vs "जयकुमार खड्का"
        # e.g. "गीता कुमारी शाही" vs "गीताकुमारी शाही"
        # e.g. "योगेन्द्र नाथ" vs "योगेन्द्रनाथ"
        if na.replace(" ", "") == nb.replace(" ", ""):
            return True

        # 3. Substring containment (e.g. full title vs short name)
        #    Minimum length guard avoids false positives like "राम" matching "सीताराम"
        if len(na) >= 6 and len(nb) >= 6 and (na in nb or nb in na):
            return True

        # 4. Token overlap
        tokens_a = set(na.split())
        tokens_b = set(nb.split())
        # Require at least 2 tokens to avoid false positives on single common words
        if len(tokens_a) < 2 or len(tokens_b) < 2:
            return False
        overlap = len(tokens_a & tokens_b)
        smaller = min(len(tokens_a), len(tokens_b))
        return overlap >= 2 and (overlap / smaller) >= 0.75

    def _is_accused(self, name: str, accused_names: set) -> bool:
        """Check if a name matches any known accused name using fuzzy matching."""
        for accused in accused_names:
            if self._names_match(name, accused):
                return True
        return False

    # ------------------------------------------------------------------
    # Case processing
    # ------------------------------------------------------------------

    @staticmethod
    def _format_case_number(case):
        """Extract case number from court_cases JSON field."""
        court_cases = getattr(case, "court_cases", None)
        if isinstance(court_cases, list) and court_cases:
            first = court_cases[0]
            if isinstance(first, str) and ":" in first:
                return first.split(":", 1)[1]
        return ""

    def _process_case(self, case, options, api_key, session, is_verbose):
        is_dry_run = options["dry_run"]
        case_number = self._format_case_number(case)
        title_trunc = (case.title or "")[:120]

        self.stdout.write(f"\n{'─'*60}")
        self.stdout.write(
            f"[CASE START] case_number={case_number} "
            f"case_id={case.case_id}\n  title={title_trunc}"
        )

        # Load existing accused relationships (for overlap check + notes update)
        accused_rels = self._get_accused_relationships(case)
        accused_names = {name for _, name in accused_rels}
        if accused_names and is_verbose:
            self.stdout.write(
                f"  Accused ({len(accused_names)}): {', '.join(sorted(accused_names))}"
            )

        content_parts = []

        # Determine court order availability first — affects how much press release we use
        co_source = self._get_court_order_source(case)

        # Press release — use more context when no court order is available
        pr_source = self._get_press_release_source(case)
        if pr_source:
            pr_md = self._convert_source_to_markdown(
                pr_source, session, is_press_release=True
            )
            if pr_md:
                # No court order → use up to PRESS_RELEASE_CHARS_NO_COURT
                # With court order → use PRESS_RELEASE_CHARS (still more than before)
                if co_source is None:
                    truncated = self._truncate_press_release(
                        pr_md, limit=PRESS_RELEASE_CHARS_NO_COURT
                    )
                else:
                    truncated = self._truncate_press_release(pr_md)
                content_parts.append("--- PRESS RELEASE ---")
                content_parts.append(truncated)
                self.stdout.write(
                    f"press_release={pr_source.source_id} "
                    f"chars={len(pr_md)} used={len(truncated)}"
                    + (" (extended — no court order)" if co_source is None else "")
                )
            elif is_verbose:
                self.stdout.write(
                    f"  Press release: {pr_source.source_id} — conversion failed"
                )
        elif is_verbose:
            self.stdout.write("  No press release source found")

        # Court order — intelligent truncation
        if co_source:
            co_md = self._convert_source_to_markdown(
                co_source, session, is_press_release=False
            )
            if co_md:
                co_len = len(co_md)
                truncated = self._truncate_court_order(co_md)
                thahar_found = "ठहर खण्ड" in co_md
                self.stdout.write(
                    f"court_order={co_source.source_id} "
                    f"chars={co_len} used={len(truncated)}"
                    + (" [ठहर खण्ड]" if thahar_found else " [head+tail]")
                )
                content_parts.append("--- COURT ORDER ---")
                content_parts.append(truncated)
            elif is_verbose:
                self.stdout.write(
                    f"  Court order: {co_source.source_id} — conversion failed"
                )
        elif is_verbose:
            self.stdout.write("  No court order source found")

        if not content_parts:
            self.stats["cases_skipped"] += 1
            self.stdout.write(
                self.style.WARNING("  SKIPPED: No document content found")
            )
            return

        user_prompt = self._enforce_prompt_budget(content_parts, is_verbose)

        self.stdout.write(f"Prompt size: {len(user_prompt)} chars")

        if user_prompt.strip() == "":
            self.stats["cases_skipped"] += 1
            self.stdout.write(
                self.style.WARNING("  SKIPPED: Empty prompt after truncation")
            )
            return

        models = [m.strip() for m in options["llm_model"].split(",") if m.strip()]
        last_error = None
        response = None

        for model in models:
            try:
                response = call_llm(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    model=model,
                    base_url=options["llm_base_url"],
                    api_key=api_key,
                    session=session,
                )
                # Success — break out of fallback loop
                break
            except CommandError as e:
                last_error = e
                err_msg = str(e).lower()
                is_timeout = "timed out" in err_msg or "read timed out" in err_msg
                if len(models) > 1 and is_timeout and model != models[-1]:
                    self.stdout.write(
                        self.style.WARNING(
                            f"  Model '{model}' timed out, trying fallback..."
                        )
                    )
                    continue
                # Non-retryable error or last model
                response = None
                break

        if response is None:
            self.stats["cases_skipped"] += 1
            msg = (
                f"  SKIPPED: {last_error}"
                if last_error
                else "  SKIPPED: All models failed"
            )
            self.stdout.write(self.style.WARNING(msg))
            return

        entities_data = parse_extraction_response(response, {"entities"})
        accused_notes_data = _parse_accused_notes(response)

        if not entities_data and not accused_notes_data:
            self.stats["cases_skipped"] += 1
            self.stdout.write(
                self.style.WARNING("  SKIPPED: No entities or accused notes extracted")
            )
            logger.debug("Raw LLM response (first 500 chars): %s", response[:500])
            return

        if entities_data:
            self._apply_entities(
                case, entities_data, is_dry_run, session, accused_names
            )

        self._apply_accused_notes(case, accused_notes_data, accused_rels, is_dry_run)
        self.stats["cases_enriched"] += 1

    # ------------------------------------------------------------------
    # NES linking
    # ------------------------------------------------------------------

    def _link_nes(self, name, session):
        """Attempt NES name search; return nes_id on confident match or None."""
        from django.conf import settings

        nes_url = getattr(settings, "NES_API_URL", None)
        if not nes_url:
            return None

        try:
            res = session.get(
                f"{nes_url.rstrip('/')}/search",
                params={"q": name},
                timeout=5,
            )
            if res.status_code == 200:
                data = res.json()
                results = data.get("results", [])
                if results:
                    best_match = results[0]
                    if best_match.get("confidence", 0) > 0.8:
                        return best_match.get("id")
        except Exception as e:
            logger.debug("NES lookup failed for %s: %s", name, e)
        return None

    # ------------------------------------------------------------------
    # Entity persistence
    # ------------------------------------------------------------------

    def _apply_accused_notes(self, case, accused_notes_data, accused_rels, is_dry_run):
        """Update notes on existing accused relationships using LLM-extracted notes.

        Only updates relationships that currently have no notes (blank or null).
        Uses fuzzy name matching to find the right relationship to update.
        """
        if not accused_notes_data:
            return

        # Build lookup: normalized_name -> relationship object (only those missing notes)
        rels_needing_notes = [
            (rel, name)
            for rel, name in accused_rels
            if not (rel.notes and rel.notes.strip())
        ]

        updated = []
        for item in accused_notes_data:
            extracted_name = (item.get("name") or "").strip()
            new_notes = (item.get("notes") or "").strip()
            if not extracted_name or not new_notes:
                continue
            # Filter garbage names: single/two-char entries like "म" (Nepali "I")
            if len(extracted_name) <= 2:
                continue
            # Cap at 200 chars — reject verbose legal charge text
            if len(new_notes) > 200:
                continue

            for rel, stored_name in rels_needing_notes:
                # Skip garbage stored names in DB (data quality issue on mass-import cases)
                if len(stored_name) <= 2:
                    continue
                if self._names_match(extracted_name, stored_name):
                    if is_dry_run:
                        self.stdout.write(
                            f"  [DRY RUN] accused_note  {stored_name}  — {new_notes}"
                        )
                    else:
                        rel.notes = new_notes
                        rel.save(update_fields=["notes"])
                        self.stats["accused_notes_updated"] += 1
                        updated.append(f"  ~ accused    {stored_name}  — {new_notes}")
                    break

        if updated:
            self.stdout.write(f"  Accused notes updated ({len(updated)}):")
            for line in updated:
                self.stdout.write(line)

    _PROCEDURAL_KEYWORDS = (
        "उपन्यायाधिवक्ता",
        "न्यायाधिवक्ता",
        "सरकारी वकिल",
        "अधिवक्ता",
        "वरिष्ठ अधिवक्ता",
        "न्यायाधीश",
        "अध्यक्ष न्यायाधीश",
        "सदस्य न्यायाधीश",
    )

    # Boilerplate entities that appear in every case — skip them
    _SKIP_ENTITIES = frozenset(
        {
            "विशेष अदालत, काठमाडौं",
            "विशेष अदालत काठमाडौं",
            "विशेष सरकारी वकिल कार्यालय",
            "विशेष सरकारी वकील कार्यालय",
            "विशेष सरकारी वकिल कार्यालय, काठमाडौं",
            "विशेष सरकारी वकील कार्यालय, काठमाडौं",
            "अख्तियार दुरुपयोग अनुसन्धान आयोग",
            "अख्तियार दुरुपयोग अनुसन्धान आयोग, टंगाल",
            "अख्तियार दुरुपयोग अनुसन्धान आयोग (CIAA)",
            "Commission for the Investigation of Abuse of Authority (CIAA)",
            "CIAA",
        }
    )

    def _apply_entities(
        self, case, entities_data, is_dry_run, session, accused_names=None
    ):
        accused_names = accused_names or set()
        skipped_accused = []
        created_summary = []

        for item in entities_data:
            name = item.get("entity_name", "").strip()
            rel_type = item.get("relationship_type")
            notes = item.get("notes", "")

            if not name or rel_type not in ("location", "related"):
                continue

            # Skip boilerplate entities present in every case
            if name in self._SKIP_ENTITIES:
                continue
            # Also catch variants by normalized name (handles spacing/punctuation differences)
            norm = self._normalize_name(name)
            if any(self._normalize_name(skip) == norm for skip in self._SKIP_ENTITIES):
                continue
            # Skip laws/acts — they are not entities
            if rel_type == "related" and (
                "ऐन," in name or "ऐन, " in name or "नियमावली," in name
            ):
                continue
            # Skip prosecutors, attorneys, judges — standard professional roles, not case participants
            if rel_type == "related" and any(
                kw in (notes or "") for kw in self._PROCEDURAL_KEYWORDS
            ):
                continue

            # Skip entities already stored as accused — avoids duplicate relationships
            if rel_type == "related" and self._is_accused(name, accused_names):
                skipped_accused.append(name)
                continue

            if is_dry_run:
                self.stdout.write(
                    f"  [DRY RUN] {rel_type:8s}  {name}"
                    + (f"  — {notes}" if notes else "")
                )
                continue

            nes_id = self._link_nes(name, session)
            with transaction.atomic():
                entity, created = JawafEntity.objects.get_or_create(
                    display_name=name,
                    defaults={"nes_id": nes_id},
                )
                if not created and not entity.nes_id and nes_id:
                    entity.nes_id = nes_id
                    entity.save(update_fields=["nes_id"])

                if created:
                    self.stats["entities_created"] += 1

                relationship_type_enum = (
                    RelationshipType.LOCATION
                    if rel_type == "location"
                    else RelationshipType.RELATED
                )
                notes_max = CaseEntityRelationship._meta.get_field("notes").max_length
                safe_notes = (notes or "")[:notes_max]
                _rel, rel_created = CaseEntityRelationship.objects.get_or_create(
                    case=case,
                    entity=entity,
                    relationship_type=relationship_type_enum,
                    defaults={"notes": safe_notes},
                )
                if rel_created:
                    self.stats["relationships_created"] += 1
                    created_summary.append(
                        f"  + {rel_type:8s}  {name}" + (f"  — {notes}" if notes else "")
                    )

        # Print clean per-case summary
        if created_summary:
            self.stdout.write(f"  Entities saved ({len(created_summary)}):")
            for line in created_summary:
                self.stdout.write(line)
        if skipped_accused:
            self.stdout.write(
                f"  Skipped {len(skipped_accused)} accused overlaps: "
                + ", ".join(skipped_accused)
            )
