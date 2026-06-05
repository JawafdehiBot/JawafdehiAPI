"""Enrich DRAFT CIAA cases with Markdown case overviews from evidence sources.

Management command: ``python manage.py enrich_case_overview [--dry-run] [--limit N] ...``

Pipeline (per case)
-------------------
1. **Eligibility** — DRAFT cases where BOTH ``short_description`` AND ``description``
   are empty/null (or ``--force`` to reprocess).
2. **Gather** — classify evidence ``DocumentSource`` records by ``source_type`` +
   keyword/URL heuristics into: charge_sheet, press_releases, court_orders,
   investigative_reports, financial_docs, media_sources, other_docs.
   Gate: at least one press_release OR one court_order. (No chargesheet available.)
3. **Convert** — download each source (uploaded file → URL fallback) and convert
   to plain-text via ``markitdown``. Truncate: press_releases/court_orders 5k chars,
   investigative 3k.
4. **Discover court cases** — query ``ngm.services.get_court_case_details()`` for
   numbers in ``Case.court_cases`` + extracted JSON metadata; match results against
   DocumentSource cache for additional court order texts.
5. **Extract** — LLM call #1: structured JSON (accused_persons, case_metadata,
   fiscal_analysis, legal_provisions, key_events, total_disputed_amount).
6. **Format** — LLM call #2: Nepali Markdown overview (क/ख/ग sections) with
   ``short_description`` + ``description``, enriched with NGM court case metadata.
7. **Validate** — hard gates: required section क, minimum lengths, Devanagari
   ratio ≥80%, no raw HTML, no placeholder tokens.
8. **Save** — write ``short_description`` and ``description`` to Case (belt-and-suspenders:
   won't overwrite existing ``description`` unless ``--force``).

Classification rules
--------------------
``OFFICIAL_GOVERNMENT`` sources are sub-classified:
  1. ``_has_charge_sheet_keywords()`` → charge_sheet (single)
  2. ``_has_press_release_keywords()`` OR ``_has_ngm_store_url()`` → press_releases
  3. Everything else → other_docs

Other ``source_type`` values map directly (LEGAL_COURT_ORDER → court_orders, etc.).

LLM backends
------------
- OpenAI-compatible proxy (``https://llm-proxy.jawafdehi.org/v1``) via ``_call_llm_stream()``
- Anthropic Messages API via ``_call_llm_anthropic()``
- Auto-detected from ``--llm-base-url`` (``anthropic.com`` in URL → Anthropic).
- 3 retries with exponential backoff on 429/503/OSError.

Key options
-----------
``--dry-run``       Preview without DB writes (banner shown at start).
``--force``         Reprocess cases that already have overview content.
``--limit N``       Process at most N cases.
``--case-id X``     Process a single case by ``case_id``.
``--llm-model``     Model name (default: ``JAWAFDEHI_ALLEGATION_MODEL`` env var).
``--llm-base-url``  LLM proxy/base URL (default: ``JAWAFDEHI_LLM_PROXY_URL`` or OpenCode).
``--llm-api-key``   API key (falls back to env vars).
``--llm-timeout``   Request timeout seconds (default 300).
``--verbose``       Enable DEBUG-level logging.

Security
--------
- SSRF protection: blocks loopback/private/link-local/reserved IPs + known
  metadata hostnames (``_validate_host_safety``).
- Redirect safety: ``_SafeRedirectHandler`` validates redirect targets.
- Path traversal: ``_confined_output_path`` enforces output stays within temp dir.
- Download limit: ``MAX_DOWNLOAD_BYTES = 25 MiB`` per source.

Dependencies
------------
- ``markitdown`` for document → text conversion.
- ``ngm.services`` (optional) for court case discovery.
- ``anthropic`` (optional) for direct Anthropic API calls.

See also
--------
- ``services/jawafdehi-api/cases/models.py`` — Case, DocumentSource models.
"""

import hashlib
import ipaddress
import json
import logging
import os
import re
import socket
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from django.core.management.base import BaseCommand, CommandError
from django.db.models import Q
from openai import OpenAI

from cases.models import Case, CaseState, CaseType, DocumentSource, SourceType
from cases.services.priority_case_loader import filter_by_priority, load_priority_cases

logger = logging.getLogger(__name__)

MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024
DOWNLOAD_CHUNK_SIZE = 16 * 1024
DEFAULT_OPENCODE_BASE = "https://llm-proxy.jawafdehi.org/v1"
DEFAULT_LLM_TIMEOUT = 300
MAX_LLM_RETRIES = 5
LLM_EMPTY_RESPONSE_RETRIES = 5  # tolerate more empty-stream retries (proxy model quirk)
COURT_ORDER_HEAD_CHARS = 12000
COURT_ORDER_TAIL_CHARS = 6000
COURT_ORDER_FULL_MAX = 18000  # if ≤ this, send whole doc; else head+tail
DEVANAGARI_ALPHABETIC_RE = re.compile(r"[ऄ-हक़-ॡ]")
ALPHABETIC_RE = re.compile(r"[^\W\d_]", re.UNICODE)
_LLM_PROMPT_SIZE_WARN = 60000
_CLOUD_METADATA_IP = "169.254.169.254"  # NOSONAR — cloud metadata link-local
_SSRF_BLOCKED_HOSTNAMES = frozenset(
    {"localhost", "metadata.google.internal", _CLOUD_METADATA_IP, "metadata", "0.0.0.0"}
)

CHARGE_SHEET_KEYWORDS = [
    "charge sheet",
    "charge-sheet",
    "chargesheet",
    "अभियोगपत्र",  # अभियोगपत्र
    "अभियोगदावी",  # अभियोगदावी
    "अभियोग पत्र",  # अभियोग पत्र
    "अभियोग दावी",  # अभियोग दावी
]
PRESS_RELEASE_KEYWORDS = [
    "press release",
    "pressrelease",
    "press-release",
    "प्रेस विज्ञप्ति",  # प्रेस विज्ञप्ति
    "विज्ञप्ति",  # विज्ञप्ति
]

GARBLED_LEGAL_TERMS = frozenset(
    {
        "अख्ततमाय",  # should be अख्तियार
        "मिज्ञमि",  # should be विज्ञप्ति
        "रविसतिादी",  # should be प्रतिवादी
        "सञ्",  # common garbled prefix
        "गयेकोर",  # should be गरेको
        "गयेफभोख्जभ",  # garbled बमोजिम
        "रविस",  # garbled prefix for प्रति-
        "रविधि",  # garbled prefix for प्रविधि
        "रविदान",  # garbled प्रदान
        "रविदेश",  # garbled प्रदेश
    }
)

EXCESSIVE_SPACED_CHARS_RE = re.compile(
    r"[ऀ-ॿ](?:\s[ऀ-ॿ]){3,}"  # Devanagari chars separated by single spaces (PDF artifact)
)

EXTRACTION_SYSTEM_PROMPT = """\
You are a Nepali legal document parser specialized in CIAA corruption cases.
Extract structured data from NIAA press releases and court orders. No charge
sheet is available.

CRITICAL — Nepali text may contain character-level corruption from
machine-extracted PDF/DOC sources (e.g. "अख्ततमाय" instead of "अख्तियार",
"रविसतिादी" instead of "प्रतिवादी"). Use context, word position, and standard
Nepali legal terminology to disambiguate. Cross-reference between sources.
When a press release conflicts with a court order, trust the court order.

Document quality expectations:
- Court orders (from .doc/.docx) are typically 95%+ clean. Primary source.
- Press releases (from .pdf) typically 60—70% character accuracy. Secondary.
- If a word or name is ambiguous due to corruption, check whether it appears
  elsewhere in clean form. If unresolvable, record the best reconstruction
  and note the uncertainty in extraction_quality_notes.

Rules:
1. Extract ONLY information explicitly present or reasonably reconstructable
   from corrupted text. Do NOT fabricate, hallucinate, or infer.
2. Preserve exact names, dates (in Nepali Vikram Samvat), amounts (in NPR),
   and act/section citations as they appear in clean text.
3. For fiscal analysis, extract EVERY fiscal year row separately. If only a
   total is available, create a single row with fiscal_year="समग्र".
4. If a field is missing or genuinely unrecoverable, set it to null. Never
   use placeholder text (no [अज्ञात], N/A, TBD, ...).
5. Dates MUST remain in original Nepali Vikram Samvat (e.g. २०८१/०३/०९).
   Do NOT convert to Gregorian.
6. Amounts: extract exact NPR figures as written (e.g. रु. ३८,६७,१७,६४०/-).
   Keep commas and decimals.
7. legal_provisions: extract the full act name, section/dafa number, a plain-
   Nepali description of what the section prohibits, and the penalty if stated.

Source reliability order (when sources conflict):
- Court order: most reliable — facts, accused identity, verdict, sentencing,
  legal provisions, dates.
- Press release: reliable for case narrative, allegation summary, amounts,
  and timeline.
- When both are available prefer the court order for all factual fields; use
  the press release to fill narrative gaps only.

Return valid JSON only. No markdown. No explanation. No code fences."""

EXTRACTION_USER_PROMPT = """\
Extract structured case data from these CIAA case documents.

Case context:
- Case ID: {case_id}
- Case title: {case_title}
- Known court cases: {court_cases}
- Known bigo amount: {bigo}

IMPORTANT: Some text below may contain character-level corruption from PDF
extraction (especially the press releases). Use context and standard Nepali
legal terminology to disambiguate garbled words. The `source_quality` hints
indicate expected accuracy per source.

PRESS RELEASES (source quality: ~60-70%):
{press_release_texts}

COURT ORDERS (source quality: ~95%+):
{court_order_texts}

OTHER DOCUMENTS:
{other_texts}

{source_quality_notes_section}

Return JSON with these exact keys:

{{
  "accused_persons": [
    {{
      "name": "string (required — full name as written)",
      "position": "string|null (e.g. मुख्य सचिव, तत्कालिन कार्यकारी निर्देशक)",
      "institution": "string|null (e.g. सञ्चार तथा सूचना प्रविधि मन्त्रालय)",
      "employment_dates": "string|null",
      "role_in_case": "string|null (e.g. मुख्य प्रतिवादी, सह-प्रतिवादी)"
    }}
  ],
  "case_metadata": {{
    "case_number": "string|null (e.g. 080-CR-0196)",
    "filing_date": "string|null (Vikram Samvat: २०८१/०३/०९)",
    "court": "string|null (e.g. विशेष अदालत, काठमाडौं)",
    "verdict_date": "string|null (कसूर ठहर मिति)",
    "sentencing_date": "string|null (सजाय निर्धारण मिति)",
    "charge_sheet_number": "string|null",
    "complaint_numbers": ["string"],
    "investigation_period": "string|null"
  }},
  "fiscal_analysis": [
    {{
      "fiscal_year": "string (required)",
      "income": "string|null",
      "expenditure": "string|null",
      "balance": "string|null",
      "source_detail": "string|null"
    }}
  ],
  "legal_provisions": [
    {{
      "act": "string (required, e.g. भ्रष्टाचार निवारण ऐन, २०५९)",
      "section": "string|null (e.g. दफा ३ को उपदफा (१) को देहाय (झ))",
      "description": "string|null (plain Nepali: what this provision prohibits or requires)",
      "penalty": "string|null (e.g. कैद र बिगो बमोजिम जरिवाना)"
    }}
  ],
  "key_events": [
    {{
      "date": "string (required, Nepali VS)",
      "description": "string (required)"
    }}
  ],
  "total_disputed_amount": "string|null (exact NPR: रु. ३८,६७,१७,६४०/-)",
  "extraction_quality_notes": "string|null (note sections where text corruption made extraction unreliable)"
}}

IMPORTANT:
- Return ONLY a valid JSON object. No markdown code fences. No explanation.
- For missing or irrecoverable data use null. Never use placeholder text.
- Every accused person MUST have name filled (reconstruct from context if
  partially garbled; mark quality concern in extraction_quality_notes).
- Fiscal years: prefer individual year rows. If only total: fiscal_year="समग्र".
- legal_provisions: include the FULL section hierarchy (दफा + उपदफा + देहाय)
  exactly as written.
- key_events: chronological order. Include filing, investigation milestones,
  verdict, sentencing dates if available.
- extraction_quality_notes: brief summary if any section relied on heavily
  garbled text (null if all text was clean).
- Prefer court order data when multiple sources conflict on facts."""

FORMATTING_SYSTEM_PROMPT = """\
You are a Nepali legal writer for JAWAFDEHI, Nepal's public corruption case
archive. Format structured CIAA case data into a Markdown case overview
entirely in Nepali Devanagari.

Rules:
1. Write entirely in Nepali Devanagari. English ONLY for: proper nouns
   (company names, brand names), legal citation numbers (080-CR-0196),
   and technical terms without a standard Nepali equivalent.
2. Format and transcribe; do NOT summarize away specific details. Every
   name, date, amount, and legal citation from the data must appear.
3. Use Markdown bold (**text**) for headings. NEVER use HTML tags.
4. Use Markdown pipe tables for fiscal data. Include ALL fiscal year rows.
5. Return ONLY: {{"short_description": "...", "description": "..."}} —
   valid JSON, no fences, no explanation.
6. No placeholder text EVER (no [AI-generated], [draft], [TODO], N/A,
   TBD, ...). If data is genuinely missing for a section, omit that section.
7. If extraction_quality_notes indicates corrupted text for a section,
   prefix with a brief inline marker like "(पाठ आंशिक रूपमा अस्पष्ट)"
   and present what IS known. Do not fabricate to fill gaps.
8. Legal style: formal Nepali, passive voice appropriate for legal writing,
   consistent terminology across cases."""

FORMATTING_USER_PROMPT = """\
Format this extracted case data into a JAWAFDEHI case overview.

EXTRACTED CASE DATA:
{extracted_json}

COURT CASE METADATA (from NGM judicial database):
{court_case_metadata}

OUTPUT STRUCTURE:

**क) अभियोगदावीको सार** (MANDATORY)
- 4-6 paragraphs of formal legal Nepali narrative.
- Paragraph 1: Open with "प्रस्तुत मुद्दामा". State accused names, positions,
  institutions, core allegation, and total disputed amount.
- Paragraph 2: Detail the alleged scheme — what was done, how, when, who.
- Paragraph 3: Investigation findings — key evidence, audit reports, expert
  opinions mentioned in the source material.
- Paragraph 4: Fiscal analysis pipe table (ONLY if fiscal_analysis has data):

  | आर्थिक वर्ष | विवरण | आय (रु.) | व्यय/खर्च (रु.) | फरक/बचत |
  |-------------|--------|----------|-----------------|----------|

- Paragraph 5 (if verdict): Court case status, verdict details, and sentencing.
- Paragraph 6: Total बिगो and confiscation demands if stated.

**ख) आकर्षित कानुनी व्यवस्था** (CONDITIONAL — only if legal_provisions non-empty)
Format each provision as: "**{{Act}}, {{section}}:** {{plain-Nepali description of
what is prohibited}}. {{Penalty if stated}}."
Number provisions: १., २., ३., ...
Example: "**भ्रष्टाचार निवारण ऐन, २०५९ को दफा ३ को उपदफा (१) को देहाय (झ):**
सार्वजनिक सेवकले गैरकानूनी रुपमा सम्पत्ति आर्जन गर्न नहुने।
सजाय: कैद र बिगो बमोजिम जरिवाना।"

**ग) प्रमाणको संक्षेप** (CONDITIONAL — only if key_events non-empty)
- Chronological bullet list with Nepali dates.
- Include: complaint source, investigation initiation, charge sheet filing date,
  court hearing dates, verdict date, sentencing date.
- Evidence types checked: documents examined, witness testimony, expert/financial
  audit reports.
- Reference court case numbers and verdict details from the court case metadata.

CRITICAL FORMATTING RULES:
- short_description: Exactly 2-3 sentences. Plain Nepali text only. No Markdown.
  Must contain: who (accused name + position + institution), what (allegation
  type), disputed amount (if known), filing date (if known), current status.
  Minimum 50 characters. Maximum 500 characters.
  Example: "विशेष अदालतमा नेपाल सरकारको वादमा [accused], [position],
  [institution] विरुद्ध [allegation] मुद्दा। दर्ता मिति: [date]। हाल चलिरहेको।"
- description: Full Markdown per section structure above. Minimum 100 characters.
  Minimum 80% Devanagari script (excluding numbers and English proper nouns).
- If extraction_quality_notes warns about corrupted sections, add inline
  "(पाठ आंशिक रूपमा अस्पष्ट)" where data is uncertain.
- Preserve ALL specific details from extracted data exactly.
- Return ONLY {{"short_description": "...", "description": "..."}}."""


def _validate_host_safety(hostname: str) -> None:
    # DNS rebinding note: _validate_host_safety resolves once, urlopen resolves
    # again. Full DNS pinning (custom HTTPConnection) would be disproportionate
    # here — URLs originate from DocumentSource records in our own DB, not from
    # untrusted user input, and the validation→connect window is sub-millisecond,
    # making a TOCTOU race infeasible. If this command later accepts ad-hoc URLs,
    # pin the resolved IPs and set the Host header on a custom opener.
    if hostname is None:
        raise ValueError("Cannot validate host: hostname is None (malformed URL)")
    host = hostname.lower().rstrip(".")
    if host in _SSRF_BLOCKED_HOSTNAMES:
        raise ValueError(f"Blocked internal host: {hostname!r}")
    try:
        addrinfo = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ValueError(f"Cannot resolve host: {hostname!r}") from exc
    for info in addrinfo:
        addr = ipaddress.ip_address(info[4][0])
        if (
            addr.is_loopback
            or addr.is_private
            or addr.is_link_local
            or addr.is_reserved
        ):
            raise ValueError(f"Blocked internal address: {hostname!r} -> {addr}")


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlparse(newurl)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise urllib.error.HTTPError(
                req.full_url, code, f"Unsafe redirect to {newurl}", headers, fp
            )
        try:
            _validate_host_safety(parsed.hostname)
        except ValueError as exc:
            raise urllib.error.HTTPError(
                req.full_url, code, f"Unsafe redirect to {newurl}: {exc}", headers, fp
            ) from exc
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _sanitize_download_filename(filename: str | None, source_id: str) -> str:
    raw = (filename or "").strip()
    if not raw:
        return f"{source_id}.bin"
    candidate = Path(urllib.parse.unquote(raw)).name.strip().replace("\x00", "")
    candidate = re.sub(r"[<>:\"/\\|?*]+", "_", candidate).rstrip(" .")
    if candidate in {"", ".", ".."}:
        return f"{source_id}.bin"
    if len(candidate) <= 200:
        return candidate
    suffix = "".join(Path(candidate).suffixes)
    stem = candidate[: -len(suffix)] if suffix else candidate
    digest = hashlib.sha256(candidate.encode()).hexdigest()[:10]
    budget = max(1, 200 - len(suffix) - len(digest) - 1)
    return f"{stem[:budget].rstrip(' .-_') or source_id}-{digest}{suffix}"[:200]


def _confined_output_path(output_dir: Path, filename: str) -> Path:
    output_dir_resolved = output_dir.resolve()
    out_path = (output_dir / filename).resolve()
    if output_dir_resolved not in out_path.parents:
        raise CommandError(f"Refusing to write outside output directory: {filename!r}")
    return out_path


def _copy_stream_to_path_with_limit(in_file: Any, out_path: Path) -> None:
    total = 0
    try:
        with out_path.open("wb") as out_file:
            while True:
                chunk = in_file.read(DOWNLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    raise CommandError(
                        f"Downloaded source exceeds max size of {MAX_DOWNLOAD_BYTES} bytes"
                    )
                out_file.write(chunk)
    except (OSError, CommandError):
        out_path.unlink(missing_ok=True)
        raise


def normalize_model(model: str) -> str:
    model = model.strip()
    for prefix in ("opencode-go/", "openai:"):
        if model.startswith(prefix):
            return model[len(prefix) :]
    return model


def normalize_base_url(url: str | None) -> str:
    url = (
        (url or os.environ.get("JAWAFDEHI_LLM_PROXY_URL", DEFAULT_OPENCODE_BASE))
        .strip()
        .rstrip("/")
    )
    if url.endswith("/zen/v1"):
        return url.replace("/zen/v1", "/zen/go/v1")
    if url.endswith("/zen/go"):
        return f"{url}/v1"
    return url


def resolve_api_key(cli_key: str | None = None, is_anthropic: bool = False) -> str:
    if cli_key and cli_key.strip():
        return cli_key.strip()
    env_vars = (
        ("ANTHROPIC_API_KEY", "JAWAFDEHI_LLM_API_KEY", "OPENCODE_API_KEY")
        if is_anthropic
        else ("JAWAFDEHI_LLM_API_KEY", "OPENCODE_API_KEY", "ANTHROPIC_API_KEY")
    )
    for env_var in env_vars:
        val = os.environ.get(env_var)
        if val:
            return val
    raise CommandError(
        "No API key provided. Set --llm-api-key, JAWAFDEHI_LLM_API_KEY, "
        "OPENCODE_API_KEY, or ANTHROPIC_API_KEY."
    )


def _llm_timeout(cli_timeout: int | None = None) -> int:
    if cli_timeout is not None:
        if cli_timeout <= 0:
            raise CommandError(f"--llm-timeout must be > 0, got {cli_timeout}")
        return cli_timeout
    env = os.environ.get("JAWAFDEHI_LLM_TIMEOUT_SECONDS")
    if env:
        try:
            val = int(env)
            if val > 0:
                return val
        except ValueError:
            pass
    return DEFAULT_LLM_TIMEOUT


def _extract_json_body(raw: str) -> str:
    """Extract JSON from LLM response, robust to markdown fences and prefixes."""
    raw = raw.strip()
    # If the response starts with conversational text then a fenced block, find the block
    fence_start = raw.find("```")
    if fence_start != -1:
        # Skip the fence opener line (which may have "json" or other language tag)
        after_fence = raw[fence_start + 3 :]
        # Strip "json" language tag if present
        after_fence = re.sub(r"^json\s*", "", after_fence, flags=re.IGNORECASE).strip()
        fence_end = after_fence.find("```")
        if fence_end != -1:
            return after_fence[:fence_end].strip()
        # No closing fence — treat everything after opening fence
        return after_fence.strip()
    # No fenced block — try to find { ... }
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start : end + 1]
    return raw




def safe_truncate(text: str | None, max_chars: int = 15000) -> str:
    """Character-safe truncation for Devanagari text.
    Python str slicing operates on Unicode code points, not bytes,
    so Devanagari grapheme clusters are preserved correctly.
    """
    if not text:
        return ""
    return text[:max_chars]


class Command(BaseCommand):
    help = (
        "Generate Markdown case overview content from CIAA evidence sources using LLM"
    )

    def __init__(self):
        super().__init__()
        self.stats = {
            "cases_processed": 0,
            "cases_enriched": 0,
            "cases_skipped": 0,
            "cases_failed": 0,
            "cases_no_content": 0,
            "llm_extraction_failures": 0,
            "llm_formatting_failures": 0,
        }
        self._markitdown = None

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Preview without saving. REQUIRED for first run — no writes happen without --dry-run removed.",
        )
        parser.add_argument(
            "--limit", type=int, default=None, help="Process only N cases"
        )
        parser.add_argument(
            "--case-id", type=str, default=None, help="Process a specific case"
        )
        parser.add_argument(
            "--verbose", action="store_true", help="Enable debug logging"
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Re-process cases with existing overview content (overwrites both short_description and description)",
        )
        parser.add_argument(
            "--llm-model",
            type=str,
            default=os.environ.get("JAWAFDEHI_ALLEGATION_MODEL", "claude-sonnet-4-5"),
            help="LLM model",
        )
        parser.add_argument(
            "--llm-base-url", type=str, default=None, help="LLM base URL"
        )
        parser.add_argument("--llm-api-key", type=str, default=None, help="LLM API key")
        parser.add_argument(
            "--llm-timeout", type=int, default=None, help="LLM timeout seconds"
        )
        parser.add_argument(
            "--priority",
            action="store_true",
            help="Enrich only cases in the priority case list",
        )

    def handle(self, *args, **options):
        priority = options.get("priority", False)
        case_id = options.get("case_id")
        if priority and case_id:
            raise CommandError("--priority and --case-id are mutually exclusive")
        if options["verbose"]:
            logger.setLevel(logging.DEBUG)
        model = normalize_model(options["llm_model"])
        base_url = normalize_base_url(options["llm_base_url"])
        is_opencode = "anthropic.com" not in base_url
        api_key = resolve_api_key(
            options.get("llm_api_key"), is_anthropic=not is_opencode
        )
        timeout = _llm_timeout(options.get("llm_timeout"))
        dry_run = options["dry_run"]
        force = options["force"]

        logger.info(
            "LLM resolution: resolved_model=%s resolved_base_url=%s is_opencode=%s "
            "env_ALLEGATION_MODEL=%s env_LLM_MODEL=%s env_PROXY_URL=%s cli_model=%s cli_base=%s",
            model,
            base_url,
            is_opencode,
            os.environ.get("JAWAFDEHI_ALLEGATION_MODEL", "(unset)"),
            os.environ.get("JAWAFDEHI_LLM_MODEL", "(unset)"),
            os.environ.get("JAWAFDEHI_LLM_PROXY_URL", "(unset)"),
            options.get("llm_model") or "(default)",
            options.get("llm_base_url") or "(default)",
        )

        if dry_run:
            self.stdout.write(self.style.WARNING("=" * 70))
            self.stdout.write(
                self.style.WARNING("  DRY RUN — No database changes will be made")
            )
            self.stdout.write(
                self.style.WARNING(
                    "  Remove --dry-run to write case overviews to the database"
                )
            )
            self.stdout.write(self.style.WARNING("=" * 70))
        else:
            self.stdout.write(
                self.style.WARNING(
                    "Writing to database. Use --dry-run to preview first."
                )
            )

        logger.info(
            "enrich_case_overview START | dry_run=%s model=%s base_url=%s limit=%s case_id=%s force=%s priority=%s",
            dry_run,
            model,
            base_url,
            options.get("limit"),
            options.get("case_id"),
            options.get("force"),
            priority,
        )
        self.stdout.write(
            self.style.WARNING(
                f"{'[DRY RUN] ' if dry_run else ''}Starting case overview enrichment..."
            )
        )
        started = time.time()
        cases = self._get_eligible_cases(
            options["limit"], options["force"], options.get("case_id"), priority
        )
        logger.info("Found %d eligible CIAA DRAFT cases", len(cases))
        self.stdout.write(f"Found {len(cases)} eligible CIAA DRAFT case(s) to process")
        self._fetch_source_cache(cases)
        logger.info(
            "step=fetch_source_cache source_ids=%d",
            len(getattr(self, "_source_lookup", {})),
        )

        for idx, case in enumerate(cases, 1):
            self.stdout.write(
                f"\n[{idx}/{len(cases)}] {case.case_id} - {case.title[:80]}..."
            )
            logger.info("[%d/%d] Processing case %s", idx, len(cases), case.case_id)
            try:
                self._process_case(
                    case, model, base_url, api_key, timeout, is_opencode, dry_run, force
                )
            except Exception as exc:
                self.stats["cases_failed"] += 1
                logger.exception("Error processing %s", case.case_id)
                self.stdout.write(
                    self.style.ERROR(
                        self._safe_output_text(f"FAILED: {case.case_id} - {exc}")
                    )
                )

        elapsed = int(time.time() - started)
        logger.info(
            "enrich_case_overview END | elapsed=%ds processed=%d enriched=%d skipped=%d no_content=%d failed=%d extraction_fail=%d formatting_fail=%d",
            elapsed,
            self.stats["cases_processed"],
            self.stats["cases_enriched"],
            self.stats["cases_skipped"],
            self.stats["cases_no_content"],
            self.stats["cases_failed"],
            self.stats["llm_extraction_failures"],
            self.stats["llm_formatting_failures"],
        )
        self._print_summary(
            dry_run,
            f"{elapsed // 60}m {elapsed % 60}s" if elapsed >= 60 else f"{elapsed}s",
        )

    def _get_eligible_cases(self, limit, force, case_id, priority=False):
        queryset = Case.objects.filter(
            state=CaseState.DRAFT, case_type=CaseType.CORRUPTION
        )
        if case_id:
            queryset = queryset.filter(case_id=case_id)
        if priority:
            priority_list = load_priority_cases()
            logger.info(
                "Priority mode: loaded %d case numbers across all fiscal years",
                len(priority_list),
            )
            queryset = filter_by_priority(queryset, priority_list)
        if not force:
            queryset = queryset.filter(
                (Q(short_description__isnull=True) | Q(short_description=""))
                & (Q(description__isnull=True) | Q(description=""))
            )
        if limit is not None:
            if limit < 0:
                raise CommandError(f"--limit must be >= 0, got {limit}")
            queryset = queryset[:limit] if limit > 0 else queryset.none()
        return list(queryset)

    def _fetch_source_cache(self, cases):
        source_ids = set()
        for case in cases:
            for entry in case.evidence or []:
                if isinstance(entry, dict) and isinstance(entry.get("source_id"), str):
                    source_ids.add(entry["source_id"])
        self._source_lookup = {
            source.source_id: source
            for source in DocumentSource.objects.filter(
                source_id__in=source_ids, is_deleted=False
            ).prefetch_related("uploaded_files")
        }

    # ── Multi-source evidence gathering ───────────────────────────

    def _gather_case_sources(self, case):
        """Return categorized dict of all relevant sources for the case."""
        gathered = {
            "charge_sheet": None,
            "press_releases": [],
            "court_orders": [],
            "procedural_docs": [],
            "financial_docs": [],
            "investigative_reports": [],
            "media_sources": [],
            "other_docs": [],
        }
        for entry in case.evidence or []:
            if not isinstance(entry, dict):
                continue
            sid = entry.get("source_id")
            if not isinstance(sid, str) or not sid.strip():
                continue
            source = self._source_lookup.get(sid)
            if not source:
                continue
            stype = source.source_type
            if stype == SourceType.OFFICIAL_GOVERNMENT:
                if _has_charge_sheet_keywords(source):
                    if not gathered["charge_sheet"]:
                        gathered["charge_sheet"] = source
                elif _has_press_release_keywords(source) or _has_ngm_store_url(source):
                    gathered["press_releases"].append(source)
                else:
                    gathered["other_docs"].append(source)
            elif stype == SourceType.LEGAL_PROCEDURAL:
                if _has_press_release_keywords(source) or _has_ngm_store_url(source):
                    gathered["press_releases"].append(source)
                else:
                    gathered["procedural_docs"].append(source)
            elif stype == SourceType.LEGAL_COURT_ORDER:
                gathered["court_orders"].append(source)
            elif stype == SourceType.FINANCIAL_FORENSIC:
                gathered["financial_docs"].append(source)
            elif stype == SourceType.INVESTIGATIVE_REPORT:
                gathered["investigative_reports"].append(source)
            elif stype == SourceType.MEDIA_NEWS:
                gathered["media_sources"].append(source)
            else:
                gathered["other_docs"].append(source)
        return gathered

    def _convert_sources_to_texts(self, gathered):
        """Convert all gathered sources to text, returning a dict of texts."""
        texts = {
            "charge_sheet": "",
            "press_releases": [],
            "court_orders": [],
            "investigative_reports": [],
            "financial_docs": [],
        }
        # Charge sheet (primary)
        if gathered["charge_sheet"]:
            texts["charge_sheet"] = (
                self._convert_one_source(gathered["charge_sheet"]) or ""
            )
        # Press releases
        for src in gathered["press_releases"]:
            t = self._convert_one_source(src)
            if t and len(t.strip()) >= 50:
                # Truncate long press releases
                texts["press_releases"].append(t[:5000])
        # Court orders — head+tail for long docs (verdict usually at end)
        for src in gathered["court_orders"]:
            t = self._convert_one_source(src)
            if t and len(t.strip()) >= 50:
                texts["court_orders"].append(_truncate_long_doc(t))
        # Investigative reports
        for src in gathered["investigative_reports"]:
            t = self._convert_one_source(src)
            if t and len(t.strip()) >= 50:
                texts["investigative_reports"].append(t[:3000])
        # Financial docs
        for src in gathered["financial_docs"]:
            t = self._convert_one_source(src)
            if t and len(t.strip()) >= 50:
                texts["financial_docs"].append(t[:5000])
        return texts

    def _convert_one_source(self, source):
        """Convert a single source to text, returning None on failure."""
        try:
            result = self._convert_source_to_markdown(source)
            if result:
                logger.debug(
                    "Source %s: converted — %d chars", source.source_id, len(result)
                )
            return result
        except (CommandError, ValueError, OSError) as exc:
            logger.warning("Source %s: conversion failed — %s", source.source_id, exc)
        # Fallback: try OLE-based extraction for legacy .doc files
        try:
            result = self._convert_source_via_ole(source)
            if result:
                logger.info(
                    "Source %s: converted via OLE fallback — %d chars",
                    source.source_id,
                    len(result),
                )
                return result
        except Exception as exc:
            logger.debug(
                "Source %s: OLE fallback also failed — %s",
                source.source_id,
                exc,
            )
        return None

    @staticmethod
    def _convert_source_via_ole(source):
        """Extract Nepali text from legacy .doc (OLE) files via binary parsing.

        Used as a fallback when markitdown cannot convert .doc files on
        platforms without antiword. Works on Composite Document File V2
        (.doc) format with UTF-16LE encoded Nepali text.
        """
        import struct

        try:
            import olefile
        except ImportError:
            logger.debug("olefile not installed — OLE fallback unavailable")
            return None
        # Read raw bytes from uploaded_file (Django FieldFile)
        uploaded = None
        if hasattr(source, "uploaded_file") and source.uploaded_file:
            uploaded = source.uploaded_file
        if not uploaded:
            uploaded_qs = getattr(source, "uploaded_files", None)
            if uploaded_qs is not None:
                first = uploaded_qs.first()
                if first and hasattr(first, "file"):
                    uploaded = first.file
        if not uploaded:
            return None
        try:
            uploaded.open("rb")
            raw = uploaded.read()
        except Exception:
            return None
        finally:
            try:
                uploaded.close()
            except Exception:
                pass
        try:
            ole = olefile.OleFileIO(raw)
        except Exception:
            return None
        if not ole.exists("WordDocument"):
            ole.close()
            return None
        try:
            wd = ole.openstream("WordDocument").read()
        except Exception:
            ole.close()
            return None
        ole.close()
        # Validate magic: must be 0xA5EC for Word binary
        if len(wd) < 2 or struct.unpack_from("<H", wd, 0)[0] != 0xA5EC:
            return None
        # Walk UTF-16LE stream, collecting Devanagari + ASCII chars
        result = []
        RELEVANT_CP = frozenset(
            {
                0x0020,
                0x0964,
                0x0965,
                0x002E,
                0x002C,
                0x0028,
                0x0029,
                0x002F,
                0x003A,
                0x003B,
                0x002D,
                0x000A,
                0x000D,
            }
        )
        for i in range(0, len(wd) - 1, 2):
            cu = struct.unpack_from("<H", wd, i)[0]
            if (
                0x0900 <= cu <= 0x097F  # Devanagari
                or 0x0030 <= cu <= 0x0039  # digits
                or 0x0041 <= cu <= 0x005A  # A-Z
                or 0x0061 <= cu <= 0x007A  # a-z
                or cu in RELEVANT_CP
            ):
                result.append(chr(cu))
        text = "".join(result)
        if len(text.strip()) < 50:
            return None
        return text

    def _discover_court_cases(self, case, extracted_json):
        """Query NGM judicial DB for court cases matching the case record.

        Uses case.court_cases (e.g. ["special:080-CR-0007"]) and extracted_json
        case metadata to find matching court case records, then searches
        DocumentSource cache for related court order files and converts them.

        Returns dict with:
        - court_cases_found: list of metadata dicts from NGM DB
        - court_order_texts: list of converted text strings
        """
        try:
            from ngm.services import get_court_case_details, normalize_case_number
        except ImportError:
            logger.warning("NGM services not available — skipping court case discovery")
            return {"court_cases_found": [], "court_order_texts": []}

        # Collect case numbers from case record + extracted JSON
        case_numbers_to_lookup = set()

        # 1. From case.court_cases field: ["special:080-CR-0007", ...]
        raw_court_cases = case.court_cases or []
        if isinstance(raw_court_cases, list):
            for entry in raw_court_cases:
                if isinstance(entry, str) and ":" in entry:
                    court_id, case_num = entry.split(":", 1)
                    try:
                        normalized = normalize_case_number(case_num.strip())
                        case_numbers_to_lookup.add((court_id.strip(), normalized))
                    except ValueError:
                        logger.debug(
                            "Case %s: unparseable court case ref %r",
                            case.case_id,
                            entry,
                        )

        # 2. From extracted JSON metadata
        extracted_case_nums = set()
        if isinstance(extracted_json, dict):
            meta = extracted_json.get("case_metadata", {})
            if isinstance(meta, dict):
                for key in ("case_number", "court_case_number", "case_numbers"):
                    val = meta.get(key)
                    if isinstance(val, str) and val.strip():
                        extracted_case_nums.add(val.strip())
                    elif isinstance(val, list):
                        for item in val:
                            if isinstance(item, str) and item.strip():
                                extracted_case_nums.add(item.strip())

            # Also check legal_provisions for case refs
            provisions = extracted_json.get("legal_provisions", [])
            if isinstance(provisions, list):
                for prov in provisions:
                    if isinstance(prov, dict):
                        for key in ("case_number", "reference"):
                            val = prov.get(key)
                            if isinstance(val, str) and val.strip():
                                extracted_case_nums.add(val.strip())

        # Normalize extracted numbers too
        for raw_num in extracted_case_nums:
            try:
                normalized = normalize_case_number(raw_num)
            except ValueError:
                continue
            # Try common court identifiers
            for court_id in ("special", "supreme", "high", "district"):
                case_numbers_to_lookup.add((court_id, normalized))

        if not case_numbers_to_lookup:
            logger.debug("Case %s: no court case numbers to look up", case.case_id)
            return {"court_cases_found": [], "court_order_texts": []}

        logger.info(
            "Case %s: step=discover looking_up=%d numbers=%s",
            case.case_id,
            len(case_numbers_to_lookup),
            sorted(f"{c}:{n}" for c, n in case_numbers_to_lookup),
        )

        # Query NGM database for each case number
        court_cases_found = []
        for court_id, case_num in case_numbers_to_lookup:
            try:
                details = get_court_case_details(court_id, case_num)
                if details:
                    court_cases_found.append(
                        {
                            "court_identifier": court_id,
                            "case_number": case_num,
                            "registration_date_bs": details.get("case", {}).get(
                                "registration_date_bs"
                            ),
                            "case_type": details.get("case", {}).get("case_type"),
                            "status": details.get("case", {}).get("case_status"),
                            "verdict_date_bs": details.get("case", {}).get(
                                "verdict_date_bs"
                            ),
                            "plaintiff": details.get("case", {}).get("plaintiff"),
                            "defendant": details.get("case", {}).get("defendant"),
                            "entities": [
                                {"name": e.get("name"), "side": e.get("side")}
                                for e in details.get("entities", [])
                            ],
                        }
                    )
                    logger.info(
                        "Case %s: found NGM record %s:%s",
                        case.case_id,
                        court_id,
                        case_num,
                    )
            except Exception:
                logger.debug(
                    "Case %s: NGM lookup failed for %s:%s",
                    case.case_id,
                    court_id,
                    case_num,
                    exc_info=True,
                )

        if not court_cases_found:
            logger.info("Case %s: no matching court cases in NGM DB", case.case_id)
            return {"court_cases_found": [], "court_order_texts": []}

        # Phase 2: Find DocumentSource records matching discovered case numbers
        court_order_texts = []
        # Search ALL LEGAL_COURT_ORDER sources in the DB (not just the current
        # case's evidence _source_lookup, which only contains press releases and
        # charge sheets).  NGM court order files are separate DocumentSource
        # records with source_type=LEGAL_COURT_ORDER.
        for record in court_cases_found:
            case_num = record["case_number"]
            # Build a query that matches the case number in title or description.
            # Use icontains for substring matching against the normalized case
            # number (e.g. "080-CR-0007" appearing anywhere in the title).
            q = Q(title__icontains=case_num)
            sources = DocumentSource.objects.filter(
                q,
                source_type=SourceType.LEGAL_COURT_ORDER,
                is_deleted=False,
            ).prefetch_related("uploaded_files")
            for source in sources:
                text = self._convert_one_source(source)
                if text:
                    court_order_texts.append(text)
                    logger.info(
                        "Case %s: converted court order source %s for %s",
                        case.case_id,
                        source.source_id,
                        case_num,
                    )
                    break  # One converted source per case number is sufficient

        return {
            "court_cases_found": court_cases_found,
            "court_order_texts": court_order_texts,
        }

    # ── Per-case processing ───────────────────────────────────────

    def _process_case(
        self, case, model, base_url, api_key, timeout, is_opencode, dry_run, force=False
    ):
        self.stats["cases_processed"] += 1
        logger.info(
            "Case %s: step=start evidence_entries=%d",
            case.case_id,
            len(case.evidence or []),
        )
        if not case.evidence:
            self.stats["cases_skipped"] += 1
            logger.warning("Case %s: step=skip reason=no_evidence", case.case_id)
            self.stdout.write(self.style.WARNING("  SKIPPED: No evidence"))
            return

        # Gather all sources
        gathered = self._gather_case_sources(case)
        logger.info(
            "Case %s: step=gather status=ok charge_sheet=%s press_releases=%d court_orders=%d investigative=%d financial=%d procedural=%d media=%d other=%d",
            case.case_id,
            bool(gathered["charge_sheet"]),
            len(gathered["press_releases"]),
            len(gathered["court_orders"]),
            len(gathered["investigative_reports"]),
            len(gathered["financial_docs"]),
            len(gathered["procedural_docs"]),
            len(gathered["media_sources"]),
            len(gathered["other_docs"]),
        )
        if not gathered["press_releases"] and not gathered["court_orders"]:
            self._skip_no_content(
                case,
                "No press releases or court orders available (chargesheet not required)",
                dry_run,
            )
            logger.warning(
                "Case %s: step=skip reason=no_press_releases_or_court_orders",
                case.case_id,
            )
            return

        # Convert all sources to texts
        source_texts = self._convert_sources_to_texts(gathered)
        logger.info(
            "Case %s: step=convert status=ok charge_sheet=%d press_releases=%d court_orders=%d investigative=%d financial=%d",
            case.case_id,
            len(source_texts["charge_sheet"]),
            len(source_texts["press_releases"]),
            len(source_texts["court_orders"]),
            len(source_texts["investigative_reports"]),
            len(source_texts["financial_docs"]),
        )

        # Need at least a press release or court order
        if not source_texts["press_releases"] and not source_texts["court_orders"]:
            self._skip_no_content(case, "Failed to convert any source to text", dry_run)
            logger.warning("Case %s: step=skip reason=conversion_failed", case.case_id)
            return

        # Primary source description for logging
        if gathered["charge_sheet"]:
            primary = gathered["charge_sheet"]
        elif gathered["press_releases"]:
            primary = gathered["press_releases"][0]
        elif gathered["court_orders"]:
            primary = gathered["court_orders"][0]
        else:
            primary = None
        if primary:
            self.stdout.write(
                f"  Source: {self._describe_source(primary)}"
            )
        # Also print court order source when primary isn't already a court order
        if gathered["court_orders"] and primary != gathered["court_orders"][0]:
            self.stdout.write(
                f"  Court order source: {self._describe_source(gathered['court_orders'][0])}"
            )
        self.stdout.write(
            f"  Gathered: charge_sheet={'yes' if source_texts['charge_sheet'] else 'no'}, "
            f"press_releases={len(source_texts['press_releases'])}, "
            f"court_orders={len(source_texts['court_orders'])}, "
            f"investigative={len(source_texts['investigative_reports'])}, "
            f"financial={len(source_texts['financial_docs'])}"
        )

        # Build supplementary source sections for the prompt
        press_text = (
            "\n\n---\n\n".join(
                f"Press Release {i+1}:\n{t}"
                for i, t in enumerate(source_texts["press_releases"])
            )
            if source_texts["press_releases"]
            else "(No press releases available)"
        )

        truncated_orders = [t[:5000] for t in source_texts["court_orders"]]
        court_joined = ""
        court_total = 0
        for i, t in enumerate(truncated_orders):
            header = f"Court Order {i+1}:\n"
            sep = "\n\n---\n\n" if court_joined else ""
            chunk = sep + header + t
            if court_total + len(chunk) > 15000:
                break
            court_joined += chunk
            court_total += len(chunk)
        court_text = court_joined if court_joined else "(No court orders available)"

        investigative_text = (
            "\n\n---\n\n".join(
                f"Investigative Report {i+1}:\n{t}"
                for i, t in enumerate(source_texts["investigative_reports"])
            )
            if source_texts["investigative_reports"]
            else "(No investigative reports available)"
        )

        financial_text = (
            "\n\n---\n\n".join(
                f"Financial Document {i+1}:\n{t}"
                for i, t in enumerate(source_texts["financial_docs"])
            )
            if source_texts["financial_docs"]
            else "(No financial documents available)"
        )

        # LLM Call #1: Extract structured data from ALL sources
        bigo = f"रू {case.bigo:,}" if case.bigo else "उल्लेख छैन"
        prompt = EXTRACTION_USER_PROMPT.format(
            case_id=case.case_id,
            case_title=case.title,
            court_cases=(
                json.dumps(case.court_cases, ensure_ascii=False)
                if case.court_cases
                else "None"
            ),
            bigo=bigo,
            press_release_texts=press_text[:10000],
            court_order_texts=court_text[:32000],
            other_texts=(
                f"Supplementary:\n{investigative_text[:3000]}\n\n{financial_text[:4000]}"
                if (
                    source_texts["investigative_reports"]
                    or source_texts["financial_docs"]
                )
                else "(No additional documents)"
            ),
            source_quality_notes_section=(
                "NOTE: Press release text may contain character-level corruption "
                "from PDF extraction (~60-70% accuracy). Court order text is "
                "typically 95%+ clean. Trust court order over press release when "
                "they conflict on facts, names, dates, or legal citations."
            ),
        )

        self.stdout.write(
            f"  [1/2] Extracting structured data "
            f"(charge_sheet={len(source_texts['charge_sheet'])} chars)..."
        )
        logger.info(
            "Case %s: step=extract status=calling charge_sheet=%d press_releases=%d court_orders=%d",
            case.case_id,
            len(source_texts["charge_sheet"]),
            len(source_texts["press_releases"]),
            len(source_texts["court_orders"]),
        )

        raw = self._call_llm(
            model,
            base_url,
            api_key,
            timeout,
            is_opencode,
            EXTRACTION_SYSTEM_PROMPT,
            prompt,
            step_tag="extract",
        )
        try:
            extracted_json = json.loads(raw or "")
        except json.JSONDecodeError:
            extracted_json = None
        if not isinstance(extracted_json, dict):
            self.stats["llm_extraction_failures"] += 1
            self.stats["cases_failed"] += 1
            snippet = (raw or "")[:500]
            logger.error(
                "Case %s: step=extract status=failed reason=invalid_json raw_len=%d snippet=%r",
                case.case_id,
                len(raw or ""),
                snippet,
            )
            self.stdout.write(
                self.style.ERROR(
                    f"  FAILED: Extraction returned invalid JSON "
                    f"(raw: {snippet[:200]})"
                )
            )
            return

        logger.info(
            "Case %s: step=extract status=ok keys=%s",
            case.case_id,
            list(extracted_json.keys()),
        )

        # Court case discovery: find matching court orders from NGM judicial DB
        self.stdout.write("  [court] Discovering matching court cases...")
        discovery = self._discover_court_cases(case, extracted_json)
        logger.info(
            "Case %s: step=discover status=ok court_cases_found=%d court_order_texts=%d",
            case.case_id,
            len(discovery.get("court_cases_found", [])),
            len(discovery.get("court_order_texts", [])),
        )
        if discovery.get("court_cases_found"):
            self.stdout.write(
                f"  Found {len(discovery['court_cases_found'])} matching court case(s) in NGM database"
            )
            # Backfill case.court_cases from NGM discoveries so future runs
            # benefit without re-querying NGM.
            existing = set(case.court_cases or [])
            new_entries = []
            for found in discovery["court_cases_found"]:
                entry = f"{found['court_identifier']}:{found['case_number']}"
                if entry not in existing:
                    new_entries.append(entry)
            if new_entries and not dry_run:
                case.court_cases = (case.court_cases or []) + new_entries
                case.save(update_fields=["court_cases", "updated_at"])
                logger.info(
                    "Case %s: backfilled court_cases with %d new entries: %s",
                    case.case_id,
                    len(new_entries),
                    new_entries,
                )

        # LLM Call #2: Format into Markdown overview
        fmt_context = {
            "extracted_json": json.dumps(extracted_json, ensure_ascii=False, indent=2),
        }
        if discovery.get("court_cases_found"):
            fmt_context["court_case_metadata"] = json.dumps(
                discovery["court_cases_found"], ensure_ascii=False, indent=2
            )
        else:
            fmt_context["court_case_metadata"] = "(No court case metadata found)"

        fmt_prompt = FORMATTING_USER_PROMPT.format(**fmt_context)
        self.stdout.write("  [2/2] Formatting Markdown overview...")
        logger.info("Case %s: step=format status=calling", case.case_id)
        raw = self._call_llm(
            model,
            base_url,
            api_key,
            timeout,
            is_opencode,
            FORMATTING_SYSTEM_PROMPT,
            fmt_prompt,
            step_tag="format",
        )
        try:
            formatted = json.loads(raw or "")
        except json.JSONDecodeError:
            formatted = None
        if not isinstance(formatted, dict):
            self.stats["llm_formatting_failures"] += 1
            self.stats["cases_failed"] += 1
            snippet = (raw or "")[:500]
            logger.error(
                "Case %s: step=format status=failed reason=invalid_json raw_len=%d snippet=%r",
                case.case_id,
                len(raw or ""),
                snippet,
            )
            self.stdout.write(
                self.style.ERROR(
                    f"  FAILED: Formatting returned invalid JSON "
                    f"(raw: {snippet[:200]})"
                )
            )
            return

        short_description = (formatted.get("short_description") or "").strip()
        description = (formatted.get("description") or "").strip()

        logger.info(
            "Case %s: step=format status=ok short_description=%d description=%d",
            case.case_id,
            len(short_description),
            len(description),
        )

        valid, issues = self._validate_overview(short_description, description)
        for issue in issues:
            self.stdout.write(
                self.style.WARNING(self._safe_output_text(f"  Quality issue: {issue}"))
            )
            logger.warning(
                "Case %s: step=validate status=issue reason=%s", case.case_id, issue
            )
        if not valid:
            self.stats["cases_failed"] += 1
            logger.error(
                "Case %s: step=validate status=failed reason=%s",
                case.case_id,
                "; ".join(issues),
            )
            self.stdout.write(
                self.style.ERROR("  FAILED: Overview failed required quality gates")
            )
            return

        if dry_run:
            self.stats["cases_enriched"] += 1
            logger.info(
                "Case %s: step=save status=dry_run description=%d",
                case.case_id,
                len(description),
            )
            self.stdout.write(
                self.style.SUCCESS(
                    f"  [DRY RUN] Would save overview ({len(description)} chars)"
                )
            )
            return

        case.short_description = short_description
        if not case.description or force:
            case.description = description
        case.save(update_fields=["short_description", "description", "updated_at"])
        self.stats["cases_enriched"] += 1
        logger.info(
            "Case %s: step=save status=ok short_description=%d description=%d",
            case.case_id,
            len(short_description),
            len(description),
        )
        self.stdout.write(
            self.style.SUCCESS(f"  ENRICHED: Saved overview ({len(description)} chars)")
        )

    # ── LLM calls ──────────────────────────────────────────────────

    def _call_llm(
        self, model, base_url, api_key, timeout, is_opencode, system_prompt, prompt,
        step_tag="LLM",
    ):
        backend = "stream" if is_opencode else "anthropic"
        logger.info(
            "LLM call: backend=%s model=%s prompt_len=%d timeout=%d",
            backend,
            model,
            len(prompt),
            timeout,
        )
        if is_opencode:
            return self._call_llm_stream(
                model, base_url, api_key, timeout, system_prompt, prompt, step_tag=step_tag,
            )
        return self._call_llm_anthropic(
            model, base_url, api_key, timeout, system_prompt, prompt
        )

    def _call_llm_stream(
        self, model, base_url, api_key, timeout, system_prompt, prompt, step_tag="LLM"
    ):
        """Call LLM via OpenAI-compatible proxy with streaming (SDK-based).

        Streaming forces Chunked Transfer Encoding at the HTTP layer,
        which resets Cloudflare's idle timer and prevents HTTP 524
        gateway timeouts during long generations.
        """
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,  # handled by our own retry loop
        )
        total_chars = len(prompt)
        logger.info(
            "LLM stream: step=%s model=%s prompt_len=%d timeout=%d",
            step_tag, model, total_chars, timeout,
        )
        if total_chars > _LLM_PROMPT_SIZE_WARN:
            logger.warning(
                "LLM stream: step=%s HIGH PAYLOAD WARNING: %d chars (>%d)",
                step_tag, total_chars, _LLM_PROMPT_SIZE_WARN,
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(1, MAX_LLM_RETRIES + 1):
            try:
                start_time = time.time()
                stream = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    temperature=0.1,
                    max_tokens=6000,
                )
                accumulated = ""
                ttft_recorded = False
                resolved_model = "unknown"
                for chunk in stream:
                    if not ttft_recorded:
                        ttft = time.time() - start_time
                        resolved_model = getattr(chunk, "model", "unknown") or "unknown"
                        logger.info(
                            "LLM stream: step=%s TTFT=%.2fs resolved_model=%s",
                            step_tag, ttft, resolved_model,
                        )
                        ttft_recorded = True
                    if chunk.choices and chunk.choices[0].delta.content is not None:
                        accumulated += chunk.choices[0].delta.content

                total_duration = time.time() - start_time
                logger.info(
                    "LLM stream: step=%s attempt=%d succeeded "
                    "gen_time=%.2fs response_len=%d model=%s",
                    step_tag, attempt, total_duration, len(accumulated), resolved_model,
                )

                if not accumulated.strip():
                    logger.warning(
                        "LLM stream: step=%s attempt=%d — empty response", step_tag, attempt,
                    )
                    if attempt < LLM_EMPTY_RESPONSE_RETRIES:
                        wait = 2**attempt + 5  # base 2s + 5s buffer for proxy cold-start
                        self.stdout.write(
                            self.style.WARNING(
                                f"  LLM empty response attempt {attempt}, retry in {wait}s..."
                            )
                        )
                        time.sleep(wait)
                        continue
                    raise CommandError(
                        f"LLM empty response after {LLM_EMPTY_RESPONSE_RETRIES} attempts"
                    )
                return _extract_json_body(accumulated)

            except Exception as exc:
                logger.warning(
                    "LLM stream: step=%s attempt=%d — %s: %s",
                    step_tag, attempt, type(exc).__name__, exc,
                )
                if attempt < MAX_LLM_RETRIES:
                    wait = 2**attempt
                    self.stdout.write(
                        self.style.WARNING(
                            f"  LLM error on attempt {attempt}, retry in {wait}s..."
                        )
                    )
                    time.sleep(wait)
                    continue
                raise CommandError(
                    f"LLM call failed after {MAX_LLM_RETRIES} attempts: {exc}"
                ) from exc
        return None

    def _call_llm_anthropic(
        self, model, base_url, api_key, timeout, system_prompt, prompt
    ):
        try:
            import anthropic
        except ImportError as exc:
            raise CommandError(
                "anthropic package is required for direct Anthropic API calls"
            ) from exc
        client = anthropic.Anthropic(api_key=api_key, base_url=base_url or None)
        for attempt in range(MAX_LLM_RETRIES):
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=6000,
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    timeout=timeout,
                )
                if not response.content:
                    logger.warning(
                        "LLM anthropic: attempt %d — empty response content",
                        attempt + 1,
                    )
                    self.stdout.write(
                        self.style.WARNING("  LLM returned empty response")
                    )
                    if attempt < MAX_LLM_RETRIES - 1:
                        wait = 2**attempt
                        time.sleep(wait)
                        continue
                    raise CommandError(
                        f"LLM returned empty response after {MAX_LLM_RETRIES} attempts"
                    )
                raw = response.content[0].text
                response_model = getattr(response, "model", "")
                logger.info(
                    "LLM anthropic: attempt %d succeeded — response_len=%d model=%s request_model=%s",
                    attempt + 1,
                    len(raw),
                    response_model,
                    model,
                )
                if response_model != model and normalize_model(response_model) != model:
                    raise ValueError(
                        f"Model mismatch: sent {model!r} but "
                        f"response from {response_model!r}"
                    )
                return _extract_json_body(raw)
            except Exception as exc:
                logger.warning(
                    "LLM anthropic: attempt %d — %s: %s",
                    attempt + 1,
                    type(exc).__name__,
                    exc,
                )
                if attempt < MAX_LLM_RETRIES - 1:
                    wait = 2**attempt
                    self.stdout.write(self.style.WARNING(f"  Retrying in {wait}s..."))
                    time.sleep(wait)
                    continue
                raise CommandError(
                    f"LLM call failed after {MAX_LLM_RETRIES} attempts: {exc}"
                ) from exc

    # ── Source conversion ──────────────────────────────────────────

    def _get_markitdown(self):
        if self._markitdown is None:
            try:
                from markitdown import MarkItDown
            except ImportError as exc:
                raise CommandError(
                    "markitdown is required for case overview enrichment"
                ) from exc
            self._markitdown = MarkItDown(enable_plugins=True)
        return self._markitdown

    def _convert_source_to_markdown(self, source):
        converter = self._get_markitdown()
        with tempfile.TemporaryDirectory(prefix="overview-enrichment-") as tmp_dir:
            output_dir = Path(tmp_dir)
            # Try uploaded files first
            try:
                temp_path = self._download_source_to_path(source, output_dir)
                if temp_path:
                    if temp_path.suffix.lower() == ".doc":
                        logger.debug(
                            "Source %s: legacy .doc file '%s' — markitdown may fail, OLE fallback will attempt next.",
                            source.source_id,
                            temp_path.name,
                        )
                    result = converter.convert_uri(temp_path.resolve().as_uri())
                    if result.text_content and len(result.text_content.strip()) >= 50:
                        return result.text_content
            except Exception:
                logger.debug(
                    "Failed to convert uploaded file for %s",
                    source.source_id,
                    exc_info=True,
                )
            # Try URLs
            for url in self._ranked_source_urls(source):
                try:
                    temp_path = self._download_url_to_path(
                        url, source.source_id, output_dir
                    )
                    if temp_path:
                        if temp_path.suffix.lower() == ".doc":
                            logger.debug(
                                "Source %s: downloaded file '%s' is .doc format — "
                                "markitdown may fail, OLE fallback will attempt next.",
                                source.source_id,
                                temp_path.name,
                            )
                        result = converter.convert_uri(temp_path.resolve().as_uri())
                        if (
                            result.text_content
                            and len(result.text_content.strip()) >= 50
                        ):
                            return result.text_content
                except Exception:
                    logger.debug(
                        "Failed to convert URL %s for %s",
                        url,
                        source.source_id,
                        exc_info=True,
                    )
                    continue
            logger.warning(
                "Source %s: no convertible content from files or URLs (description=%d chars, skipped)",
                source.source_id,
                len(source.description or ""),
            )
            raise CommandError(
                f"Unable to convert source {source.source_id}: no convertible content"
            )

    def _download_source_to_path(self, source, output_dir):
        if source.uploaded_file:
            filename = _sanitize_download_filename(
                source.uploaded_filename or source.uploaded_file.name,
                source.source_id,
            )
            out_path = _confined_output_path(output_dir, filename)
            with source.uploaded_file.open("rb") as in_file:
                _copy_stream_to_path_with_limit(in_file, out_path)
            return out_path
        uploaded = source.uploaded_files.first()
        if uploaded and uploaded.file:
            filename = _sanitize_download_filename(
                uploaded.filename or uploaded.file.name, source.source_id
            )
            out_path = _confined_output_path(output_dir, filename)
            with uploaded.file.open("rb") as in_file:
                _copy_stream_to_path_with_limit(in_file, out_path)
            return out_path
        return None

    def _download_url_to_path(self, url, source_id, output_dir):
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"Invalid URL {url!r}")
        _validate_host_safety(parsed.hostname)
        out_path = _confined_output_path(
            output_dir, _sanitize_download_filename(parsed.path, source_id)
        )
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/146.0.0.0 Safari/537.36"
                    )
                },
            )
            with urllib.request.build_opener(_SafeRedirectHandler()).open(
                request, timeout=30
            ) as response:
                _copy_stream_to_path_with_limit(response, out_path)
            return out_path
        except OSError:
            out_path.unlink(missing_ok=True)
            return None

    def _ranked_source_urls(self, source):
        urls = [
            url.strip()
            for url in (source.url or [])
            if isinstance(url, str) and url.strip()
        ]
        direct = [url for url in urls if _is_direct_document_url(url)]
        other = [url for url in urls if url not in direct]
        direct.sort(key=_source_url_priority, reverse=True)
        return direct + other

    # ── Validation ─────────────────────────────────────────────────

    def _validate_overview(self, short_description, description):
        issues = []
        valid = True

        # Hard gate: required section क
        if "क) अभियोगदावीको सार" not in description:
            issues.append("Missing required section: क) अभियोगदावीको सार")
            valid = False

        # Hard gate: description has content
        if not description or len(description) < 100:
            issues.append("Description too short")
            valid = False

        # Hard gate: short_description minimum length
        if not short_description or len(short_description) < 50:
            issues.append("short_description too short (< 50 chars)")
            valid = False

        # Hard gate: short_description maximum length
        if len(short_description) > 1000:
            issues.append("short_description too long (> 1000 chars)")
            valid = False

        # Hard gate: Devanagari ratio ≥80% of alphabetic characters
        alphabetic_chars = ALPHABETIC_RE.findall(description)
        if alphabetic_chars:
            ratio = len(DEVANAGARI_ALPHABETIC_RE.findall(description)) / len(
                alphabetic_chars
            )
            if ratio < 0.80:
                issues.append(f"Devanagari alphabetic ratio {ratio:.2f} below 80%")
                valid = False

        # Hard gate: no raw HTML tags
        if re.search(
            r"<\s*(h[1-6]|table|tr|td|th|div|p|span|br|ul|ol|li|a)\b",
            description,
            re.IGNORECASE,
        ):
            issues.append("Raw HTML tags found in description")
            valid = False

        # Hard gate: no placeholder text (expanded set)
        if any(
            token.lower() in description.lower()
            for token in [
                "[insert]",
                "[tbd]",
                "[todo]",
                "[draft]",
                "[अज्ञात]",
                "[placeholder]",
                "[n/a]",
                "[ai-generated]",
                "[ai generated]",
                "[add content]",
                "[to be written]",
            ]
        ):
            issues.append("Placeholder text found")
            valid = False

        # Soft gate: detect common PDF-corruption garbling patterns
        # (non-Devanagari chars in suspicious positions, garbled legal terms)
        garbled_indicators = detect_corrupted_text(description)
        for indicator in garbled_indicators:
            issues.append(f"Corrupted text detected: {indicator}")
            # Soft gate — does not fail validation, only warns

        return valid, issues

    # ── Helpers ────────────────────────────────────────────────────

    def _skip_no_content(self, case, note, dry_run):
        self.stats["cases_no_content"] += 1
        logger.warning("Case %s: skipped — %s", case.case_id, note)
        if not dry_run:
            self._record_missing_details(case, f"enrich_case_overview: {note}")
        self.stdout.write(
            self.style.WARNING(self._safe_output_text(f"  SKIPPED: {note}"))
        )

    def _record_missing_details(self, case, note):
        current = case.missing_details or ""
        if note not in current:
            case.missing_details = f"{current}\n{note}" if current else note
            case.save(update_fields=["missing_details", "updated_at"])

    @staticmethod
    def _safe_output_text(text: str) -> str:
        """Strip non-ASCII for console safety (prevents UnicodeEncodeError on cp1252)."""
        return text.encode("ascii", errors="replace").decode("ascii")

    def _describe_source(self, source):
        if source.uploaded_file:
            return (
                f"uploaded file: "
                f"{source.uploaded_filename or source.uploaded_file.name} "
                f"({source.source_id})"
            )
        uploaded = source.uploaded_files.first()
        if uploaded and uploaded.file:
            return (
                f"uploaded file: {uploaded.filename or uploaded.file.name} "
                f"({source.source_id})"
            )
        urls = self._ranked_source_urls(source)
        if urls:
            parsed = urllib.parse.urlsplit(urls[0])
            return (
                f"URL: {urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, '', ''))} "
                f"({source.source_id})"
            )
        if source.description and len(source.description.strip()) >= 500:
            return f"description fallback ({source.source_id})"
        return f"{source.source_id} (no content)"

    def _print_summary(self, dry_run, elapsed_str):
        self.stdout.write("\n" + "=" * 60)
        self.stdout.write(
            self.style.WARNING(f"{'[DRY RUN] ' if dry_run else ''}SUMMARY")
        )
        self.stdout.write("=" * 60)
        self.stdout.write(f"Total time:              {elapsed_str}")
        self.stdout.write(f"Cases processed:         {self.stats['cases_processed']}")
        self.stdout.write(
            self.style.SUCCESS(
                f"Cases enriched:          {self.stats['cases_enriched']}"
            )
        )
        self.stdout.write(
            self.style.WARNING(
                f"Cases skipped:           {self.stats['cases_skipped']}"
            )
        )
        self.stdout.write(
            self.style.WARNING(
                f"Cases no content:        {self.stats['cases_no_content']}"
            )
        )
        if self.stats["cases_failed"]:
            self.stdout.write(
                self.style.ERROR(
                    f"Cases failed:            {self.stats['cases_failed']}"
                )
            )
        if self.stats["llm_extraction_failures"]:
            self.stdout.write(
                self.style.WARNING(
                    f"LLM extraction failures: {self.stats['llm_extraction_failures']}"
                )
            )
        if self.stats["llm_formatting_failures"]:
            self.stdout.write(
                self.style.WARNING(
                    f"LLM formatting failures: {self.stats['llm_formatting_failures']}"
                )
            )
        self.stdout.write("=" * 60)
        if dry_run:
            self.stdout.write(
                self.style.WARNING(
                    "\nThis was a dry run. No changes were made to the database."
                )
            )


def _has_charge_sheet_keywords(source) -> bool:
    corpus = (source.title + " " + (source.description or "")).lower()
    return any(kw in corpus for kw in CHARGE_SHEET_KEYWORDS)


def detect_corrupted_text(text: str) -> list:
    """Detect common PDF-corruption patterns in Nepali legal text.

    Returns a list of human-readable issue strings (empty if clean).
    These are *soft* gates — they do not cause validation failure,
    only quality warnings.
    """
    issues = []
    # 1. Known garbled legal terms
    for term in GARBLED_LEGAL_TERMS:
        if term in text:
            issues.append(f"garbled term '{term}' (PDF extraction artifact)")
    # 2. Excessive single-space-separated Devanagari characters
    #    (PDF often inserts spaces between adjacent Nepali characters)
    if EXCESSIVE_SPACED_CHARS_RE.search(text):
        issues.append(
            "excessive single-space separation between Devanagari characters "
            "(likely PDF extraction artifact)"
        )
    return issues


def _has_press_release_keywords(source) -> bool:
    corpus = (source.title + " " + (source.description or "")).lower()
    return any(kw in corpus for kw in PRESS_RELEASE_KEYWORDS)


def _has_ngm_store_url(source) -> bool:
    urls = source.url or []
    return any(
        "ngm-store.jawafdehi.org" in (url or "")
        for url in (urls if isinstance(urls, list) else [])
    )


def _is_direct_document_url(url):
    path = urllib.parse.unquote(urllib.parse.urlparse(url).path).lower()
    return path.endswith((".pdf", ".doc", ".docx"))


def _source_url_priority(url):
    parsed = urllib.parse.urlparse(url)
    path = urllib.parse.unquote(parsed.path).lower()
    return (
        int(parsed.netloc.lower() == "ngm-store.jawafdehi.org"),
        int(path.endswith(".pdf")),
        int(path.endswith((".pdf", ".doc", ".docx"))),
    )


def _truncate_long_doc(text: str) -> str:
    """Head+tail extraction for long court orders.

    Short docs (≤COURT_ORDER_FULL_MAX): return full text.
    Long docs: return head (identity, charges, narrative) + tail (verdict, sentencing).
    Middle sections (witness playback, evidence replay) are skipped — the LLM
    captures those from the head's summary paragraphs.
    """
    if len(text) <= COURT_ORDER_FULL_MAX:
        return text
    head = text[:COURT_ORDER_HEAD_CHARS]
    tail = text[-COURT_ORDER_TAIL_CHARS:]
    return (
        head + "\n\n[... मध्य भाग संक्षिप्त गरिएको — तल फैसला/सजाय खण्ड ...]\n\n" + tail
    )
