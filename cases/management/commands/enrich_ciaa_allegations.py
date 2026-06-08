"""
Management command to enrich CIAA DRAFT cases with key allegations extracted
via LLM from press release markdown content.

Usage::

    python manage.py enrich_ciaa_allegations --dry-run
    python manage.py enrich_ciaa_allegations --limit 10
    python manage.py enrich_ciaa_allegations --llm-model claude-sonnet-4-5 --verbose

Environment variables::

    ANTHROPIC_API_KEY        — API key for Anthropic (fallback)
    JAWAFDEHI_LLM_API_KEY    — API key for Jawafdehi LLM proxy
    JAWAFDEHI_LLM_PROXY_URL  — base URL for Jawafdehi LLM proxy
    JAWAFDEHI_LLM_TIMEOUT_SECONDS — timeout in seconds (default 300)
    OPENCODE_API_KEY         — API key for OpenCode Go
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

from cases.models import Case, CaseState, DocumentSource, SourceType
from cases.services.priority_case_loader import filter_by_priority, load_priority_cases

logger = logging.getLogger(__name__)

MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024
DOWNLOAD_CHUNK_SIZE = 16 * 1024
DEFAULT_OPENCODE_BASE = "https://opencode.ai/zen/go/v1"
DEFAULT_LLM_TIMEOUT = 300
MAX_LLM_RETRIES = 3
MINIMAX_MODELS = frozenset({"minimax-m2.5", "minimax-m2.7"})

# SSRF protection: block well-known internal/metadata endpoints.
# These are not configurable infrastructure — they are RFC-defined
# special-purpose addresses used by all major cloud providers.
_CLOUD_METADATA_IP = "169.254.169.254"  # NOSONAR — link-local, not configurable

_SSRF_BLOCKED_HOSTNAMES = frozenset(
    {
        "localhost",
        "metadata.google.internal",
        _CLOUD_METADATA_IP,
        "metadata",
        "0.0.0.0",
    }
)


def _validate_host_safety(hostname: str) -> None:
    host = hostname.lower().rstrip(".")
    if host in _SSRF_BLOCKED_HOSTNAMES:
        raise ValueError(
            f"Blocked internal host: {hostname!r}. "
            "Download sources must target public hosts only."
        )
    try:
        addrinfo = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ValueError(
            f"Cannot resolve host: {hostname!r}. "
            "Only resolvable public hosts are allowed for source downloads."
        ) from exc
    for info in addrinfo:
        addr = ipaddress.ip_address(info[4][0])
        if (
            addr.is_loopback
            or addr.is_private
            or addr.is_link_local
            or addr.is_reserved
        ):
            raise ValueError(
                f"Blocked internal address: {hostname!r} → {addr}. "
                "Download sources must target public IPs only."
            )


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlparse(newurl)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise urllib.error.HTTPError(
                req.full_url,
                code,
                f"Unsafe redirect scheme/host to {newurl}",
                headers,
                fp,
            )
        _validate_host_safety(parsed.hostname)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def normalize_model(model: str) -> str:
    model = model.strip()
    for prefix in ("opencode-go/", "openai:"):
        if model.startswith(prefix):
            model = model[len(prefix) :]
    return model


def normalize_base_url(url: str | None) -> str:
    if url and url.strip():
        url = url.strip().rstrip("/")
    else:
        url = os.environ.get("JAWAFDEHI_LLM_PROXY_URL", DEFAULT_OPENCODE_BASE).rstrip(
            "/"
        )
    if url.endswith("/zen/v1"):
        url = url.replace("/zen/v1", "/zen/go/v1")
    elif url.endswith("/zen/go"):
        url += "/v1"
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


def _llm_endpoint(base_url: str, model: str) -> str:
    normalized = normalize_model(model)
    base = base_url.rstrip("/")
    if normalized in MINIMAX_MODELS:
        return f"{base}/messages"
    return f"{base}/chat/completions"


def _llm_timeout(cli_timeout: int | None = None) -> int:
    def _validate(val: int) -> int:
        if val <= 0:
            raise CommandError(f"--llm-timeout must be > 0, got {val}")
        return val

    if cli_timeout is not None:
        return _validate(cli_timeout)
    env = os.environ.get("JAWAFDEHI_LLM_TIMEOUT_SECONDS")
    if env is not None:
        try:
            return _validate(int(env))
        except ValueError:
            pass
    return DEFAULT_LLM_TIMEOUT


_FATAL_CMDER_MSG_FRAGMENTS = frozenset(
    {"markitdown", "path confinement", "Refusing to write", "exceeds max size"}
)


def _is_missing_content_cmderror(exc: CommandError) -> bool:
    msg = str(exc)
    return not any(fragment in msg for fragment in _FATAL_CMDER_MSG_FRAGMENTS)


def _build_llm_opencode_body(
    normalized_model: str, is_minimax: bool, prompt: str
) -> str:
    if is_minimax:
        body = {
            "model": normalized_model,
            "max_tokens": 6000,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        }
    else:
        body = {
            "model": normalized_model,
            "max_tokens": 6000,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
        }
    return json.dumps(body)


def _parse_llm_opencode_response(payload: dict, is_minimax: bool) -> str:
    if is_minimax:
        return payload.get("content", [{}])[0].get("text", "")
    return payload.get("choices", [{}])[0].get("message", {}).get("content", "")


def _parse_llm_response_json(raw_text: str) -> dict:
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(raw_text):
        try:
            obj, end = decoder.raw_decode(raw_text, idx)
        except json.JSONDecodeError:
            break
        if isinstance(obj, dict) and ("choices" in obj or "content" in obj):
            return obj
        idx = end
    return decoder.raw_decode(raw_text)[0]


def _format_llm_http_error(status: int, body_snippet: str) -> str:
    msg = f"LLM HTTP {status}: {body_snippet[:300]}"
    if status == 429:
        msg += (
            " Hint: OpenCode Go usage limits may apply. "
            "Try reducing --limit or switching models."
        )
    return msg


def _extract_fenced_block(raw: str) -> str | None:
    if not raw.startswith("```"):
        return None
    lines = raw.split("\n")
    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip().startswith("```"):
            end_idx = i
            break
    if end_idx is None:
        return None
    inner = "\n".join(lines[1:end_idx]).strip()
    return inner if inner else None


SYSTEM_PROMPT = """You are a Nepali legal analyst extracting structured key allegations
from CIAA (Commission for the Investigation of Abuse of Authority) press releases.

Every allegation MUST:
1. Be factually grounded in the provided press release — NO fabrication
2. Be written in professional, accessible Nepali (नेपाली)
3. Focus on accused entities and alleged acts, not long name lists
4. Group accused people by office/role when many names appear, naming only the principal accused or role group when enough
5. Describe related entities by role when possible (for example, "एक निजी कम्पनी", "निर्माण व्यवसायी", or "सम्बन्धित उपभोक्ता समिति") unless the press release makes a name essential
6. Describe the specific misconduct mechanism (what was done and how)
7. Include the disputed amount (बिगो) when mentioned in the source, using readable Nepali-scale wording when possible (for example, "रु. ३८ करोडभन्दा बढी" instead of "रु. ३८,६७,१७,६४०")
8. Include the time period (date range or fiscal year) when specified
9. Be self-contained — understandable without additional context
10. Follow the established Jawafdehi allegation style (see examples below)

Return 2-3 allegations. Each allegation MUST be exactly one sentence.
The first allegation MUST be the most descriptive overview of the primary allegation.
The first allegation MUST mention the core institution/property/transaction, the alleged scheme, and the financial harm; it MUST NOT spend most of the sentence listing accused names.
The second and third allegations, if present, MUST be shorter supporting allegations that add a different mechanism or actor role; they MUST NOT restate the first allegation with different wording.
Use formal but clear Nepali.

DO NOT:
- Fabricate or embellish beyond the source text
- Use legal jargon without explanation
- State legal conclusions about guilt or innocence
- Write vague statements like "भ्रष्टाचार गरेको"
- Mix multiple unrelated misconducts into one allegation
- Start with a long comma-separated list of accused names when a role group can carry the allegation
- Produce near-duplicate allegations that repeat the same accused list and same misconduct
- Write remedies, requests, or procedural outcomes as allegations, such as asset-return demands, confiscation requests, charge filing, or punishment requests
- End allegations with attribution phrases such as "उल्लेख छ", "भनिएको छ", "जनाइएको छ", "देखिन्छ", or "आरोप छ"
- Include multiple sentences in one allegation
- List related entity names when a descriptive role is enough
- Use long comma-formatted Nepali amounts when a readable crore/lakh approximation is clearer

STRUCTURE the first allegation in Nepali as:
"मुख्य पदाधिकारी/भूमिका समूहले — कुन संस्था/सम्पत्ति/कारोबारमा — के योजना/कृत्य गरे — कसरी — कति रकम/हानि — कुन अवधिमा"
(Principal role group — institution/property/transaction — alleged scheme/action — mechanism — amount/harm — period)

STRUCTURE supporting allegations as shorter statements describing secondary mechanisms, supporting acts, or specific misuse patterns.
Each supporting allegation MUST still describe alleged misconduct by an accused actor, not a legal remedy or requested court outcome.
If the source lists many accused names, compress them into a role group such as "तत्कालीन अध्यक्ष र सञ्चालक समिति सदस्यहरू" unless one person's name is necessary to identify the case.

REFERENCE EXAMPLES from published Jawafdehi cases:

Example 1 (Illegal property accumulation):
"कमल राज गौतमले मिति २०५५/०१/०७ देखि २०७९/१२/२४ सम्म सार्वजनिक पद धारण
गर्दा वैध आयभन्दा रु. २,५१,७८,६८७.७१ बढी सम्पत्ति खर्च तथा लगानी गरी
गैरकानूनी रूपमा सम्पत्ति आर्जन गरेको।"

Example 2 (Procurement fraud):
"प्रतिवादीहरूको मिलेमतोमा काठमाडौं महानगरपालिकाको NCBW-KMC को ठेक्कामा
Pending Litigation नहुने विषयलाई Pending Litigation रहेको भनी गलत मूल्याङ्कन
प्रतिवेदन खडा गरी सार्वजनिक सम्पत्ति बदनियतपूर्वक हानि नोक्सानी पुर्याएको।"

Example 3 (Bribery and money laundering):
"मोहनबहादुर बस्नेतले नगर प्रमुख पदको दुरुपयोग गरी पद्मा कम्पनीहरू र राजु
प्रसाद कँडेललाई कर छुट र जग्गा उपलब्धता लगायत अनुचित लाभ पुर्याई सो बापत
करिब रु. ९.२२ करोड घुस/रिसवत लिएको।"

Example 4 (Embezzlement):
"प्रतिवादीहरूको मिलेमतोमा हुलाक बचत बैङ्कमा बचतकर्ताहरूको निक्षेप रकम
बैङ्क दाखिला नगरी अपचलन गरी हिनामिना गरेको।"
"""

USER_PROMPT_TEMPLATE = """Extract 2-3 key allegation statements from this CIAA press release.

Case title: {case_title}
Bigo amount: {bigo}

Instructions:
- Each allegation must be exactly one complete, self-contained sentence in Nepali
- Do not end any allegation with attribution wording such as "उल्लेख छ", "भनिएको छ", "जनाइएको छ", "देखिन्छ", or "आरोप छ"
- Make the first allegation a descriptive overview of the primary allegation
- Make the first allegation about substance: institution/property/transaction, alleged scheme, mechanism, amount or harm, and period when available
- Make the second and third allegations shorter supporting allegations
- Make each allegation distinct; do not repeat the same accused list and same misconduct in multiple sentences
- Each allegation must describe alleged misconduct by accused actors, not remedies or procedural outcomes such as asset-return demands, confiscation requests, charge filing, or punishment requests
- Focus on accused entities and their acts; do not include related entity names unless essential
- Prefer role descriptions for related entities, such as "एक निजी कम्पनी", "निर्माण व्यवसायी", or "सम्बन्धित उपभोक्ता समिति"
- When many accused names are listed, group them by role such as "तत्कालीन अध्यक्ष र सञ्चालक समिति सदस्यहरू"; include individual names only when needed to identify the principal accused or a distinct act
- Include names and positions of accused entities when available, but do not let name lists dominate the allegation
- Include amounts and time periods when available, but express large amounts readably in Nepali scale when possible, such as "रु. ३८ करोडभन्दा बढी" instead of "रु. ३८,६७,१७,६४०"
- Extract distinct allegations, not variations of the same claim

Press release text:

{press_release}

IMPORTANT: Return ONLY a valid JSON object with an "allegations" key.
Example:
{{"allegations": ["पहिलो मुख्य आरोप...", "दोस्रो मुख्य आरोप..."]}}
No explanations, no markdown, no text outside the JSON object."""


def _sanitize_download_filename(filename: str | None, source_id: str) -> str:
    raw = (filename or "").strip()
    if not raw:
        return f"{source_id}.bin"

    decoded = urllib.parse.unquote(raw)
    candidate = Path(decoded).name.strip()

    if candidate in {"", ".", ".."}:
        return f"{source_id}.bin"

    candidate = candidate.replace("\x00", "")
    candidate = re.sub(r"[<>:\"/\\|?*]+", "_", candidate).rstrip(" .")
    if candidate in {"", ".", ".."}:
        return f"{source_id}.bin"

    max_len = 200
    if len(candidate) <= max_len:
        return candidate

    suffix = "".join(Path(candidate).suffixes)
    stem = candidate[: -len(suffix)] if suffix else candidate
    digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()[:10]

    stem_budget = max_len - len(suffix) - len(digest) - 1
    if stem_budget < 1:
        return f"{source_id}-{digest}{suffix}"[:max_len]

    truncated_stem = stem[:stem_budget].rstrip(" .-_")
    if not truncated_stem:
        truncated_stem = source_id

    return f"{truncated_stem}-{digest}{suffix}"


def _confined_output_path(output_dir: Path, filename: str) -> Path:
    output_dir_resolved = output_dir.resolve()
    out_path = (output_dir / filename).resolve()
    if output_dir_resolved not in out_path.parents:
        raise CommandError(f"Refusing to write outside output directory: '{filename}'")
    return out_path


def _copy_stream_to_path_with_limit(in_file: Any, out_path: Path) -> None:
    total_bytes = 0
    try:
        with out_path.open("wb") as out_file:
            while True:
                chunk = in_file.read(DOWNLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_DOWNLOAD_BYTES:
                    raise CommandError(
                        f"Downloaded source exceeds max size of {MAX_DOWNLOAD_BYTES} bytes."
                    )
                out_file.write(chunk)
    except OSError:
        out_path.unlink(missing_ok=True)
        raise
    except CommandError:
        out_path.unlink(missing_ok=True)
        raise


class Command(BaseCommand):
    help = "Extract key allegations from CIAA press release content using LLM"

    def __init__(self):
        super().__init__()
        self.stats = {
            "cases_processed": 0,
            "cases_enriched": 0,
            "cases_skipped": 0,
            "cases_failed": 0,
            "cases_no_content": 0,
            "allegation_counts": {},
        }

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
            help="Process only N cases (useful for testing)",
        )
        parser.add_argument(
            "--llm-model",
            type=str,
            default=os.environ.get("JAWAFDEHI_ALLEGATION_MODEL", "claude-sonnet-4-5"),
            help="Model id (accepts opencode-go/ prefix, defaults to claude-sonnet-4-5)",
        )
        parser.add_argument(
            "--llm-base-url",
            type=str,
            default=None,
            help="Base URL for LLM API (env: JAWAFDEHI_LLM_PROXY_URL, "
            "default: https://opencode.ai/zen/go/v1)",
        )
        parser.add_argument(
            "--llm-api-key",
            type=str,
            default=None,
            help="API key (env: JAWAFDEHI_LLM_API_KEY, OPENCODE_API_KEY, "
            "ANTHROPIC_API_KEY)",
        )
        parser.add_argument(
            "--llm-timeout",
            type=int,
            default=None,
            help="LLM request timeout in seconds (env: JAWAFDEHI_LLM_TIMEOUT_SECONDS, "
            "default: 300)",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enable detailed debug logging",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Re-process cases that already have key_allegations",
        )
        parser.add_argument(
            "--case-id",
            type=str,
            default=None,
            help="Process a specific case by case_id",
        )
        parser.add_argument(
            "--priority",
            action="store_true",
            help="Enrich only cases in the priority case list",
        )
        parser.add_argument(
            "--all",
            action="store_true",
            dest="all_cases",
            help="Enrich all DRAFT CIAA cases (explicit, same as default)",
        )
        parser.add_argument(
            "--base-url",
            type=str,
            default=None,
            help="Deprecated: use --llm-base-url instead. Base URL for LLM API.",
        )

    def handle(self, *args, **options):
        dry_run = options["dry_run"]
        limit = options["limit"]
        model = options["llm_model"]
        base_url = options["llm_base_url"] or options.get("base_url")
        llm_api_key = options.get("llm_api_key")
        llm_timeout = options.get("llm_timeout")
        verbose = options["verbose"]
        force = options["force"]
        case_id = options.get("case_id")
        priority = options["priority"]
        all_cases_flag = options.get("all_cases")

        if priority and case_id:
            raise CommandError("--priority and --case-id are mutually exclusive")

        if not priority and not all_cases_flag:
            self.stdout.write(
                self.style.NOTICE(
                    "Processing all DRAFT CIAA cases (default). "
                    "Use --all to make this explicit or --priority to filter."
                )
            )

        if verbose:
            logger.setLevel(logging.DEBUG)

        model = normalize_model(model)
        base_url = normalize_base_url(base_url)
        is_opencode = "anthropic.com" not in base_url
        api_key = resolve_api_key(llm_api_key, is_anthropic=not is_opencode)
        timeout = _llm_timeout(llm_timeout)

        self.stdout.write(
            self.style.WARNING(
                f"{'[DRY RUN] ' if dry_run else ''}Starting CIAA allegation enrichment..."
            )
        )

        start_time = time.time()

        cases = self._get_eligible_cases(limit, force, case_id, priority)
        self.stdout.write(f"Found {len(cases)} eligible CIAA DRAFT case(s) to process")

        self._fetch_source_cache(cases)

        is_opencode = "anthropic.com" not in base_url

        for idx, case in enumerate(cases, 1):
            try:
                self.stdout.write(
                    f"\n[{idx}/{len(cases)}] {case.case_id} - {case.title[:80]}..."
                )
                self._process_case(
                    case, model, base_url, api_key, timeout, is_opencode, dry_run
                )
            except Exception as e:
                self.stats["cases_failed"] += 1
                logger.exception(f"Error processing {case.case_id}: {e}")
                self.stdout.write(self.style.ERROR(f"FAILED: {case.case_id} - {e}"))

        elapsed = time.time() - start_time
        mins, secs = divmod(int(elapsed), 60)
        elapsed_str = f"{mins}m {secs}s" if mins else f"{secs}s"
        self._print_summary(dry_run, elapsed_str)

    def _init_client(self, api_key, base_url):
        if "anthropic.com" in (base_url or ""):
            import anthropic

            return anthropic.Anthropic(api_key=api_key, base_url=base_url or None)
        return None

    def _build_llm_headers(self, api_key: str) -> dict:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "JawafdehiAPI/1.0 enrich_ciaa_allegations",
        }

    def _call_llm_opencode(
        self,
        model: str,
        base_url: str,
        api_key: str,
        timeout: int,
        prompt: str,
    ) -> str | None:
        endpoint = _llm_endpoint(base_url, model)
        normalized_model = normalize_model(model)
        headers = self._build_llm_headers(api_key)
        is_minimax = normalized_model in MINIMAX_MODELS
        body = _build_llm_opencode_body(normalized_model, is_minimax, prompt)

        last_status = None
        last_body_snippet = ""

        for attempt in range(1, MAX_LLM_RETRIES + 1):
            try:
                req = urllib.request.Request(
                    endpoint,
                    data=body.encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    raw_text = resp.read().decode("utf-8")
                payload = _parse_llm_response_json(raw_text)
                raw = _parse_llm_opencode_response(payload, is_minimax)
                logger.debug(f"LLM response: {raw[:100]}...")
                return raw

            except urllib.error.HTTPError as e:
                last_status = e.code
                try:
                    last_body_snippet = e.read().decode("utf-8", errors="replace")[:500]
                except Exception:
                    last_body_snippet = "<unreadable>"
                if self._should_retry_llm_http(attempt, last_status):
                    continue
                raise CommandError(
                    _format_llm_http_error(last_status, last_body_snippet)
                ) from e

            except OSError as e:
                if self._should_retry_llm_network(attempt, e):
                    continue
                raise CommandError(
                    f"LLM connection failed after {MAX_LLM_RETRIES} attempts: {e}"
                ) from e

        return None

    def _should_retry_llm_http(self, attempt: int, status: int) -> bool:
        if attempt >= MAX_LLM_RETRIES:
            return False
        if status not in (429, 503):
            return False
        wait = 2**attempt
        hint = ""
        if status == 429:
            hint = (
                " (OpenCode Go usage limits may apply; "
                "consider reducing --limit or using a different model)"
            )
        self.stdout.write(
            self.style.WARNING(
                f"  LLM {status} on attempt {attempt}, retrying in {wait}s...{hint}"
            )
        )
        time.sleep(wait)
        return True

    def _should_retry_llm_network(self, attempt: int, error: Exception) -> bool:
        if attempt >= MAX_LLM_RETRIES:
            return False
        wait = 2**attempt
        self.stdout.write(
            self.style.WARNING(
                f"  LLM connection error on attempt {attempt} "
                f"({error}), retrying in {wait}s..."
            )
        )
        time.sleep(wait)
        return True

    def _get_eligible_cases(self, limit, force, case_id, priority=False):
        queryset = Case.objects.filter(state=CaseState.DRAFT)

        if case_id:
            queryset = queryset.filter(case_id=case_id)

        if priority:
            priority_list = load_priority_cases()
            logger.info(
                "Priority mode: loaded %d case numbers across all fiscal years",
                len(priority_list),
            )
            queryset = filter_by_priority(queryset, priority_list)

        eligible = self._filter_eligible(list(queryset), force)

        if limit is not None:
            if limit < 0:
                raise CommandError(f"--limit must be >= 0, got {limit}")
            eligible = eligible[:limit] if limit > 0 else []

        return eligible

    def _filter_eligible(self, cases, force):
        result = []
        for case in cases:
            if not force and case.key_allegations:
                continue
            if case.court_cases and isinstance(case.court_cases, list):
                if any(
                    isinstance(ref, str) and ref.startswith("special:")
                    for ref in case.court_cases
                ):
                    result.append(case)
            else:
                result.append(case)
        return result

    def _fetch_source_cache(self, cases):
        source_ids = set()
        for case in cases:
            if not case.evidence or not isinstance(case.evidence, (list, tuple)):
                continue
            for entry in case.evidence:
                if not isinstance(entry, dict):
                    continue
                if isinstance((sid := entry.get("source_id")), str) and sid.strip():
                    source_ids.add(sid)

        self._source_lookup = {
            source.source_id: source
            for source in DocumentSource.objects.filter(
                source_id__in=source_ids, is_deleted=False
            ).prefetch_related("uploaded_files")
        }
        logger.debug(f"Cached {len(self._source_lookup)} DocumentSource records")

    def _process_case(
        self, case, model, base_url, api_key, timeout, is_opencode, dry_run
    ):
        self.stats["cases_processed"] += 1

        if not case.evidence:
            self.stats["cases_skipped"] += 1
            self.stdout.write(self.style.WARNING("  SKIPPED: No evidence"))
            return

        press_release_text = self._acquire_press_release_text(case, dry_run)
        if press_release_text is None:
            return

        bigo = case.bigo
        bigo_display = f"रू {bigo:,}" if bigo else "उल्लेख छैन"

        prompt = USER_PROMPT_TEMPLATE.format(
            case_title=case.title,
            bigo=bigo_display,
            press_release=press_release_text[:60000],
        )

        logger.debug(f"Prompt length: {len(prompt)} chars")
        self.stdout.write(
            f"  Sending to LLM ({len(press_release_text)} chars of content)..."
        )

        if is_opencode:
            raw = self._call_llm_opencode(model, base_url, api_key, timeout, prompt)
            if not raw:
                self.stats["cases_failed"] += 1
                self.stdout.write(self.style.ERROR("  FAILED: No LLM response"))
                return
            allegations = self._parse_allegations(raw)
        else:
            raw = None
            client = self._init_client(api_key, base_url)
            allegations = self._call_llm_anthropic(client, model, prompt, timeout)
        if not allegations:
            self.stats["cases_failed"] += 1
            if raw:
                self.stdout.write(
                    self.style.ERROR(
                        f"  FAILED: No allegations extracted. "
                        f"Raw response: {raw[:200]}..."
                    )
                )
            else:
                self.stdout.write(
                    self.style.ERROR("  FAILED: No allegations extracted")
                )
            return

        allegations = self._trim_allegations(allegations)
        count = len(allegations)
        if count < 2 or count > 3:
            self.stats["cases_failed"] += 1
            self.stdout.write(
                self.style.ERROR(
                    f"  FAILED: {count} allegation(s) extracted; expected 2-3"
                )
            )
            return
        self.stats["allegation_counts"][count] = (
            self.stats["allegation_counts"].get(count, 0) + 1
        )
        self._report_allegations(allegations)

        if not dry_run:
            case.key_allegations = allegations
            case.save(update_fields=["key_allegations"])
            self.stats["cases_enriched"] += 1
            self.stdout.write(
                self.style.SUCCESS(f"  ENRICHED: {len(allegations)} allegation(s)")
            )
        else:
            self.stats["cases_enriched"] += 1
            self.stdout.write(
                self.style.SUCCESS(
                    f"  [DRY RUN] Would save {len(allegations)} allegation(s)"
                )
            )

    def _record_missing_details(self, case, note):
        if not note:
            return
        current = case.missing_details or ""
        if note not in current:
            case.missing_details = f"{current}\n{note}" if current else note
            case.save(update_fields=["missing_details"])

    def _acquire_press_release_text(self, case, dry_run):
        source = self._select_press_release_source(case)
        if not source:
            self.stats["cases_no_content"] += 1
            if not dry_run:
                self._record_missing_details(
                    case,
                    "enrich_ciaa_allegations: No press release source found in evidence",
                )
            self.stdout.write(
                self.style.WARNING("  SKIPPED: No press release source found")
            )
            return None

        self.stdout.write(f"  Source: {self._describe_source(source)}")

        try:
            press_release_text = self._convert_source_to_markdown(source)
        except CommandError as e:
            if not _is_missing_content_cmderror(e):
                raise
            self.stats["cases_no_content"] += 1
            if not dry_run:
                self._record_missing_details(
                    case,
                    f"enrich_ciaa_allegations: Failed to convert source to markdown: {e!s}",
                )
            self.stdout.write(
                self.style.WARNING(
                    f"  SKIPPED: Failed to convert source to markdown: {e!s}"
                )
            )
            return None
        except (ValueError, OSError) as e:
            self.stats["cases_no_content"] += 1
            if not dry_run:
                self._record_missing_details(
                    case,
                    f"enrich_ciaa_allegations: Failed to convert source to markdown: {e!s}",
                )
            self.stdout.write(
                self.style.WARNING(
                    f"  SKIPPED: Failed to convert source to markdown: {e!s}"
                )
            )
            return None

        if not press_release_text or len(press_release_text.strip()) < 50:
            self.stats["cases_no_content"] += 1
            if not dry_run:
                self._record_missing_details(
                    case,
                    "enrich_ciaa_allegations: No press release markdown content "
                    "available for LLM extraction",
                )
            self.stdout.write(
                self.style.WARNING("  SKIPPED: No press release markdown content")
            )
            return None

        return press_release_text

    def _trim_allegations(self, allegations):
        if len(allegations) < 2:
            self.stdout.write(
                self.style.WARNING(
                    f"  WARNING: Only {len(allegations)} allegation(s) extracted (want 2-3)"
                )
            )
        if len(allegations) > 3:
            allegations = allegations[:3]
            self.stdout.write(self.style.WARNING("  Truncated to 3 allegations"))
        return allegations

    def _report_allegations(self, allegations):
        self.stdout.write(f"  Extracted {len(allegations)} allegation(s):")
        for a in allegations:
            self.stdout.write(f"    - {a}")

    def _select_press_release_source(self, case: Case) -> DocumentSource | None:
        source_ids = [
            item["source_id"]
            for item in (case.evidence or [])
            if isinstance(item, dict) and isinstance(item.get("source_id"), str)
        ]
        if not source_ids:
            return None

        sources = [
            s
            for s in (self._source_lookup.get(sid) for sid in source_ids)
            if s is not None
        ]
        if not sources:
            return None

        ranked = sorted(
            ((self._score_source_for_press_release(s), s) for s in sources),
            key=lambda row: row[0],
            reverse=True,
        )
        best_score, best_source = ranked[0]
        return best_source if best_score > 0 else None

    def _describe_source(self, source: DocumentSource) -> str:
        if source.uploaded_file:
            name = source.uploaded_filename or source.uploaded_file.name
            return f"uploaded file: {name} ({source.source_id})"
        uploaded = source.uploaded_files.first()
        if uploaded and uploaded.file:
            name = uploaded.filename or uploaded.file.name
            return f"uploaded file: {name} ({source.source_id})"
        urls = self._ranked_source_urls(source)
        if urls:
            parsed = urllib.parse.urlsplit(urls[0])
            redacted = urllib.parse.urlunsplit(
                (parsed.scheme, parsed.netloc, parsed.path, "", "")
            )
            return f"URL: {redacted} ({source.source_id})"
        if source.description and len(source.description.strip()) >= 500:
            title = (source.title or "<untitled>").strip()[:120]
            return f"description fallback ({source.source_id} — {title})"
        return f"{source.source_id} (no content)"

    def _score_source_for_press_release(self, source: DocumentSource) -> int:
        upload_names = [
            file.filename or Path(file.file.name).name
            for file in source.uploaded_files.all()
        ]
        url_text = " ".join(source.url_links)
        corpus = " ".join(
            [
                source.title or "",
                source.description or "",
                source.uploaded_filename or "",
                url_text,
                " ".join(upload_names),
            ]
        ).lower()

        score = 0
        press_keywords = [
            "press release",
            "pressrelease",
            "press-release",
            "प्रेस विज्ञप्ति",
            "विज्ञप्ति",
        ]
        ciaa_keywords = ["ciaa", "अख्तियार"]

        if any(keyword in corpus for keyword in press_keywords):
            score += 5
        if any(keyword in corpus for keyword in ciaa_keywords):
            score += 3
        if source.source_type == SourceType.OFFICIAL_GOVERNMENT:
            score += 1
        return score

    def _convert_source_to_markdown(self, source: DocumentSource) -> str:
        try:
            from markitdown import MarkItDown
        except ImportError as exc:
            raise CommandError(
                "markitdown is required for allegation enrichment conversion. "
                "Install conversion dependencies (markitdown + likhit plugin)."
            ) from exc

        converter = MarkItDown(enable_plugins=True)
        with tempfile.TemporaryDirectory(prefix="allegation-enrichment-") as tmp_dir:
            temp_path = self._download_source_to_path(source, Path(tmp_dir))
            if temp_path:
                logger.debug("Converting uploaded source file for %s", source.source_id)
                result = converter.convert_uri(temp_path.resolve().as_uri())
                if result.text_content and len(result.text_content.strip()) >= 50:
                    return result.text_content

            ranked_urls = self._ranked_source_urls(source)

            last_error = None
            for url in ranked_urls:
                try:
                    logger.debug(
                        "Converting source URL for %s: %s", source.source_id, url
                    )
                    temp_path = self._download_url_to_path(
                        url, source.source_id, Path(tmp_dir)
                    )
                    if not temp_path:
                        last_error = f"download failed for {url}"
                        continue
                    result = converter.convert_uri(temp_path.resolve().as_uri())
                    if result.text_content and len(result.text_content.strip()) >= 50:
                        return result.text_content
                    last_error = f"insufficient content from {url}"
                except (OSError, ValueError) as e:
                    last_error = f"{url}: {e}"
                    continue

            if source.description and len(source.description.strip()) >= 500:
                logger.debug(
                    "Using long source.description fallback for %s",
                    source.source_id,
                )
                return source.description

            if not ranked_urls:
                raise CommandError(
                    f"No downloadable URLs found for source {source.source_id}"
                )

            raise CommandError(
                f"Unable to convert source {source.source_id}: {last_error}"
            )

    def _download_url_to_path(
        self, url: str, source_id: str, output_dir: Path
    ) -> Path | None:
        url = self._validate_url_scheme(url)
        parsed = urllib.parse.urlparse(url)
        guessed_name = _sanitize_download_filename(parsed.path, source_id)
        out_path = _confined_output_path(output_dir, guessed_name)
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"
                },
            )
            opener = urllib.request.build_opener(_SafeRedirectHandler())
            with opener.open(request, timeout=30) as response:
                _copy_stream_to_path_with_limit(response, out_path)
            return out_path
        except OSError:
            out_path.unlink(missing_ok=True)
            return None
        except CommandError:
            out_path.unlink(missing_ok=True)
            raise

    def _download_source_to_path(
        self, source: DocumentSource, output_dir: Path
    ) -> Path | None:
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
                uploaded.filename or uploaded.file.name,
                source.source_id,
            )
            out_path = _confined_output_path(output_dir, filename)
            with uploaded.file.open("rb") as in_file:
                _copy_stream_to_path_with_limit(in_file, out_path)
            return out_path

        return None

    def _pick_source_url(self, source: DocumentSource) -> str | None:
        urls = self._ranked_source_urls(source)
        return urls[0] if urls else None

    def _ranked_source_urls(self, source: DocumentSource) -> list[str]:
        urls = [url.strip() for url in source.url_links if url.strip()]
        if not urls:
            return []

        direct_urls = [url for url in urls if self._is_direct_document_url(url)]
        non_direct_urls = [url for url in urls if url not in direct_urls]

        direct_urls.sort(key=self._source_url_priority, reverse=True)
        return direct_urls + non_direct_urls

    def _is_direct_document_url(self, url: str) -> bool:
        parsed = urllib.parse.urlparse(url)
        path = urllib.parse.unquote(parsed.path).lower()
        return path.endswith((".pdf", ".doc", ".docx"))

    def _source_url_priority(self, url: str) -> tuple[int, int, int]:
        parsed = urllib.parse.urlparse(url)
        host = parsed.netloc.lower()
        path = urllib.parse.unquote(parsed.path).lower()
        is_ngm_store = int(host == "ngm-store.jawafdehi.org")
        is_pdf = int(path.endswith(".pdf"))
        is_direct_document = int(path.endswith((".pdf", ".doc", ".docx")))
        return (is_ngm_store, is_pdf, is_direct_document)

    def _validate_url_scheme(self, url: str) -> str:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            _validate_host_safety(parsed.hostname)
            return url
        raise ValueError(
            f"Invalid URL '{url}'. Only http and https URLs are allowed with a host."
        )

    def _call_llm_anthropic(self, client, model, prompt, timeout=DEFAULT_LLM_TIMEOUT):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=6000,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    timeout=timeout,
                )
                raw = response.content[0].text
                logger.debug(f"LLM response: {raw[:100]}...")
                return self._parse_allegations(raw)
            except Exception as e:
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    wait = 2**attempt
                    self.stdout.write(self.style.WARNING(f"  Retrying in {wait}s..."))
                    time.sleep(wait)
                else:
                    self.stdout.write(
                        self.style.ERROR(
                            f"  LLM call failed after {max_retries} attempts: {e}"
                        )
                    )
                    return None

    def _extract_json_body(self, raw):
        raw = raw.strip()
        body = _extract_fenced_block(raw)
        if body is not None:
            return body
        brace_start = raw.find("{")
        brace_end = raw.rfind("}")
        if brace_start != -1 and brace_end != -1:
            return raw[brace_start : brace_end + 1]
        return raw

    def _parse_json_to_allegations(self, raw):
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            result = data.get("allegations", data.get("response", []))
            if not isinstance(result, list):
                return [str(result)]
            return result
        return []

    def _parse_fallback_allegations(self, raw):
        lines = [
            line.strip().lstrip("0123456789.-) ")
            for line in raw.split("\n")
            if line.strip()
        ]
        allegations = [line for line in lines if len(line) > 10]
        return allegations if allegations else None

    def _flatten_allegation_items(self, allegations):
        flat = []
        for a in allegations:
            if isinstance(a, str) and a.strip():
                flat.append(a.strip())
            elif isinstance(a, dict):
                vals = [str(v) for v in a.values() if v]
                if vals:
                    flat.append("; ".join(vals))
        return flat

    def _parse_allegations(self, raw):
        body = self._extract_json_body(raw)
        try:
            allegations = self._parse_json_to_allegations(body)
        except json.JSONDecodeError:
            return self._parse_fallback_allegations(body)

        flat = self._flatten_allegation_items(allegations)
        if not flat:
            return None
        return flat[:3]

    def _print_summary(self, dry_run, elapsed_str="0s"):
        self.stdout.write("\n" + "=" * 60)
        self.stdout.write(
            self.style.WARNING(f"{'[DRY RUN] ' if dry_run else ''}SUMMARY")
        )
        self.stdout.write("=" * 60)
        self.stdout.write(f"Total time:       {elapsed_str}")
        self.stdout.write(f"Cases processed:  {self.stats['cases_processed']}")
        self.stdout.write(
            self.style.SUCCESS(f"Cases enriched:   {self.stats['cases_enriched']}")
        )
        self.stdout.write(
            self.style.WARNING(f"Cases skipped:    {self.stats['cases_skipped']}")
        )
        self.stdout.write(
            self.style.WARNING(f"Cases no content:  {self.stats['cases_no_content']}")
        )
        if self.stats["cases_failed"] > 0:
            self.stdout.write(
                self.style.ERROR(f"Cases failed:     {self.stats['cases_failed']}")
            )
        if self.stats["allegation_counts"]:
            self.stdout.write("Allegations per case:")
            for count in sorted(self.stats["allegation_counts"]):
                self.stdout.write(
                    f"  {count} allegation(s): {self.stats['allegation_counts'][count]}"
                )
        self.stdout.write("=" * 60)

        if dry_run:
            self.stdout.write(
                self.style.WARNING(
                    "\nThis was a dry run. No changes were made to the database."
                )
            )
            self.stdout.write("Run without --dry-run to apply changes.")
