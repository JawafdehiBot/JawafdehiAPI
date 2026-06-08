"""Enrich missing BIGO values for DRAFT cases using press releases + LLM extraction."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db.models import Q

from cases.models import Case, CaseState, DocumentSource, SourceType
from cases.services.priority_case_loader import filter_by_priority, load_priority_cases

MAX_LIMIT = 1000
MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024
DOWNLOAD_CHUNK_SIZE = 16 * 1024
BIGO_CONTEXT_KEYWORDS = (
    "बिगो",
    "मागदाबी",
    "हानि",
    "हानी",
    "नोक्सानी",
    "क्षति",
    "damage claim",
    "loss amount",
    "corruption loss",
)

_NEPALI_TO_ASCII_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def _first_env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


class Command(BaseCommand):
    help = (
        "Find DRAFT cases with missing BIGO, extract amount from CIAA press release "
        "content, and PATCH BIGO via API."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--limit",
            type=int,
            default=50,
            help=f"Max cases to process (1-{MAX_LIMIT}).",
        )
        parser.add_argument(
            "--case-id",
            type=str,
            default=None,
            help="Optional exact case_id to process.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help=(
                "Preview eligible cases and selected source without downloading, "
                "calling the LLM, or PATCHing cases. Use --dry-run-extract to "
                "also run source conversion and LLM extraction."
            ),
        )
        parser.add_argument(
            "--dry-run-extract",
            action="store_true",
            help=(
                "With --dry-run, run source download/conversion and LLM extraction, "
                "but still do not PATCH cases."
            ),
        )
        parser.add_argument(
            "--allow-production",
            action="store_true",
            help="Required when DEBUG=False to run this command in production.",
        )
        parser.add_argument(
            "--api-base-url",
            type=str,
            default=os.getenv("JAWAFDEHI_API_BASE_URL", "http://127.0.0.1:8000"),
            help="Jawafdehi API base URL (root or /api).",
        )
        parser.add_argument(
            "--api-token",
            type=str,
            default=os.getenv("JAWAFDEHI_API_TOKEN"),
            help="Jawafdehi API token. Defaults to JAWAFDEHI_API_TOKEN.",
        )
        parser.add_argument(
            "--llm-api-key",
            type=str,
            default=os.getenv("JAWAFDEHI_LLM_API_KEY"),
            help="LLM API key for native Anthropic or the OpenAI-compatible proxy. Defaults to JAWAFDEHI_LLM_API_KEY.",
        )
        parser.add_argument(
            "--anthropic-api-key",
            type=str,
            default=os.getenv("ANTHROPIC_API_KEY"),
            help="Deprecated alias for --llm-api-key. Defaults to ANTHROPIC_API_KEY.",
        )
        parser.add_argument(
            "--llm-model",
            type=str,
            default=_first_env(
                "BIGO_ENRICHMENT_MODEL",
                "JAWAFDEHI_CASEWORK_MODEL",
                default="claude-sonnet-4-5",
            ),
            help=(
                "LLM model used for BIGO extraction. Defaults to BIGO_ENRICHMENT_MODEL, "
                "JAWAFDEHI_CASEWORK_MODEL, or claude-sonnet-4-5."
            ),
        )
        parser.add_argument(
            "--llm-base-url",
            type=str,
            default=_first_env(
                "BIGO_ENRICHMENT_LLM_BASE_URL",
                "BIGO_ENRICHMENT_BASE_URL",
                "JAWAFDEHI_CASEWORK_BASE_URL",
                "JAWAFDEHI_LLM_PROXY_URL",
            ),
            help=(
                "LLM API base URL (for OpenAI-compatible proxy). Defaults to "
                "BIGO_ENRICHMENT_LLM_BASE_URL, BIGO_ENRICHMENT_BASE_URL, "
                "JAWAFDEHI_CASEWORK_BASE_URL, or JAWAFDEHI_LLM_PROXY_URL."
            ),
        )
        parser.add_argument(
            "--llm-timeout",
            type=float,
            default=float(os.getenv("BIGO_ENRICHMENT_LLM_TIMEOUT", "120")),
            help="LLM request timeout in seconds. Defaults to BIGO_ENRICHMENT_LLM_TIMEOUT or 120.",
        )
        parser.add_argument(
            "--llm-max-tokens",
            type=int,
            default=int(os.getenv("BIGO_ENRICHMENT_LLM_MAX_TOKENS", "2000")),
            help="LLM response token budget. Defaults to BIGO_ENRICHMENT_LLM_MAX_TOKENS or 2000.",
        )
        parser.add_argument(
            "--download-timeout",
            type=float,
            default=float(os.getenv("BIGO_ENRICHMENT_DOWNLOAD_TIMEOUT", "30")),
            help="Source download timeout in seconds. Defaults to BIGO_ENRICHMENT_DOWNLOAD_TIMEOUT or 30.",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enable detailed per-case logging for enrichment flow.",
        )
        parser.add_argument(
            "--min-confidence",
            choices=["high", "medium", "low"],
            default="medium",
            help="Minimum accepted extraction confidence.",
        )
        parser.add_argument(
            "--priority",
            action="store_true",
            help="Enrich only cases in the priority case list.",
        )

    def handle(self, *args, **options):
        self._verbose = bool(options.get("verbose"))
        self._validate_guardrails(options)
        self._validate_runtime_inputs(options)
        self._log_info(
            "Starting BIGO enrichment run "
            f"(limit={options['limit']}, dry_run={options['dry_run']}, "
            f"dry_run_extract={options['dry_run_extract']}, "
            f"case_id={options['case_id'] or 'ALL'}, min_confidence={options['min_confidence']})",
            always=True,
        )

        llm_api_key = self._resolve_llm_api_key(options)

        priority = options["priority"]
        case_id = options.get("case_id")
        if priority and case_id:
            self.stderr.write(
                self.style.ERROR("--priority and --case-id are mutually exclusive")
            )
            return

        queryset = (
            Case.objects.filter(state=CaseState.DRAFT)
            .filter(Q(bigo__isnull=True) | Q(bigo=0))
            .order_by("-created_at")
        )
        if case_id:
            queryset = queryset.filter(case_id=case_id)
        elif priority:
            priority_list = load_priority_cases()
            queryset = filter_by_priority(queryset, priority_list)

        cases = list(queryset[: options["limit"]])
        if not cases:
            self.stdout.write("No eligible DRAFT case found for BIGO enrichment.")
            return

        updated = 0
        skipped = 0
        failed = 0

        for case in cases:
            try:
                self._log_info(f"Processing case {case.case_id}")
                source = self._select_press_release_source(case)
                if source is None:
                    skipped += 1
                    self.stdout.write(
                        self.style.WARNING(
                            f"[SKIP] {case.case_id}: no press release source found."
                        )
                    )
                    continue

                self._log_info(
                    f"Selected source {source.source_id} for case {case.case_id}; "
                    f"court_cases={case.court_cases or []}; title={source.title!r}"
                )
                self._log_source_diagnostics(source)
                if options["dry_run"] and not options["dry_run_extract"]:
                    skipped += 1
                    self.stdout.write(
                        self.style.WARNING(
                            f"[DRY-RUN] {case.case_id}: selected source={source.source_id}; "
                            "would download/convert, extract BIGO with LLM, and PATCH if a "
                            "reliable BIGO is found. Use --dry-run-extract to run external "
                            "download and LLM extraction."
                        )
                    )
                    continue

                markdown = self._convert_source_to_markdown(
                    source,
                    download_timeout=options["download_timeout"],
                )
                self._log_info(
                    f"Converted source {source.source_id} to markdown ({len(markdown)} chars)"
                )
                bigo = self._extract_bigo_from_source_metadata(source)
                if bigo is not None:
                    self._log_info(
                        f"Extracted BIGO from source metadata for {case.case_id}: {bigo}"
                    )
                else:
                    self._log_info(
                        f"No explicit BIGO found in source metadata for {case.case_id}; "
                        "checking converted markdown."
                    )
                    bigo = self._extract_explicit_bigo_from_markdown(markdown)
                    if bigo is not None:
                        self._log_info(
                            f"Extracted BIGO from converted markdown scan for {case.case_id}: {bigo}"
                        )

                if bigo is None:
                    self._log_bigo_snippets(markdown)
                    bigo = self._extract_bigo_from_markdown(
                        markdown=markdown,
                        case=case,
                        source=source,
                        model=options["llm_model"],
                        anthropic_api_key=llm_api_key,
                        min_confidence=options["min_confidence"],
                        llm_base_url=options.get("llm_base_url"),
                        llm_timeout=options["llm_timeout"],
                        llm_max_tokens=options["llm_max_tokens"],
                    )
                self._log_info(f"Extracted BIGO candidate for {case.case_id}: {bigo}")
                if bigo is None:
                    skipped += 1
                    self.stdout.write(
                        self.style.WARNING(
                            f"[SKIP] {case.case_id}: could not extract a reliable BIGO."
                        )
                    )
                    continue

                if options["dry_run"]:
                    self.stdout.write(
                        self.style.WARNING(
                            f"[DRY-RUN] {case.case_id}: would PATCH BIGO={bigo}"
                        )
                    )
                else:
                    self._patch_case_bigo(
                        case=case,
                        bigo=bigo,
                        api_base_url=options["api_base_url"],
                        api_token=options["api_token"],
                    )
                    self.stdout.write(
                        self.style.SUCCESS(f"[UPDATED] {case.case_id}: BIGO={bigo}")
                    )
                updated += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                self.stdout.write(
                    self.style.ERROR(
                        f"[FAIL] {case.case_id}: {type(exc).__name__}: {exc}"
                    )
                )

        self.stdout.write(
            f"Processed={len(cases)} Updated={updated} Skipped={skipped} Failed={failed}"
        )

    def _log_info(self, message: str, *, always: bool = False) -> None:
        if always or getattr(self, "_verbose", False):
            self.stdout.write(f"[INFO] {message}")

    def _log_source_diagnostics(self, source: DocumentSource) -> None:
        if not getattr(self, "_verbose", False):
            return
        urls = [urllib.parse.unquote(url) for url in source.url_links]
        upload_names = self._source_upload_names(source)
        self._log_info(
            "Source diagnostics: "
            f"type={source.source_type}; description={source.description!r}; "
            f"uploaded_filename={source.uploaded_filename!r}; urls={urls}; uploads={upload_names}"
        )

    def _log_bigo_snippets(self, markdown: str) -> None:
        if not getattr(self, "_verbose", False):
            return
        normalized = markdown.translate(_NEPALI_TO_ASCII_DIGITS)
        matches = list(
            re.finditer(
                r"बिगो|मागदाबी|हानि|हानी|नोक्सानी|क्षति",
                normalized,
                flags=re.IGNORECASE,
            )
        )
        if not matches:
            self._log_info("No BIGO-context keywords found in converted markdown.")
            return
        snippets = []
        for match in matches[:5]:
            start = max(0, match.start() - 120)
            end = min(len(normalized), match.end() + 180)
            snippets.append(re.sub(r"\s+", " ", normalized[start:end]).strip())
        self._log_info("BIGO-context markdown snippets: " + " || ".join(snippets))

    def _validate_guardrails(self, options: dict[str, Any]) -> None:
        limit = options["limit"]
        if limit < 1 or limit > MAX_LIMIT:
            raise CommandError(f"--limit must be between 1 and {MAX_LIMIT}.")

        if not settings.DEBUG and not options["allow_production"]:
            raise CommandError(
                "This command refuses to run in production unless --allow-production is provided."
            )

    def _validate_runtime_inputs(self, options: dict[str, Any]) -> None:
        if options["dry_run_extract"] and not options["dry_run"]:
            raise CommandError("--dry-run-extract requires --dry-run.")

        if options["download_timeout"] <= 0:
            raise CommandError("--download-timeout must be greater than 0.")
        if options["llm_timeout"] <= 0:
            raise CommandError("--llm-timeout must be greater than 0.")
        if options["llm_max_tokens"] <= 0:
            raise CommandError("--llm-max-tokens must be greater than 0.")

        if options["dry_run"] and not options["dry_run_extract"]:
            return

        if not options["api_token"] and not options["dry_run"]:
            raise CommandError(
                "JAWAFDEHI API token is required. Set --api-token or JAWAFDEHI_API_TOKEN."
            )
        if not self._resolve_llm_api_key(options):
            raise CommandError(
                "LLM API key is required. Set --llm-api-key, JAWAFDEHI_LLM_API_KEY, "
                "--anthropic-api-key, or ANTHROPIC_API_KEY."
            )

    def _resolve_llm_api_key(self, options: dict[str, Any]) -> str | None:
        return options.get("llm_api_key") or options.get("anthropic_api_key")

    def _select_press_release_source(self, case: Case) -> DocumentSource | None:
        source_ids = [
            item["source_id"]
            for item in (case.evidence or [])
            if isinstance(item, dict) and isinstance(item.get("source_id"), str)
        ]
        if not source_ids:
            return None

        sources = list(
            DocumentSource.objects.filter(
                source_id__in=source_ids,
                is_deleted=False,
            ).prefetch_related("uploaded_files")
        )
        if not sources:
            return None

        ranked = sorted(
            (
                (self._score_source_for_press_release(source), source)
                for source in sources
            ),
            key=lambda row: row[0],
            reverse=True,
        )
        best_score, best_source = ranked[0]

        # If we have a positive score, return the best match
        if best_score > 0:
            return best_source

        # Fallback: if there's only one source and it has a PDF, use it
        # This handles cases where the source isn't properly labeled
        if len(sources) == 1:
            source = sources[0]
            # Check if it has an uploaded PDF file
            has_pdf = (
                source.uploaded_file
                and source.uploaded_file.name.lower().endswith(".pdf")
            ) or any(
                f.file.name.lower().endswith(".pdf")
                for f in source.uploaded_files.all()
            )
            if has_pdf:
                self._log_info(
                    f"Using single PDF source {source.source_id} for {case.case_id} "
                    "(no press release keywords found, but only one source available)"
                )
                return source

        return None

    def _score_source_for_press_release(self, source: DocumentSource) -> int:
        upload_names = [
            file.filename or Path(file.file.name).name
            for file in source.uploaded_files.all()
        ]
        url_text = " ".join(
            u.get("link") if isinstance(u, dict) else str(u) for u in (source.url or [])
        )
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

    def _convert_source_to_markdown(
        self,
        source: DocumentSource,
        *,
        download_timeout: float = 30,
    ) -> str:
        try:
            from markitdown import MarkItDown
        except ImportError as exc:  # pragma: no cover - env dependent
            raise CommandError(
                "markitdown is required for BIGO enrichment conversion. "
                "Install conversion dependencies (markitdown + likhit plugin)."
            ) from exc

        converter = MarkItDown(enable_plugins=True)
        with tempfile.TemporaryDirectory(prefix="bigo-enrichment-") as tmp_dir:
            temp_path = self._download_source_to_path(
                source,
                Path(tmp_dir),
                timeout=download_timeout,
            )
            if temp_path:
                result = converter.convert_uri(temp_path.resolve().as_uri())
                return result.markdown

            raise CommandError(
                f"No downloadable source found for source_id={source.source_id}."
            )

    def _download_source_to_path(
        self,
        source: DocumentSource,
        output_dir: Path,
        *,
        timeout: float = 30,
    ) -> Path | None:
        if source.uploaded_file:
            filename = self._sanitize_download_filename(
                source.uploaded_filename or source.uploaded_file.name,
                source.source_id,
            )
            out_path = self._confined_output_path(output_dir, filename)
            with source.uploaded_file.open("rb") as in_file:
                self._copy_stream_to_path_with_limit(in_file, out_path)
            return out_path

        uploaded = source.uploaded_files.first()
        if uploaded and uploaded.file:
            filename = self._sanitize_download_filename(
                uploaded.filename or uploaded.file.name,
                source.source_id,
            )
            out_path = self._confined_output_path(output_dir, filename)
            with uploaded.file.open("rb") as in_file:
                self._copy_stream_to_path_with_limit(in_file, out_path)
            return out_path

        source_url = self._pick_source_url(source)
        if not source_url:
            return None

        source_url = self._validate_url_scheme(source_url)
        parsed = urllib.parse.urlparse(source_url)
        guessed_name = self._sanitize_download_filename(parsed.path, source.source_id)
        out_path = self._confined_output_path(output_dir, guessed_name)
        try:
            request = urllib.request.Request(
                source_url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"
                },
            )
            with urllib.request.urlopen(  # NOSONAR
                request, timeout=timeout
            ) as response:  # noqa: S310
                headers = getattr(response, "headers", {})
                content_length = headers.get("Content-Length")
                if content_length and int(content_length) > MAX_DOWNLOAD_BYTES:
                    raise CommandError(
                        f"Source is too large ({content_length} bytes); max is {MAX_DOWNLOAD_BYTES} bytes."
                    )
                self._copy_stream_to_path_with_limit(response, out_path)
            return out_path
        except OSError as exc:
            out_path.unlink(missing_ok=True)
            self._log_info(f"Download failed for {source.source_id}: {exc}")
            return None
        except CommandError:
            out_path.unlink(missing_ok=True)
            raise

    def _copy_stream_to_path_with_limit(self, in_file: Any, out_path: Path) -> None:
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

    def _pick_source_url(self, source: DocumentSource) -> str | None:
        urls = [url.strip() for url in source.url_links if url.strip()]
        if not urls:
            return None

        # Prefer direct downloadable files over CIAA HTML pressrelease pages. CIAA pages
        # can be summaries with a "Download" button; the BIGO text is usually in the
        # linked PDF/DOCX, and NGM-mapped URLs often point directly to that file.
        direct_file_urls = [url for url in urls if self._is_direct_document_url(url)]
        if direct_file_urls:
            return max(direct_file_urls, key=self._source_url_priority)

        return urls[0]

    def _is_direct_document_url(self, url: str) -> bool:
        parsed = urllib.parse.urlparse(url)
        path = urllib.parse.unquote(parsed.path).lower()
        return path.endswith((".pdf", ".doc", ".docx"))

    def _source_url_priority(self, url: str) -> tuple[int, int]:
        parsed = urllib.parse.urlparse(url)
        host = parsed.netloc.lower()
        path = urllib.parse.unquote(parsed.path).lower()
        is_ngm_store = int(host == "ngm-store.jawafdehi.org")
        is_pdf = int(path.endswith(".pdf"))
        return (is_ngm_store, is_pdf)

    def _validate_url_scheme(self, url: str) -> str:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme == "https" and parsed.netloc:
            return url
        if (
            parsed.scheme == "http"
            and parsed.netloc
            and self._is_loopback_host(parsed.hostname)
        ):
            return url
        raise ValueError(
            f"Invalid URL '{url}'. Only https URLs are allowed for non-local hosts."
        )

    @staticmethod
    def _is_loopback_host(hostname: str | None) -> bool:
        if not hostname:
            return False
        host = hostname.lower().rstrip(".")
        if host == "localhost":
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False

    def _sanitize_download_filename(self, filename: str | None, source_id: str) -> str:
        raw = (filename or "").strip()
        if not raw:
            return f"{source_id}.bin"

        # URL paths often contain percent-encoded Unicode; decoding keeps filenames readable
        # and avoids hitting filesystem filename-length limits (e.g., 255 chars on macOS).
        decoded = urllib.parse.unquote(raw)
        candidate = Path(decoded).name.strip()

        if candidate in {"", ".", ".."}:
            return f"{source_id}.bin"

        # Basic cross-platform safety: remove NUL and replace Windows-forbidden characters.
        candidate = candidate.replace("\x00", "")
        candidate = re.sub(r"[<>:\"/\\|?*]+", "_", candidate).rstrip(" .")
        if candidate in {"", ".", ".."}:
            return f"{source_id}.bin"

        # Keep well under typical per-filename limits (255) and leave room for temp dir paths.
        max_len = 200
        if len(candidate) <= max_len:
            return candidate

        suffix = "".join(Path(candidate).suffixes)
        stem = candidate[: -len(suffix)] if suffix else candidate
        digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()[:10]

        stem_budget = max_len - len(suffix) - len(digest) - 1  # for '-'
        if stem_budget < 1:
            # Worst-case fallback: ensure a short, unique name.
            return f"{source_id}-{digest}{suffix}"[:max_len]

        truncated_stem = stem[:stem_budget].rstrip(" .-_")
        if not truncated_stem:
            truncated_stem = source_id

        return f"{truncated_stem}-{digest}{suffix}"

    def _confined_output_path(self, output_dir: Path, filename: str) -> Path:
        output_dir_resolved = output_dir.resolve()
        out_path = (output_dir / filename).resolve()
        if output_dir_resolved not in out_path.parents:
            raise CommandError(
                f"Refusing to write outside output directory: '{filename}'"
            )
        return out_path

    def _extract_bigo_from_source_metadata(self, source: DocumentSource) -> int | None:
        """Extract BIGO from source title/URL/file metadata when explicitly labeled.

        NGM-store press-release URLs commonly preserve the CIAA filename, e.g.
        "...उपर बिगो रु. २,००,००० कायम...pdf". Prefer this deterministic signal
        before asking the LLM to read potentially noisy PDF conversion output.
        """
        snippets = [
            source.title or "",
            source.description or "",
            source.uploaded_filename or "",
        ]
        snippets.extend(source.url_links)
        snippets.extend(self._source_upload_names(source))

        for snippet in snippets:
            bigo = self._extract_explicit_bigo_from_text(snippet)
            if bigo is not None:
                return bigo
        return None

    def _extract_explicit_bigo_from_text(self, text: str | None) -> int | None:
        if not text:
            return None

        normalized = self._normalize_text_for_bigo(text)
        return self._best_bigo_candidate(
            self._collect_explicit_bigo_candidates(normalized)
        )

    def _normalize_text_for_bigo(self, text: str | None) -> str:
        decoded = urllib.parse.unquote(str(text or ""))
        embedded_http = re.search(r"https?://", decoded)
        if embedded_http and not decoded.lstrip().lower().startswith("http"):
            decoded = decoded[embedded_http.start() :]
        return decoded.translate(_NEPALI_TO_ASCII_DIGITS)

    def _collect_explicit_bigo_candidates(self, text: str) -> list[tuple[int, int]]:
        candidates: list[tuple[int, int]] = []
        strong_patterns = [
            (
                r"बिगो\s*(?:रू\.|रु\.|रू|रु|rs\.?|npr)\s*[.:：]?\s*[:\-–—]?\s*([0-9][0-9,]*(?:/[0-9]+)?)",
                120,
            ),
            (
                r"(?:damage claim|loss amount|corruption loss)\s*(?:rs\.?|npr)\s*[.:：]?\s*[:\-–—]?\s*([0-9][0-9,]*(?:/[0-9]+)?)",
                100,
            ),
        ]
        for pattern, base_score in strong_patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                candidate = self._bigo_candidate_from_amount(
                    amount_text=match.group(1),
                    context=text[max(0, match.start() - 80) : match.end() + 80],
                    base_score=base_score,
                )
                if candidate is not None:
                    candidates.append(candidate)

        for marker in re.finditer(r"बिगो", text, flags=re.IGNORECASE):
            window = text[marker.start() : marker.start() + 220]
            for amount_match in re.finditer(r"\d[\d,]*(?:/\d+)?", window):
                prefix = window[: amount_match.start()]
                if not re.search(
                    r"रू\.?|रु\.?|rs\.?|npr|कायम|मागदाबी|दायर",
                    prefix,
                    flags=re.IGNORECASE,
                ):
                    continue
                candidate = self._bigo_candidate_from_amount(
                    amount_text=amount_match.group(0),
                    context=window,
                    base_score=60,
                )
                if candidate is not None:
                    candidates.append(candidate)
        return candidates

    def _bigo_candidate_from_amount(
        self,
        amount_text: str,
        context: str,
        base_score: int,
    ) -> tuple[int, int] | None:
        amount_before_paisa = amount_text.split("/", 1)[0]
        digits_only = re.sub(r"[^\d]", "", amount_before_paisa)
        if not digits_only:
            return None
        bigo = int(digits_only)
        if not self._is_plausible_deterministic_bigo(bigo):
            return None

        score = base_score + min(len(digits_only), 12)
        if "," in amount_before_paisa:
            score += 20
        if re.search(r"कायम|मागदाबी|दायर|प्रतिवादीउपर", context, flags=re.IGNORECASE):
            score += 15
        return score, bigo

    def _is_plausible_deterministic_bigo(self, bigo: int) -> bool:
        return bigo >= 1000

    def _best_bigo_candidate(self, candidates: list[tuple[int, int]]) -> int | None:
        if not candidates:
            return None
        return max(candidates, key=lambda candidate: candidate[0])[1]

    def _extract_explicit_bigo_from_markdown(self, markdown: str) -> int | None:
        normalized = self._normalize_text_for_bigo(markdown)
        candidates = self._collect_explicit_bigo_candidates(normalized)

        table_patterns = [
            r"बिगो\s*(?:रू\.|रु\.|रू|रु)?[^\n\d]{0,40}\n\s*([0-9][0-9,]*(?:/[0-9]+)?)",
            r"बिगो\s*(?:रू\.|रु\.|रू|रु)?[^0-9]{0,80}([0-9][0-9,]*(?:/[0-9]+)?)",
        ]
        for pattern in table_patterns:
            for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
                candidate = self._bigo_candidate_from_amount(
                    amount_text=match.group(1),
                    context=normalized[max(0, match.start() - 80) : match.end() + 80],
                    base_score=80,
                )
                if candidate is not None:
                    candidates.append(candidate)
        return self._best_bigo_candidate(candidates)

    def _extract_bigo_from_markdown(
        self,
        markdown: str,
        case: Case,
        source: DocumentSource | None,
        model: str,
        anthropic_api_key: str,
        min_confidence: str,
        llm_base_url: str | None = None,
        llm_timeout: float = 120,
        llm_max_tokens: int = 2000,
    ) -> int | None:
        prompt = self._build_bigo_prompt(markdown=markdown, case=case, source=source)

        # Use OpenAI-compatible client if base_url provided (Jawafdehi proxy)
        if llm_base_url:
            try:
                from openai import OpenAI

                client = OpenAI(
                    api_key=anthropic_api_key,
                    base_url=llm_base_url,
                    timeout=llm_timeout,
                )
                response = client.chat.completions.create(
                    model=model,
                    max_tokens=llm_max_tokens,
                    temperature=0,
                    response_format={"type": "json_object"},
                    messages=[{"role": "user", "content": prompt}],
                )
                text = self._openai_response_text(response)
            except CommandError:
                raise
            except Exception as exc:
                self._raise_llm_proxy_command_error(exc, llm_base_url, model)
        else:
            # Use native Anthropic client
            import anthropic

            client = anthropic.Anthropic(api_key=anthropic_api_key)
            response = client.messages.create(
                model=model,
                max_tokens=llm_max_tokens,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(
                block.text
                for block in response.content
                if getattr(block, "type", "") == "text"
            )

        payload = self._parse_json_response(text)
        confidence = str(payload.get("confidence", "")).strip().lower()
        if self._confidence_rank(confidence) < self._confidence_rank(min_confidence):
            return None

        evidence_quote = payload.get("evidence_quote")
        if not self._is_explicit_bigo_context(evidence_quote):
            self._log_info(
                f"Rejected extraction for {case.case_id}: evidence_quote lacked explicit BIGO context."
            )
            return None

        return self._coerce_bigo_int(payload.get("bigo"))

    def _raise_llm_proxy_command_error(
        self,
        exc: Exception,
        llm_base_url: str,
        model: str,
    ) -> None:
        exc_name = type(exc).__name__
        status_code = getattr(exc, "status_code", None)
        if status_code is None:
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)

        if status_code in {401, 403} or exc_name in {
            "AuthenticationError",
            "PermissionDeniedError",
        }:
            raise CommandError(
                f"LLM proxy authentication/authorization failed at {llm_base_url}. "
                "Use a key authorized for the proxy via --llm-api-key or "
                "JAWAFDEHI_LLM_API_KEY, and verify that the requested model is allowed. "
                f"Requested model: {model}. Original error: {exc}"
            ) from exc

        if status_code is not None:
            raise CommandError(
                f"LLM proxy request failed at {llm_base_url} "
                f"(status {status_code}). Requested model: {model}. "
                f"Original error: {exc}"
            ) from exc

        raise CommandError(
            f"LLM proxy request failed at {llm_base_url}. Requested model: {model}. "
            f"Original error: {exc}"
        ) from exc

    def _openai_response_text(self, response: Any) -> str:
        if not response:
            raise CommandError("OpenAI-compatible LLM response was empty.")

        output_text = self._value(response, "output_text")
        text = self._text_from_content(output_text)
        if text:
            return text

        choices = self._value(response, "choices")
        if not isinstance(choices, (list, tuple)) or not choices:
            raise CommandError(
                "OpenAI-compatible LLM response missing choices content."
            )

        first_choice = choices[0]
        message = self._value(first_choice, "message")
        if message is not None:
            content = self._value(message, "content")
            text = self._text_from_content(content)
            if text:
                return text

            # Some OpenAI-compatible proxies expose final answer text under
            # additional message fields instead of message.content. Do not use
            # reasoning_content as the final answer; reasoning-only text often
            # contains no strict JSON and should fail with a clear parse error.
            for field_name in ("text", "output_text"):
                text = self._text_from_content(self._value(message, field_name))
                if text:
                    return text

        # Legacy completion-shaped responses and some proxy adapters can put
        # text directly on the choice instead of choice.message.content.
        for field_name in ("text", "content", "output_text"):
            text = self._text_from_content(self._value(first_choice, field_name))
            if text:
                return text

        finish_reason = self._value(first_choice, "finish_reason")
        message_keys = self._keys_for_debug(message) if message is not None else "none"
        choice_keys = self._keys_for_debug(first_choice)
        raise CommandError(
            "OpenAI-compatible LLM response missing text content "
            f"(finish_reason={finish_reason or 'unknown'}, "
            f"message_keys={message_keys}, choice_keys={choice_keys})."
        )

    def _value(self, obj: Any, key: str) -> Any:
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    def _keys_for_debug(self, obj: Any) -> str:
        if isinstance(obj, dict):
            keys = sorted(str(key) for key in obj.keys())
        else:
            keys = sorted(
                key
                for key in dir(obj)
                if not key.startswith("_") and not callable(getattr(obj, key, None))
            )
        return ",".join(keys[:12]) or "none"

    def _text_from_content(self, content: Any) -> str | None:
        if content is None:
            return None

        if isinstance(content, str):
            text = content.strip()
            return text or None

        if isinstance(content, dict):
            for field_name in ("text", "content", "output_text"):
                text = self._text_from_content(content.get(field_name))
                if text:
                    return text
            return None

        if isinstance(content, (list, tuple)):
            parts: list[str] = []
            for block in content:
                text = self._text_from_content(block)
                if text:
                    parts.append(text)
            joined = "".join(parts).strip()
            return joined or None

        block_text = self._value(content, "text")
        text = self._text_from_content(block_text)
        if text:
            return text

        block_content = self._value(content, "content")
        text = self._text_from_content(block_content)
        if text:
            return text

        return None

    def _build_bigo_prompt(
        self,
        markdown: str,
        case: Case,
        source: DocumentSource | None = None,
    ) -> str:
        source_context = self._build_source_context(source)
        return f"""You extract BIGO (बिगो), the damage claim amount / मागदाबी, from CIAA press release content.

Return STRICT JSON only with this schema:
{{
  "bigo": <integer or null>,
  "confidence": "high" | "medium" | "low",
  "evidence_quote": "<short quote from text that supports the amount>",
  "press_release_type": "sting_operation" | "appeal_review" | "charge_filing" | "other"
}}

CRITICAL RULES (apply in order):

Rule 1 — Output format
BIGO must be an NPR integer only. No commas, no currency symbols (रू/Rs/NPR), no paisa suffix (/90, /39, etc.), no floats.
If the extracted amount has a paisa portion (e.g. १,४६,८१,२२५/९०), strip everything after / before returning.

Rule 2 — Numeral normalization
Before any matching, normalize Devanagari digits to Arabic (०→0, १→1, ... ९→9). CIAA PDFs mix both in the same number.
Then strip commas. Then strip the paisa suffix. Then parse as integer.

Rule 3 — Type-routing first (CHECK THIS BEFORE READING TEXT)
Before reading any text, determine the press release type:
- Sting Operation (रंगेहात, sting, caught red-handed) → return null (high confidence). The amounts in sting releases are physical cash caught during arrest — bribe/unexplained cash — not a formally established bigo.
- Appeal/Review (पुनरावेदन, अपील, appeal, review) → return null (high confidence). These record CIAA appealing a court verdict; bigo was defined at charge-sheet stage and is not re-stated here.
- Charge Filing (अभियोग दायर, charge filed, मुद्दा दर्ता) → proceed to extraction rules below.
- Other → proceed to extraction rules below.

Rule 4 — Null with low confidence
If no reliable bigo signal exists after all checks, return null with low confidence. Do not guess.

Rule 5 — No ranges, floats, or formatted strings
Never return ranges (१-५ लाख), floats (1.5 करोड), or formatted strings. Integer only.

Rule 6 — Priority signal hierarchy (apply in order, stop at first match)
1. Title contains "बिगो रू.[AMOUNT] कायम गरी" → extract AMOUNT → high confidence
2. PDF table row under "बिगो रू." column header → extract amount → high confidence
3. PDF body sentence "कूल आय भन्दा कूल व्यय ... [AMOUNT] ले बढी" (excess of expenditure over income) → extract AMOUNT → medium confidence, verify it matches signal 2
4. No match → null

Note: "बिगो बमोजिम जरिवाना" is a reference to an already-stated bigo, not a declaration of it — use it only to confirm, not as a primary source.

Rule 7 — Multiple amounts: label is mandatory
When multiple monetary amounts appear in the text, extract only the one explicitly labeled as bigo using the hierarchy in Rule 6.
If multiple amounts are present and none carries a clear bigo label, return null — do not pick the largest, the first, or the one that "seems right."

Rule 8 — Ignore list (NEVER extract these as bigo)
Amount type | Nepali marker
------------|---------------
Bribe received/demanded | घुस/रिसवत रकम रू.
Unexplained cash seized | स्रोत नखुलेको रकम रू.
Lawful income subtotal | जम्मा/कूल आय रू.
Expenditure subtotal | जम्मा/कूल व्यय रू.
Fine/penalty | जरिवाना रू.
Contract/budget amount | ठेक्का रकम, बजेट रकम
Asset seizure value | जफत गर्ने सम्पत्ति रू.
Co-accused row with — | no amount; asset forfeiture only

IMPORTANT: Income and expenditure subtotals are always larger than bigo in illegal property cases. If you accidentally extract either, the number will be bigger than the bigo stated in the table. Use this as a sanity check.

Rule 9 — Vague/verbal amounts → null
If the amount is expressed in vague prose (करोडौं, अरबौं, लाखौं) with no accompanying numeric, return null.
Word-amount parsing of Nepali number words is error-prone and CIAA's structured documents always pair prose amounts with numerics when a formal bigo exists.

Case ID: {case.case_id}
Case title: {case.title}
Source metadata (title, description, filenames, URLs):
{source_context}

Press release markdown:
{markdown[:100000]}
"""

    def _build_source_context(self, source: DocumentSource | None) -> str:
        if source is None:
            return ""

        parts = [
            f"source_id: {source.source_id}",
            f"title: {source.title or ''}",
            f"description: {source.description or ''}",
            f"uploaded_filename: {source.uploaded_filename or ''}",
        ]
        parts.extend(f"url: {urllib.parse.unquote(url)}" for url in source.url_links)
        parts.extend(
            f"uploaded_file: {name}" for name in self._source_upload_names(source)
        )
        return "\n".join(parts)[:20000]

    def _source_upload_names(self, source: DocumentSource) -> list[str]:
        if not source.pk:
            return []
        return [
            upload.filename or Path(upload.file.name).name
            for upload in source.uploaded_files.all()
        ]

    def _is_explicit_bigo_context(self, evidence_quote: Any) -> bool:
        if not isinstance(evidence_quote, str):
            return False
        normalized_quote = evidence_quote.strip().lower()
        if not normalized_quote:
            return False
        return any(keyword in normalized_quote for keyword in BIGO_CONTEXT_KEYWORDS)

    def _parse_json_response(self, content: str) -> dict[str, Any]:
        content = content.strip()
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        fenced = re.search(r"```(?:json)?\s*(\{[^}]*\})\s*```", content, re.DOTALL)
        candidates = [fenced.group(1)] if fenced else []
        candidates.extend(
            re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content, re.DOTALL)
        )

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed

        preview = re.sub(r"\s+", " ", content)[:200]
        raise ValueError(
            "LLM response did not contain a JSON object. "
            f"Response preview: {preview!r}"
        )

    def _confidence_rank(self, confidence: str) -> int:
        rank = {"low": 1, "medium": 2, "high": 3}
        return rank.get(confidence, 0)

    def _coerce_bigo_int(self, value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, int):
            return value if value > 0 else None
        if isinstance(value, float):
            return int(value) if value > 0 else None
        if not isinstance(value, str):
            return None

        normalized = value.translate(_NEPALI_TO_ASCII_DIGITS)
        digits_only = re.sub(r"[^\d]", "", normalized)
        if not digits_only:
            return None
        bigo = int(digits_only)
        return bigo if bigo > 0 else None

    def _patch_case_bigo(
        self,
        case: Case,
        bigo: int,
        api_base_url: str,
        api_token: str,
    ) -> None:
        url = self._case_patch_url(api_base_url, case.slug)
        payload = [{"op": "replace", "path": "/bigo", "value": bigo}]
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=url,
            method="PATCH",
            data=data,
            headers={
                "Authorization": f"Token {api_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30):  # NOSONAR / noqa: S310
                return
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise CommandError(
                f"PATCH failed for case {case.case_id} (status {exc.code}): {body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise CommandError(
                f"PATCH failed for case {case.case_id}: {exc.reason}"
            ) from exc

    def _case_patch_url(self, api_base_url: str, case_slug: str) -> str:
        parsed = urllib.parse.urlparse((api_base_url or "").strip())
        if not (
            parsed.scheme == "https"
            or (parsed.scheme == "http" and self._is_loopback_host(parsed.hostname))
        ):
            raise ValueError(
                f"Invalid api_base_url '{api_base_url}': use https for non-local hosts."
            )
        if not parsed.netloc:
            raise ValueError(
                f"Invalid api_base_url '{api_base_url}': URL must include a host."
            )
        path = parsed.path.rstrip("/")
        base = urllib.parse.urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))
        quoted_slug = urllib.parse.quote(str(case_slug).strip(), safe="")
        if base.endswith("/api"):
            return f"{base}/cases/{quoted_slug}/"
        return f"{base}/api/cases/{quoted_slug}/"
