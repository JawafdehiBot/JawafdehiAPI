"""Tests for enrich_case_overview management command.

Groups:
  A — Case eligibility & selection (_get_eligible_cases)
  B — Source gathering & categorization (_gather_case_sources)
  C — Source text conversion (_convert_sources_to_texts, _convert_one_source)
  D — LLM extraction call (_call_llm, prompt construction)
  E — LLM formatting call
  F — Validation & quality gates (_validate_overview)
  G — Source safety & URL handling
  H — Pipeline integration (end-to-end via call_command)
  I — Edge cases & error handling
"""

import json
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from django.core.management import CommandError, call_command

from cases.management.commands.enrich_case_overview import (
    _CLOUD_METADATA_IP,
    Command,
    _confined_output_path,
    _copy_stream_to_path_with_limit,
    _extract_json_body,
    _has_charge_sheet_keywords,
    _has_ngm_store_url,
    _has_press_release_keywords,
    _is_direct_document_url,
    _llm_endpoint,
    _llm_timeout,
    _SafeRedirectHandler,
    _sanitize_download_filename,
    _source_url_priority,
    _validate_host_safety,
    normalize_base_url,
    normalize_model,
)
from cases.models import Case, CaseState, CaseType, DocumentSource, SourceType

# ── test fixtures ───────────────────────────────────────────────────────────


def _make_case(
    case_id="case-test-001",
    title="Test Case",
    state=CaseState.DRAFT,
    bigo=None,
    evidence=None,
    short_description=None,
    description=None,
    court_cases=None,
):
    return Case.objects.create(
        case_id=case_id,
        case_type=CaseType.CORRUPTION,
        state=state,
        title=title,
        timeline=[],
        evidence=evidence or [],
        bigo=bigo,
        short_description=short_description or "",
        description=description or "",
        court_cases=court_cases,
    )


def _make_source(
    source_id="source:test:001",
    title="Test Source",
    source_type=SourceType.OFFICIAL_GOVERNMENT,
    url=None,
    description=None,
    uploaded_file=None,
    is_deleted=False,
    publication_date=None,
):
    from datetime import date

    return DocumentSource.objects.create(
        source_id=source_id,
        title=title,
        source_type=source_type,
        url=url or [],
        description=description or "",
        is_deleted=is_deleted,
        publication_date=publication_date or date.today(),
    )


_PRIVATE_IP_CLASS_A = "10.0.0.1"  # NOSONAR
_PRIVATE_IP_CLASS_B = "172.16.0.1"  # NOSONAR
_PRIVATE_IP_CLASS_C = "192.168.1.1"  # NOSONAR


# ═══════════════════════════════════════════════════════════════════════════════
# Group G: Source Safety & URL Handling
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidateHostSafety:
    def test_blocks_loopback(self):
        for host in ("127.0.0.1", "localhost"):
            with pytest.raises(ValueError, match="Blocked internal"):
                _validate_host_safety(host)

    def test_blocks_private_ranges(self):
        for host in (_PRIVATE_IP_CLASS_A, _PRIVATE_IP_CLASS_C, _PRIVATE_IP_CLASS_B):
            with pytest.raises(ValueError, match="Blocked internal"):
                _validate_host_safety(host)

    def test_blocks_cloud_metadata(self):
        with pytest.raises(ValueError, match="Blocked internal"):
            _validate_host_safety(_CLOUD_METADATA_IP)

    def test_blocks_static_blocked_hostnames(self):
        for host in ("metadata.google.internal", "metadata", "0.0.0.0"):
            with pytest.raises(ValueError, match="Blocked internal"):
                _validate_host_safety(host)

    def test_allows_public_host(self):
        _validate_host_safety("example.com")
        _validate_host_safety("ciaa.gov.np")

    def test_none_hostname_raises(self):
        with pytest.raises(ValueError, match="hostname is None"):
            _validate_host_safety(None)

    def test_unresolvable_hostname_raises(self):
        with pytest.raises(ValueError, match="Cannot resolve host"):
            _validate_host_safety("this-does-not-exist.example.invalid.foo")


class TestSafeRedirectHandler:
    def test_blocks_redirect_to_internal(self):
        handler = _SafeRedirectHandler()
        req = urllib.request.Request("https://example.com/page")
        with pytest.raises(urllib.error.HTTPError, match="Unsafe redirect"):
            handler.redirect_request(
                req, None, 302, "Found", {}, "http://localhost/admin"
            )

    def test_allows_redirect_to_public(self):
        handler = _SafeRedirectHandler()
        req = urllib.request.Request("https://example.com/page")
        result = handler.redirect_request(
            req, None, 301, "Moved", {}, "https://example.com/page"
        )
        assert result is not None

    def test_rejects_non_http_redirect(self):
        handler = _SafeRedirectHandler()
        req = urllib.request.Request("https://example.com/page")
        with pytest.raises(urllib.error.HTTPError, match="Unsafe redirect"):
            handler.redirect_request(
                req, None, 302, "Found", {}, "file:///tmp/evil.txt"
            )

    def test_rejects_empty_host_redirect(self):
        handler = _SafeRedirectHandler()
        req = urllib.request.Request("https://example.com/page")
        with pytest.raises(urllib.error.HTTPError, match="Unsafe redirect"):
            handler.redirect_request(req, None, 302, "Found", {}, "https://")


class TestSanitizeDownloadFilename:
    def test_preserves_extension(self):
        result = _sanitize_download_filename("press-release.pdf", "src-001")
        assert result.endswith(".pdf")

    def test_handles_path_traversal(self):
        result = _sanitize_download_filename("../../../etc/passwd", "src-001")
        assert "etc" not in result.lower() or "passwd" not in result

    def test_handles_null_bytes(self):
        result = _sanitize_download_filename("file\x00name.pdf", "src-001")
        assert "\x00" not in result

    def test_handles_dot_and_dotdot(self):
        for name in (".", ".."):
            result = _sanitize_download_filename(name, "src-001")
            assert result == "src-001.bin"

    def test_handles_empty_and_none(self):
        assert _sanitize_download_filename("", "src-001") == "src-001.bin"
        assert _sanitize_download_filename(None, "src-001") == "src-001.bin"

    def test_truncates_long_filenames_with_hash(self):
        long_name = ("a" * 500) + ".pdf"
        result = _sanitize_download_filename(long_name, "src-001")
        assert len(result) <= 200
        assert result.endswith(".pdf")
        assert "-" in result


class TestConfinedOutputPath:
    def test_prevents_escape(self):
        output_dir = Path(tempfile.gettempdir(), "test-output")
        with pytest.raises(CommandError, match="Refusing to write outside"):
            _confined_output_path(output_dir, "../../etc/passwd")


# ═══════════════════════════════════════════════════════════════════════════════
# Group A: Case Eligibility & Selection
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.django_db
class TestGetEligibleCases:
    def test_returns_only_draft_cases_without_overview(self):
        _make_case("case-draft-no-overview", "Draft no overview")
        _make_case(
            "case-draft-with-overview",
            "Draft with overview",
            short_description="exists",
        )
        _make_case("case-published", "Published", state=CaseState.PUBLISHED)

        cmd = Command()
        cases = cmd._get_eligible_cases(limit=None, force=False, case_id=None)
        ids = {c.case_id for c in cases}
        assert "case-draft-no-overview" in ids
        assert "case-draft-with-overview" not in ids
        assert "case-published" not in ids

    def test_force_reprocesses_existing_overviews(self):
        _make_case(
            "case-draft-with-overview", "Has overview", short_description="exists"
        )

        cmd = Command()
        cases = cmd._get_eligible_cases(limit=None, force=True, case_id=None)
        ids = {c.case_id for c in cases}
        assert "case-draft-with-overview" in ids

    def test_filters_by_specific_case_id(self):
        _make_case("case-a", "A")
        _make_case("case-b", "B")
        _make_case("case-c", "C")

        cmd = Command()
        cases = cmd._get_eligible_cases(limit=None, force=False, case_id="case-b")
        assert len(cases) == 1
        assert cases[0].case_id == "case-b"

    def test_respects_limit(self):
        for i in range(10):
            _make_case(f"case-{i:03d}", f"Case {i}")

        cmd = Command()
        cases = cmd._get_eligible_cases(limit=3, force=False, case_id=None)
        assert len(cases) == 3

    def test_limit_zero_returns_empty(self):
        _make_case("case-a", "A")
        cmd = Command()
        cases = cmd._get_eligible_cases(limit=0, force=False, case_id=None)
        assert len(cases) == 0

    def test_excludes_cases_with_description_filled(self):
        """Cases with description (even without short_description) should be excluded."""
        _make_case(
            "case-no-short",
            "Has description only",
            short_description="",
            description="Filled description",
        )
        _make_case(
            "case-empty-both", "Empty both", short_description="", description=""
        )

        cmd = Command()
        cases = cmd._get_eligible_cases(limit=None, force=False, case_id=None)
        ids = {c.case_id for c in cases}
        assert "case-no-short" not in ids  # description filled → excluded
        assert "case-empty-both" in ids  # both empty → eligible

    def test_excludes_cases_with_short_description_filled(self):
        """Cases with short_description (even without description) should be excluded."""
        _make_case(
            "case-with-short",
            "Has short desc only",
            short_description="Short desc",
            description="",
        )
        cmd = Command()
        cases = cmd._get_eligible_cases(limit=None, force=False, case_id=None)
        ids = {c.case_id for c in cases}
        assert "case-with-short" not in ids


# ═══════════════════════════════════════════════════════════════════════════════
# Group B: Source Gathering & Categorization
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.django_db
class TestGatherCaseSources:
    def _make_cmd_with_cache(self, sources):
        cmd = Command()
        cmd._source_lookup = {s.source_id: s for s in sources}
        return cmd

    def test_categorizes_charge_sheet_from_official_government(self):
        source = _make_source(
            "src-cs-001",
            "अभियोगपत्र — Special Court",
            SourceType.OFFICIAL_GOVERNMENT,
        )
        case = _make_case(evidence=[{"source_id": source.source_id}])
        cmd = self._make_cmd_with_cache([source])

        gathered = cmd._gather_case_sources(case)
        assert gathered["charge_sheet"] is not None
        assert gathered["charge_sheet"].source_id == "src-cs-001"

    def test_categorizes_press_release_without_chargesheet_keywords(self):
        source = _make_source(
            "src-pr-001",
            "Press Release — CIAA Investigation",
            SourceType.OFFICIAL_GOVERNMENT,
        )
        case = _make_case(evidence=[{"source_id": source.source_id}])
        cmd = self._make_cmd_with_cache([source])

        gathered = cmd._gather_case_sources(case)
        assert gathered["charge_sheet"] is None
        assert len(gathered["press_releases"]) == 1
        assert gathered["press_releases"][0].source_id == "src-pr-001"

    def test_categorizes_court_orders(self):
        source = _make_source("src-co-001", "Verdict", SourceType.LEGAL_COURT_ORDER)
        case = _make_case(evidence=[{"source_id": source.source_id}])
        cmd = self._make_cmd_with_cache([source])

        gathered = cmd._gather_case_sources(case)
        assert len(gathered["court_orders"]) == 1

    def test_categorizes_investigative_reports(self):
        source = _make_source(
            "src-ir-001", "Investigation Report", SourceType.INVESTIGATIVE_REPORT
        )
        case = _make_case(evidence=[{"source_id": source.source_id}])
        cmd = self._make_cmd_with_cache([source])

        gathered = cmd._gather_case_sources(case)
        assert len(gathered["investigative_reports"]) == 1

    def test_categorizes_financial_docs(self):
        source = _make_source(
            "src-fd-001", "Audit Report", SourceType.FINANCIAL_FORENSIC
        )
        case = _make_case(evidence=[{"source_id": source.source_id}])
        cmd = self._make_cmd_with_cache([source])

        gathered = cmd._gather_case_sources(case)
        assert len(gathered["financial_docs"]) == 1

    def test_categorizes_media_news(self):
        source = _make_source("src-mn-001", "News Article", SourceType.MEDIA_NEWS)
        case = _make_case(evidence=[{"source_id": source.source_id}])
        cmd = self._make_cmd_with_cache([source])

        gathered = cmd._gather_case_sources(case)
        assert len(gathered["media_sources"]) == 1

    def test_categorizes_procedural(self):
        source = _make_source(
            "src-proc-001", "Procedural Doc", SourceType.LEGAL_PROCEDURAL
        )
        case = _make_case(evidence=[{"source_id": source.source_id}])
        cmd = self._make_cmd_with_cache([source])

        gathered = cmd._gather_case_sources(case)
        assert len(gathered["procedural_docs"]) == 1

    def test_categorizes_other_types(self):
        source = _make_source("src-oth-001", "Misc", SourceType.OTHER_VISUAL)
        case = _make_case(evidence=[{"source_id": source.source_id}])
        cmd = self._make_cmd_with_cache([source])

        gathered = cmd._gather_case_sources(case)
        assert len(gathered["other_docs"]) == 1

    def test_ignores_missing_sources(self):
        case = _make_case(evidence=[{"source_id": "src:nonexistent"}])
        cmd = Command()
        cmd._source_lookup = {}

        gathered = cmd._gather_case_sources(case)
        assert gathered["charge_sheet"] is None
        assert gathered["press_releases"] == []
        assert gathered["court_orders"] == []

    def test_ignores_deleted_sources(self):
        source = _make_source("src-deleted-001", "Deleted", is_deleted=True)
        case = _make_case(evidence=[{"source_id": source.source_id}])
        cmd = Command()
        cmd._source_lookup = (
            {}
        )  # deleted sources not in cache (filtered by is_deleted=False)

        gathered = cmd._gather_case_sources(case)
        assert gathered["charge_sheet"] is None

    def test_official_govt_with_ngm_store_url_becomes_press_release(self):
        """OFFICIAL_GOVERNMENT with ngm-store URL but no press release keywords → press_releases."""
        source = _make_source(
            "src-ngm-001",
            "Notice — CIAA",
            SourceType.OFFICIAL_GOVERNMENT,
            url=["https://ngm-store.jawafdehi.org/080-081/file.pdf"],
        )
        case = _make_case(evidence=[{"source_id": source.source_id}])
        cmd = self._make_cmd_with_cache([source])

        gathered = cmd._gather_case_sources(case)
        assert gathered["charge_sheet"] is None
        assert len(gathered["press_releases"]) == 1
        assert gathered["press_releases"][0].source_id == "src-ngm-001"

    def test_official_govt_catch_all_goes_to_other_docs(self):
        """OFFICIAL_GOVERNMENT without charge sheet or press release keywords → other_docs."""
        source = _make_source(
            "src-generic-001",
            "General Government Notice",
            SourceType.OFFICIAL_GOVERNMENT,
        )
        case = _make_case(evidence=[{"source_id": source.source_id}])
        cmd = self._make_cmd_with_cache([source])

        gathered = cmd._gather_case_sources(case)
        assert gathered["charge_sheet"] is None
        assert len(gathered["press_releases"]) == 0
        assert len(gathered["other_docs"]) == 1
        assert gathered["other_docs"][0].source_id == "src-generic-001"

    def test_handles_empty_evidence(self):
        case = _make_case(evidence=None)
        cmd = Command()
        cmd._source_lookup = {}

        gathered = cmd._gather_case_sources(case)
        for key, val in gathered.items():
            if isinstance(val, list):
                assert val == []
            else:
                assert val is None

    def test_handles_malformed_evidence_entries(self):
        case = _make_case(evidence=[None, "not-a-dict", 123, {"no_source_id": True}])
        cmd = Command()
        cmd._source_lookup = {}

        gathered = cmd._gather_case_sources(case)
        assert gathered["charge_sheet"] is None


# ═══════════════════════════════════════════════════════════════════════════════
# Group C: Source Text Conversion
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.django_db
class TestConvertSourcesToTexts:
    def test_handles_all_empty_sources(self):
        cmd = Command()
        gathered = {
            "charge_sheet": None,
            "press_releases": [],
            "court_orders": [],
            "investigative_reports": [],
            "financial_docs": [],
        }
        texts = cmd._convert_sources_to_texts(gathered)
        assert texts["charge_sheet"] == ""
        assert texts["press_releases"] == []
        assert texts["court_orders"] == []
        assert texts["investigative_reports"] == []
        assert texts["financial_docs"] == []

    def test_truncates_press_releases_to_5k(self):
        cmd = Command()
        source = _make_source("src-pr-001", "PR", SourceType.OFFICIAL_GOVERNMENT)
        long_text = "X" * 10000
        with patch.object(cmd, "_convert_one_source", return_value=long_text):
            texts = cmd._convert_sources_to_texts(
                {
                    "charge_sheet": None,
                    "press_releases": [source],
                    "court_orders": [],
                    "investigative_reports": [],
                    "financial_docs": [],
                }
            )
        assert len(texts["press_releases"][0]) == 5000

    def test_head_tail_for_long_court_orders(self):
        from cases.management.commands.enrich_case_overview import (
            COURT_ORDER_FULL_MAX,
            COURT_ORDER_HEAD_CHARS,
            COURT_ORDER_TAIL_CHARS,
        )

        cmd = Command()
        source = _make_source("src-co-001", "CO", SourceType.LEGAL_COURT_ORDER)
        # Shorter than COURT_ORDER_FULL_MAX → full text preserved
        short_text = "X" * 10000
        with patch.object(cmd, "_convert_one_source", return_value=short_text):
            texts = cmd._convert_sources_to_texts(
                {
                    "charge_sheet": None,
                    "press_releases": [],
                    "court_orders": [source],
                    "investigative_reports": [],
                    "financial_docs": [],
                }
            )
        assert len(texts["court_orders"][0]) == 10000  # short → full

        # Longer than COURT_ORDER_FULL_MAX → head+tail
        long_text = "X" * (COURT_ORDER_FULL_MAX + 5000)
        with patch.object(cmd, "_convert_one_source", return_value=long_text):
            texts2 = cmd._convert_sources_to_texts(
                {
                    "charge_sheet": None,
                    "press_releases": [],
                    "court_orders": [source],
                    "investigative_reports": [],
                    "financial_docs": [],
                }
            )
        truncated = texts2["court_orders"][0]
        expected_len = (
            COURT_ORDER_HEAD_CHARS
            + COURT_ORDER_TAIL_CHARS
            + len("\n\n[... मध्य भाग संक्षिप्त गरिएको — तल फैसला/सजाय खण्ड ...]\n\n")
        )
        assert len(truncated) == expected_len  # long → head+tail
        assert truncated.startswith("X" * COURT_ORDER_HEAD_CHARS)

    def test_truncates_investigative_reports_to_3k(self):
        cmd = Command()
        source = _make_source("src-ir-001", "IR", SourceType.INVESTIGATIVE_REPORT)
        long_text = "X" * 10000
        with patch.object(cmd, "_convert_one_source", return_value=long_text):
            texts = cmd._convert_sources_to_texts(
                {
                    "charge_sheet": None,
                    "press_releases": [],
                    "court_orders": [],
                    "investigative_reports": [source],
                    "financial_docs": [],
                }
            )
        assert len(texts["investigative_reports"][0]) == 3000

    def test_filters_under_50_chars(self):
        cmd = Command()
        source = _make_source("src-short-001", "Short", SourceType.OFFICIAL_GOVERNMENT)
        with patch.object(cmd, "_convert_one_source", return_value="too short"):
            texts = cmd._convert_sources_to_texts(
                {
                    "charge_sheet": None,
                    "press_releases": [source],
                    "court_orders": [],
                    "investigative_reports": [],
                    "financial_docs": [],
                }
            )
        assert len(texts["press_releases"]) == 0

    def test_handles_convert_one_source_failure(self):
        cmd = Command()
        source = _make_source("src-fail-001", "Fail", SourceType.OFFICIAL_GOVERNMENT)
        with patch.object(cmd, "_convert_one_source", return_value=None):
            texts = cmd._convert_sources_to_texts(
                {
                    "charge_sheet": None,
                    "press_releases": [source],
                    "court_orders": [],
                    "investigative_reports": [],
                    "financial_docs": [],
                }
            )
        assert len(texts["press_releases"]) == 0

    def test_handles_none_text_content_gracefully(self):
        cmd = Command()
        source = _make_source("src-none-001", "None", SourceType.OFFICIAL_GOVERNMENT)
        with patch.object(cmd, "_convert_one_source", return_value="  "):
            texts = cmd._convert_sources_to_texts(
                {
                    "charge_sheet": None,
                    "press_releases": [source],
                    "court_orders": [],
                    "investigative_reports": [],
                    "financial_docs": [],
                }
            )
        assert len(texts["press_releases"]) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Group D: LLM Extraction Call
# ═══════════════════════════════════════════════════════════════════════════════


class TestExtractJsonBody:
    def test_extracts_valid_json(self):
        result = _extract_json_body('{"key": "value"}')
        assert result == '{"key": "value"}'

    def test_extracts_json_from_markdown_fence(self):
        result = _extract_json_body('```json\n{"key": "value"}\n```')
        assert json.loads(result) == {"key": "value"}

    def test_extracts_json_from_fence_without_language_tag(self):
        result = _extract_json_body('```\n{"key": "value"}\n```')
        assert json.loads(result) == {"key": "value"}

    def test_extracts_json_with_conversational_prefix(self):
        result = _extract_json_body('Here is the extraction:\n{"key": "value"}\nDone.')
        assert json.loads(result) == {"key": "value"}

    def test_extracts_json_from_middle_of_text(self):
        raw = 'Prefix text ```json\n{"a": 1}\n``` suffix text'
        result = _extract_json_body(raw)
        assert json.loads(result) == {"a": 1}

    def test_returns_stripped_input_on_failure(self):
        result = _extract_json_body("  just some text with no json  ")
        assert result == "just some text with no json"

    def test_handles_fence_with_no_closing(self):
        result = _extract_json_body('```json\n{"key": "value"}')
        assert json.loads(result) == {"key": "value"}

    def test_handles_nested_braces(self):
        result = _extract_json_body('{"outer": {"inner": [1, 2, 3]}}')
        assert json.loads(result) == {"outer": {"inner": [1, 2, 3]}}


class TestNormalizeModel:
    def test_strips_opencode_prefix(self):
        assert normalize_model("opencode-go/claude-sonnet-4-5") == "claude-sonnet-4-5"

    def test_strips_openai_colon_prefix(self):
        assert normalize_model("openai:gpt-5") == "gpt-5"

    def test_leaves_bare_model_unchanged(self):
        assert normalize_model("claude-sonnet-4-5") == "claude-sonnet-4-5"


class TestNormalizeBaseUrl:
    def test_replaces_zen_v1_with_zen_go_v1(self):
        result = normalize_base_url("https://opencode.ai/zen/v1")
        assert result == "https://opencode.ai/zen/go/v1"

    def test_adds_v1_suffix(self):
        result = normalize_base_url("https://opencode.ai/zen/go")
        assert result == "https://opencode.ai/zen/go/v1"

    def test_strips_trailing_slash(self):
        result = normalize_base_url("https://example.com/api/")
        assert result == "https://example.com/api"


class TestLlmEndpoint:
    def test_standard_model_uses_chat_completions(self):
        url = _llm_endpoint("https://api.example.com", "claude-sonnet-4-5")
        assert url.endswith("/chat/completions")


class TestLlmTimeout:
    def test_default(self):
        assert _llm_timeout(None) == 300

    def test_cli_wins(self):
        assert _llm_timeout(60) == 60

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("JAWAFDEHI_LLM_TIMEOUT_SECONDS", "120")
        assert _llm_timeout(None) == 120

    def test_invalid_env_fallback(self, monkeypatch):
        monkeypatch.setenv("JAWAFDEHI_LLM_TIMEOUT_SECONDS", "not-a-number")
        assert _llm_timeout(None) == 300

    def test_zero_env_fallback(self, monkeypatch):
        monkeypatch.setenv("JAWAFDEHI_LLM_TIMEOUT_SECONDS", "0")
        assert _llm_timeout(None) == 300

    def test_zero_cli_raises(self):
        with pytest.raises(CommandError, match="must be > 0"):
            _llm_timeout(0)

    def test_negative_cli_raises(self):
        with pytest.raises(CommandError, match="must be > 0"):
            _llm_timeout(-5)


@pytest.mark.django_db
class TestLlmPromptConstruction:
    def _make_cmd(self):
        cmd = Command()
        return cmd

    def test_prompt_includes_court_cases_when_present(self):
        case = _make_case(court_cases=["Special Court 081-CR-0123"])

        from cases.management.commands.enrich_case_overview import (
            EXTRACTION_USER_PROMPT,
        )

        prompt = EXTRACTION_USER_PROMPT.format(
            case_id=case.case_id,
            case_title=case.title,
            court_cases=json.dumps(case.court_cases, ensure_ascii=False),
            bigo="उल्लेख छैन",
            press_release_texts="PR text",
            court_order_texts="CO text",
            other_texts="(No additional documents)",
            source_quality_notes_section="",
        )
        assert "081-CR-0123" in prompt

    def test_prompt_includes_bigo_when_present(self):
        case = _make_case(bigo=123456)

        from cases.management.commands.enrich_case_overview import (
            EXTRACTION_USER_PROMPT,
        )

        prompt = EXTRACTION_USER_PROMPT.format(
            case_id=case.case_id,
            case_title=case.title,
            court_cases="None",
            bigo="रू 123,456",
            press_release_texts="PR text",
            court_order_texts="CO text",
            other_texts="(No additional documents)",
            source_quality_notes_section="",
        )
        assert "रू 123,456" in prompt

    def test_prompt_handles_null_court_cases(self):
        case = _make_case(court_cases=None)

        from cases.management.commands.enrich_case_overview import (
            EXTRACTION_USER_PROMPT,
        )

        prompt = EXTRACTION_USER_PROMPT.format(
            case_id=case.case_id,
            case_title=case.title,
            court_cases="None",
            bigo="उल्लेख छैन",
            press_release_texts="PR text",
            court_order_texts="CO text",
            other_texts="(No additional documents)",
            source_quality_notes_section="",
        )
        assert "None" in prompt or "Known court cases" in prompt

    def test_prompt_handles_null_bigo(self):
        case = _make_case(bigo=None)

        from cases.management.commands.enrich_case_overview import (
            EXTRACTION_USER_PROMPT,
        )

        prompt = EXTRACTION_USER_PROMPT.format(
            case_id=case.case_id,
            case_title=case.title,
            court_cases="None",
            bigo="उल्लेख छैन",
            press_release_texts="(No press releases available)",
            court_order_texts="(No court orders available)",
            other_texts="(No additional documents)",
            source_quality_notes_section="",
        )
        assert "उल्लेख छैन" in prompt


# ═══════════════════════════════════════════════════════════════════════════════
# Group E: LLM Formatting Call
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.django_db
class TestProcessCaseFormatting:
    def _make_cmd(self):
        return Command()

    def test_formatting_succeeds_with_valid_response(self):
        source = _make_source(
            "src-pr-001", "CIAA Press Release", SourceType.OFFICIAL_GOVERNMENT
        )
        case = _make_case(
            "case-format-001",
            "Formatting Success",
            evidence=[{"source_id": source.source_id}],
        )

        cmd = self._make_cmd()
        cmd._source_lookup = {source.source_id: source}

        extraction_json = {
            "accused_persons": [],
            "case_metadata": {},
            "fiscal_analysis": [],
            "legal_provisions": [],
            "key_events": [],
            "total_disputed_amount": None,
        }
        format_json = {
            "short_description": "संक्षिप्त विवरण भ्रष्टाचार मुद्दाको। यो नेपालीमा छ।",
            "description": "क) अभियोगदावीको सार\n\n"
            + (
                "यो मुद्दा सार्वजनिक संस्थामा भएको भ्रष्टाचारको हो। विस्तृत विवरण सहित। "
                * 4
            ),
        }

        with (
            patch.object(cmd, "_convert_one_source", return_value="test content" * 10),
            patch.object(cmd, "_call_llm") as mock_llm,
        ):
            mock_llm.side_effect = [
                json.dumps(extraction_json),
                json.dumps(format_json),
            ]
            cmd._process_case(
                case,
                "claude-sonnet-4-5",
                "https://api.example.com",
                "key",
                300,
                True,
                False,
            )

        case.refresh_from_db()
        assert case.short_description == format_json["short_description"]
        assert case.description == format_json["description"].strip()

    def test_formatting_invalid_json_fails(self):
        source = _make_source(
            "src-pr-002", "CIAA Press Release", SourceType.OFFICIAL_GOVERNMENT
        )
        case = _make_case(
            "case-fail-001",
            "Formatting Fail",
            evidence=[{"source_id": source.source_id}],
        )

        cmd = self._make_cmd()
        cmd._source_lookup = {source.source_id: source}

        extraction_json = {"accused_persons": []}

        with (
            patch.object(cmd, "_convert_one_source", return_value="test content" * 10),
            patch.object(cmd, "_call_llm") as mock_llm,
            patch.object(cmd.stdout, "write"),
        ):
            mock_llm.side_effect = [json.dumps(extraction_json), "not valid json {{{"]
            cmd._process_case(
                case,
                "claude-sonnet-4-5",
                "https://api.example.com",
                "key",
                300,
                True,
                False,
            )

        assert cmd.stats["llm_formatting_failures"] == 1
        assert cmd.stats["cases_failed"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Group F: Validation & Quality Gates
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidateOverview:
    def _make_cmd(self):
        return Command()

    def test_requires_section_ka(self):
        cmd = self._make_cmd()
        valid, issues = cmd._validate_overview(
            "पर्याप्त लामो नेपाली संक्षिप्त विवरण भ्रष्टाचारको मुद्दाको बारेमा जानकारी।",
            "ख) आकर्षित कानुनी व्यवस्था\n\nकानूनी प्रावधानहरूको विवरण।",
        )
        assert not valid
        assert any("क) अभियोगदावीको सार" in i for i in issues)

    def test_requires_description_minimum_length(self):
        cmd = self._make_cmd()
        valid, issues = cmd._validate_overview("short desc", "tiny")
        assert not valid
        assert any("short" in i for i in issues)

    def test_requires_short_description_minimum_length(self):
        cmd = self._make_cmd()
        valid, issues = cmd._validate_overview(
            "छोटो", "क) अभियोगदावीको सार\n\n" + ("विस्तृत नेपाली विवरण।" * 20)
        )
        assert not valid
        assert any("short_description" in i for i in issues)

    def test_rejects_long_short_description(self):
        cmd = self._make_cmd()
        valid, issues = cmd._validate_overview(
            "नेपाली " * 300, "क) अभियोगदावीको सार\n\n" + ("विस्तृत नेपाली विवरण।" * 20)
        )
        assert not valid
        assert any("short_description" in i for i in issues)

    def test_rejects_placeholder_text(self):
        cmd = self._make_cmd()
        valid, issues = cmd._validate_overview(
            "पर्याप्त लामो नेपाली संक्षिप्त विवरण सहितको टेक्स्ट।",
            "क) अभियोगदावीको सार\n\n" + ("विस्तृत नेपाली विवरण। " * 20) + "[TBD]",
        )
        assert not valid
        assert any("Placeholder" in i for i in issues)

    def test_rejects_todo_and_insert_placeholders(self):
        cmd = self._make_cmd()
        for token in ["[TODO]", "[insert]"]:
            valid, issues = cmd._validate_overview(
                "पर्याप्त लामो नेपाली संक्षिप्त विवरण सहितको टेक्स्ट।",
                "क) अभियोगदावीको सार\n\n" + ("विस्तृत नेपाली विवरण। " * 20) + token,
            )
            assert not valid

    def test_warns_on_low_devanagari_ratio(self):
        cmd = self._make_cmd()
        description = (
            "क) अभियोगदावीको सार\n\n"
            + "This is mostly English text for a Nepali document. " * 20
        )
        valid, issues = cmd._validate_overview(
            "पर्याप्त लामो नेपाली संक्षिप्त विवरण सहितको टेक्स्ट।", description
        )
        assert any("Devanagari" in i for i in issues)
        assert not valid  # Hard gate as of ec83ba7

    def test_warns_on_raw_html(self):
        cmd = self._make_cmd()
        valid, issues = cmd._validate_overview(
            "पर्याप्त लामो नेपाली संक्षिप्त विवरण भ्रष्टाचार मुद्दाको।",
            "क) अभियोगदावीको सार\n\n"
            + ("विस्तृत नेपाली विवरण। " * 20)
            + "<table><tr><td>data</td></tr></table>",
        )
        assert any("HTML" in i for i in issues)
        assert not valid  # Hard gate as of ec83ba7

    def test_passes_valid_nepali_content(self):
        cmd = self._make_cmd()
        valid, issues = cmd._validate_overview(
            "भ्रष्टाचार मुद्दाको संक्षिप्त विवरण। यसमा आरोपित व्यक्ति र बिगोको जानकारी समावेश छ।",
            "क) अभियोगदावीको सार\n\n"
            + ("यो मुद्दा विशेष अदालतमा दायर भएको भ्रष्टाचारको मुद्दा हो। ") * 10,
        )
        assert valid
        assert len(issues) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Group H: Pipeline Integration (End-to-End)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.django_db
class TestPipelineIntegration:
    @pytest.fixture(autouse=True)
    def _set_api_key(self, monkeypatch):
        monkeypatch.setenv("JAWAFDEHI_LLM_API_KEY", "test-api-key")

    def test_dry_run_previews_without_modifying_db(self):
        source = _make_source(
            "src-pr-001", "CIAA Press Release", SourceType.OFFICIAL_GOVERNMENT
        )
        case = _make_case(
            "case-dry-001", "Dry Run Case", evidence=[{"source_id": source.source_id}]
        )

        extraction_json = {
            "accused_persons": [],
            "case_metadata": {},
            "fiscal_analysis": [],
        }
        format_json = {
            "short_description": "भ्रष्टाचार मुद्दाको संक्षिप्त विवरण। नेपाली भाषामा लेखिएको।",
            "description": "क) अभियोगदावीको सार\n\n" + ("विस्तृत नेपाली विवरण। " * 10),
        }

        with (
            patch(
                "cases.management.commands.enrich_case_overview.Command._convert_one_source",
                return_value="test content " * 10,
            ),
            patch(
                "cases.management.commands.enrich_case_overview.Command._call_llm"
            ) as mock_llm,
        ):
            mock_llm.side_effect = [
                json.dumps(extraction_json),
                json.dumps(format_json),
            ]
            call_command("enrich_case_overview", "--dry-run")

        case.refresh_from_db()
        assert case.short_description == ""

    def test_full_pipeline_enriches_case_with_press_release_only(self):
        source = _make_source(
            "src-pr-002", "Press Release", SourceType.OFFICIAL_GOVERNMENT
        )
        case = _make_case(
            "case-pr-only-001",
            "PR-Only Case",
            evidence=[{"source_id": source.source_id}],
        )

        extraction_json = {
            "accused_persons": [{"name": "Test Person", "position": "Officer"}],
            "case_metadata": {
                "case_number": "081-CR-0123",
                "filing_date": "2081-04-01",
            },
            "fiscal_analysis": [],
            "legal_provisions": [],
            "key_events": [],
            "total_disputed_amount": "50000",
        }
        format_json = {
            "short_description": "भ्रष्टाचार मुद्दाको संक्षिप्त विवरण। यो प्रेस विज्ञप्तिमा आधारित छ।",
            "description": "क) अभियोगदावीको सार\n\n"
            + ("यो विशेष अदालतमा दायर भएको भ्रष्टाचार मुद्दा हो। ") * 10,
        }

        with (
            patch(
                "cases.management.commands.enrich_case_overview.Command._convert_one_source",
                return_value="test content " * 10,
            ),
            patch(
                "cases.management.commands.enrich_case_overview.Command._call_llm"
            ) as mock_llm,
        ):
            mock_llm.side_effect = [
                json.dumps(extraction_json),
                json.dumps(format_json),
            ]
            call_command("enrich_case_overview")

        case.refresh_from_db()
        assert case.short_description == format_json["short_description"]
        assert "क) अभियोगदावीको सार" in case.description

    def test_pipeline_skips_case_with_no_evidence(self):
        case = _make_case("case-no-evidence-001", "No Evidence")

        out = StringIO()
        call_command("enrich_case_overview", stdout=out)
        output = out.getvalue()

        assert "No evidence" in output or case.case_id in output

    def test_pipeline_skips_case_with_only_media_sources(self):
        source = _make_source("src-media-001", "News", SourceType.MEDIA_NEWS)
        _make_case(
            "case-media-only-001",
            "Media Only",
            evidence=[{"source_id": source.source_id}],
        )

        out = StringIO()
        with patch.object(
            Command, "_convert_one_source", return_value="media text " * 10
        ):
            call_command("enrich_case_overview", stdout=out)

        output = out.getvalue()
        assert "No press releases or court orders" in output

    def test_case_id_filter_only_processes_specified_case(self):
        source = _make_source(
            "src-pr-003", "Press Release — CIAA", SourceType.OFFICIAL_GOVERNMENT
        )
        case_a = _make_case(
            "case-a-001", "A", evidence=[{"source_id": source.source_id}]
        )
        case_b = _make_case(
            "case-b-001", "B", evidence=[{"source_id": source.source_id}]
        )

        extraction_json = {
            "accused_persons": [],
            "case_metadata": {},
            "fiscal_analysis": [],
        }
        format_json = {
            "short_description": "भ्रष्टाचार मुद्दाको संक्षिप्त विवरण। नेपालीमा लेखिएको।",
            "description": "क) अभियोगदावीको सार\n\n" + ("विस्तृत विवरण। " * 10),
        }

        with (
            patch.object(
                Command, "_convert_one_source", return_value="test content " * 10
            ),
            patch.object(Command, "_call_llm") as mock_llm,
        ):
            mock_llm.side_effect = [
                json.dumps(extraction_json),
                json.dumps(format_json),
            ]
            call_command("enrich_case_overview", "--case-id=case-a-001")

        case_a.refresh_from_db()
        case_b.refresh_from_db()
        assert case_a.short_description != ""
        assert case_b.short_description == ""

    def test_force_flag_reprocesses_existing_overviews(self):
        source = _make_source(
            "src-pr-004", "Press Release — CIAA", SourceType.OFFICIAL_GOVERNMENT
        )
        case = _make_case(
            "case-force-001",
            "Force Case",
            evidence=[{"source_id": source.source_id}],
            short_description="existing",
            description="existing content",
        )

        extraction_json = {"accused_persons": [], "case_metadata": {}}
        result_json = {
            "short_description": "नयाँ संक्षिप्त विवरण। भ्रष्टाचार मुद्दाको बारेमा पूर्ण जानकारी सहित।",
            "description": "क) अभियोगदावीको सार\n\nनयाँ विवरण।" + ("विस्तृत। " * 10),
        }

        with (
            patch.object(
                Command, "_convert_one_source", return_value="test content " * 10
            ),
            patch.object(Command, "_call_llm") as mock_llm,
        ):
            mock_llm.side_effect = [
                json.dumps(extraction_json),
                json.dumps(result_json),
            ]
            call_command("enrich_case_overview", "--force")

        case.refresh_from_db()
        assert case.short_description == result_json["short_description"]

    def test_limit_flag_stops_after_n_cases(self):
        source = _make_source(
            "src-pr-005", "Press Release — CIAA", SourceType.OFFICIAL_GOVERNMENT
        )
        for i in range(5):
            _make_case(
                f"case-limit-{i:03d}",
                f"Case {i}",
                evidence=[{"source_id": source.source_id}],
            )

        extraction_json = {"accused_persons": [], "case_metadata": {}}
        format_json = {
            "short_description": "भ्रष्टाचार मुद्दाको संक्षिप्त विवरण। नेपाली भाषामा।",
            "description": "क) अभियोगदावीको सार\n\n" + ("विस्तृत नेपाली विवरण। " * 10),
        }

        with (
            patch.object(
                Command, "_convert_one_source", return_value="test content " * 10
            ),
            patch.object(Command, "_call_llm") as mock_llm,
        ):
            mock_llm.side_effect = [
                json.dumps(extraction_json),
                json.dumps(format_json),
            ] * 3
            call_command("enrich_case_overview", "--limit=2")

        enriched = (
            Case.objects.filter(state=CaseState.DRAFT)
            .exclude(short_description="")
            .count()
        )
        assert enriched == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Group I: Edge Cases & Error Handling
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.django_db
class TestErrorHandling:
    def test_llm_call_retries_on_429(self):
        cmd = Command()

        class FakeResponse:
            def read(self):
                return b"rate limited"

            def close(self):
                pass

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = [
                urllib.error.HTTPError(
                    "url", 429, "Too Many Requests", {}, FakeResponse()
                ),
                urllib.error.HTTPError(
                    "url", 429, "Too Many Requests", {}, FakeResponse()
                ),
                urllib.error.HTTPError(
                    "url", 429, "Too Many Requests", {}, FakeResponse()
                ),
            ]

            with patch("time.sleep", return_value=None):
                with pytest.raises(CommandError, match="LLM HTTP 429"):
                    cmd._call_llm_opencode(
                        "claude-sonnet-4-5",
                        "https://api.example.com",
                        "key",
                        5,
                        "system",
                        "prompt",
                    )

            assert mock_urlopen.call_count == 3

    def test_llm_call_raises_after_max_retries(self):
        cmd = Command()

        class FakeResponse:
            def read(self):
                return b"server error"

            def close(self):
                pass

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.HTTPError(
                "url", 503, "Service Unavailable", {}, FakeResponse()
            )
            with patch("time.sleep", return_value=None):
                with pytest.raises(CommandError, match="LLM HTTP 503"):
                    cmd._call_llm_opencode(
                        "claude-sonnet-4-5",
                        "https://api.example.com",
                        "key",
                        5,
                        "system",
                        "prompt",
                    )

            assert mock_urlopen.call_count == 3

    def test_llm_call_handles_extraction_json_parse_failure(self):
        source = _make_source(
            "src-errtest-001", "Press Release — CIAA", SourceType.OFFICIAL_GOVERNMENT
        )
        case = _make_case(
            "case-err-json-001", "JSON Fail", evidence=[{"source_id": source.source_id}]
        )

        cmd = Command()
        cmd._source_lookup = {source.source_id: source}
        cmd.stdout = StringIO()

        with (
            patch.object(cmd, "_convert_one_source", return_value="test content " * 10),
            patch.object(cmd, "_call_llm", return_value="not json at all {{{"),
        ):
            cmd._process_case(
                case,
                "claude-sonnet-4-5",
                "https://api.example.com",
                "key",
                300,
                True,
                False,
            )

        assert cmd.stats["llm_extraction_failures"] == 1
        assert cmd.stats["cases_failed"] == 1

    def test_extraction_result_not_dict_is_handled(self):
        source = _make_source(
            "src-errtest-002", "Press Release — CIAA", SourceType.OFFICIAL_GOVERNMENT
        )
        case = _make_case(
            "case-err-type-001", "Type Fail", evidence=[{"source_id": source.source_id}]
        )

        cmd = Command()
        cmd._source_lookup = {source.source_id: source}
        cmd.stdout = StringIO()

        with (
            patch.object(cmd, "_convert_one_source", return_value="test content " * 10),
            patch.object(cmd, "_call_llm", return_value="[1, 2, 3]"),
        ):
            cmd._process_case(
                case,
                "claude-sonnet-4-5",
                "https://api.example.com",
                "key",
                300,
                True,
                False,
            )

        assert cmd.stats["llm_extraction_failures"] == 1

    def test_llm_call_retries_on_oserror(self):
        cmd = Command()
        call_count = [0]

        def mock_urlopen(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 2:
                raise OSError("connection refused")
            response = MagicMock()
            response.__enter__ = MagicMock(return_value=response)
            response.__exit__ = MagicMock(return_value=False)
            response.read.return_value = json.dumps(
                {"choices": [{"message": {"content": '{"result": "ok"}'}}]}
            ).encode()
            response.status = 200
            return response

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            with patch("time.sleep", return_value=None):
                result = cmd._call_llm_opencode(
                    "claude-sonnet-4-5",
                    "https://api.example.com",
                    "key",
                    5,
                    "system",
                    "prompt",
                )

        assert call_count[0] == 2
        assert json.loads(result) == {"result": "ok"}

    def test_download_source_to_path_enforces_max_bytes(self):
        cmd = Command()

        class StreamResponse:
            headers = {}

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self, size=-1):
                if not hasattr(self, "_called"):
                    self._called = True
                    return b"x" * 16
                return b"x" * 16

        opener = MagicMock()
        opener.open.return_value = StreamResponse()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            with (
                patch(
                    "cases.management.commands.enrich_case_overview.MAX_DOWNLOAD_BYTES",
                    16,
                    create=True,
                ),
                patch("urllib.request.build_opener", return_value=opener),
            ):
                with pytest.raises(CommandError, match="exceeds max size"):
                    cmd._download_url_to_path(
                        "https://example.com/test.pdf", "src-001", out_dir
                    )

    def test_gathered_sources_handle_none_source_type(self):
        source = _make_source("src-nulltype-001", "Null Type")
        source.source_type = None
        source.save()

        case = _make_case(evidence=[{"source_id": source.source_id}])
        cmd = Command()
        cmd._source_lookup = {source.source_id: source}

        gathered = cmd._gather_case_sources(case)
        assert gathered["charge_sheet"] is None
        assert len(gathered["other_docs"]) == 1

    def test_call_llm_opencode_empty_choices_before_success(self):
        cmd = Command()
        call_count = [0]

        def mock_urlopen(*args, **kwargs):
            call_count[0] += 1
            response = MagicMock()
            response.__enter__ = MagicMock(return_value=response)
            response.__exit__ = MagicMock(return_value=False)
            if call_count[0] < 2:
                response.read.return_value = json.dumps({"choices": []}).encode()
                response.status = 200
            else:
                response.read.return_value = json.dumps(
                    {
                        "choices": [
                            {"message": {"content": '{"result": "ok"}'}}
                        ]
                    }
                ).encode()
                response.status = 200
            return response

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            with patch("time.sleep", return_value=None):
                result = cmd._call_llm_opencode(
                    "claude-sonnet-4-5",
                    "https://api.example.com",
                    "key",
                    5,
                    "system",
                    "prompt",
                )

        assert json.loads(result) == {"result": "ok"}
        assert call_count[0] == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Helper Function Unit Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestHasChargeSheetKeywords:
    def test_detects_english_keywords(self):
        source = DocumentSource(
            title="Charge Sheet for Case 123", description="Details"
        )
        assert _has_charge_sheet_keywords(source) is True

    def test_detects_nepali_keywords(self):
        source = DocumentSource(title="अभियोगपत्र मुद्दा १२३", description="")
        assert _has_charge_sheet_keywords(source) is True

    def test_detects_keyword_with_hyphen(self):
        source = DocumentSource(title="charge-sheet filed today", description="")
        assert _has_charge_sheet_keywords(source) is True

    def test_returns_false_for_no_match(self):
        source = DocumentSource(title="Press Release", description="General notice")
        assert _has_charge_sheet_keywords(source) is False

    def test_matches_in_description(self):
        source = DocumentSource(
            title="Notice", description="This is a chargesheet document"
        )
        assert _has_charge_sheet_keywords(source) is True


class TestHasPressReleaseKeywords:
    def test_detects_english_keyword(self):
        source = DocumentSource(
            title="Press Release Regarding Investigation", description=""
        )
        assert _has_press_release_keywords(source) is True

    def test_detects_nepali_keywords(self):
        source = DocumentSource(
            title="प्रेस विज्ञप्ति — अख्तियार दुरुपयोग अनुसन्धान आयोग", description=""
        )
        assert _has_press_release_keywords(source) is True

    def test_detects_vigyapti(self):
        source = DocumentSource(title="विज्ञप्ति", description="")
        assert _has_press_release_keywords(source) is True

    def test_returns_false_for_no_match(self):
        source = DocumentSource(title="Charge Sheet", description="अभियोगपत्र")
        assert _has_press_release_keywords(source) is False

    def test_matches_in_description(self):
        source = DocumentSource(
            title="Notice", description="This is a pressrelease from CIAA"
        )
        assert _has_press_release_keywords(source) is True

    def test_handles_none_description(self):
        source = DocumentSource(title="Press Release", description=None)
        assert _has_press_release_keywords(source) is True


class TestHasNgmStoreUrl:
    def test_detects_ngm_store_url(self):
        source = DocumentSource(
            title="Press Release",
            url=["https://ngm-store.jawafdehi.org/080-081/somefile.pdf"],
        )
        assert _has_ngm_store_url(source) is True

    def test_rejects_other_urls(self):
        source = DocumentSource(
            title="Press Release", url=["https://example.com/doc.pdf"]
        )
        assert _has_ngm_store_url(source) is False

    def test_handles_empty_urls(self):
        source = DocumentSource(title="Press Release", url=[])
        assert _has_ngm_store_url(source) is False

    def test_handles_none_url(self):
        source = DocumentSource(title="Press Release", url=None)
        assert _has_ngm_store_url(source) is False


class TestIsDirectDocumentUrl:
    def test_detects_pdf(self):
        assert _is_direct_document_url("https://example.com/doc.pdf") is True

    def test_detects_docx(self):
        assert _is_direct_document_url("https://example.com/doc.docx") is True

    def test_rejects_html(self):
        assert _is_direct_document_url("https://example.com/page") is False

    def test_rejects_empty(self):
        assert _is_direct_document_url("") is False


class TestSourceUrlPriority:
    def test_prioritizes_ngm_store(self):
        assert _source_url_priority(
            "https://ngm-store.jawafdehi.org/doc.pdf"
        ) > _source_url_priority("https://other.com/doc.pdf")

    def test_prioritizes_pdf_over_docx(self):
        a = _source_url_priority("https://example.com/doc.pdf")
        b = _source_url_priority("https://example.com/doc.docx")
        assert a >= b

    def test_prioritizes_direct_doc_over_webpage(self):
        a = _source_url_priority("https://example.com/doc.pdf")
        b = _source_url_priority("https://example.com/page.html")
        assert a > b


class TestCopyStreamToPathWithLimit:
    def test_copies_within_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "test.bin"

            class InFile:
                def __init__(self, data):
                    self._data = iter(data)

                def read(self, size=-1):
                    return next(self._data, b"")

            in_file = InFile([b"hello", b"world"])
            _copy_stream_to_path_with_limit(in_file, out_path)
            assert out_path.exists()
            assert out_path.read_bytes() == b"helloworld"

    def test_cleans_partial_on_overflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "test.bin"

            class InFile:
                _called = 0

                def read(self, size=-1):
                    self._called += 1
                    return b"x" * 16 if self._called <= 1 else b""

            with patch(
                "cases.management.commands.enrich_case_overview.MAX_DOWNLOAD_BYTES",
                8,
                create=True,
            ):
                with pytest.raises(CommandError, match="exceeds max size"):
                    _copy_stream_to_path_with_limit(InFile(), out_path)

            assert not out_path.exists()


# ═══════════════════════════════════════════════════════════════════════════════
# Group J: Court Case Discovery
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.django_db
class TestDiscoverCourtCases:
    def test_returns_empty_when_no_court_cases_field(self):
        cmd = Command()
        case = _make_case("case-no-court", "No court cases", court_cases=None)
        result = cmd._discover_court_cases(case, {})
        assert result["court_cases_found"] == []
        assert result["court_order_texts"] == []

    def test_returns_empty_when_court_cases_empty_list(self):
        cmd = Command()
        case = _make_case("case-empty-court", "Empty court cases", court_cases=[])
        result = cmd._discover_court_cases(case, {})
        assert result["court_cases_found"] == []
        assert result["court_order_texts"] == []

    def test_parses_court_cases_field(self):
        cmd = Command()
        case = _make_case(
            "case-with-court",
            "Has court cases",
            court_cases=["special:080-CR-0007"],
        )
        with patch(
            "ngm.services.get_court_case_details",
            return_value=None,
        ):
            result = cmd._discover_court_cases(case, {})
        assert result["court_cases_found"] == []
        assert result["court_order_texts"] == []

    def test_finds_ngm_record_for_known_case(self):
        cmd = Command()
        case = _make_case(
            "case-found",
            "Found in NGM",
            court_cases=["special:080-CR-0007"],
        )
        mock_details = {
            "case": {
                "registration_date_bs": "2080-01-15",
                "case_type": "corruption",
                "case_status": "disposed",
                "verdict_date_bs": "2081-03-20",
                "plaintiff": "CIAA",
                "defendant": "Test Defendant",
            },
            "hearings": [],
            "entities": [],
        }
        with patch(
            "ngm.services.get_court_case_details",
            return_value=mock_details,
        ):
            result = cmd._discover_court_cases(case, {})
        assert len(result["court_cases_found"]) == 1
        assert result["court_cases_found"][0]["court_identifier"] == "special"
        assert result["court_cases_found"][0]["case_number"] == "080-CR-0007"
        assert result["court_cases_found"][0]["plaintiff"] == "CIAA"

    def test_handles_ngm_import_error_gracefully(self):
        """When ngm.services cannot be imported, discovery returns empty gracefully."""
        cmd = Command()
        case = _make_case(
            "case-no-ngm",
            "NGM unavailable",
            court_cases=["special:080-CR-0007"],
        )

        with patch.dict("sys.modules", {"ngm.services": None}):
            result = cmd._discover_court_cases(case, {})
        # If ngm was already cached, we may still get results; the key is no crash
        assert isinstance(result, dict)
        assert "court_cases_found" in result
        assert "court_order_texts" in result

    def test_skips_unparseable_case_numbers(self):
        """Unparseable case numbers are skipped without calling NGM."""
        cmd = Command()
        case = _make_case(
            "case-bad-format",
            "Bad format",
            court_cases=["special:not-a-valid-case-number"],
        )
        with patch(
            "ngm.services.get_court_case_details",
        ) as mock_get:
            result = cmd._discover_court_cases(case, {})
        # normalize_case_number raises ValueError → entry skipped, no NGM query
        mock_get.assert_not_called()
        assert result["court_cases_found"] == []

    def test_converts_matching_source_when_found(self):
        cmd = Command()
        # Create a LEGAL_COURT_ORDER source with the case number in its title.
        # The DB query in Phase 2C will find it via title__icontains.
        _make_source(
            "src-court-001",
            "Verdict for 080-CR-0007",
            SourceType.LEGAL_COURT_ORDER,
        )

        case = _make_case(
            "case-with-source",
            "Has matching source",
            court_cases=["special:080-CR-0007"],
        )
        mock_details = {
            "case": {
                "registration_date_bs": "2080-01-15",
                "case_type": "corruption",
                "case_status": "disposed",
                "verdict_date_bs": None,
                "plaintiff": "CIAA",
                "defendant": "Test",
            },
            "hearings": [],
            "entities": [],
        }
        with patch(
            "ngm.services.get_court_case_details",
            return_value=mock_details,
        ):
            with patch.object(
                cmd, "_convert_one_source", return_value="Mocked court order text"
            ):
                result = cmd._discover_court_cases(case, {})
        assert len(result["court_cases_found"]) == 1
        assert len(result["court_order_texts"]) == 1
        assert result["court_order_texts"][0] == "Mocked court order text"

    def test_extracts_case_numbers_from_extracted_json(self):
        cmd = Command()
        case = _make_case("case-from-json", "From JSON metadata")
        extracted = {
            "case_metadata": {
                "case_number": "081-CR-0081",
            },
        }
        with patch(
            "ngm.services.get_court_case_details",
            return_value=None,
        ) as mock_get:
            cmd._discover_court_cases(case, extracted)
        # Should attempt lookups for the extracted case number across multiple court identifiers
        assert mock_get.call_count >= 1
        # First call should be with the extracted number
        call_args = [c[0] for c in mock_get.call_args_list]
        assert any(args[1] == "081-CR-0081" for args in call_args)
