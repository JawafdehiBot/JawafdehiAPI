"""
Tests for enrich_ciaa_news_articles management command and NewsEnricher service.

Phase 2d of CIAA FY 080/081 Case Enrichment pipeline.
Covers: dry-run safety, case filtering, priority list, article acceptance/rejection,
duplicate prevention, image storage, error handling, and idempotency.
"""

import json
import uuid
from datetime import date
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest
from django.core.management import call_command

from cases.management.commands.enrich_ciaa_news_articles import Command
from cases.models import Case, CaseState, CaseType, DocumentSource, SourceType
from cases.services.news_enricher import (
    NewsEnricher,
    _extract_images_from_html,
    _extract_text_from_html,
    _extract_title_from_html,
    _generate_query_variations,
    _guess_outlet,
    _parse_llm_json,
)


@pytest.mark.django_db
class TestNewsEnricherService:
    """Test the NewsEnricher service class."""

    def _create_case(self, **overrides):
        defaults = {
            "case_type": CaseType.CORRUPTION,
            "state": CaseState.DRAFT,
            "title": "CIAA Special Court Case 080-CR-0007 Test Case",
            "case_id": "case-test-001",
            "court_cases": ["special:080-CR-0007"],
            "key_allegations": [
                "Test allegation about bribery involving government official.",
            ],
            "evidence": [],
        }
        defaults.update(overrides)
        return Case.objects.create(**defaults)

    def _mock_llm_response(self, relevant=True, confidence="high", reason="Match"):
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "relevant": relevant,
                                "confidence": confidence,
                                "reason": reason,
                            }
                        )
                    }
                }
            ]
        }
        return mock_response

    def _get_sample_html(self, title="Test Article", body=None):
        if body is None:
            body = "This is a test article body with enough text. " * 5
        return f"""<!DOCTYPE html>
<html>
<head><title>{title}</title>
<meta property="article:published_time" content="2025-01-15T12:00:00Z">
<meta name="date" content="2025-01-15">
</head>
<body>
<article>
<h1>{title}</h1>
<p>{body}</p>
<img src="https://example.com/image1.jpg" alt="Test image">
<img src="https://example.com/image2.jpg" alt="Second image">
</article>
</body>
</html>"""

    def _mock_search_results(self, prefix=""):
        if not prefix:
            prefix = f"https://example-{uuid.uuid4().hex[:6]}.com"
        elif not prefix.startswith("https://"):
            prefix = f"https://example-{prefix}.com"
        return [
            {
                "title": "Test News Article",
                "url": f"{prefix}/news/article1",
                "snippet": "A test news article about corruption.",
            },
            {
                "title": "Unrelated Article",
                "url": f"{prefix}/news/other",
                "snippet": "Something completely different.",
            },
        ]

    def _mock_fetch(self, html_content=None):
        if html_content is None:
            html_content = self._get_sample_html()

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.text = html_content
        mock_response.headers = {"content-type": "text/html; charset=utf-8"}
        return mock_response

    def _create_enricher(self, max_articles_per_case=3):
        return NewsEnricher(
            llm_api_key="test-key",
            llm_base_url="https://test-llm.example.com/v1",
            max_articles_per_case=max_articles_per_case,
        )

    _FETCH_UNSET = object()

    def _mock_setup(
        self,
        search_results=None,
        fetch_html=_FETCH_UNSET,
        llm_relevant=True,
        confidence="high",
        reason="Match",
    ):
        if search_results is None:
            search_results = self._mock_search_results()
        if fetch_html is self._FETCH_UNSET:
            fetch_html = self._get_sample_html()

        p1 = patch(
            "cases.services.news_enricher._search_duckduckgo",
            return_value=search_results,
        )
        p2 = patch(
            "cases.services.news_enricher._fetch_article_content",
            return_value=fetch_html,
        )
        p3 = patch("requests.post")
        p3.return_value = self._mock_llm_response(
            relevant=llm_relevant,
            confidence=confidence,
            reason=reason,
        )
        return p1, p2, p3

    def test_generate_query_variations_with_case_number(self):
        case = self._create_case(
            court_cases=["special:080-CR-0007"],
            key_allegations=["घुस रिश्वत सम्बन्धी भ्रष्टाचार मुद्दा"],
        )
        queries = _generate_query_variations(case)
        assert len(queries) > 0
        assert any("080-CR-0007" in q for q in queries)

    def test_generate_query_variations_no_court_cases(self):
        case = self._create_case(
            court_cases=[],
            title="Some case without court data",
        )
        queries = _generate_query_variations(case)
        assert isinstance(queries, list)

    def test_extract_text_from_html(self):
        html = self._get_sample_html(body="This is article content. " * 20)
        text = _extract_text_from_html(html)
        assert len(text) > 100
        assert "Test Article" in text

    def test_extract_title_from_html(self):
        html = self._get_sample_html(title="Nepal Corruption Case Verdict")
        title = _extract_title_from_html(html)
        assert title == "Nepal Corruption Case Verdict"

    def test_extract_images_from_html(self):
        html = self._get_sample_html()
        images = _extract_images_from_html(html)
        assert len(images) == 2
        assert images[0]["url"] == "https://example.com/image1.jpg"
        assert images[1]["url"] == "https://example.com/image2.jpg"

    def test_extract_images_with_base_url(self):
        html = '<img src="/images/photo.jpg" alt="Photo">'
        images = _extract_images_from_html(html, base_url="https://example.com/news/")
        assert len(images) >= 1
        assert any("example.com" in img["url"] for img in images)

    def test_guess_outlet(self):
        assert "Ekantipur" == _guess_outlet("https://ekantipur.com/news/article")
        assert "Onlinekhabar" == _guess_outlet("https://www.onlinekhabar.com/content")
        assert "Example" == _guess_outlet("https://example.com/story")

    def test_parse_llm_json_relevant(self):
        result = _parse_llm_json(
            '{"relevant": true, "confidence": "high", "reason": "Matches case."}'
        )
        assert result["relevant"] is True
        assert result["confidence"] == "high"

    def test_parse_llm_json_not_relevant(self):
        result = _parse_llm_json('{"relevant": false, "reason": "Different case."}')
        assert result["relevant"] is False

    def test_enrich_case_accepts_relevant_article(self):
        case = self._create_case()
        enricher = self._create_enricher()
        search_results = self._mock_search_results(prefix="accept-test")
        fetch_html = self._get_sample_html(body="Test article " * 50)

        p1, p2, p3 = self._mock_setup(
            search_results=search_results,
            fetch_html=fetch_html,
            llm_relevant=True,
            reason="Case number matches.",
        )
        with p1, p2, p3:
            stats = enricher.enrich_case(case, dry_run=False, case_num=1, total_cases=1)

            assert stats["accepted"] >= 1
            assert stats["new_sources"] >= 1

            sources = list(DocumentSource.objects.all())
            assert len(sources) >= 1
            all_urls = set()
            for s in sources:
                all_urls.update(s.url)
            assert "https://example-accept-test.com/news/article1" in all_urls
            case.refresh_from_db()
            assert len(case.evidence) >= 1

    def test_enrich_case_rejects_unrelated_article(self):
        case = self._create_case()
        enricher = self._create_enricher()

        p1, p2, p3 = self._mock_setup(
            llm_relevant=False, confidence="low", reason="Unrelated case."
        )
        with p1, p2, p3:
            stats = enricher.enrich_case(case, dry_run=False, case_num=1, total_cases=1)

            assert stats["accepted"] == 0
            assert stats["rejected"] >= 1
            assert DocumentSource.objects.count() == 0

    def test_dry_run_does_not_save(self):
        case = self._create_case()
        enricher = self._create_enricher()

        p1, p2, p3 = self._mock_setup()
        with p1, p2, p3:
            stats = enricher.enrich_case(case, dry_run=True, case_num=1, total_cases=1)

            assert stats["accepted"] >= 1
            assert stats["new_sources"] == 0
            assert DocumentSource.objects.count() == 0
            case.refresh_from_db()
            assert case.evidence == []

    def test_duplicate_article_not_saved_twice(self):
        case = self._create_case()
        enricher = self._create_enricher()
        search_results = self._mock_search_results(prefix="dup-test")
        fetch_html = self._get_sample_html(body="Test content. " * 50)

        p1, p2, p3 = self._mock_setup(
            search_results=search_results, fetch_html=fetch_html
        )
        with p1, p2, p3:
            enricher.enrich_case(case, dry_run=False, case_num=1, total_cases=1)
            first_source_count = DocumentSource.objects.count()
            enricher.enrich_case(case, dry_run=False, case_num=1, total_cases=1)
            second_source_count = DocumentSource.objects.count()
            assert first_source_count == second_source_count

    def test_duplicate_evidence_not_created(self):
        case = self._create_case()
        enricher = NewsEnricher(
            llm_api_key="test-key",
            llm_base_url="https://test-llm.example.com/v1",
            max_articles_per_case=3,
        )

        search_results = self._mock_search_results(prefix="evdup-test")
        with patch(
            "cases.services.news_enricher._search_duckduckgo",
            return_value=search_results,
        ), patch(
            "cases.services.news_enricher._fetch_article_content",
            return_value=self._get_sample_html(body="Evidence dedup test. " * 50),
        ), patch(
            "requests.post",
        ) as mock_post:
            mock_post.return_value = self._mock_llm_response()

            enricher.enrich_case(case, dry_run=False, case_num=1, total_cases=1)

            case.refresh_from_db()
            first_evidence_count = len(case.evidence)

            enricher.enrich_case(case, dry_run=False, case_num=1, total_cases=1)

            case.refresh_from_db()
            second_evidence_count = len(case.evidence)
            assert first_evidence_count == second_evidence_count

    def test_images_stored_in_description(self):
        case = self._create_case()
        enricher = self._create_enricher()
        search_results = self._mock_search_results(prefix="img-test")
        html = self._get_sample_html(body="Test article with images. " * 50)

        p1, p2, p3 = self._mock_setup(search_results=search_results, fetch_html=html)
        with p1, p2, p3:
            enricher.enrich_case(case, dry_run=False, case_num=1, total_cases=1)

            source = DocumentSource.objects.first()
            assert source is not None
            assert "Image:" in source.description
            assert "https://example.com/image1.jpg" in source.description
            assert "https://example.com/image2.jpg" in source.description

    def test_publication_date_stored(self):
        case = self._create_case()
        enricher = self._create_enricher()

        p1, p2, p3 = self._mock_setup()
        with p1, p2, p3:
            enricher.enrich_case(case, dry_run=False, case_num=1, total_cases=1)

            source = DocumentSource.objects.first()
            assert source is not None
            assert source.publication_date is not None

    def test_missing_publication_date_handled(self):
        case = self._create_case()
        enricher = self._create_enricher()

        html_without_date = (
            "<html><head><title>No Date Article</title></head><body><p>"
            + "Article body content. " * 20
            + "</p></body></html>"
        )
        search_results = [
            {
                "title": "No Date",
                "url": "https://example-nodate.com/nodate",
                "snippet": "...",
            }
        ]
        p1, p2, p3 = self._mock_setup(
            search_results=search_results, fetch_html=html_without_date
        )
        with p1, p2, p3:
            enricher.enrich_case(case, dry_run=False, case_num=1, total_cases=1)

            source = DocumentSource.objects.first()
            assert source is not None
            assert source.publication_date is not None

    def test_search_error_counted_not_fatal(self):
        case = self._create_case()
        enricher = self._create_enricher()

        with patch(
            "cases.services.news_enricher._search_duckduckgo",
            side_effect=Exception("Search failed"),
        ), patch(
            "requests.post",
        ) as mock_post:
            mock_post.return_value = self._mock_llm_response()

            stats = enricher.enrich_case(case, dry_run=False, case_num=1, total_cases=1)
            assert stats["errors"] >= 1

    def test_fetch_error_does_not_stop_batch(self):
        case = self._create_case()
        enricher = self._create_enricher()

        p1, p2, p3 = self._mock_setup(fetch_html=None)
        with p1, p2, p3:
            stats = enricher.enrich_case(case, dry_run=False, case_num=1, total_cases=1)
            assert stats["status"] in ("processed", "no_articles")

    def test_llm_error_counted_not_fatal(self):
        case = self._create_case()
        enricher = self._create_enricher()

        p1, p2, _ = self._mock_setup()
        with p1, p2, patch("requests.post", side_effect=Exception("LLM API error")):
            stats = enricher.enrich_case(case, dry_run=False, case_num=1, total_cases=1)
            assert stats["accepted"] == 0

    def test_already_linked_url_not_fetched_again(self):
        case = self._create_case()
        DocumentSource.objects.create(
            title="Existing Article",
            source_type=SourceType.MEDIA_NEWS,
            url=["https://example-already.com/news/article1"],
            publication_date=date(2025, 1, 15),
        )

        enricher = self._create_enricher()
        search_results = [
            {
                "title": "Article1",
                "url": "https://example-already.com/news/article1",
                "snippet": "...",
            },
            {
                "title": "Article2",
                "url": "https://example-already.com/news/other",
                "snippet": "...",
            },
        ]

        with patch(
            "cases.services.news_enricher._search_duckduckgo",
            return_value=search_results,
        ), patch(
            "cases.services.news_enricher._fetch_article_content",
        ) as mock_fetch:
            mock_fetch.return_value = self._get_sample_html()

            with patch("requests.post") as mock_post:
                mock_post.return_value = self._mock_llm_response()
                stats = enricher.enrich_case(
                    case, dry_run=False, case_num=1, total_cases=1
                )

            assert stats["already_linked"] >= 1
            assert mock_fetch.call_count <= 1

    def test_source_description_contains_required_fields(self):
        case = self._create_case()
        enricher = self._create_enricher()

        p1, p2, p3 = self._mock_setup()
        with p1, p2, p3:
            enricher.enrich_case(case, dry_run=False, case_num=1, total_cases=1)

            source = DocumentSource.objects.first()
            assert source is not None
            desc = source.description
            assert len(desc) > 0
            assert "https://example" in desc
            assert "Image:" in desc if source.description else True

    def test_evidence_description_present(self):
        case = self._create_case()
        enricher = self._create_enricher()

        p1, p2, p3 = self._mock_setup()
        with p1, p2, p3:
            enricher.enrich_case(case, dry_run=False, case_num=1, total_cases=1)

            case.refresh_from_db()
            assert len(case.evidence) >= 1
            entry = case.evidence[0]
            assert entry["source_id"]
            assert entry["description"]

    def test_max_articles_per_case_enforced(self):
        case = self._create_case()
        enricher = self._create_enricher(max_articles_per_case=2)

        many_results = [
            {
                "title": f"Article {i}",
                "url": f"https://example.com/{i}",
                "snippet": "...",
            }
            for i in range(5)
        ]
        p1, p2, p3 = self._mock_setup(search_results=many_results)
        with p1, p2, p3:
            stats = enricher.enrich_case(case, dry_run=False, case_num=1, total_cases=1)
            assert stats["accepted"] <= 2


@pytest.mark.django_db
class TestEnrichCiaaNewsArticlesCommand:
    """Test the management command."""

    def _create_case(self, **overrides):
        defaults = {
            "case_type": CaseType.CORRUPTION,
            "state": CaseState.DRAFT,
            "title": "CIAA Test Case 080-CR-0007",
            "case_id": "case-test-001",
            "court_cases": ["special:080-CR-0007"],
            "key_allegations": ["Test allegation."],
            "evidence": [],
        }
        defaults.update(overrides)
        return Case.objects.create(**defaults)

    def _get_minimal_search_results(self):
        return [
            {"title": "News", "url": "https://example.com/news", "snippet": "test"},
        ]

    def test_cli_flags_registered(self):
        parser = MagicMock()
        cmd = Command()
        cmd.add_arguments(parser)

        expected_flags = [
            "--dry-run",
            "--case-id",
            "--priority",
            "--all",
            "--force",
            "--limit",
            "--max-articles",
            "--verbose",
        ]
        calls = [str(call) for call in parser.add_argument.call_args_list]
        for flag in expected_flags:
            assert any(flag in c for c in calls), f"Flag {flag} not registered"

    def test_dry_run_no_db_changes(self):
        case = self._create_case()

        with patch(
            "cases.services.news_enricher._search_duckduckgo",
            return_value=self._get_minimal_search_results(),
        ), patch(
            "cases.services.news_enricher._fetch_article_content",
            return_value="<html><head><title>T</title></head><body><p>"
            + "Test. " * 50
            + "</p></body></html>",
        ), patch(
            "requests.post",
        ) as mock_post:
            mock_llm = MagicMock()
            mock_llm.raise_for_status.return_value = None
            mock_llm.json.return_value = {
                "choices": [
                    {
                        "message": {
                            "content": '{"relevant": true, "confidence": "high", "reason": "Match"}'
                        }
                    }
                ]
            }
            mock_post.return_value = mock_llm

            out = StringIO()
            call_command(
                "enrich_ciaa_news_articles",
                "--dry-run",
                f"--case-id={case.case_id}",
                "--llm-api-key=test-key",
                "--llm-base-url=https://test.example.com/v1",
                stdout=out,
            )

        assert DocumentSource.objects.count() == 0
        case.refresh_from_db()
        assert case.evidence == []
        output = out.getvalue()
        assert "DRY-RUN" in output or "DRY RUN" in output

    def test_case_id_limits_to_single_case(self):
        case_a = self._create_case(case_id="case-a")
        self._create_case(case_id="case-b")

        with patch(
            "cases.services.news_enricher._search_duckduckgo",
            return_value=self._get_minimal_search_results(),
        ), patch(
            "cases.services.news_enricher._fetch_article_content",
            return_value="<html><head><title>T</title></head><body><p>"
            + "Test. " * 50
            + "</p></body></html>",
        ), patch(
            "requests.post",
        ) as mock_post:
            mock_llm = MagicMock()
            mock_llm.raise_for_status.return_value = None
            mock_llm.json.return_value = {
                "choices": [
                    {
                        "message": {
                            "content": '{"relevant": false, "reason": "No match"}'
                        }
                    }
                ]
            }
            mock_post.return_value = mock_llm

            out = StringIO()
            call_command(
                "enrich_ciaa_news_articles",
                "--dry-run",
                f"--case-id={case_a.case_id}",
                "--llm-api-key=test-key",
                "--llm-base-url=https://test.example.com/v1",
                stdout=out,
            )

        output = out.getvalue()
        assert "Total cases:     1" in output

    def test_priority_mutually_exclusive_with_case_id(self):
        out = StringIO()
        err = StringIO()
        call_command(
            "enrich_ciaa_news_articles",
            "--priority",
            "--case-id=test-001",
            stdout=out,
            stderr=err,
        )
        assert "mutually exclusive" in err.getvalue()

    def test_no_cases_prints_message(self, caplog):
        out = StringIO()
        call_command(
            "enrich_ciaa_news_articles",
            "--case-id=nonexistent",
            "--llm-api-key=test-key",
            stdout=out,
        )
        assert "No cases" in out.getvalue()

    def test_no_api_key_shows_error(self, caplog):
        out = StringIO()
        err = StringIO()
        with patch.dict("os.environ", {}, clear=True):
            call_command(
                "enrich_ciaa_news_articles",
                "--case-id=test-001",
                stdout=out,
                stderr=err,
            )
        assert "No LLM API key" in err.getvalue()

    def test_limit_enforced(self):
        for i in range(3):
            self._create_case(
                case_id=f"test-limit-{i}",
                court_cases=["special:1"],
            )

        with patch(
            "cases.services.news_enricher._search_duckduckgo",
            return_value=[
                {"title": "N", "url": "https://example.com/n", "snippet": "x"}
            ],
        ), patch(
            "cases.services.news_enricher._fetch_article_content",
            return_value="<html><head><title>T</title></head><body><p>"
            + "Test. " * 50
            + "</p></body></html>",
        ), patch(
            "requests.post",
        ) as mock_post:
            mock_llm = MagicMock()
            mock_llm.raise_for_status.return_value = None
            mock_llm.json.return_value = {
                "choices": [
                    {"message": {"content": '{"relevant": false, "reason": "No"}'}}
                ]
            }
            mock_post.return_value = mock_llm

            out = StringIO()
            call_command(
                "enrich_ciaa_news_articles",
                "--limit=1",
                "--dry-run",
                "--llm-api-key=test-key",
                "--llm-base-url=https://test.example.com/v1",
                stdout=out,
            )

        output = out.getvalue()
        assert "Total cases:     1" in output

    def test_summary_printed(self):
        case = self._create_case()

        with patch(
            "cases.services.news_enricher._search_duckduckgo",
            return_value=[],
        ):
            out = StringIO()
            call_command(
                "enrich_ciaa_news_articles",
                f"--case-id={case.case_id}",
                "--dry-run",
                "--llm-api-key=test-key",
                "--llm-base-url=https://test.example.com/v1",
                stdout=out,
            )

        output = out.getvalue()
        assert "news article enrichment" in output.lower()

    def test_priority_flag_loads_priority_cases(self, caplog):
        self._create_case(
            case_id="priority-test",
            court_cases=["special:080-CR-0007"],
        )
        self._create_case(
            case_id="non-priority-test",
            court_cases=["special:999-CR-9999"],
        )

        with patch(
            "cases.services.news_enricher._search_duckduckgo",
            return_value=[],
        ):
            out = StringIO()
            call_command(
                "enrich_ciaa_news_articles",
                "--priority",
                "--dry-run",
                "--llm-api-key=test-key",
                "--llm-base-url=https://test.example.com/v1",
                stdout=out,
            )

        assert "Priority mode" in caplog.text

    def test_verbose_enables_debug(self, caplog):
        case = self._create_case()

        with patch(
            "cases.services.news_enricher._search_duckduckgo",
            return_value=[],
        ):
            out = StringIO()
            call_command(
                "enrich_ciaa_news_articles",
                f"--case-id={case.case_id}",
                "--verbose",
                "--dry-run",
                "--llm-api-key=test-key",
                "--llm-base-url=https://test.example.com/v1",
                stdout=out,
            )

        assert "DRY-RUN" in caplog.text or "DRY RUN" in caplog.text
