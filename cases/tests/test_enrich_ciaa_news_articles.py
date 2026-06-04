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
from cases.services.news_enricher import (
    _EVENT_APPEAL,
    _EVENT_FILING,
    _EVENT_HEARING,
    _EVENT_INVESTIGATION,
    _EVENT_QUERY_TEMPLATES,
    _EVENT_VERDICT,
    NewsEnricher,
    _detect_case_events,
    _extract_org_name_from_title,
    _extract_text_from_html,
    _extract_title_from_html,
    _fix_mojibake,
    _generate_query_variations,
    _get_accused_names,
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

    def _mock_llm_response(
        self, relevant=True, confidence="high", reason="Match", summary=""
    ):
        if not summary:
            summary = (
                "A corruption case article summary."
                if relevant
                else "Unrelated article summary."
            )
        inner_json = json.dumps(
            {
                "relevant": relevant,
                "confidence": confidence,
                "reason": reason,
                "summary": summary,
            }
        )
        outer_payload = json.dumps({"choices": [{"message": {"content": inner_json}}]})
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = json.loads(outer_payload)
        mock_response.text = outer_payload
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
        summary="",
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
        p3 = patch(
            "requests.post",
            return_value=self._mock_llm_response(
                relevant=llm_relevant,
                confidence=confidence,
                reason=reason,
                summary=summary,
            ),
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

    def test_generate_query_variations_reserves_event_slots(self):
        case = self._create_case(
            title="लामो शीर्षक भ्रष्टाचार अनियमितता घुस रकम जग्गा खरिद निर्माण ठेक्का",
            court_cases=["special:080-CR-0007"],
            case_start_date=date(2023, 7, 1),
            case_end_date=date(2024, 6, 12),
            key_allegations=["घुस रिश्वत भ्रष्टाचार अनियमितता अकुत सम्पत्ति"],
        )

        queries = _generate_query_variations(case)

        assert len(queries) <= 15
        assert any("मुद्दा दायर" in query for query in queries)
        assert any("फैसला विशेष अदालत" in query for query in queries)

    def test_generate_query_variations_cap_is_15(self):
        case = self._create_case(
            title="लामो शीर्षक भ्रष्टाचार अनियमितता घुस रकम जग्गा खरिद निर्माण ठेक्का",
            court_cases=["special:080-CR-0007"],
            case_start_date=date(2023, 7, 1),
            case_end_date=date(2024, 6, 12),
            key_allegations=["घुस रिश्वत भ्रष्टाचार अनियमितता अकुत सम्पत्ति"],
        )
        queries = _generate_query_variations(case)
        assert len(queries) <= 15

    def test_generate_query_variations_include_english_romanized_queries(self):
        case = self._create_case()
        entity = JawafEntity.objects.create(display_name="राजु पुरी")
        CaseEntityRelationship.objects.create(
            case=case, entity=entity, relationship_type=RelationshipType.ACCUSED
        )

        queries = _generate_query_variations(case)
        english_queries = [q for q in queries if "राजु" not in q and "पुरी" not in q]

        assert len(english_queries) >= 4
        assert any("raju puri CIAA case Nepal" in q for q in queries)
        assert any("raju puri corruption case Nepal" in q for q in queries)

    def test_generate_query_variations_adds_nepal_keyword_to_all_queries(self):
        case = self._create_case(
            court_cases=["special:080-CR-0007"],
            case_start_date=date(2023, 7, 1),
            case_end_date=date(2024, 6, 12),
            key_allegations=["घुस रिश्वत भ्रष्टाचार अनियमितता अकुत सम्पत्ति"],
        )
        queries = _generate_query_variations(case)

        assert queries
        assert all("Nepal" in q or "नेपाल" in q for q in queries)

    def test_extract_org_name_from_title(self):
        title = "नेपाल सरकार विरुद्ध साझा भण्डार सहकारी संस्था लिमिटेड मुद्दा"
        org = _extract_org_name_from_title(title)
        assert org == "साझा भण्डार सहकारी संस्था लिमिटेड"

    def test_extract_org_name_from_title_no_match(self):
        assert _extract_org_name_from_title("") == ""
        assert _extract_org_name_from_title("राम विरुद्ध श्याम मुद्दा") == ""

    def test_get_accused_names_from_entity_relationships(self):
        case = self._create_case()
        entity = JawafEntity.objects.create(display_name="गोपाल पराजुली")
        CaseEntityRelationship.objects.create(
            case=case, entity=entity, relationship_type=RelationshipType.ACCUSED
        )
        names = _get_accused_names(case)
        assert "गोपाल पराजुली" in names

    def test_get_accused_names_primary_first(self):
        case = self._create_case()
        primary = JawafEntity.objects.create(display_name="राजु पुरी")
        CaseEntityRelationship.objects.create(
            case=case, entity=primary, relationship_type=RelationshipType.ACCUSED
        )
        coaccused = JawafEntity.objects.create(display_name="श्रृजना गिरी")
        CaseEntityRelationship.objects.create(
            case=case, entity=coaccused, relationship_type=RelationshipType.ACCUSED
        )
        names = _get_accused_names(case)
        assert names[0] == "राजु पुरी"
        assert names[1] == "श्रृजना गिरी"

    def test_event_cap_constant(self):
        from cases.services.news_enricher import _MAX_ARTICLES_PER_EVENT_TYPE

        assert _MAX_ARTICLES_PER_EVENT_TYPE == 2

    def test_evidence_entry_stores_event_type(self):
        enricher = self._create_enricher()
        entry = enricher._build_evidence_entry(
            "source-1",
            {"url": "https://example.com/news", "event_type": "appeal"},
        )
        assert entry["source_id"] == "source-1"
        assert entry["event_type"] == "appeal"
        assert entry["description"]

    def test_existing_event_type_counts_only_media_news_with_event_type(self):
        case = self._create_case()
        media_filing = DocumentSource.objects.create(
            title="Filing",
            source_type=SourceType.MEDIA_NEWS,
            url=["https://example.com/filing"],
            publication_date=date(2025, 1, 15),
        )
        old_media = DocumentSource.objects.create(
            title="Old",
            source_type=SourceType.MEDIA_NEWS,
            url=["https://example.com/old"],
            publication_date=date(2025, 1, 16),
        )
        other_source = DocumentSource.objects.create(
            title="Other",
            source_type=SourceType.OFFICIAL_GOVERNMENT,
            url=["https://ciaa.gov.np/press"],
            publication_date=date(2025, 1, 17),
        )
        case.evidence = [
            {"source_id": media_filing.source_id, "event_type": "filing"},
            {"source_id": old_media.source_id},
            {"source_id": other_source.source_id, "event_type": "filing"},
        ]
        case.save(update_fields=["evidence"])

        enricher = self._create_enricher()
        counts = enricher._get_existing_event_type_counts(case)

        assert counts == {"filing": 1}

    def test_search_candidates_runs_queries_sequentially_with_delay(self):
        enricher = self._create_enricher()
        enricher.search_delay = 2.0
        stats = {"searched": 0, "errors": 0}
        queries = ["q1", "q2", "q3"]

        with patch(
            "cases.services.news_enricher._search_duckduckgo",
            side_effect=[
                [{"url": "https://example.com/1", "title": "one", "snippet": ""}],
                [{"url": "https://example.com/2", "title": "two", "snippet": ""}],
                [{"url": "https://example.com/1", "title": "dup", "snippet": ""}],
            ],
        ) as search_mock, patch(
            "cases.services.news_enricher.time.sleep"
        ) as sleep_mock:
            results = enricher._search_candidates(queries, stats)

        assert search_mock.call_count == 3
        assert sleep_mock.call_count == 2
        sleep_mock.assert_any_call(2.0)
        assert stats["searched"] == 3
        assert len(results) == 2

    def test_detect_case_events_always_returns_all_event_types(self):
        case = self._create_case(
            case_start_date=None,
            case_end_date=None,
            court_cases=[],
        )
        events = _detect_case_events(case)
        assert _EVENT_INVESTIGATION in events
        assert _EVENT_FILING in events
        assert _EVENT_HEARING in events
        assert _EVENT_VERDICT in events
        assert _EVENT_APPEAL in events
        assert len(events) == 5

    def test_detect_case_events_with_start_date_also_returns_all(self):
        case = self._create_case(
            case_start_date=date(2023, 7, 1),
            case_end_date=None,
            court_cases=[],
        )
        events = _detect_case_events(case)
        assert _EVENT_HEARING in events
        assert _EVENT_VERDICT in events
        assert _EVENT_APPEAL in events

    def test_detect_case_events_with_end_date_also_returns_all(self):
        case = self._create_case(
            case_start_date=date(2023, 7, 1),
            case_end_date=date(2024, 6, 12),
            court_cases=[],
        )
        events = _detect_case_events(case)
        assert _EVENT_VERDICT in events
        assert _EVENT_APPEAL in events

    def test_appeal_query_templates_include_ciaa_specific(self):
        templates = _EVENT_QUERY_TEMPLATES.get(_EVENT_APPEAL, [])
        formatted = set()
        for t in templates:
            formatted.add(t.format(name="गोपाल"))
        assert "गोपाल अख्तियार पुनरावेदन" in formatted
        assert "गोपाल सर्वोच्च अदालत फैसला" in formatted

    def test_extract_text_from_html(self):
        html = self._get_sample_html(body="This is article content. " * 20)
        text = _extract_text_from_html(html)
        assert len(text) > 100
        assert "Test Article" in text

    def test_extract_title_from_html(self):
        html = self._get_sample_html(title="Nepal Corruption Case Verdict")
        title = _extract_title_from_html(html)
        assert title == "Nepal Corruption Case Verdict"

    def test_fix_mojibake_repairs_latin1_utf8_double_encode(self):
        # "अख्तियार दुरुपयोग" as UTF-8 bytes decoded as Latin-1
        correct = "अख्तियार दुरुपयोग"
        mangled = correct.encode("utf-8").decode("latin-1")
        repaired = _fix_mojibake(mangled)
        assert "अख्तियार" in repaired
        assert "दुरुपयोग" in repaired

    def test_fix_mojibake_passes_clean_text(self):
        clean = "अख्तियार दुरुपयोग अनुसन्धान आयोगले"
        assert _fix_mojibake(clean) == clean

    def test_fix_mojibake_passes_ascii(self):
        assert _fix_mojibake("CIAA corruption case") == "CIAA corruption case"

    def test_fix_mojibake_empty_string(self):
        assert _fix_mojibake("") == ""

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
            # new_sources counts would-be-saved articles in dry-run mode
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

    def test_source_description_contains_article_summary(self):
        case = self._create_case()
        enricher = self._create_enricher()
        search_results = self._mock_search_results(prefix="desc-test")
        html = self._get_sample_html(body="Test article for description check. " * 50)

        p1, p2, p3 = self._mock_setup(
            search_results=search_results,
            fetch_html=html,
            reason="Case number matches.",
            summary="CIAA filed a case against the survey office chief at Chabahil for illegal assets.",
        )
        with p1, p2, p3:
            enricher.enrich_case(case, dry_run=False, case_num=1, total_cases=1)

            source = DocumentSource.objects.first()
            assert source is not None
            assert (
                source.description
                == "CIAA filed a case against the survey office chief at Chabahil for illegal assets."
            )

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
        existing = DocumentSource.objects.create(
            title="Existing Article",
            source_type=SourceType.MEDIA_NEWS,
            url=["https://example-already.com/news/article1"],
            publication_date=date(2025, 1, 15),
        )
        # Link the source into the case's evidence so the code sees it as already-linked
        case.evidence = [
            {"source_id": existing.source_id, "description": "News article"}
        ]
        case.save(update_fields=["evidence"])

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
            assert len(source.description) > 0

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

    def test_retry_on_zero_articles_first_retry_succeeds(self):
        """When initial search yields 0 accepted, retry with fallback queries finds articles."""
        case = self._create_case()
        enricher = self._create_enricher(max_articles_per_case=3)

        fallback_results = [
            {
                "title": "Fallback Article",
                "url": "https://example-fallback.com/news/1",
                "snippet": "...",
            },
        ]

        # _search_candidates returns empty on first call, results on subsequent calls
        _ = enricher._search_candidates
        call_count = [0]

        def _patched_search_candidates(queries, stats_dict):
            call_count[0] += 1
            if call_count[0] == 1:
                return []  # initial search: empty
            # retry attempts: return fallback results
            return fallback_results

        enricher._search_candidates = _patched_search_candidates

        with patch(
            "cases.services.news_enricher._fetch_article_content",
            return_value=self._get_sample_html(body="Fallback article content. " * 50),
        ), patch("requests.post") as mock_post:
            mock_post.return_value = self._mock_llm_response(
                relevant=True,
                reason="Match via fallback.",
                summary="A fallback-matched article summary.",
            )

            stats = enricher.enrich_case(case, dry_run=False, case_num=1, total_cases=1)

            assert stats["accepted"] >= 1
            assert stats["new_sources"] >= 1
            case.refresh_from_db()
            assert len(case.evidence) >= 1

    def test_retry_on_zero_articles_all_retries_exhausted(self):
        """When all retries return 0 articles, case is marked as no_articles."""
        case = self._create_case()
        enricher = self._create_enricher(max_articles_per_case=3)

        with patch(
            "cases.services.news_enricher._search_duckduckgo",
            return_value=[],  # all searches return empty
        ):
            stats = enricher.enrich_case(case, dry_run=False, case_num=1, total_cases=1)

            assert stats["status"] == "no_articles"
            assert stats["accepted"] == 0
            assert stats["new_sources"] == 0

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

    def test_priority_flag_loads_priority_cases(self):
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

        assert "Priority mode" in out.getvalue()

    def test_verbose_enables_debug(self):
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

        output = out.getvalue()
        assert "DRY-RUN" in output or "DRY RUN" in output
