from cases.models import (
    Case,
    SourceType,
)
from cases.services.ciaa_draft_case_service import CIAADraftCaseService


class TestCreateDocumentSources:
    def test_abhiyog_patras_created(self, db):
        service = CIAADraftCaseService()
        case = Case.objects.create(
            case_type="CORRUPTION",
            state="DRAFT",
            title="Test Case",
            court_cases=["special:081-CR-0999"],
        )
        ciaa_json = {
            "case_no": "081-CR-0999",
            "case_title": "Test",
            "ciaa": {
                "press_releases": [],
                "abhiyogPatras": [
                    {
                        "case_number": "081-CR-0999",
                        "title": "AG Charge Sheet Test",
                        "filing_date": "2081-10-15",
                        "pdf_url": "https://ag.gov.np/storage/test.pdf",
                        "court_office": "Special Court",
                    }
                ],
            },
            "court_case": {"faisala_link": []},
        }

        sources = service.create_document_sources(ciaa_json, case)

        assert len(sources) == 1
        source = sources[0]
        assert source.source_type == SourceType.OFFICIAL_GOVERNMENT
        assert "https://ag.gov.np/storage/test.pdf" in source.url_links
        assert "AG Charge Sheet Test" in source.title
        assert source.publication_date is not None

    def test_press_release_date_populated(self, db):
        service = CIAADraftCaseService()
        case = Case.objects.create(
            case_type="CORRUPTION",
            state="DRAFT",
            title="Test Case",
            court_cases=["special:081-CR-0999"],
        )
        ciaa_json = {
            "case_no": "081-CR-0999",
            "case_title": "Test",
            "ciaa": {
                "press_releases": [
                    {
                        "release_id": "3345",
                        "title": "CIAA Press Release",
                        "url": "https://ciaa.gov.np/pressrelease/3345",
                        "date": "2081-12-15",
                    }
                ],
                "abhiyogPatras": [],
            },
            "court_case": {"faisala_link": []},
        }

        sources = service.create_document_sources(ciaa_json, case)

        assert len(sources) == 1
        source = sources[0]
        assert source.publication_date is not None
        assert source.source_type == SourceType.LEGAL_PROCEDURAL

    def test_empty_json_no_sources(self, db):
        service = CIAADraftCaseService()
        case = Case.objects.create(
            case_type="CORRUPTION",
            state="DRAFT",
            title="Empty Case",
            court_cases=["special:081-CR-0001"],
        )
        ciaa_json = {
            "case_no": "081-CR-0001",
            "case_title": "Empty",
            "ciaa": {"press_releases": [], "abhiyogPatras": []},
            "court_case": {"faisala_link": []},
        }

        sources = service.create_document_sources(ciaa_json, case)
        assert len(sources) == 0
        assert case.evidence == []

    def test_abhiyog_patras_attached_to_evidence(self, db):
        service = CIAADraftCaseService()
        case = Case.objects.create(
            case_type="CORRUPTION",
            state="DRAFT",
            title="Evidence Test",
            court_cases=["special:082-CR-0002"],
        )
        ciaa_json = {
            "case_no": "082-CR-0002",
            "case_title": "Evidence",
            "ciaa": {
                "press_releases": [],
                "abhiyogPatras": [
                    {
                        "case_number": "082-CR-0002",
                        "title": "AG Sheet",
                        "filing_date": "2082-05-11",
                        "pdf_url": "https://ag.gov.np/test.pdf",
                        "court_office": "Special Court",
                    }
                ],
            },
            "court_case": {"faisala_link": []},
        }

        service.create_document_sources(ciaa_json, case)
        case.refresh_from_db()

        assert len(case.evidence) == 1
        assert case.evidence[0]["description"] == "AG Charge Sheet - 082-CR-0002"
        assert case.evidence[0]["source_id"] is not None
