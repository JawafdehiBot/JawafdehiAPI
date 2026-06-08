"""Tests for map_press_release_files management command."""

from io import StringIO
from unittest.mock import MagicMock, patch

import pytest
from django.core.management import call_command

from cases.models import Case, CaseState, CaseType, DocumentSource, SourceType


@pytest.mark.django_db
class TestMapPressReleaseFiles:
    """Test suite for map_press_release_files command."""

    @pytest.fixture
    def mock_press_release_index(self):
        """Mock NGM press release index response."""
        return {
            "name": "ciaa-press-releases",
            "path": "/ciaa-press-releases",
            "manuscripts": [
                {
                    "url": "https://ngm-store.jawafdehi.org/uploads/ciaa/press-releases/files/3173. test - 1.pdf",
                    "file_name": "3173. test - 1.pdf",
                    "metadata": {
                        "press_id": 3173,
                        "title": "Test Press Release",
                        "publication_date": "2082-04-19",
                        "full_text": "Test content",
                        "source_url": "https://ciaa.gov.np/pressrelease/3173",
                        "file_names": [
                            "3173. test - 1.pdf",
                            "3173. test - 2.docx",
                        ],
                    },
                },
                {
                    "url": "https://ngm-store.jawafdehi.org/uploads/ciaa/press-releases/files/3173. test - 2.docx",
                    "file_name": "3173. test - 2.docx",
                    "metadata": {
                        "press_id": 3173,
                        "title": "Test Press Release",
                        "publication_date": "2082-04-19",
                        "full_text": "Test content",
                        "source_url": "https://ciaa.gov.np/pressrelease/3173",
                        "file_names": [
                            "3173. test - 1.pdf",
                            "3173. test - 2.docx",
                        ],
                    },
                },
            ],
        }

    @pytest.fixture
    def mock_root_index(self):
        """Mock NGM root index response."""
        return {
            "children": [
                {
                    "name": "ciaa-press-releases",
                    "$ref": "https://ngm-store.jawafdehi.org/indices/2026-05-04/index.ciaa-press-releases.json",
                }
            ]
        }

    @pytest.fixture
    def case_with_press_release_evidence(self):
        """Create a case with press release evidence."""
        # Create press release source
        pr_source = DocumentSource.objects.create(
            title="CIAA Press Release",
            url=["https://ciaa.gov.np/pressrelease/3173"],
            source_type=SourceType.LEGAL_PROCEDURAL,
        )

        # Create case with evidence pointing to press release
        case = Case.objects.create(
            case_type=CaseType.CORRUPTION,
            state=CaseState.DRAFT,
            title="Test Case",
            evidence=[
                {
                    "source_id": pr_source.source_id,
                    "description": "CIAA Press Release (ID: 3173)",
                }
            ],
        )

        return case, pr_source

    def test_dry_run_mode(
        self,
        mock_root_index,
        mock_press_release_index,
        case_with_press_release_evidence,
    ):
        """Test that dry-run mode doesn't modify database."""
        case, pr_source = case_with_press_release_evidence
        original_evidence = case.evidence.copy()

        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            # First call returns root index, second call returns press release index
            mock_response.json.side_effect = [mock_root_index, mock_press_release_index]
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            out = StringIO()
            call_command("map_press_release_files", "--dry-run", stdout=out)

            # Refresh case from database
            case.refresh_from_db()

            # Evidence should not be changed in dry-run mode
            assert case.evidence == original_evidence

            # Output should indicate dry-run
            output = out.getvalue()
            assert "[DRY RUN]" in output

    def test_map_press_release_evidence(
        self,
        mock_root_index,
        mock_press_release_index,
        case_with_press_release_evidence,
    ):
        """Test that command maps press release evidence to actual files."""
        case, pr_source = case_with_press_release_evidence

        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            # First call returns root index, second call returns press release index
            mock_response.json.side_effect = [mock_root_index, mock_press_release_index]
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            out = StringIO()
            call_command("map_press_release_files", stdout=out)

            # Refresh case from database
            case.refresh_from_db()

            # Evidence should be updated with ONE source per press release (not per file)
            assert len(case.evidence) == 1  # One source containing both files

            # Check that new source was created
            evidence_entry = case.evidence[0]
            source = DocumentSource.objects.get(source_id=evidence_entry["source_id"])

            # Source should have both the web URL and file URLs
            assert len(source.url) == 3  # web URL + 2 file URLs
            links = [u["link"] if isinstance(u, dict) else u for u in source.url]
            assert any("ciaa.gov.np/pressrelease/3173" in link for link in links)
            assert sum("ngm-store.jawafdehi.org" in link for link in links) == 2

            # Evidence description should show file count
            assert "2 documents" in evidence_entry["description"]

            # Output should indicate success
            output = out.getvalue()
            assert "✓ Cases mapped:" in output

    def test_skip_cases_without_press_release_evidence(self, mock_root_index):
        """Test that cases without press release evidence are skipped."""
        # Create case with regular evidence (not press release)
        regular_source = DocumentSource.objects.create(
            title="Regular Document",
            url=["https://example.com/document.pdf"],
            source_type=SourceType.LEGAL_COURT_ORDER,
        )

        case = Case.objects.create(
            case_type=CaseType.CORRUPTION,
            state=CaseState.DRAFT,
            title="Test Case",
            evidence=[
                {
                    "source_id": regular_source.source_id,
                    "description": "Regular document",
                }
            ],
        )

        original_evidence = case.evidence.copy()

        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            # First call returns root index, second call returns empty press release index
            mock_response.json.side_effect = [mock_root_index, {"manuscripts": []}]
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            out = StringIO()
            call_command("map_press_release_files", stdout=out)

            # Refresh case from database
            case.refresh_from_db()

            # Evidence should not be changed
            assert case.evidence == original_evidence

    def test_specific_case_id(
        self,
        mock_root_index,
        mock_press_release_index,
        case_with_press_release_evidence,
    ):
        """Test mapping a specific case by case_id."""
        case, pr_source = case_with_press_release_evidence

        # Create another case that should not be processed
        other_source = DocumentSource.objects.create(
            title="Other Press Release",
            url=["https://ciaa.gov.np/pressrelease/9999"],
            source_type=SourceType.LEGAL_PROCEDURAL,
        )
        other_case = Case.objects.create(
            case_type=CaseType.CORRUPTION,
            state=CaseState.DRAFT,
            title="Other Case",
            evidence=[
                {
                    "source_id": other_source.source_id,
                    "description": "Other press release",
                }
            ],
        )
        other_original_evidence = other_case.evidence.copy()

        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            # First call returns root index, second call returns press release index
            mock_response.json.side_effect = [mock_root_index, mock_press_release_index]
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            out = StringIO()
            call_command(
                "map_press_release_files",
                f"--case-id={case.case_id}",
                stdout=out,
            )

            # Refresh cases from database
            case.refresh_from_db()
            other_case.refresh_from_db()

            # Target case should be fixed (one source per press release)
            assert len(case.evidence) == 1

            # Other case should not be changed
            assert other_case.evidence == other_original_evidence

    def test_limit_option(self, mock_root_index, mock_press_release_index):
        """Test that limit option works correctly."""
        # Create multiple cases with press release evidence
        cases = []
        for i in range(5):
            pr_source = DocumentSource.objects.create(
                title=f"Press Release {i}",
                url=["https://ciaa.gov.np/pressrelease/3173"],
                source_type=SourceType.LEGAL_PROCEDURAL,
            )
            case = Case.objects.create(
                case_type=CaseType.CORRUPTION,
                state=CaseState.DRAFT,
                title=f"Test Case {i}",
                evidence=[
                    {
                        "source_id": pr_source.source_id,
                        "description": f"Press release {i}",
                    }
                ],
            )
            cases.append(case)

        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            # First call returns root index, second call returns press release index
            mock_response.json.side_effect = [mock_root_index, mock_press_release_index]
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            out = StringIO()
            call_command("map_press_release_files", "--limit=2", stdout=out)

            # Check that only 2 cases were processed
            output = out.getvalue()
            assert "Cases processed:     2" in output

    def test_handle_missing_press_release_in_index(
        self, mock_root_index, case_with_press_release_evidence
    ):
        """Test handling of press releases not found in NGM index."""
        case, pr_source = case_with_press_release_evidence
        original_evidence = case.evidence.copy()

        # Mock empty index (press release not found)
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            # First call returns root index, second call returns empty press release index
            mock_response.json.side_effect = [mock_root_index, {"manuscripts": []}]
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            out = StringIO()
            call_command("map_press_release_files", stdout=out)

            # Refresh case from database
            case.refresh_from_db()

            # Evidence should not be changed if press release not found
            assert case.evidence == original_evidence

            # Output should indicate no files found
            output = out.getvalue()
            assert "No files found" in output or "Skipped" in output
