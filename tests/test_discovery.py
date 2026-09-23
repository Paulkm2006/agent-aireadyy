from __future__ import annotations

import csv
import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from agent.cli import app
from agent.discovery.diversity import select_diverse_items
from agent.discovery.manifest import MANIFEST_COLUMNS, write_dataset_manifest
from agent.discovery.memory import DiscoveryMemory, DiscoveryReviewDecision, now_utc_iso
from agent.discovery.models import DatasetManifest, DatasetRequest, DiscoveredFile, DiscoveredProject
from agent.discovery.pride_discovery import discover_pride_dataset
from agent.discovery.query_builder import build_pride_queries, prepare_pride_search_queries
from agent.discovery.scoring import (
    build_discovered_project,
    classify_file_role,
    file_type_for_name,
    is_result_or_report_file,
    is_supported_raw_file,
    score_file,
    score_project,
)


def _project(accession: str, *, title: str, description: str = "") -> dict[str, Any]:
    return {
        "accession": accession,
        "title": title,
        "projectDescription": description,
        "sampleProcessingProtocol": "TiO2 phosphopeptide enrichment on a 90 min LC gradient",
        "dataProcessingProtocol": "DDA shotgun proteomics search with HCD fragmentation",
        "keywords": ["phosphoproteomics"],
        "organisms": [{"name": "Homo sapiens"}],
        "experimentTypes": [{"name": "shotgun proteomics"}],
        "instruments": [{"name": "Q Exactive HF"}],
    }


def _file(name: str, *, size: int = 1000) -> dict[str, Any]:
    return {
        "fileName": name,
        "fileSizeBytes": size,
        "publicFileLocations": [{"value": f"ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2026/06/{name}"}],
    }


def _discovered_project() -> DiscoveredProject:
    request = DatasetRequest(goal="ptm", ptm_type="phospho", max_projects=1, max_files=1)
    raw_project = _project("PXD000001", title="Human phosphoproteomics DDA")
    score = score_project(raw_project, request)
    return build_discovered_project(raw_project, request, score)


def test_discovery_defaults_separate_review_and_file_delivery_batch_sizes() -> None:
    request = DatasetRequest()

    assert request.inspection_batch_size == 30
    assert request.partial_delivery_batch_size == 500


def test_query_builder_phospho_request_generates_pride_keywords():
    request = DatasetRequest(goal="ptm", ptm_type="phospho", species=["human"], acquisition_mode="dda")

    queries = build_pride_queries(request)

    assert "phosphoproteomics" in queries
    assert "phosphopeptide enrichment" in queries
    assert any("TiO2" in query or "Ti-IMAC" in query for query in queries)
    assert any("phosphotyrosine" in query for query in queries)
    # PTS: species is a post-pool filter — not an equal PRIDE seed.
    assert not any("Homo sapiens" in query for query in queries)
    assert not any(query.strip().casefold() == "dda" for query in queries)


def test_query_builder_multi_ptm_request_generates_all_selected_ptm_keywords():
    request = DatasetRequest(goal="ptm", ptm_type="phospho", ptm_types=["phospho", "acetyl"], acquisition_mode="dda")

    queries = build_pride_queries(request)

    assert any("phosphoproteomics" in query for query in queries)
    assert any("acetylome" in query or "acetylation" in query for query in queries)
    # PTS: DDA is a filter, not a cross-product seed peer.
    assert not any(query.strip().endswith(" DDA") for query in queries)


def test_query_builder_general_request_uses_free_text_terms():
    request = DatasetRequest(
        goal="general",
        query_terms=["drug treatment", "kinase inhibitor"],
        species=["human"],
        acquisition_mode="dda",
    )

    queries = build_pride_queries(request)

    assert "drug treatment" in queries
    assert "kinase inhibitor" in queries
    assert not any("drug treatment DDA" in query for query in queries)
    assert not any("Homo sapiens drug treatment" in query for query in queries)


def test_pride_query_adapter_turns_compound_agent_queries_into_distinct_seeds():
    queries = prepare_pride_search_queries(
        [
            "human DDA Orbitrap label-free",
            "HeLa DDA Orbitrap",
            "HEK293 DDA Orbitrap label-free",
            "human DDA Orbitrap gradient length",
            "human label-free DDA Q Exactive",
            "human DDA Orbitrap Fusion Lumos",
        ]
    )

    # WP-A: expand ALL distinct seeds from the portfolio, not first-seed-only.
    assert "human" in queries
    assert "HeLa" in queries
    assert "HEK293" in queries
    assert "DDA" in queries
    assert "label-free" in queries
    assert "Orbitrap" in queries
    assert "q exactive" in queries
    assert "fusion lumos" in queries
    # No silent first-only collapse: multi-token inputs contribute multiple seeds.
    assert len(queries) >= 6


def test_prepare_pride_expands_all_atomic_seeds_not_first_only():
    seeds = prepare_pride_search_queries(["human DDA phospho"])
    assert seeds[:3] == ["human", "DDA", "phospho"] or set(seeds) >= {"human", "DDA", "phospho"}
    assert len(seeds) >= 3


def test_discovery_executes_high_recall_queries_with_balanced_page_sizes():
    class SearchRecordingClient:
        def __init__(self):
            self.calls: list[tuple[str, int]] = []

        def search_projects(self, keyword: str, page_size: int = 100):
            self.calls.append((keyword, page_size))
            return []

        def close(self):
            return None

    client = SearchRecordingClient()
    request = DatasetRequest(max_candidate_projects=25, max_projects=5, max_files=10)

    discover_pride_dataset(
        request,
        client=client,
        queries=[
            "human DDA Orbitrap label-free",
            "HeLa DDA Orbitrap",
            "HEK293 DDA Orbitrap label-free",
            "human DDA Orbitrap gradient length",
            "human label-free DDA Q Exactive",
            "human DDA Orbitrap Fusion Lumos",
        ],
    )

    # WP-A multi-seed portfolio: each compound query expands to multiple seeds.
    # Require core high-recall seeds; additional instrument/label seeds are allowed.
    keywords = [keyword for keyword, _page in client.calls]
    for required in ("human", "HeLa", "HEK293"):
        assert required in keywords
    assert all(page == 100 for _keyword, page in client.calls)
    assert len(client.calls) >= 6


def test_discovery_ranks_relevant_older_candidates_before_inspection():
    target = _project(
        "PXD000900",
        title="Confetti: A Multi-protease Map of the HeLa Proteome",
    )
    distractors = [
        _project(f"PXD9{index:05d}", title=f"Unrelated recent project {index}")
        for index in range(19)
    ]

    class RankedSearchClient:
        def __init__(self):
            self.inspected: list[str] = []

        def search_projects(self, keyword: str, page_size: int = 100):
            return [*distractors, target][:page_size]

        def get_project(self, accession: str):
            self.inspected.append(accession)
            return target if accession == "PXD000900" else distractors[0]

        def list_project_files(self, accession: str, **_kwargs):
            return []

        def close(self):
            return None

    client = RankedSearchClient()
    request = DatasetRequest(
        goal="general",
        query_terms=["HeLa", "multi-protease"],
        species=["Homo sapiens"],
        species_policy="include_only",
        max_candidate_projects=1,
        max_projects=1,
        max_files=1,
    )

    discover_pride_dataset(request, client=client, queries=["multi-protease"])

    assert client.inspected == ["PXD000900"]


def test_scoring_prefers_phospho_project_metadata():
    request = DatasetRequest(species=["human"], acquisition_mode="dda")
    phospho_project = _project("PXD000001", title="Human phosphoproteomics by DDA")
    regular_project = {
        "accession": "PXD000002",
        "title": "Human whole proteome DDA benchmark",
        "projectDescription": "Standard proteomics dataset",
        "dataProcessingProtocol": "DDA shotgun proteomics search",
        "organisms": [{"name": "Homo sapiens"}],
    }

    phospho_score = score_project(phospho_project, request)
    regular_score = score_project(regular_project, request)

    assert phospho_score.project_score > regular_score.project_score
    assert phospho_score.ptm_type == "phospho"
    assert not phospho_score.excluded


def test_general_discovery_scores_text_evidence_without_ptm_requirement():
    request = DatasetRequest(
        goal="general",
        query_terms=["drug treatment", "kinase inhibitor"],
        species=["human"],
        acquisition_mode="dda",
        max_projects=1,
        max_files=1,
    )
    raw_project = {
        "accession": "PXD000010",
        "title": "Human drug treatment DDA proteomics",
        "projectDescription": "Kinase inhibitor perturbation in a cancer cell line.",
        "sampleProcessingProtocol": "Compound treatment followed by 60 min LC gradient.",
        "dataProcessingProtocol": "DDA shotgun proteomics search with HCD fragmentation.",
        "keywords": ["drug treatment", "kinase inhibitor", "DDA"],
        "organisms": [{"name": "Homo sapiens"}],
        "experimentTypes": [{"name": "shotgun proteomics"}],
        "instruments": [{"name": "Q Exactive HF"}],
    }

    score = score_project(raw_project, request)
    project = build_discovered_project(raw_project, request, score)
    scored_file = score_file(_file("drug_treatment_01.raw"), project, request)

    assert score.project_score > 0
    assert any(item.source == "general_query" for item in score.evidence)
    assert scored_file is not None
    assert scored_file.validity_status in {"valid", "weak_keep"}
    assert "general_discovery_target" in scored_file.validity_reasons
    assert "weak_ptm_evidence" not in scored_file.validity_reasons
    assert "strong_ptm_evidence" not in scored_file.validity_reasons


def test_scoring_excludes_dia_when_dda_requested():
    request = DatasetRequest(
        acquisition_mode="dda",
        hard_constraint_fields=["repository", "acquisition_mode"],
    )
    dia_project = {
        "accession": "PXD000003",
        "title": "Human phosphoproteomics DIA SWATH",
        "projectDescription": "Data independent acquisition phosphoproteomics.",
        "sampleProcessingProtocol": "TiO2 phosphopeptide enrichment.",
        "dataProcessingProtocol": "SWATH processing with targeted extraction.",
        "keywords": ["phosphoproteomics", "DIA", "SWATH"],
        "organisms": [{"name": "Homo sapiens"}],
    }

    score = score_project(dia_project, request)

    assert score.excluded
    assert score.needs_review


def test_scoring_keeps_mixed_dda_dia_project_for_file_level_review():
    request = DatasetRequest(
        acquisition_mode="dda",
        hard_constraint_fields=["repository", "acquisition_mode"],
    )
    mixed_project = _project("PXD000004", title="Human phosphoproteomics DDA and DIA SWATH")

    score = score_project(mixed_project, request)
    project = build_discovered_project(mixed_project, request, score)

    assert not score.excluded
    assert not score.needs_review
    assert any(item.source == "mixed_acquisition" for item in score.evidence)
    assert project.validity_status == "weak_keep"
    assert project.needs_review is False
    assert "mixed_acquisition_project" in project.validity_reasons


def test_file_level_acquisition_resolves_mixed_project_candidates():
    request = DatasetRequest(
        acquisition_mode="dda",
        hard_constraint_fields=["repository", "acquisition_mode"],
    )
    mixed_project = _project("PXD000005", title="Human phosphoproteomics DDA and DIA SWATH")
    score = score_project(mixed_project, request)
    project = build_discovered_project(mixed_project, request, score)

    assert score_file(_file("fraction_DIA_01.raw"), project, request) is None

    dda_file = score_file(_file("fraction_DDA_01.raw"), project, request)
    unknown_file = score_file(_file("fraction_01.raw"), project, request)

    assert dda_file is not None
    assert dda_file.validity_status != "exclude"
    assert dda_file.needs_review is False
    assert "needs_file_level_acquisition_confirmation" not in dda_file.validity_reasons
    assert unknown_file is not None
    assert unknown_file.validity_status == "needs_review"
    assert "needs_file_level_acquisition_confirmation" in unknown_file.validity_reasons


def test_file_filter_keeps_raw_mzml_and_excludes_result_tables():
    project = _discovered_project()
    request = DatasetRequest()

    assert is_supported_raw_file("HeLa_01.raw")
    assert is_supported_raw_file("HeLa_01.mzML")
    assert is_supported_raw_file("sample.wiff")
    assert is_supported_raw_file("sample.d")
    assert file_type_for_name("sample.d.zip") == ".d"
    assert is_result_or_report_file("protein_groups.tsv")
    assert is_result_or_report_file("search_result.xlsx")

    assert score_file(_file("HeLa_01.raw"), project, request) is not None
    assert score_file(_file("HeLa_01.mzML"), project, request) is not None
    assert score_file(_file("protein_groups.tsv"), project, request) is None


def test_file_role_classifies_raw_peaklist_and_metadata():
    assert classify_file_role("HeLa_01.raw").role == "raw_acquisition"
    assert classify_file_role("HeLa_01.mzML").role == "converted_peaklist"
    assert classify_file_role("sample.d.zip").role == "raw_acquisition"
    assert classify_file_role("fraction_01.MGF").role == "converted_peaklist"
    assert classify_file_role("PXD000001.sdrf.tsv").role == "metadata"


def test_file_filter_excludes_derived_identification_mgf_names():
    project = _discovered_project()
    request = DatasetRequest()
    derived_name = "sample.raw__F001_.mzid_sample.raw__F001_.MGF"

    assert classify_file_role(derived_name).role == "search_result"
    assert is_result_or_report_file(derived_name)
    assert score_file(_file(derived_name), project, request) is None


def test_clean_mgf_is_retained_as_converted_peaklist():
    project = _discovered_project()
    request = DatasetRequest()

    scored = score_file(_file("fraction_01.MGF"), project, request)

    assert scored is not None
    assert scored.file_role == "converted_peaklist"
    assert scored.validity_status == "weak_keep"
    assert "converted_peaklist" in scored.validity_reasons


def test_file_name_species_conflict_is_quarantined_from_usable(tmp_path: Path):
    project = _discovered_project()
    request = DatasetRequest(species=["human"], species_policy="include_only", max_projects=1, max_files=1)
    scored = score_file(_file("Tcell_mouse_TMTpro_IMAC_F02.raw"), project, request)

    assert scored is not None
    assert scored.validity_status == "needs_review"
    assert "file_name_species_conflict" in scored.validity_reasons

    manifest = DatasetManifest(request=request, projects=[project], files=[scored])
    paths = write_dataset_manifest(manifest, tmp_path)
    assert paths["dataset_manifest_usable_csv"].read_text(encoding="utf-8").splitlines() == [
        ",".join(MANIFEST_COLUMNS)
    ]


class FakePrideClient:
    def __init__(self) -> None:
        self.projects = {
            "PXD000001": _project("PXD000001", title="Human phosphoproteomics DDA 1"),
            "PXD000002": _project("PXD000002", title="Human phosphoproteomics DDA 2"),
            "PXD000003": _project("PXD000003", title="Human phosphoproteomics DDA 3"),
        }
        self.files = {
            accession: [_file(f"{accession}_A.raw"), _file(f"{accession}_B.mzML"), _file(f"{accession}_report.tsv")]
            for accession in self.projects
        }

    def search_projects(self, keyword: str, page_size: int = 100) -> list[dict[str, Any]]:
        return [{"accession": accession} for accession in self.projects]

    def get_project(self, accession: str) -> dict[str, Any]:
        return self.projects[accession]

    def list_project_files(
        self,
        accession: str,
        keyword: str | None = None,
        page_size: int = 1000,
        max_files: int | None = None,
    ) -> list[dict[str, Any]]:
        records = self.files[accession]
        return records[:max_files] if max_files is not None else records

    def download_text(self, url: str) -> str:
        return ""

    def close(self) -> None:
        return None


def test_manifest_respects_project_and_file_limits():
    request = DatasetRequest(max_projects=2, max_files=3, max_files_per_project=1, max_candidate_projects=3)

    manifest = discover_pride_dataset(request, client=FakePrideClient())

    assert len(manifest.projects) == 2
    assert len(manifest.files) == 2
    assert all(project.selected_file_count == 1 for project in manifest.projects)
    assert all(file.file_name.endswith((".raw", ".mzML")) for file in manifest.files)
    assert manifest.summary["search_pagination"]
    assert manifest.summary["search_pagination"][0]["mode"] == "budgeted"
    assert manifest.summary["search_pagination"][0]["exhausted"] is True
    assert all(
        "file_inventory_truncated" in outcome
        for outcome in manifest.summary["inspection_outcomes"]
        if outcome["category"] != "not_inspected"
    )


def test_discovery_reports_project_metadata_score_and_evidence_fields():
    messages: list[str] = []
    request = DatasetRequest(
        goal="ptm",
        ptm_type="phospho",
        max_projects=1,
        max_files=1,
    )

    discover_pride_dataset(
        request,
        client=FakePrideClient(),
        report=messages.append,
    )

    scored = next(message for message in messages if "metadata scored" in message)
    assert "PXD000001" in scored
    assert "retrieval score" in scored
    assert "confidence" in scored
    assert "evidence fields" in scored


def test_discovery_extracts_instrument_and_fragmentation_from_metadata():
    request = DatasetRequest(goal="ptm", ptm_type="phospho", max_projects=1, max_files=1)
    manifest = discover_pride_dataset(request, client=FakePrideClient())

    file = manifest.files[0]

    project = manifest.projects[0]
    assert "Q Exactive HF" in project.instrument_names
    assert "orbitrap" in project.instrument_families
    assert "HCD" in project.fragmentation_methods
    # Project-level instrument metadata must not be broadcast to every file.
    assert file.instrument_names == []
    assert file.instrument_families == []
    assert "HCD" in file.fragmentation_methods
    assert file.lc_gradient_minutes == 90.0
    assert file.trust_score > 0
    assert "instrument:orbitrap" in project.diversity_tags
    assert file.validity_status == "weak_keep"
    assert file.evidence_level == "project"
    assert "project_level_evidence_only" in file.validity_reasons
    assert "strong_ptm_evidence" in file.validity_reasons


def test_validity_gate_marks_missing_fragmentation_as_weak_keep():
    request = DatasetRequest(max_projects=1, max_files=1)
    client = FakePrideClient()
    client.projects["PXD000001"] = {
        **client.projects["PXD000001"],
        "dataProcessingProtocol": "DDA shotgun proteomics search",
    }
    client.projects = {"PXD000001": client.projects["PXD000001"]}
    client.files = {"PXD000001": client.files["PXD000001"]}

    manifest = discover_pride_dataset(request, client=client)

    file = next(item for item in manifest.files if item.project_accession == "PXD000001")
    assert file.validity_status == "weak_keep"
    assert "missing_fragmentation" in file.validity_reasons


def test_validity_summary_counts_statuses_and_reasons():
    request = DatasetRequest(goal="ptm", ptm_type="phospho", max_projects=1, max_files=2, max_files_per_project=2)

    manifest = discover_pride_dataset(request, client=FakePrideClient())

    assert "validity_status_counts" in manifest.summary
    assert manifest.summary["validity_status_counts"]["weak_keep"] >= 1
    assert "strong_ptm_evidence" in manifest.summary["validity_reason_counts"]
    assert "evidence_level_distribution" in manifest.summary
    assert "sdrf_match_status_distribution" in manifest.summary


class FakeSdrfPrideClient(FakePrideClient):
    def __init__(self) -> None:
        super().__init__()
        self.projects = {
            "PXD000001": {
                **_project("PXD000001", title="Human phosphoproteomics DDA"),
                "instruments": [],
                "sampleProcessingProtocol": "TiO2 phosphopeptide enrichment",
                "dataProcessingProtocol": "DDA shotgun proteomics search",
            }
        }
        self.files = {
            "PXD000001": [
                _file("HeLa_01.raw"),
                {
                    "fileName": "PXD000001.sdrf.tsv",
                    "fileSizeBytes": 100,
                    "publicFileLocations": [{"value": "https://ftp.pride.ebi.ac.uk/PXD000001.sdrf.tsv"}],
                },
            ]
        }

    def download_text(self, url: str) -> str:
        return (
            "comment[data file]\tcharacteristics[cell line]\tcharacteristics[organism]"
            "\tcharacteristics[disease]\tfactor value[treatment]\tcomment[data acquisition method]"
            "\tcomment[fraction identifier]\tcomment[instrument]\tcomment[fragmentation method]"
            "\tcomment[LC gradient]\tcharacteristics[organism part]\tcomment[enzyme]"
            "\tcomment[isolation window]\tcomment[resolution]\tcomment[collision energy]"
            "\tcomment[scan range]\n"
            "HeLa_01.raw\tHeLa\tHomo sapiens\tcervical cancer\tDMSO\tDDA\t1"
            "\tOrbitrap Fusion Lumos\tCID\t120 min\tcervix\tTrypsin"
            "\t1.6 m/z\t30000\t30 NCE\t100-1600 m/z\n"
        )


def test_discovery_extracts_file_level_sdrf_features():
    request = DatasetRequest(max_projects=1, max_files=1)

    manifest = discover_pride_dataset(request, client=FakeSdrfPrideClient())

    file = manifest.files[0]
    assert file.instrument_names[0] == "Orbitrap Fusion Lumos"
    assert "orbitrap" in file.instrument_families
    assert file.fragmentation_methods == ["CID"]
    assert file.lc_gradient_minutes == 120.0
    assert file.tissue == "cervix"
    assert file.enzyme == "Trypsin"
    assert file.isolation_window == 1.6
    assert file.resolution == 30000.0
    assert file.collision_energy == 30.0
    assert file.scan_range == "100-1600 m/z"
    summary = manifest.projects[0].sdrf_summary
    expected_text = FakeSdrfPrideClient().download_text("unused")
    assert summary["status"] == "available"
    assert summary["content_sha256"] == sha256(expected_text.encode("utf-8")).hexdigest()
    assert summary["row_count"] == 1
    assert summary["match_status_counts"] == {
        "matched": 1,
        "no_file_match": 0,
        "no_sdrf": 0,
    }
    assert summary["canonical_fields"]["cell_line"] == ["HeLa"]
    assert summary["canonical_fields"]["disease"] == ["cervical cancer"]
    assert summary["canonical_fields"]["treatment"] == ["DMSO"]
    assert summary["canonical_fields"]["assay"] == ["DDA"]
    assert summary["file_match_examples"][0] == {
        "file_name": "HeLa_01.raw",
        "status": "matched",
        "matched_row_count": 1,
    }


def test_discovery_surfaces_sdrf_parse_errors_without_losing_project() -> None:
    class BrokenSdrfPrideClient(FakeSdrfPrideClient):
        def download_text(self, url: str) -> str:
            return ""

    manifest = discover_pride_dataset(
        DatasetRequest(max_projects=1, max_files=1),
        client=BrokenSdrfPrideClient(),
    )

    summary = manifest.projects[0].sdrf_summary
    assert summary["status"] == "parse_error"
    assert summary["content_sha256"] == sha256(b"").hexdigest()
    assert summary["row_count"] == 0
    assert summary["match_status_counts"]["no_sdrf"] == 1
    assert summary["errors"]


def test_memory_prior_is_neutral_for_unseen_project(tmp_path: Path):
    request = DatasetRequest(max_projects=1, max_files=1)
    memory = DiscoveryMemory(tmp_path / "memory")

    manifest = discover_pride_dataset(request, client=FakePrideClient(), memory=memory)

    assert manifest.projects[0].memory_prior == 0
    assert manifest.files[0].memory_prior == 0


def test_memory_prior_boosts_keep_and_penalizes_reject(tmp_path: Path):
    memory = DiscoveryMemory(tmp_path / "memory")
    memory.append_review_decisions(
        [
            DiscoveryReviewDecision(
                review_id="r1",
                run_id="old",
                created_at=now_utc_iso(),
                project_accession="PXD000001",
                file_name="PXD000001_A.raw",
                decision="keep",
                reason="correct",
            ),
            DiscoveryReviewDecision(
                review_id="r2",
                run_id="old",
                created_at=now_utc_iso(),
                project_accession="PXD000002",
                file_name="PXD000002_A.raw",
                decision="reject",
                reason="wrong_ptm",
            ),
        ]
    )
    request = DatasetRequest(max_projects=3, max_files=3, max_files_per_project=1)

    manifest = discover_pride_dataset(request, client=FakePrideClient(), memory=memory)
    priors = {file.project_accession: file.memory_prior for file in manifest.files}

    assert priors["PXD000001"] > 0
    assert priors["PXD000002"] < 0


def test_diversity_selector_prefers_new_instrument_family_when_scores_are_close():
    request = DatasetRequest(max_projects=2, max_files=2, max_files_per_project=2)
    project_a = DiscoveredProject(project_accession="PXA", project_score=80, trust_score=0.9)
    project_b = DiscoveredProject(project_accession="PXB", project_score=80, trust_score=0.9)
    file_a1 = DiscoveredFile(
        project_accession="PXA",
        file_name="A1.raw",
        file_type=".raw",
        file_score=60,
        confidence=0.9,
        trust_score=0.9,
        species=["human"],
        instrument_families=["orbitrap"],
        fragmentation_methods=["HCD"],
    )
    file_a2 = file_a1.model_copy(update={"file_name": "A2.raw", "file_score": 59})
    file_b = DiscoveredFile(
        project_accession="PXB",
        file_name="B1.raw",
        file_type=".raw",
        file_score=58,
        confidence=0.9,
        trust_score=0.9,
        species=["human"],
        instrument_families=["tof"],
        fragmentation_methods=["HCD"],
    )

    selected = select_diverse_items([(project_a, [file_a1, file_a2]), (project_b, [file_b])], request)
    selected_files = [file.file_name for _project, files in selected for file in files]

    assert "B1.raw" in selected_files
    assert len(selected_files) == 2


def test_instrument_preference_changes_ranking_using_observed_instrument_generation():
    newer_project = DiscoveredProject(
        project_accession="PXD_NEW",
        project_score=80,
        trust_score=0.9,
        instrument_generation_score=0.95,
        instrument_generation_label="current",
    )
    classic_project = DiscoveredProject(
        project_accession="PXD_CLASSIC",
        project_score=80,
        trust_score=0.9,
        instrument_generation_score=0.15,
        instrument_generation_label="legacy",
    )
    newer_file = DiscoveredFile(
        project_accession="PXD_NEW",
        file_name="new.raw",
        file_type=".raw",
        file_score=60,
        trust_score=0.9,
        validity_status="valid",
        instrument_generation_score=0.95,
        instrument_generation_label="current",
    )
    classic_file = DiscoveredFile(
        project_accession="PXD_CLASSIC",
        file_name="classic.raw",
        file_type=".raw",
        file_score=60,
        trust_score=0.9,
        validity_status="valid",
        instrument_generation_score=0.15,
        instrument_generation_label="legacy",
    )
    items = [(newer_project, [newer_file]), (classic_project, [classic_file])]

    newer = select_diverse_items(
        items,
        DatasetRequest(
            max_projects=1,
            max_files=1,
            max_files_per_project=1,
            instrument_preference="newer",
        ),
    )
    classic = select_diverse_items(
        items,
        DatasetRequest(
            max_projects=1,
            max_files=1,
            max_files_per_project=1,
            instrument_preference="classic",
        ),
    )

    assert newer[0][0].project_accession == "PXD_NEW"
    assert classic[0][0].project_accession == "PXD_CLASSIC"


def test_diversity_selector_respects_limits():
    request = DatasetRequest(max_projects=1, max_files=2, max_files_per_project=1)
    project_a = DiscoveredProject(project_accession="PXA", project_score=80, trust_score=0.9)
    project_b = DiscoveredProject(project_accession="PXB", project_score=80, trust_score=0.9)
    file_a1 = DiscoveredFile(project_accession="PXA", file_name="A1.raw", file_type=".raw", file_score=60, trust_score=0.9)
    file_a2 = DiscoveredFile(project_accession="PXA", file_name="A2.raw", file_type=".raw", file_score=59, trust_score=0.9)
    file_b = DiscoveredFile(project_accession="PXB", file_name="B1.raw", file_type=".raw", file_score=58, trust_score=0.9)

    selected = select_diverse_items([(project_a, [file_a1, file_a2]), (project_b, [file_b])], request)
    selected_files = [file.file_name for _project, files in selected for file in files]

    assert len(selected) == 1
    assert len(selected_files) == 1


def test_discover_dataset_cli_writes_manifest_files(monkeypatch, tmp_path: Path):
    request = DatasetRequest(max_projects=1, max_files=1)
    project = _discovered_project()
    file = DiscoveredFile(
        project_accession=project.project_accession,
        project_title=project.project_title,
        file_name="HeLa_01.raw",
        download_url="https://ftp.pride.ebi.ac.uk/HeLa_01.raw",
        file_type=".raw",
        expected_size_bytes=1000,
        species=["human"],
        acquisition_mode="dda",
        ptm_type="phospho",
        project_score=project.project_score,
        file_score=50,
        confidence=0.8,
    )
    manifest = DatasetManifest(
        request=request,
        projects=[project],
        files=[file],
        summary={"selected_projects": 1, "selected_files": 1},
    )

    def fake_discover(received_request: DatasetRequest, memory=None) -> DatasetManifest:
        return manifest.model_copy(update={"request": received_request})

    monkeypatch.setattr("agent.cli.discover_pride_dataset", fake_discover)
    runner = CliRunner()
    output_dir = tmp_path / "discovery"

    result = runner.invoke(
        app,
        [
            "discover-dataset",
            "--goal",
            "ptm",
            "--ptm",
            "phospho",
            "--species",
            "human",
            "--acquisition",
            "dda",
            "--max-projects",
            "1",
            "--max-files",
            "1",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (output_dir / "dataset_manifest.json").exists()
    assert (output_dir / "dataset_manifest.csv").exists()
    assert (output_dir / "batch_inputs.txt").read_text(encoding="utf-8").strip() == "https://ftp.pride.ebi.ac.uk/HeLa_01.raw"


def test_manifest_writer_outputs_expected_files(tmp_path: Path):
    request = DatasetRequest(max_projects=1, max_files=1)
    project = _discovered_project()
    file = score_file(_file("HeLa_01.raw"), project, request)
    assert file is not None
    manifest = DatasetManifest(
        request=request,
        projects=[project.model_copy(update={"selected_file_count": 1})],
        files=[file],
        summary={"selected_projects": 1, "selected_files": 1},
    )

    paths = write_dataset_manifest(manifest, tmp_path)

    assert paths["dataset_request"].exists()
    assert paths["candidate_projects"].exists()
    assert paths["dataset_manifest_json"].exists()
    assert paths["dataset_manifest_csv"].exists()
    assert paths["dataset_manifest_valid_csv"].exists()
    assert paths["dataset_manifest_usable_csv"].exists()
    assert paths["dataset_manifest_task_ready_csv"].exists()
    assert paths["batch_inputs"].exists()
    assert paths["batch_inputs_valid"].exists()
    assert paths["batch_inputs_usable"].exists()
    assert paths["batch_inputs_task_ready"].exists()
    assert paths["discovery_summary"].exists()
    assert paths["quality_report"].exists()


def test_manifest_includes_trust_and_diversity_fields(tmp_path: Path):
    request = DatasetRequest(max_projects=1, max_files=1)
    manifest = discover_pride_dataset(request, client=FakePrideClient())

    paths = write_dataset_manifest(manifest, tmp_path)

    with paths["dataset_manifest_csv"].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert "retrieval_trust_score" in rows[0]
    assert "file_role" in rows[0]
    assert "file_role_reasons" in rows[0]
    assert "sdrf_match_status" in rows[0]
    assert "evidence_level" in rows[0]
    assert "file_level_evidence_count" in rows[0]
    assert "project_level_evidence_count" in rows[0]
    assert "evidence_warnings" in rows[0]
    assert "instrument_families" in rows[0]
    assert "fragmentation_methods" in rows[0]
    assert "diversity_tags" in rows[0]
    assert "validity_status" in rows[0]
    assert "validity_reasons" in rows[0]
    assert "task_type" in rows[0]
    assert "task_profile" in rows[0]
    assert "task_readiness_status" in rows[0]
    assert "task_readiness_reasons" in rows[0]
    assert "missing_task_requirements" in rows[0]
    assert "label_source_status" in rows[0]
    assert "spectra_requirement_status" in rows[0]
    assert "metadata_requirement_status" in rows[0]
    assert "next_pipeline_steps" in rows[0]
    assert "ai_ready_target_schema" in rows[0]


def test_project_level_only_evidence_is_weak_keep():
    project = _discovered_project()
    request = DatasetRequest()

    scored = score_file(_file("sample_01.raw"), project, request, sdrf_match_status="no_sdrf")

    assert scored is not None
    assert scored.evidence_level == "project"
    assert scored.sdrf_match_status == "no_sdrf"
    assert scored.file_level_evidence_count == 0
    assert scored.project_level_evidence_count > 0
    assert scored.validity_status == "weak_keep"
    assert "project_level_evidence_only" in scored.validity_reasons


def test_file_name_evidence_creates_mixed_evidence_level():
    project = _discovered_project()
    request = DatasetRequest()

    scored = score_file(_file("sample_phospho_DDA_01.raw"), project, request, sdrf_match_status="no_sdrf")

    assert scored is not None
    assert scored.evidence_level == "mixed"
    assert scored.file_level_evidence_count > 0
    assert scored.project_level_evidence_count > 0


def test_manifest_writer_outputs_valid_and_usable_subsets(tmp_path: Path):
    request = DatasetRequest(max_projects=1, max_files=3, max_files_per_project=3)
    project = _discovered_project()
    valid_file = DiscoveredFile(
        project_accession=project.project_accession,
        file_name="valid.raw",
        file_type=".raw",
        validity_status="valid",
        validity_reasons=["strong_ptm_evidence"],
    )
    weak_file = DiscoveredFile(
        project_accession=project.project_accession,
        file_name="weak.mzML",
        file_type=".mzML",
        validity_status="weak_keep",
        validity_reasons=["missing_fragmentation"],
    )
    review_file = DiscoveredFile(
        project_accession=project.project_accession,
        file_name="review.raw",
        file_type=".raw",
        validity_status="needs_review",
        validity_reasons=["missing_species_evidence"],
        needs_review=True,
    )
    weak_review_file = DiscoveredFile(
        project_accession=project.project_accession,
        file_name="weak-review.raw",
        file_type=".raw",
        validity_status="weak_keep",
        validity_reasons=["project_level_evidence_only"],
        needs_review=True,
    )
    manifest = DatasetManifest(
        request=request,
        projects=[project],
        files=[valid_file, weak_file, review_file, weak_review_file],
        summary={"selected_projects": 1, "selected_files": 4, "excluded_files": 2},
    )

    paths = write_dataset_manifest(manifest, tmp_path)

    with paths["dataset_manifest_valid_csv"].open("r", encoding="utf-8", newline="") as handle:
        valid_rows = list(csv.DictReader(handle))
    with paths["dataset_manifest_usable_csv"].open("r", encoding="utf-8", newline="") as handle:
        usable_rows = list(csv.DictReader(handle))
    quality = json.loads(paths["quality_report"].read_text(encoding="utf-8"))

    assert [row["file_name"] for row in valid_rows] == ["valid.raw"]
    assert [row["file_name"] for row in usable_rows] == ["valid.raw"]
    assert paths["batch_inputs_valid"].read_text(encoding="utf-8").splitlines() == ["PXD000001/valid.raw"]
    assert paths["batch_inputs_usable"].read_text(encoding="utf-8").splitlines() == [
        "PXD000001/valid.raw"
    ]
    assert quality["valid_files"] == 1
    assert quality["usable_files"] == 1
    assert quality["needs_review_files"] == 2
    assert quality["excluded_files"] == 2
    assert quality["task_readiness_applicability"] == "not_applicable_task_undecided"
    assert "task_ready_manifest_csv" not in quality["recommended_outputs"]
