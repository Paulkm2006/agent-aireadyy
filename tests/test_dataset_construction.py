from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner
from sqlalchemy import create_engine, inspect, select
from sqlalchemy.orm import Session

from agent.dataset_construction import (
    DatasetCatalog,
    DatasetContractError,
    ObservationRecord,
    SplitAllocation,
    DatasetConstructionBase,
    DatasetReleaseRow,
    DatasetSplitAllocationRow,
    SplitPolicy,
    audit_split,
    build_dataset_release,
    ingest_existing_batch,
    plan_split_suite,
    validate_catalog,
    build_identity_ledger,
)
from agent.operations.config import OperationsSettings
from agent.operations.database import OperationsDatabase
from agent.dataset_construction.agent_runtime import build_dataset_construction_agent
from agent.dataset_construction.benchmark import (
    DENOVO_BENCHMARK_PROFILE,
    curate_denovo_benchmark,
)
from agent.dataset_construction.cli import app as dataset_cli


def _write_existing_batch(batch_dir: Path) -> Path:
    parquet = batch_dir / "01_run" / "task_runs" / "denovo" / "denovo.parquet"
    parquet.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "spectrum_id": "scan=1",
                "peptide_sequence": "PEPTIDEK",
                "modified_sequence": "PEPTIDEK",
                "charge": 2,
                "q_value": 0.005,
            },
            {
                "spectrum_id": "scan=2",
                "peptide_sequence": "PEPTIDER",
                "modified_sequence": "PEPTIDER",
                "charge": 3,
                "q_value": 0.009,
            },
        ]
    ).to_parquet(parquet, index=False)
    (batch_dir / "mini_e2e_batch_summary.json").write_text(
        json.dumps(
            {
                "run_results": [
                    {
                        "run_name": "run_1",
                        "project_accession": "PXD000001",
                        "source_file": "sample_A.raw",
                        "sample_id": "sample_A",
                        "technical_replicate_id": "rep_1",
                        "task_statuses": {"denovo": "completed"},
                        "rows_out": {"denovo": 2},
                        "task_files": {
                            "denovo": {"denovo_train_parquet": str(parquet)}
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return parquet


def test_existing_batch_resolves_relative_artifacts_from_batch_root(tmp_path: Path) -> None:
    batch_dir = tmp_path / "batch"
    parquet = _write_existing_batch(batch_dir)
    summary_path = batch_dir / "mini_e2e_batch_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["run_results"][0]["task_files"]["denovo"]["denovo_train_parquet"] = (
        parquet.relative_to(batch_dir).as_posix()
    )
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    catalog = ingest_existing_batch(batch_dir)

    assert len(catalog.observations) == 2
    assert {row.source_artifact_uri for row in catalog.observations} == {
        str(parquet.resolve())
    }


def test_existing_batch_aggregates_per_item_summaries_below_product_batch_root(
    tmp_path: Path,
) -> None:
    product_root = tmp_path / "product-batch"
    for index in (1, 2):
        item = product_root / f"item-{index}"
        _write_existing_batch(item)
        summary_path = item / "mini_e2e_batch_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["run_results"][0]["project_accession"] = f"PXD{index:06d}"
        summary["run_results"][0]["source_file"] = f"sample_{index}.raw"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

    catalog = ingest_existing_batch(product_root)

    assert len(catalog.observations) == 4
    assert {Path(row.source_artifact_uri).parts[-5] for row in catalog.observations} == {
        "item-1",
        "item-2",
    }


def test_existing_batch_filters_multi_task_artifacts_and_preserves_task_label(
    tmp_path: Path,
) -> None:
    batch_dir = tmp_path / "batch"
    denovo = _write_existing_batch(batch_dir)
    rt = batch_dir / "01_run" / "task_runs" / "rt" / "rt.parquet"
    rt.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "project_accession": "PXD000001",
                "source_file": "sample_A.raw",
                "spectrum_id": "scan=1",
                "peptide_sequence": "PEPTIDEK",
                "modified_sequence": "PEPTIDEK",
                "charge": 2,
                "retention_time": 17.5,
                "retention_time_unit": "minutes",
                "q_value": 0.005,
            }
        ]
    ).to_parquet(rt, index=False)
    summary_path = batch_dir / "mini_e2e_batch_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["run_results"][0]["task_files"] = {
        "denovo": {"denovo_train_parquet": str(denovo)},
        "rt_prediction": {"rt_train_parquet": str(rt)},
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    catalog = ingest_existing_batch(batch_dir, task_types=["rt"])

    assert len(catalog.observations) == 1
    assert catalog.observations[0].task_type == "rt_prediction"
    assert catalog.observations[0].label_type == "retention_time"
    assert catalog.observations[0].label_payload == {
        "retention_time": 17.5,
        "unit": "minutes",
        "unit_source": "",
    }


def test_catalog_contract_enforces_task_specific_learning_target() -> None:
    base = _observation(
        1,
        project="PXD1",
        file_name="a",
        sample="sample-a",
        lab="lab-a",
        instrument="inst-a",
        organism="human",
        peptide="PEPTIDE",
        modification="none",
        acquisition="DDA",
    ).model_copy(
        update={
            "task_type": "rt_prediction",
            "label_type": "retention_time",
            "label_payload": {"retention_time": 12.3, "unit": "minutes"},
        }
    )

    evidence = validate_catalog(
        DatasetCatalog(source_batch_dir="/batch", observations=[base]),
        task_spec={"task_type": "rt_prediction"},
    )
    assert evidence["label_policy"]["status"] == "pass"

    missing_unit = base.model_copy(
        update={"label_payload": {"retention_time": 12.3, "unit": ""}}
    )
    with pytest.raises(DatasetContractError, match="missing_retention_time_unit"):
        validate_catalog(
            DatasetCatalog(source_batch_dir="/batch", observations=[missing_unit]),
            task_spec={"task_type": "rt_prediction"},
        )


def test_identity_ledger_records_canonical_value_source_confidence_and_missing_reason(
    tmp_path: Path,
) -> None:
    batch_dir = tmp_path / "batch"
    _write_existing_batch(batch_dir)
    summary_path = batch_dir / "mini_e2e_batch_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["run_results"][0].update(
        {
            "species": "Homo sapiens",
            "instrument": "Thermo Fisher Q Exactive HF",
            "acquisition_mode": "DDA-HCD",
        }
    )
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    ledger = build_identity_ledger(ingest_existing_batch(batch_dir))

    organism = next(
        row
        for row in ledger.assertions
        if row.observation_id.endswith(ledger.assertions[0].observation_id.split(":")[-1])
        and row.dimension == "organism_id"
    )
    assert organism.canonical_values == ["ncbi:9606"]
    assert organism.source_kind == "batch_summary"
    assert organism.confidence == "reported"
    lab = next(row for row in ledger.assertions if row.dimension == "lab_id")
    assert lab.status == "missing"
    assert lab.missing_reason == "not_reported_by_batch_or_source_artifact"
    instrument_summary = next(
        row for row in ledger.dimensions if row.dimension == "instrument_id"
    )
    assert instrument_summary.present_count == 2
    assert instrument_summary.unique_count == 1
    assert instrument_summary.coverage == 1.0


def test_existing_batch_is_ingested_at_observation_level_without_mutating_source(
    tmp_path: Path,
) -> None:
    batch_dir = tmp_path / "batch"
    source_parquet = _write_existing_batch(batch_dir)
    original_checksum = source_parquet.read_bytes()

    catalog = ingest_existing_batch(batch_dir)

    assert len(catalog.observations) == 2
    assert {row.spectrum_id for row in catalog.observations} == {"scan=1", "scan=2"}
    assert {row.project_id for row in catalog.observations} == {"PXD000001"}
    assert len({row.file_family_id for row in catalog.observations}) == 1
    assert all(row.source_artifact_uri == str(source_parquet.resolve()) for row in catalog.observations)
    assert source_parquet.read_bytes() == original_checksum


def _observation(
    number: int,
    *,
    project: str,
    file_name: str,
    sample: str,
    lab: str,
    instrument: str,
    organism: str,
    peptide: str,
    modification: str,
    acquisition: str,
) -> ObservationRecord:
    return ObservationRecord(
        observation_id=f"obs-{number}",
        task_type="denovo",
        project_id=project,
        source_file_id=file_name,
        file_family_id=f"family-{file_name}",
        source_artifact_uri=f"/{file_name}.parquet",
        source_row_number=number,
        spectrum_id=f"scan={number}",
        sample_id=sample,
        lab_id=lab,
        instrument_id=instrument,
        organism_id=organism,
        acquisition_id=acquisition,
        peptide=peptide,
        modified_peptide=f"{peptide}[{modification}]",
        modification_classes=[modification],
        q_value=0.005,
    )


def test_split_suite_exposes_all_protocols_and_never_silently_falls_back() -> None:
    observations = [
        _observation(
            index,
            project=f"PXD{index:06d}",
            file_name=f"file-{index}",
            sample=f"sample-{index}",
            lab=f"lab-{index}",
            instrument=f"instrument-{index}",
            organism=f"organism-{index}",
            peptide=f"PEPTIDE{index}",
            modification=f"PTM-{index}",
            acquisition=f"acquisition-{index}",
        )
        for index in range(1, 10)
    ]
    catalog = DatasetCatalog(source_batch_dir="/batch", observations=observations)

    suite = plan_split_suite(catalog, ratios=(0.6, 0.2, 0.2), seed=17)

    assert set(suite.protocols) == {
        "row_random_control",
        "file_disjoint",
        "project_disjoint",
        "lab_disjoint",
        "instrument_disjoint",
        "organism_disjoint",
        "peptide_disjoint",
        "protein_family_disjoint",
        "modification_disjoint",
        "acquisition_disjoint",
    }
    assert suite.protocols["protein_family_disjoint"].status == "inconclusive"
    assert suite.protocols["protein_family_disjoint"].reasons == [
        "missing_required_identity:protein_family_ids"
    ]
    assert all(
        plan.status == "ready"
        for name, plan in suite.protocols.items()
        if name != "protein_family_disjoint"
    )
    for name, plan in suite.protocols.items():
        if name == "protein_family_disjoint":
            continue
        assert plan.resolved_protocol == name
        assert {row.split for row in plan.allocations} == {"train", "validation", "test"}

    incomplete = catalog.model_copy(
        update={
            "observations": [row.model_copy(update={"lab_id": ""}) for row in observations]
        }
    )
    incomplete_suite = plan_split_suite(incomplete, seed=17)

    assert incomplete_suite.protocols["lab_disjoint"].status == "inconclusive"
    assert incomplete_suite.protocols["lab_disjoint"].resolved_protocol == "lab_disjoint"
    assert incomplete_suite.protocols["lab_disjoint"].allocations == []


def test_file_family_and_sample_components_are_never_split() -> None:
    observations = [
        _observation(
            1,
            project="PXD1",
            file_name="a",
            sample="shared-sample",
            lab="lab-a",
            instrument="inst-a",
            organism="human",
            peptide="PEPA",
            modification="none",
            acquisition="DDA",
        ),
        _observation(
            2,
            project="PXD2",
            file_name="b",
            sample="shared-sample",
            lab="lab-b",
            instrument="inst-b",
            organism="mouse",
            peptide="PEPB",
            modification="phospho",
            acquisition="DIA",
        ),
        *[
            _observation(
                index,
                project=f"PXD{index}",
                file_name=f"f{index}",
                sample=f"s{index}",
                lab=f"l{index}",
                instrument=f"i{index}",
                organism=f"o{index}",
                peptide=f"PEP{index}",
                modification=f"m{index}",
                acquisition=f"a{index}",
            )
            for index in range(3, 10)
        ],
    ]

    suite = plan_split_suite(
        DatasetCatalog(source_batch_dir="/batch", observations=observations),
        seed=9,
    )

    for protocol, plan in suite.protocols.items():
        if plan.status != "ready" or protocol == "row_random_control":
            continue
        by_observation = {row.observation_id: row.split for row in plan.allocations}
        assert by_observation["obs-1"] == by_observation["obs-2"]


def test_independent_auditor_detects_tampered_manifest_and_reports_evidence() -> None:
    observations = [
        _observation(
            index,
            project=f"PXD{index}",
            file_name=f"f{index}",
            sample=f"s{index}",
            lab=f"l{index}",
            instrument=f"i{index}",
            organism="human",
            peptide=f"PEP{index}",
            modification=f"m{index}",
            acquisition="DDA",
        )
        for index in range(1, 7)
    ]
    # Two observations deliberately share a project. A valid project split must
    # keep them together; the forged manifest below separates them.
    observations[1] = observations[1].model_copy(update={"project_id": "PXD1"})
    catalog = DatasetCatalog(source_batch_dir="/batch", observations=observations)
    plan = plan_split_suite(catalog, seed=11).protocols["project_disjoint"]
    assert plan.status == "ready"
    assert audit_split(catalog, plan).status == "pass"

    forged = plan.model_copy(
        update={
            "allocations": [
                SplitAllocation(
                    observation_id=row.observation_id,
                    component_id=row.component_id,
                    split=(
                        "train"
                        if row.observation_id == "obs-1"
                        else "test"
                        if row.observation_id == "obs-2"
                        else row.split
                    ),
                )
                for row in plan.allocations
            ]
        }
    )

    audit = audit_split(catalog, forged)

    assert audit.status == "fail"
    finding = next(row for row in audit.findings if row.dimension == "project_id")
    assert finding.status == "fail"
    assert finding.overlap_count == 1
    assert finding.affected_identities == ["pxd1"]
    assert {"obs-1", "obs-2"} <= set(finding.affected_observation_ids)


def test_independent_auditor_enforces_available_base_must_link_identity() -> None:
    observations = [
        _observation(
            index,
            project=f"PXD{index}",
            file_name=f"f{index}",
            sample=("shared-sample" if index <= 2 else f"s{index}"),
            lab=f"l{index}",
            instrument=f"i{index}",
            organism="human",
            peptide=f"PEP{index}",
            modification="none",
            acquisition="DDA",
        )
        for index in range(1, 7)
    ]
    catalog = DatasetCatalog(source_batch_dir="/batch", observations=observations)
    plan = plan_split_suite(catalog, seed=12).protocols["project_disjoint"]
    forged = plan.model_copy(
        update={
            "allocations": [
                allocation.model_copy(update={"split": "train"})
                if allocation.observation_id == "obs-1"
                else allocation.model_copy(update={"split": "test"})
                if allocation.observation_id == "obs-2"
                else allocation
                for allocation in plan.allocations
            ]
        }
    )

    audit = audit_split(catalog, forged)

    finding = next(row for row in audit.findings if row.dimension == "sample_id")
    assert audit.status == "fail"
    assert finding.requirement == "zero_overlap"
    assert finding.affected_identities == ["shared-sample"]


def test_auditor_marks_missing_required_identity_inconclusive() -> None:
    observations = [
        _observation(
            index,
            project=f"PXD{index}",
            file_name=f"f{index}",
            sample=f"s{index}",
            lab="",
            instrument=f"i{index}",
            organism="human",
            peptide=f"PEP{index}",
            modification="none",
            acquisition="DDA",
        )
        for index in range(1, 5)
    ]
    catalog = DatasetCatalog(source_batch_dir="/batch", observations=observations)
    plan = plan_split_suite(catalog).protocols["lab_disjoint"]

    audit = audit_split(catalog, plan)

    assert audit.status == "inconclusive"
    assert any(row.dimension == "lab_id" and row.status == "inconclusive" for row in audit.findings)


def test_auditor_rejects_incomplete_duplicate_and_unknown_allocations() -> None:
    observations = [
        _observation(
            index,
            project=f"PXD{index}",
            file_name=f"f{index}",
            sample=f"s{index}",
            lab=f"l{index}",
            instrument=f"i{index}",
            organism="human",
            peptide=f"PEP{index}",
            modification="none",
            acquisition="DDA",
        )
        for index in range(1, 7)
    ]
    catalog = DatasetCatalog(source_batch_dir="/batch", observations=observations)
    plan = plan_split_suite(catalog, seed=21).protocols["project_disjoint"]
    forged = plan.model_copy(
        update={
            "allocations": [
                *[row for row in plan.allocations if row.observation_id != "obs-1"],
                plan.allocations[1],
                SplitAllocation(
                    observation_id="unknown-observation",
                    component_id="forged-component",
                    split="holdout",
                ),
            ]
        }
    )

    audit = audit_split(catalog, forged)

    assert audit.status == "fail"
    finding = next(row for row in audit.findings if row.dimension == "manifest_integrity")
    assert finding.status == "fail"
    assert any(value == "missing:obs-1" for value in finding.affected_identities)
    assert any(value.startswith("duplicate:") for value in finding.affected_identities)
    assert "unknown:unknown-observation" in finding.affected_identities
    assert "invalid_split:unknown-observation" in finding.affected_identities


def test_dataset_release_is_frozen_with_checksums_and_sql_index(tmp_path: Path) -> None:
    observations = [
        _observation(
            index,
            project=f"PXD{index}",
            file_name=f"f{index}",
            sample=f"s{index}",
            lab=f"l{index}",
            instrument=f"i{index}",
            organism=f"o{index}",
            peptide=f"PEP{index}",
            modification=f"m{index}",
            acquisition=f"a{index}",
        )
        for index in range(1, 10)
    ]
    catalog = DatasetCatalog(source_batch_dir="/batch", observations=observations)
    suite = plan_split_suite(catalog, seed=31)
    engine = create_engine(f"sqlite:///{(tmp_path / 'catalog.sqlite').as_posix()}")
    DatasetConstructionBase.metadata.create_all(engine)

    release = build_dataset_release(
        catalog,
        suite,
        output_dir=tmp_path / "release",
        release_id="release-001",
        task_spec={"task_type": "denovo", "version": 1},
        engine=engine,
    )

    assert release.status == "released"
    assert set(release.protocol_statuses) == set(suite.protocols)
    assert Path(release.files["catalog_parquet"]).is_file()
    assert Path(release.files["release_manifest_json"]).is_file()
    assert Path(release.files["checksums_sha256"]).is_file()
    assert Path(release.files["catalog_contract_json"]).is_file()
    assert Path(release.files["benchmark_quality_report_json"]).is_file()
    assert Path(release.files["benchmark_index_parquet"]).is_file()
    assert Path(release.files["identity_ledger_parquet"]).is_file()
    identity_summary = json.loads(
        Path(release.files["identity_ledger_summary_json"]).read_text(encoding="utf-8")
    )
    assert identity_summary["schema_version"] == "identity-ledger/v1"
    assert any(
        row["dimension"] == "peptide" and row["coverage"] == 1.0
        for row in identity_summary["dimensions"]
    )
    assert Path(release.files["prov_json"]).is_file()
    assert Path(release.files["ro_crate_metadata_json"]).is_file()
    checksums = Path(release.files["checksums_sha256"]).read_text(encoding="utf-8")
    assert "catalog/observations.parquet" in checksums
    assert "split_manifests/project_disjoint.parquet" in checksums
    manifest = json.loads(Path(release.files["release_manifest_json"]).read_text(encoding="utf-8"))
    assert manifest["release_id"] == "release-001"
    assert manifest["immutable"] is True
    assert manifest["split_policy"]["peptide_identity_mode"] == "il_equivalent"
    assert manifest["protocols"]["project_disjoint"]["audit_status"] == "pass"
    prov = json.loads(Path(release.files["prov_json"]).read_text(encoding="utf-8"))
    assert "entity" in prov
    crate = json.loads(
        Path(release.files["ro_crate_metadata_json"]).read_text(encoding="utf-8")
    )
    assert crate["@context"]

    with Session(engine) as session:
        stored = session.scalar(
            select(DatasetReleaseRow).where(DatasetReleaseRow.release_id == "release-001")
        )
        assert stored is not None
        assert stored.status == "released"
        assert stored.observation_count == 9

    try:
        build_dataset_release(
            catalog,
            suite,
            output_dir=tmp_path / "another-dir",
            release_id="release-001",
            task_spec={"task_type": "denovo", "version": 1},
            engine=engine,
        )
    except ValueError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("an immutable release ID must not be overwritten")

    with Session(engine) as session:
        allocations = session.scalars(
            select(DatasetSplitAllocationRow).where(
                DatasetSplitAllocationRow.release_id == "release-001",
                DatasetSplitAllocationRow.protocol == "project_disjoint",
            )
        ).all()
        assert len(allocations) == 9


def test_report_only_overlap_is_visible_but_does_not_fail_protocol() -> None:
    observations = [
        _observation(
            index,
            project=f"PXD{index}",
            file_name=f"f{index}",
            sample=f"s{index}",
            lab="shared-lab",
            instrument="shared-instrument",
            organism="human",
            peptide="SHAREDPEPTIDE",
            modification="none",
            acquisition="DDA",
        )
        for index in range(1, 7)
    ]
    catalog = DatasetCatalog(source_batch_dir="/batch", observations=observations)
    plan = plan_split_suite(catalog, seed=4).protocols["project_disjoint"]

    audit = audit_split(catalog, plan)

    assert audit.status == "pass"
    peptide = next(row for row in audit.findings if row.dimension == "peptide")
    assert peptide.requirement == "report_only"
    assert peptide.status == "reported_overlap"
    assert peptide.overlap_count == 1
    assert next(row for row in audit.findings if row.dimension == "project_id").status == "pass"


def test_release_refuses_ready_protocol_with_failed_independent_audit(tmp_path: Path) -> None:
    observations = [
        _observation(
            index,
            project=f"PXD{index}",
            file_name=f"f{index}",
            sample=f"s{index}",
            lab=f"l{index}",
            instrument=f"i{index}",
            organism=f"o{index}",
            peptide=f"PEP{index}",
            modification=f"m{index}",
            acquisition=f"a{index}",
        )
        for index in range(1, 7)
    ]
    observations[1] = observations[1].model_copy(update={"project_id": "PXD1"})
    catalog = DatasetCatalog(source_batch_dir="/batch", observations=observations)
    suite = plan_split_suite(catalog, seed=8)
    project_plan = suite.protocols["project_disjoint"]
    suite.protocols["project_disjoint"] = project_plan.model_copy(
        update={
            "allocations": [
                allocation.model_copy(update={"split": "train"})
                if allocation.observation_id == "obs-1"
                else allocation.model_copy(update={"split": "test"})
                if allocation.observation_id == "obs-2"
                else allocation
                for allocation in project_plan.allocations
            ]
        }
    )

    try:
        build_dataset_release(
            catalog,
            suite,
            output_dir=tmp_path / "release",
            release_id="bad-release",
            task_spec={"task_type": "denovo"},
        )
    except ValueError as exc:
        assert "failed leakage audit" in str(exc)
    else:
        raise AssertionError("failed leakage audit must block release")


def test_peptide_identity_policy_controls_il_equivalence() -> None:
    observations = [
        _observation(
            index,
            project=f"PXD{index}",
            file_name=f"f{index}",
            sample=f"s{index}",
            lab=f"l{index}",
            instrument=f"i{index}",
            organism="human",
            peptide=("PEPTIDEI" if index == 1 else "PEPTIDEL" if index == 2 else f"PEP{index}"),
            modification="none",
            acquisition="DDA",
        )
        for index in range(1, 7)
    ]
    catalog = DatasetCatalog(source_batch_dir="/batch", observations=observations)

    strict = plan_split_suite(
        catalog,
        seed=15,
        policy=SplitPolicy(peptide_identity_mode="il_equivalent"),
    ).protocols["peptide_disjoint"]
    exact = plan_split_suite(
        catalog,
        seed=15,
        policy=SplitPolicy(peptide_identity_mode="exact"),
    ).protocols["peptide_disjoint"]

    strict_allocations = {row.observation_id: row for row in strict.allocations}
    exact_allocations = {row.observation_id: row for row in exact.allocations}
    assert strict_allocations["obs-1"].component_id == strict_allocations["obs-2"].component_id
    assert strict.identity_policy["peptide_identity_mode"] == "il_equivalent"
    assert exact_allocations["obs-1"].component_id != exact_allocations["obs-2"].component_id

    forged = strict.model_copy(
        update={
            "allocations": [
                allocation.model_copy(update={"split": "train"})
                if allocation.observation_id == "obs-1"
                else allocation.model_copy(update={"split": "test"})
                if allocation.observation_id == "obs-2"
                else allocation
                for allocation in strict.allocations
            ]
        }
    )
    finding = next(
        row for row in audit_split(catalog, forged).findings if row.dimension == "peptide"
    )
    assert finding.status == "fail"
    assert finding.affected_identities == ["peptjdej"]


def test_catalog_contract_blocks_duplicate_observation_ids() -> None:
    observation = _observation(
        1,
        project="PXD1",
        file_name="f1",
        sample="s1",
        lab="l1",
        instrument="i1",
        organism="human",
        peptide="PEPTIDE",
        modification="none",
        acquisition="DDA",
    )
    catalog = DatasetCatalog(
        source_batch_dir="/batch",
        observations=[observation, observation.model_copy()],
    )

    try:
        validate_catalog(catalog)
    except DatasetContractError as exc:
        assert "observation_id" in str(exc)
    else:
        raise AssertionError("duplicate observation identity must block release")


def test_task_label_policy_blocks_unreliable_peptide_labels() -> None:
    observation = _observation(
        1,
        project="PXD1",
        file_name="f1",
        sample="s1",
        lab="l1",
        instrument="i1",
        organism="human",
        peptide="PEPTIDE",
        modification="none",
        acquisition="DDA",
    ).model_copy(update={"q_value": 0.03})

    try:
        validate_catalog(
            DatasetCatalog(source_batch_dir="/batch", observations=[observation]),
            task_spec={"task_type": "denovo"},
        )
    except DatasetContractError as exc:
        assert "q_value_above_threshold:obs-1:0.03" in str(exc)
    else:
        raise AssertionError("an unreliable peptide label must block release")


def test_diverse_denovo_profile_caps_repeated_modified_peptide_deterministically() -> None:
    observations = [
        _observation(
            index,
            project=f"PXD{index}",
            file_name="same-file",
            sample=f"s{index}",
            lab="lab",
            instrument="Q Exactive",
            organism="human",
            peptide="PEPTIDEK",
            modification="phospho",
            acquisition="DDA",
        ).model_copy(
            update={
                "peak_count": 20 + index,
                "total_ion_current": float(index),
                "fragment_coverage": index / 100,
                "search_engines": ["fragpipe", "sage"],
                "engine_q_values": {"fragpipe": 0.005, "sage": 0.006},
            }
        )
        for index in range(1, 13)
    ]
    catalog = curate_denovo_benchmark(
        DatasetCatalog(source_batch_dir="/batch", observations=observations),
        {"task_type": "denovo", "benchmark_profile": DENOVO_BENCHMARK_PROFILE},
    )

    assert len(catalog.observations) == 10
    assert catalog.curation_report["filter_counts"]["peptidoform_frequency_cap"] == 2
    assert [row.representative_rank for row in catalog.observations] == list(range(1, 11))
    assert catalog.observations[0].observation_id == "obs-12"


def test_diverse_denovo_profile_validates_required_coverage() -> None:
    instruments = [
        "Q Exactive HF", "Orbitrap Fusion Lumos", "Orbitrap Eclipse",
        "Bruker timsTOF Pro", "SCIEX TripleTOF 6600",
    ]
    fragmentations = ["HCD", "CID", "EThcD"]
    enzymes = ["Trypsin", "GluC", "AspN", "Chymotrypsin"]
    ptms = ["phosphorylation", "acetylation", "ubiquitination"]
    observations = []
    for index in range(1, 11):
        observations.append(
            _observation(
                index,
                project=f"PXD{index}",
                file_name=f"f{index}",
                sample=f"s{index}",
                lab=f"l{index}",
                instrument=instruments[(index - 1) % len(instruments)],
                organism=("Arabidopsis thaliana" if index == 1 else "E. coli" if index == 2 else "human"),
                peptide=f"PEPTIDE{index}",
                modification=ptms[(index - 1) % len(ptms)],
                acquisition="DDA",
            ).model_copy(
                update={
                    "fragmentation_method": fragmentations[(index - 1) % len(fragmentations)],
                    "lc_gradient_minutes": 30.0 if index % 2 else 120.0,
                    "enzyme": enzymes[(index - 1) % len(enzymes)],
                    "peak_count": 50,
                    "spectrum_mz": [float(value) for value in range(50)],
                    "spectrum_intensity": [float(value + 1) for value in range(50)],
                    "scan_number": str(index),
                    "precursor_mz": 500.0 + index,
                    "charge": 2,
                    "tissue": "whole organism",
                    "isolation_window": 1.6,
                    "resolution": 30000.0,
                    "collision_energy": 30.0,
                    "scan_range": "100-1600 m/z",
                    "modification_sites": [f"3:{ptms[(index - 1) % len(ptms)]}"],
                    "search_engines": ["fragpipe", "sage"],
                    "engine_q_values": {"fragpipe": 0.004, "sage": 0.008},
                    "protein_family_ids": [f"family-{index}"],
                }
            )
        )
    task_spec = {
        "task_type": "denovo",
        "benchmark_profile": DENOVO_BENCHMARK_PROFILE,
    }
    catalog = curate_denovo_benchmark(
        DatasetCatalog(source_batch_dir="/batch", observations=observations),
        task_spec,
    )

    evidence = validate_catalog(catalog, task_spec=task_spec)

    assert evidence["benchmark_profile"]["status"] == "pass"
    assert evidence["benchmark_profile"]["blocking_issues"] == []


def test_modification_identity_policy_can_hold_out_exact_peptidoforms() -> None:
    observations = [
        _observation(
            index,
            project=f"PXD{index}",
            file_name=f"f{index}",
            sample=f"s{index}",
            lab=f"l{index}",
            instrument=f"i{index}",
            organism="human",
            peptide=("SHARED" if index <= 2 else f"PEP{index}"),
            modification=("phospho" if index <= 2 else f"m{index}"),
            acquisition="DDA",
        )
        for index in range(1, 7)
    ]
    catalog = DatasetCatalog(source_batch_dir="/batch", observations=observations)

    plan = plan_split_suite(
        catalog,
        seed=6,
        policy=SplitPolicy(modification_identity_mode="peptidoform"),
    ).protocols["modification_disjoint"]

    by_id = {row.observation_id: row for row in plan.allocations}
    assert by_id["obs-1"].component_id == by_id["obs-2"].component_id
    assert plan.identity_policy["modification_identity_mode"] == "peptidoform"


def test_operations_migration_adds_dataset_tables_without_replacing_existing_tables(
    tmp_path: Path,
) -> None:
    settings = OperationsSettings(
        database_path=tmp_path / "operations.sqlite",
        queue_path=tmp_path / "queue.sqlite",
        artifact_root=tmp_path / "artifacts",
    )
    database = OperationsDatabase(settings)
    try:
        database.migrate()
        tables = set(inspect(database.engine).get_table_names())
    finally:
        database.dispose()

    assert {"jobs", "project_reviews", "batches"} <= tables
    assert {
        "dataset_releases",
        "dataset_split_protocols",
        "dataset_audit_findings",
        "dataset_split_allocations",
    } <= tables


def test_dataset_cli_constructs_release_from_existing_batch(tmp_path: Path) -> None:
    batch_dir = tmp_path / "batch"
    _write_existing_batch(batch_dir)
    task_spec = tmp_path / "task-spec.json"
    task_spec.write_text(
        json.dumps({"task_type": "denovo", "version": 1}),
        encoding="utf-8",
    )

    operations_db = tmp_path / "operations.sqlite"
    result = CliRunner().invoke(
        dataset_cli,
        [
            "release",
            "--batch-dir",
            str(batch_dir),
            "--output-dir",
            str(tmp_path / "release"),
            "--release-id",
            "cli-release",
            "--task-spec",
            str(task_spec),
        ],
        env={
            "AGENT_OPERATIONS_DB": str(operations_db),
            "AGENT_QUEUE_DB": str(tmp_path / "queue.sqlite"),
            "AGENT_OPERATIONS_ARTIFACTS": str(tmp_path / "operations-artifacts"),
        },
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["release_id"] == "cli-release"
    assert (tmp_path / "release" / "release_manifest.json").is_file()
    engine = create_engine(f"sqlite:///{operations_db.as_posix()}")
    with Session(engine) as session:
        assert session.get(DatasetReleaseRow, "cli-release") is not None
    engine.dispose()


def test_dataset_construction_agent_uses_sdk_tools_and_approval_gate() -> None:
    specialist = build_dataset_construction_agent(model="test-model")

    assert specialist.name == "Proteomics Dataset Construction Agent"
    assert [tool.name for tool in specialist.tools] == [
        "inspect_dataset_batch",
        "preview_dataset_split_protocols",
        "queue_dataset_release",
        "get_dataset_construction_job",
    ]
    publish_tool = specialist.tools[-2]
    assert publish_tool.needs_approval is True
