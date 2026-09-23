from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from agent.dataset_construction.contracts import validate_catalog
from agent.dataset_construction.benchmark import benchmark_overlap_report
from agent.dataset_construction.identity_ledger import build_identity_ledger
from agent.dataset_construction.leakage import audit_split
from agent.dataset_construction.models import (
    DatasetCatalog,
    DatasetReleaseResult,
    SplitSuite,
)
from agent.dataset_construction.persistence import (
    DatasetAuditFindingRow,
    DatasetProtocolRow,
    DatasetReleaseRow,
    DatasetSplitAllocationRow,
)
from agent.dataset_construction.provenance import write_prov_document, write_ro_crate


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = [
        {
            key: (
                json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, dict)
                else value
            )
            for key, value in row.items()
        }
        for row in rows
    ]
    pq.write_table(pa.Table.from_pylist(normalized), path, compression="zstd")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def register_dataset_release(
    catalog: DatasetCatalog,
    suite: SplitSuite,
    *,
    release_dir: str | Path,
    release_id: str,
    task_spec: dict[str, Any],
    engine: Engine,
) -> None:
    """Register an already frozen release in one SQL transaction."""

    root = Path(release_dir).resolve()
    manifest_path = root / "release_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"release manifest does not exist: {manifest_path}")
    audits = {
        protocol: audit_split(catalog, plan)
        for protocol, plan in suite.protocols.items()
    }
    with Session(engine) as session:
        if session.scalar(
            select(DatasetReleaseRow.release_id).where(
                DatasetReleaseRow.release_id == release_id
            )
        ):
            raise ValueError(f"dataset release already exists: {release_id}")
        session.add(
            DatasetReleaseRow(
                release_id=release_id,
                status="released",
                task_spec=task_spec,
                source_batch_dir=catalog.source_batch_dir,
                observation_count=len(catalog.observations),
                release_dir=str(root),
                manifest_sha256=sha256_file(manifest_path),
            )
        )
        session.flush()
        for protocol, plan in suite.protocols.items():
            audit = audits[protocol]
            session.add(
                DatasetProtocolRow(
                    release_id=release_id,
                    protocol=protocol,
                    split_status=plan.status,
                    audit_status=audit.status,
                    group_count=plan.group_count,
                    allocation_count=len(plan.allocations),
                )
            )
            session.add_all(
                DatasetAuditFindingRow(
                    release_id=release_id,
                    protocol=protocol,
                    dimension=finding.dimension,
                    status=finding.status,
                    overlap_count=finding.overlap_count,
                    missing_count=finding.missing_count,
                    severity=finding.severity,
                    evidence={
                        "affected_identities": finding.affected_identities,
                        "affected_observation_ids": finding.affected_observation_ids,
                    },
                )
                for finding in audit.findings
            )
            session.add_all(
                DatasetSplitAllocationRow(
                    release_id=release_id,
                    protocol=protocol,
                    observation_id=allocation.observation_id,
                    component_id=allocation.component_id,
                    split=allocation.split,
                )
                for allocation in plan.allocations
            )
        session.commit()


def registered_dataset_release(
    release_id: str,
    *,
    engine: Engine,
) -> dict[str, Any] | None:
    """Return the small persisted release projection used for crash recovery."""

    with Session(engine) as session:
        row = session.get(DatasetReleaseRow, release_id)
        if row is None:
            return None
        return {
            "release_id": row.release_id,
            "release_dir": row.release_dir,
            "observation_count": row.observation_count,
            "manifest_sha256": row.manifest_sha256,
        }


def build_dataset_release(
    catalog: DatasetCatalog,
    suite: SplitSuite,
    *,
    output_dir: str | Path,
    release_id: str,
    task_spec: dict[str, Any],
    engine: Engine | None = None,
) -> DatasetReleaseResult:
    """Freeze catalog, split manifests and independent audits into one release."""

    root = Path(output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"release output directory already contains files: {root}")
    if engine is not None:
        with Session(engine) as session:
            if session.scalar(
                select(DatasetReleaseRow.release_id).where(
                    DatasetReleaseRow.release_id == release_id
                )
            ):
                raise ValueError(f"dataset release already exists: {release_id}")
    contract_evidence = validate_catalog(catalog, task_spec=task_spec)
    audits = {
        protocol: audit_split(catalog, plan)
        for protocol, plan in suite.protocols.items()
    }
    for protocol, plan in suite.protocols.items():
        audit = audits[protocol]
        if plan.status == "ready" and audit.status != "pass":
            raise ValueError(
                f"protocol {protocol} failed leakage audit with status {audit.status}"
            )
    root.mkdir(parents=True, exist_ok=True)
    catalog_path = root / "catalog" / "observations.parquet"
    _write_parquet(
        catalog_path,
        [row.model_dump(mode="json") for row in catalog.observations],
    )
    task_spec_path = root / "task_spec_snapshot.json"
    _write_json(task_spec_path, task_spec)
    contract_path = root / "validation" / "catalog_contract.json"
    _write_json(contract_path, contract_evidence)
    benchmark_report_path = root / "validation" / "benchmark_quality_report.json"
    strict_benchmark = bool(catalog.curation_report)
    benchmark_report = {
        "status": catalog.curation_report.get("status", "not_requested"),
        "curation": catalog.curation_report,
        "overlap": benchmark_overlap_report(catalog, suite),
        "provenance_pipeline": (
            [
                "raw_file",
                "peaklist_mgf",
                "fragpipe_and_sage_search",
                "one_percent_fdr_consensus",
                "spectrum_quality_control",
                "representative_scan_cap",
                "leakage_aware_split",
                "immutable_release",
            ]
            if strict_benchmark
            else ["existing_batch", "catalog_contract", "leakage_aware_split", "immutable_release"]
        ),
    }
    _write_json(benchmark_report_path, benchmark_report)
    benchmark_index_path = root / "catalog" / "benchmark_index.parquet"
    observations_by_id = {
        row.observation_id: row.model_dump(mode="json")
        for row in catalog.observations
    }
    _write_parquet(
        benchmark_index_path,
        [
            {
                **observations_by_id[allocation.observation_id],
                "protocol": protocol,
                "split": allocation.split,
                "component_id": allocation.component_id,
            }
            for protocol, plan in suite.protocols.items()
            for allocation in plan.allocations
        ],
    )
    identity_ledger = build_identity_ledger(catalog, policy=suite.policy)
    identity_ledger_path = root / "identity_ledger" / "assertions.parquet"
    _write_parquet(
        identity_ledger_path,
        [row.model_dump(mode="json") for row in identity_ledger.assertions],
    )
    identity_summary_path = root / "identity_ledger" / "summary.json"
    _write_json(
        identity_summary_path,
        {
            "schema_version": identity_ledger.schema_version,
            "source_batch_dir": identity_ledger.source_batch_dir,
            "observation_count": identity_ledger.observation_count,
            "dimensions": [
                row.model_dump(mode="json") for row in identity_ledger.dimensions
            ],
        },
    )
    protocol_manifest: dict[str, Any] = {}
    for protocol, plan in suite.protocols.items():
        split_path = root / "split_manifests" / f"{protocol}.parquet"
        _write_parquet(
            split_path,
            [row.model_dump(mode="json") for row in plan.allocations],
        )
        audit = audits[protocol]
        audit_path = root / "audits" / f"{protocol}.json"
        _write_json(audit_path, audit.model_dump(mode="json"))
        protocol_manifest[protocol] = {
            "split_status": plan.status,
            "audit_status": audit.status,
            "group_count": plan.group_count,
            "allocation_count": len(plan.allocations),
        }
    manifest_path = root / "release_manifest.json"
    _write_json(
        manifest_path,
        {
            "release_id": release_id,
            "immutable": True,
            "source_batch_dir": catalog.source_batch_dir,
            "observation_count": len(catalog.observations),
            "task_spec": task_spec,
            "benchmark": catalog.curation_report,
            "ratios": suite.ratios,
            "seed": suite.seed,
            "split_policy": suite.policy.model_dump(mode="json"),
            "protocols": protocol_manifest,
        },
    )
    prov_path = write_prov_document(
        root,
        release_id=release_id,
        catalog=catalog,
        suite=suite,
        audits=audits,
    )
    crate_path = write_ro_crate(root, release_id=release_id, task_spec=task_spec)
    checksums_path = root / "checksums.sha256"
    checksums_path.write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n"
            for path in _relative_files(root)
            if path != checksums_path
        ),
        encoding="utf-8",
    )
    if engine is not None:
        register_dataset_release(
            catalog,
            suite,
            release_dir=root,
            release_id=release_id,
            task_spec=task_spec,
            engine=engine,
        )
    return DatasetReleaseResult(
        release_id=release_id,
        status="released",
        protocol_statuses={
            protocol: audit.status for protocol, audit in audits.items()
        },
        files={
            "catalog_parquet": str(catalog_path),
            "task_spec_snapshot_json": str(task_spec_path),
            "catalog_contract_json": str(contract_path),
            "benchmark_quality_report_json": str(benchmark_report_path),
            "benchmark_index_parquet": str(benchmark_index_path),
            "identity_ledger_parquet": str(identity_ledger_path),
            "identity_ledger_summary_json": str(identity_summary_path),
            "prov_json": str(prov_path),
            "ro_crate_metadata_json": str(crate_path),
            "release_manifest_json": str(manifest_path),
            "checksums_sha256": str(checksums_path),
        },
    )
