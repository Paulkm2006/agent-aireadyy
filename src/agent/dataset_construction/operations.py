from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from agent.dataset_construction.contracts import validate_catalog
from agent.dataset_construction.benchmark import (
    curate_denovo_benchmark,
    denovo_benchmark_policy,
)
from agent.dataset_construction.identity_ledger import build_identity_ledger
from agent.dataset_construction.ingestion import ingest_existing_batch
from agent.dataset_construction.leakage import audit_split
from agent.dataset_construction.models import DatasetConstructionJobSpec, SplitPolicy
from agent.dataset_construction.release import (
    build_dataset_release,
    register_dataset_release,
    registered_dataset_release,
    sha256_file,
)
from agent.dataset_construction.splitting import plan_split_suite
from agent.operations.repository import OperationsRepository
from agent.operations.runtime import get_operations_repository


class DatasetJobCancelled(RuntimeError):
    pass


def submit_dataset_construction_job(
    spec: DatasetConstructionJobSpec,
    *,
    repository: OperationsRepository | None = None,
    enqueue: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Persist and enqueue one idempotent request through the shared job plane."""

    repository = repository or get_operations_repository()
    batch_dir = Path(spec.batch_dir).resolve()
    output_dir = Path(spec.output_dir).resolve()
    idempotency_key = spec.idempotency_key.strip() or hashlib.sha256(
        json.dumps(
            spec.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    payload = {
        "objective": f"Construct dataset release {spec.release_id}",
        "batch_dir": str(batch_dir),
        "output_dir": str(output_dir),
        "release_id": spec.release_id,
        "task_spec": spec.task_spec,
        "ratios": list(spec.ratios),
        "seed": spec.seed,
        "policy": spec.policy,
    }
    existing = repository.get_job_by_idempotency_key(idempotency_key)
    if existing is not None:
        if repository.get_job_payload(existing["job_id"]) != payload:
            raise ValueError("idempotency_key_conflict")
        return existing
    if not batch_dir.is_dir():
        raise FileNotFoundError("batch_dir_not_found")
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError("release_output_is_not_a_directory")
        if any(output_dir.iterdir()):
            raise FileExistsError("release_output_not_empty")
    job_id = f"dataset_{uuid4().hex}"
    created = repository.create_job(
        job_id=job_id,
        job_type="dataset_construction",
        idempotency_key=idempotency_key,
        payload=payload,
    )
    if created["job_id"] != job_id:
        return created
    if enqueue is None:
        from agent.operations.queue import enqueue_operations_job

        enqueue = enqueue_operations_job
    try:
        queue_task_id = enqueue(job_id)
    except Exception as exc:
        repository.transition_job(
            job_id,
            "failed",
            phase="failed",
            reason="Dataset construction could not enter the durable queue.",
            event_type="job_enqueue_failed",
            level="error",
            error_code=type(exc).__name__,
            error_message=str(exc),
            resumable=True,
        )
        raise
    repository.append_event(
        job_id,
        event_type="job_enqueued",
        actor="queue",
        phase="queued",
        message="Dataset construction job entered the durable queue.",
        payload={"queue_task_id": queue_task_id},
    )
    return repository.get_job(job_id) or created


def _cancel_if_requested(repository: OperationsRepository, job_id: str) -> None:
    if repository.cancel_requested(job_id):
        raise DatasetJobCancelled("dataset_construction_cancelled")


def _event(
    repository: OperationsRepository,
    job_id: str,
    *,
    event_type: str,
    phase: str,
    message: str,
    payload: dict[str, Any] | None = None,
) -> None:
    repository.append_event(
        job_id,
        event_type=event_type,
        actor="dataset-construction-worker",
        phase=phase,
        message=message,
        payload=payload or {},
    )


def _release_result_from_disk(
    output_dir: Path,
    *,
    release_id: str,
    task_spec: dict[str, Any],
    ratios: tuple[float, float, float],
    seed: int,
    repository: OperationsRepository,
) -> dict[str, Any] | None:
    """Reconcile a release committed before a worker process was interrupted."""

    registered = registered_dataset_release(
        release_id,
        engine=repository.database.engine,
    )
    manifest_path = output_dir / "release_manifest.json"
    if registered is None or not manifest_path.is_file():
        return None
    if Path(str(registered["release_dir"])).resolve() != output_dir:
        return None
    if sha256_file(manifest_path) != registered["manifest_sha256"]:
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("release_id") != release_id:
        return None
    if manifest.get("task_spec") != task_spec:
        return None
    if tuple(manifest.get("ratios") or ()) != ratios or int(manifest.get("seed", -1)) != seed:
        return None
    known_files = {
        "catalog_parquet": output_dir / "catalog" / "observations.parquet",
        "task_spec_snapshot_json": output_dir / "task_spec_snapshot.json",
        "catalog_contract_json": output_dir / "validation" / "catalog_contract.json",
        "benchmark_quality_report_json": output_dir / "validation" / "benchmark_quality_report.json",
        "benchmark_index_parquet": output_dir / "catalog" / "benchmark_index.parquet",
        "identity_ledger_parquet": output_dir / "identity_ledger" / "assertions.parquet",
        "identity_ledger_summary_json": output_dir / "identity_ledger" / "summary.json",
        "prov_json": output_dir / "provenance" / "prov.json",
        "ro_crate_metadata_json": output_dir / "ro-crate-metadata.json",
        "release_manifest_json": manifest_path,
        "checksums_sha256": output_dir / "checksums.sha256",
    }
    if not all(path.is_file() for path in known_files.values()):
        return None
    checksums: dict[str, str] = {}
    for line in known_files["checksums_sha256"].read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if separator and digest and relative:
            checksums[relative] = digest
    for path in output_dir.rglob("*"):
        if path.is_file() and path != known_files["checksums_sha256"]:
            relative = path.relative_to(output_dir).as_posix()
            if checksums.get(relative) != sha256_file(path):
                return None
    return {
        "release_id": release_id,
        "observation_count": int(registered["observation_count"]),
        "protocol_statuses": {
            name: str(details.get("audit_status") or "")
            for name, details in dict(manifest.get("protocols") or {}).items()
            if isinstance(details, dict)
        },
        **{key: str(path) for key, path in known_files.items()},
    }


def execute_dataset_construction_job(
    job_id: str,
    *,
    repository: OperationsRepository | None = None,
) -> dict[str, Any]:
    """Execute one durable dataset job with resumable phase boundaries."""

    repository = repository or get_operations_repository()
    snapshot = repository.get_job(job_id)
    if snapshot is None:
        raise KeyError(job_id)
    if snapshot["job_type"] != "dataset_construction":
        raise ValueError(f"unsupported_job_type:{snapshot['job_type']}")
    if snapshot["status"] == "completed":
        return dict(snapshot.get("result") or {})
    if snapshot["status"] == "cancelled" and snapshot["cancel_requested"]:
        return {"job_id": job_id, "status": "cancelled"}
    payload = repository.get_job_payload(job_id)
    output_dir = Path(str(payload["output_dir"])).resolve()
    task_spec = dict(payload["task_spec"])
    ratios = cast(
        tuple[float, float, float],
        tuple(float(value) for value in payload.get("ratios") or (0.7, 0.15, 0.15)),
    )
    seed = int(payload.get("seed", 42))
    staging_dir = output_dir.with_name(f".{output_dir.name}.{job_id}.{uuid4().hex}.staging")
    try:
        repository.transition_job(
            job_id,
            "searching",
            phase="ingesting",
            reason="Dataset worker claimed the persisted job.",
            event_type="dataset_ingestion_started",
            resumable=True,
        )
        recovered = _release_result_from_disk(
            output_dir,
            release_id=str(payload["release_id"]),
            task_spec=task_spec,
            ratios=ratios,
            seed=seed,
            repository=repository,
        )
        if recovered is not None:
            repository.transition_job(
                job_id,
                "finalizing",
                phase="finalizing",
                reason="Reconciling an already committed immutable release.",
                event_type="dataset_release_recovered",
                resumable=True,
            )
            repository.set_job_result(job_id, recovered)
            repository.transition_job(
                job_id,
                "completed",
                phase="completed",
                reason="Recovered release and durable job state are consistent.",
                event_type="dataset_release_completed",
                payload={
                    "release_id": payload["release_id"],
                    "observation_count": recovered["observation_count"],
                },
                resumable=False,
            )
            return recovered
        _cancel_if_requested(repository, job_id)
        task_type = str(task_spec.get("task_type") or "").strip()
        catalog = ingest_existing_batch(
            payload["batch_dir"],
            task_types=[task_type] if task_type else None,
        )
        catalog = curate_denovo_benchmark(catalog, task_spec)
        _event(
            repository,
            job_id,
            event_type="dataset_ingestion_completed",
            phase="ingesting",
            message="Existing Batch artifacts were ingested and benchmark curation was applied when requested.",
            payload={
                "observation_count": len(catalog.observations),
                "warning_count": len(catalog.warnings),
                "benchmark_profile": catalog.curation_report,
            },
        )
        _cancel_if_requested(repository, job_id)
        contract = validate_catalog(catalog, task_spec=task_spec)
        _event(repository, job_id, event_type="dataset_contract_validated", phase="validating", message="Catalog and task label contract passed.", payload=contract)
        policy = SplitPolicy.model_validate(payload.get("policy") or {})
        if denovo_benchmark_policy(task_spec) is not None:
            policy = policy.model_copy(
                update={"modification_identity_mode": "peptidoform"}
            )
        ledger = build_identity_ledger(catalog, policy=policy)
        _event(repository, job_id, event_type="dataset_identity_ledger_built", phase="identity_ledger", message="Identity provenance ledger was built.", payload={"observation_count": ledger.observation_count, "dimension_count": len(ledger.dimensions), "incomplete_dimensions": [row.dimension for row in ledger.dimensions if row.missing_count]})
        _cancel_if_requested(repository, job_id)
        suite = plan_split_suite(
            catalog,
            ratios=ratios,
            seed=seed,
            policy=policy,
        )
        _event(repository, job_id, event_type="dataset_split_suite_planned", phase="planning", message="All leakage-aware split protocols were planned.", payload={"protocol_statuses": {name: plan.status for name, plan in suite.protocols.items()}})
        audits = {name: audit_split(catalog, plan) for name, plan in suite.protocols.items()}
        _event(repository, job_id, event_type="dataset_leakage_audited", phase="auditing", message="Independent leakage audit completed.", payload={"audit_statuses": {name: audit.status for name, audit in audits.items()}})
        _cancel_if_requested(repository, job_id)
        repository.transition_job(job_id, "finalizing", phase="finalizing", reason="Freezing immutable dataset release.", event_type="dataset_release_started", resumable=True)
        release = build_dataset_release(catalog, suite, output_dir=staging_dir, release_id=str(payload["release_id"]), task_spec=task_spec, engine=None)
        if output_dir.exists():
            if any(output_dir.iterdir()):
                raise ValueError(f"release output already exists: {output_dir}")
            output_dir.rmdir()
        staging_dir.replace(output_dir)
        files = {
            key: str(output_dir / Path(path).relative_to(staging_dir))
            for key, path in release.files.items()
        }
        result = {
            "release_id": release.release_id,
            "observation_count": len(catalog.observations),
            "protocol_statuses": release.protocol_statuses,
            **files,
        }
        try:
            register_dataset_release(
                catalog,
                suite,
                release_dir=output_dir,
                release_id=str(payload["release_id"]),
                task_spec=task_spec,
                engine=repository.database.engine,
            )
        except Exception:
            shutil.rmtree(output_dir, ignore_errors=True)
            raise
        repository.set_job_result(job_id, result)
        repository.transition_job(job_id, "completed", phase="completed", reason="Immutable audited dataset release completed.", event_type="dataset_release_completed", payload={"release_id": release.release_id, "observation_count": len(catalog.observations)}, resumable=False)
        return result
    except DatasetJobCancelled:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        current = repository.get_job(job_id)
        if current and current["status"] != "cancelled":
            repository.transition_job(job_id, "cancelled", phase="cancelled", reason="Dataset construction stopped at a safe phase boundary.", event_type="dataset_job_cancelled", resumable=True)
        return {"job_id": job_id, "status": "cancelled"}
    except Exception as exc:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        current = repository.get_job(job_id)
        if current and current["status"] not in {"failed", "cancelled"}:
            repository.transition_job(job_id, "failed", phase="failed", reason="Dataset construction failed; no release was published.", event_type="dataset_job_failed", level="error", error_code=type(exc).__name__, error_message=str(exc), resumable=True)
        raise
