from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import Engine

from agent.dataset_construction.ingestion import ingest_existing_batch
from agent.dataset_construction.benchmark import (
    curate_denovo_benchmark,
    denovo_benchmark_policy,
)
from agent.dataset_construction.models import DatasetReleaseResult, SplitPolicy, SplitSuite
from agent.dataset_construction.release import build_dataset_release
from agent.dataset_construction.splitting import plan_split_suite


def preview_split_suite_from_batch(
    batch_dir: str | Path,
    *,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    policy: SplitPolicy | None = None,
    task_type: str | None = None,
    task_spec: dict[str, Any] | None = None,
) -> SplitSuite:
    """Read an existing Batch and compute all split protocols without publishing."""

    catalog = ingest_existing_batch(
        batch_dir,
        task_types=[task_type] if task_type else None,
    )
    catalog = curate_denovo_benchmark(catalog, task_spec)
    if denovo_benchmark_policy(task_spec) is not None:
        policy = (policy or SplitPolicy()).model_copy(
            update={"modification_identity_mode": "peptidoform"}
        )
    return plan_split_suite(catalog, ratios=ratios, seed=seed, policy=policy)


def construct_dataset_release_from_batch(
    batch_dir: str | Path,
    *,
    output_dir: str | Path,
    release_id: str,
    task_spec: dict[str, Any],
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    policy: SplitPolicy | None = None,
    engine: Engine | None = None,
) -> DatasetReleaseResult:
    """Run the deterministic ingest-plan-audit-release workflow downstream of Batch."""

    task_type = str(task_spec.get("task_type") or "").strip()
    catalog = ingest_existing_batch(
        batch_dir,
        task_types=[task_type] if task_type else None,
    )
    catalog = curate_denovo_benchmark(catalog, task_spec)
    if denovo_benchmark_policy(task_spec) is not None:
        policy = (policy or SplitPolicy()).model_copy(
            update={"modification_identity_mode": "peptidoform"}
        )
    suite = plan_split_suite(catalog, ratios=ratios, seed=seed, policy=policy)
    return build_dataset_release(
        catalog,
        suite,
        output_dir=output_dir,
        release_id=release_id,
        task_spec=task_spec,
        engine=engine,
    )
