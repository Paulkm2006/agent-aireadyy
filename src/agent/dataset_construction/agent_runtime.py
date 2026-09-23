from __future__ import annotations

import json
from typing import Any

from agent.dataset_construction.benchmark import (
    curate_denovo_benchmark,
    denovo_benchmark_policy,
)
from agent.dataset_construction.ingestion import ingest_existing_batch
from agent.dataset_construction.models import DatasetConstructionJobSpec, SplitPolicy
from agent.dataset_construction.operations import submit_dataset_construction_job
from agent.dataset_construction.splitting import plan_split_suite
from agent.operations.runtime import get_operations_repository


def inspect_dataset_batch(batch_dir: str) -> dict[str, Any]:
    """Inspect an existing Batch and report identities available for fair splitting."""

    catalog = ingest_existing_batch(batch_dir)
    identity_fields = (
        "project_id",
        "file_family_id",
        "lab_id",
        "instrument_id",
        "organism_id",
        "peptide",
        "modification_classes",
        "acquisition_id",
    )
    return {
        "source_batch_dir": catalog.source_batch_dir,
        "observation_count": len(catalog.observations),
        "warnings": catalog.warnings,
        "identity_coverage": {
            field: sum(bool(getattr(row, field)) for row in catalog.observations)
            for field in identity_fields
        },
    }


def preview_dataset_split_protocols(
    batch_dir: str,
    task_type: str = "denovo",
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    benchmark_profile: str = "",
) -> dict[str, Any]:
    """Plan every leakage-aware protocol and return statuses without writing a release."""

    task_spec = {
        "task_type": task_type,
        **({"benchmark_profile": benchmark_profile} if benchmark_profile else {}),
    }
    catalog = curate_denovo_benchmark(
        ingest_existing_batch(batch_dir, task_types=[task_type] if task_type else None),
        task_spec,
    )
    policy = SplitPolicy()
    if denovo_benchmark_policy(task_spec) is not None:
        policy = policy.model_copy(update={"modification_identity_mode": "peptidoform"})
    suite = plan_split_suite(
        catalog,
        ratios=(train_ratio, validation_ratio, test_ratio),
        seed=seed,
        policy=policy,
    )
    return {
        "ratios": suite.ratios,
        "seed": suite.seed,
        "benchmark_profile": catalog.curation_report,
        "protocols": {
            name: {
                "status": plan.status,
                "holdout_identity": plan.holdout_identity,
                "group_count": plan.group_count,
                "missing_identity_count": plan.missing_identity_count,
                "actual_counts": plan.actual_counts,
                "reasons": plan.reasons,
            }
            for name, plan in suite.protocols.items()
        },
    }


def queue_dataset_release(
    batch_dir: str,
    output_dir: str,
    release_id: str,
    task_spec_json: str,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, Any]:
    """Queue an approved immutable release and return its durable job record."""

    task_spec = json.loads(task_spec_json)
    if not isinstance(task_spec, dict):
        raise ValueError("task_spec_json must decode to a JSON object")
    return submit_dataset_construction_job(
        DatasetConstructionJobSpec(
            batch_dir=batch_dir,
            output_dir=output_dir,
            release_id=release_id,
            task_spec=task_spec,
            ratios=(train_ratio, validation_ratio, test_ratio),
            seed=seed,
        )
    )


def get_dataset_construction_job(job_id: str) -> dict[str, Any]:
    """Read durable status, phase, error, progress and release result for one job."""

    snapshot = get_operations_repository().get_job(job_id)
    if snapshot is None:
        raise KeyError(job_id)
    return snapshot


def build_dataset_construction_agent(*, model: Any = None):
    """Build the product's OpenAI Agents SDK specialist for dataset construction."""

    from agents import Agent, ModelSettings, function_tool

    return Agent(
        name="Proteomics Dataset Construction Agent",
        instructions=(
            "Construct scientifically fair proteomics datasets downstream of an existing "
            "Batch. Inspect identity coverage first, preview all protocols, explain every "
            "inconclusive or infeasible protocol, and never silently fall back to a weaker "
            "split. Release submission is durable, resumable, immutable, and requires "
            "explicit approval. After submission, use the status tool instead of "
            "submitting duplicates. For a diversity-controlled de novo benchmark, use "
            "benchmark_profile='diverse_denovo_v1'. Treat missing instrument, "
            "fragmentation, LC gradient, enzyme, PTM, FragPipe/Sage consensus, or "
            "per-engine q-value evidence as a release blocker; report species balance "
            "and overlap instead of inventing metadata."
        ),
        model=model,
        tools=[
            function_tool(inspect_dataset_batch),
            function_tool(preview_dataset_split_protocols),
            function_tool(queue_dataset_release, needs_approval=True),
            function_tool(get_dataset_construction_job),
        ],
        model_settings=ModelSettings(parallel_tool_calls=False),
    )
