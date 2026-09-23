from __future__ import annotations

import json
from pathlib import Path

import typer

from agent.dataset_construction.workflow import (
    construct_dataset_release_from_batch,
    preview_split_suite_from_batch,
)
from agent.operations.database import OperationsDatabase


app = typer.Typer(help="Build immutable leakage-aware datasets from existing Batch outputs.")


@app.command("preview")
def preview_command(
    batch_dir: Path = typer.Option(..., exists=True, file_okay=False),
    task_type: str = typer.Option("denovo"),
    train_ratio: float = typer.Option(0.7),
    validation_ratio: float = typer.Option(0.15),
    test_ratio: float = typer.Option(0.15),
    seed: int = typer.Option(42),
    benchmark_profile: str = typer.Option(""),
) -> None:
    """Plan all protocols without writing or registering a release."""

    suite = preview_split_suite_from_batch(
        batch_dir,
        ratios=(train_ratio, validation_ratio, test_ratio),
        seed=seed,
        task_type=task_type,
        task_spec={
            "task_type": task_type,
            **({"benchmark_profile": benchmark_profile} if benchmark_profile else {}),
        },
    )
    typer.echo(suite.model_dump_json(indent=2))


@app.command("release")
def release_command(
    batch_dir: Path = typer.Option(..., exists=True, file_okay=False),
    output_dir: Path = typer.Option(...),
    release_id: str = typer.Option(...),
    task_spec: Path = typer.Option(..., exists=True, dir_okay=False),
    train_ratio: float = typer.Option(0.7),
    validation_ratio: float = typer.Option(0.15),
    test_ratio: float = typer.Option(0.15),
    seed: int = typer.Option(42),
) -> None:
    """Construct, audit, and freeze all supported split protocols."""

    task_spec_payload = json.loads(task_spec.read_text(encoding="utf-8"))
    if not isinstance(task_spec_payload, dict):
        raise typer.BadParameter("task spec must be a JSON object", param_hint="--task-spec")
    database = OperationsDatabase()
    try:
        database.migrate()
        result = construct_dataset_release_from_batch(
            batch_dir,
            output_dir=output_dir,
            release_id=release_id,
            task_spec=task_spec_payload,
            ratios=(train_ratio, validation_ratio, test_ratio),
            seed=seed,
            engine=database.engine,
        )
    finally:
        database.dispose()
    typer.echo(result.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
