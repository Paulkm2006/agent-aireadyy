from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from agent.discovery.models import DatasetManifest, DiscoveredFile
from agent.discovery.pipeline_handoff import HANDOFF_FILE_ROLES
from agent.discovery.task_profiles import TaskProfile, get_task_profile
from agent.discovery.task_readiness import annotate_manifest_task_readiness
from agent.models import JsonModel
from agent.utils import write_json


TaskBuildSelection = Literal["auto", "task_ready", "usable", "valid", "review", "all"]
TaskCandidateTier = Literal[
    "label_generation_candidate",
    "future_task_candidate",
    "review_before_use",
    "not_candidate",
]

USABLE_VALIDITY = {"valid", "weak_keep"}
TASK_READY_STATUSES = {"ready"}
PIPELINE_ELIGIBLE_STATUSES = {"ready", "weak_ready"}


def _instrument_vendor(names: list[str]) -> str | None:
    text = " ".join(names).casefold()
    if any(token in text for token in ("q exactive", "orbitrap", "thermo")):
        return "Thermo Fisher Scientific"
    if any(token in text for token in ("timstof", "bruker")):
        return "Bruker"
    if any(token in text for token in ("sciex", "tripletof")):
        return "SCIEX"
    return None

TASK_BUILD_COLUMNS = [
    "run_id",
    "task_type",
    "task_profile",
    "implementation_status",
    "repository",
    "project_accession",
    "project_title",
    "file_name",
    "download_url",
    "file_type",
    "file_role",
    "species",
    "species_policy",
    "canonical_species",
    "organism_taxon_id",
    "ptm_type",
    "ptm_subtype",
    "ptm_evidence_terms",
    "ptm_enrichment_methods",
    "semantic_metadata_confidence",
    "semantic_interpretation_trace",
    "modification_scope",
    "immunopeptide_scope",
    "hla_class",
    "hla_alleles",
    "immunopeptide_evidence_terms",
    "immunopeptide_enrichment_methods",
    "immunopeptide_metadata_confidence",
    "labeling_strategy",
    "validity_status",
    "task_readiness_status",
    "candidate_tier",
    "recommended_entrypoint",
    "recommended_run_mode",
    "next_pipeline_steps",
    "ai_ready_target_schema",
    "required_labels",
    "missing_task_requirements",
    "task_build_reasons",
    "trust_score",
    "file_score",
    "evidence_level",
    "sdrf_match_status",
    "tissue",
    "enzyme",
    "instrument_names",
    "instrument_vendor",
    "instrument_families",
    "fragmentation_methods",
    "acquisition_mode",
    "isolation_window",
    "resolution",
    "collision_energy",
    "scan_range",
    "lc_gradient_minutes",
]


class TaskBuildFile(JsonModel):
    run_id: str | None = None
    task_type: str
    task_profile: str
    implementation_status: str
    repository: str = "pride"
    project_accession: str
    project_title: str | None = None
    file_name: str
    download_url: str | None = None
    file_type: str
    file_role: str
    species: list[str] = Field(default_factory=list)
    species_policy: str = "open"
    canonical_species: list[str] = Field(default_factory=list)
    organism_taxon_id: list[str] = Field(default_factory=list)
    ptm_type: str | None = None
    ptm_subtype: str | None = None
    ptm_evidence_terms: list[str] = Field(default_factory=list)
    ptm_enrichment_methods: list[str] = Field(default_factory=list)
    semantic_metadata_confidence: float = 0.0
    semantic_interpretation_trace: list[str] = Field(default_factory=list)
    modification_scope: str | None = None
    immunopeptide_scope: str | None = None
    hla_class: list[str] = Field(default_factory=list)
    hla_alleles: list[str] = Field(default_factory=list)
    immunopeptide_evidence_terms: list[str] = Field(default_factory=list)
    immunopeptide_enrichment_methods: list[str] = Field(default_factory=list)
    immunopeptide_metadata_confidence: float = 0.0
    labeling_strategy: str | None = None
    validity_status: str
    task_readiness_status: str | None = None
    candidate_tier: TaskCandidateTier
    recommended_entrypoint: str
    recommended_run_mode: str
    next_pipeline_steps: list[str] = Field(default_factory=list)
    ai_ready_target_schema: str
    required_labels: list[str] = Field(default_factory=list)
    missing_task_requirements: list[str] = Field(default_factory=list)
    task_build_reasons: list[str] = Field(default_factory=list)
    trust_score: float = 0.0
    file_score: float = 0.0
    evidence_level: str = "unknown"
    sdrf_match_status: str = "not_checked"
    tissue: str | None = None
    enzyme: str | None = None
    instrument_names: list[str] = Field(default_factory=list)
    instrument_vendor: str | None = None
    instrument_families: list[str] = Field(default_factory=list)
    fragmentation_methods: list[str] = Field(default_factory=list)
    acquisition_mode: str | None = None
    isolation_window: float | None = None
    resolution: float | None = None
    collision_energy: float | None = None
    scan_range: str | None = None
    lc_gradient_minutes: float | None = None


class TaskBuildPlan(JsonModel):
    run_id: str | None = None
    task_type: str
    task_profile: str
    implementation_status: str
    selection: TaskBuildSelection
    resolved_selection: TaskBuildSelection
    target_schema: str
    required_input_files: list[str] = Field(default_factory=list)
    required_labels: list[str] = Field(default_factory=list)
    required_metadata: list[str] = Field(default_factory=list)
    preferred_acquisition: list[str] = Field(default_factory=list)
    preferred_fragmentation: list[str] = Field(default_factory=list)
    next_pipeline_steps: list[str] = Field(default_factory=list)
    quality_gate: list[str] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    files: list[TaskBuildFile] = Field(default_factory=list)


def build_task_build_plan(
    manifest: DatasetManifest,
    task_type: str,
    *,
    selection: TaskBuildSelection = "auto",
    max_files: int | None = None,
) -> TaskBuildPlan:
    profile = get_task_profile(task_type)
    manifest = _ensure_task_readiness(manifest, profile)
    resolved = _resolve_selection(profile, selection)
    files = _select_files(manifest.files, selection=resolved, max_files=max_files)
    rows = [_row_from_file(file, profile=profile, run_id=manifest.run_id) for file in files]
    summary = _summary(manifest, profile=profile, rows=rows, selection=selection, resolved=resolved)
    return TaskBuildPlan(
        run_id=manifest.run_id,
        task_type=profile.task_type,
        task_profile=profile.display_name,
        implementation_status=profile.implementation_status,
        selection=selection,
        resolved_selection=resolved,
        target_schema=profile.ai_ready_target_schema,
        required_input_files=profile.required_input_files,
        required_labels=profile.required_labels,
        required_metadata=profile.required_metadata,
        preferred_acquisition=profile.preferred_acquisition,
        preferred_fragmentation=profile.preferred_fragmentation,
        next_pipeline_steps=profile.next_pipeline_steps,
        quality_gate=profile.quality_gate,
        summary=summary,
        files=rows,
    )


def write_task_build_plan(
    manifest: DatasetManifest,
    output_dir: str | Path,
    task_type: str,
    *,
    selection: TaskBuildSelection = "auto",
    max_files: int | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = build_task_build_plan(manifest, task_type, selection=selection, max_files=max_files)
    paths = {
        "task_build_plan": output_dir / "discovery_task_build_plan.json",
        "task_build_files": output_dir / "discovery_task_build_files.csv",
        "task_build_summary": output_dir / "task_build_summary.json",
        "task_build_candidates": output_dir / "task_build_candidates.txt",
        "ai_ready_schema_requirements": output_dir / "ai_ready_schema_requirements.json",
    }
    write_json(paths["task_build_plan"], plan.model_dump(mode="json"))
    write_json(paths["task_build_summary"], plan.summary)
    write_json(
        paths["ai_ready_schema_requirements"],
        {
            "task_type": plan.task_type,
            "task_profile": plan.task_profile,
            "implementation_status": plan.implementation_status,
            "target_schema": plan.target_schema,
            "required_input_files": plan.required_input_files,
            "required_labels": plan.required_labels,
            "required_metadata": plan.required_metadata,
            "next_pipeline_steps": plan.next_pipeline_steps,
            "quality_gate": plan.quality_gate,
            "notes": [
                "Discovery can identify candidate inputs and missing labels, but does not generate final AI-ready parquet.",
                "Files listed as candidates still need downstream search/export before model training.",
            ],
        },
    )
    _write_task_build_csv(paths["task_build_files"], plan.files)
    _write_candidate_file_names(paths["task_build_candidates"], plan.files)
    return paths


def _ensure_task_readiness(manifest: DatasetManifest, profile: TaskProfile) -> DatasetManifest:
    if all(file.task_type == profile.task_type for file in manifest.files if file.task_type):
        if any(file.task_type == profile.task_type for file in manifest.files):
            return manifest
    return annotate_manifest_task_readiness(manifest, profile.task_type)


def _resolve_selection(profile: TaskProfile, selection: TaskBuildSelection) -> TaskBuildSelection:
    if selection != "auto":
        return selection
    if profile.implementation_status == "active":
        return "task_ready"
    return "usable"


def _select_files(
    files: list[DiscoveredFile],
    *,
    selection: TaskBuildSelection,
    max_files: int | None,
) -> list[DiscoveredFile]:
    if selection == "task_ready":
        selected = [file for file in files if file.task_readiness_status in PIPELINE_ELIGIBLE_STATUSES]
    elif selection == "usable":
        selected = [file for file in files if file.validity_status in USABLE_VALIDITY]
    elif selection == "valid":
        selected = [file for file in files if file.validity_status == "valid"]
    elif selection == "review":
        selected = [file for file in files if file.validity_status == "needs_review" or file.needs_review]
    elif selection == "all":
        selected = list(files)
    elif selection == "auto":
        raise ValueError("Selection must be resolved before file selection.")
    else:
        raise ValueError(f"Unsupported task build selection: {selection}")

    selected = sorted(
        selected,
        key=lambda file: (
            -float(file.trust_score or 0.0),
            -float(file.file_score or 0.0),
            file.project_accession,
            file.file_name,
        ),
    )
    if max_files is not None:
        selected = selected[:max_files]
    return selected


def _row_from_file(file: DiscoveredFile, *, profile: TaskProfile, run_id: str | None) -> TaskBuildFile:
    tier, entrypoint, run_mode, reasons = _task_build_decision(file, profile)
    return TaskBuildFile(
        run_id=run_id,
        task_type=profile.task_type,
        task_profile=profile.display_name,
        implementation_status=profile.implementation_status,
        repository=file.repository,
        project_accession=file.project_accession,
        project_title=file.project_title,
        file_name=file.file_name,
        download_url=file.download_url,
        file_type=file.file_type,
        file_role=file.file_role,
        species=file.species,
        species_policy=file.species_policy,
        canonical_species=file.canonical_species,
        organism_taxon_id=file.organism_taxon_id,
        ptm_type=file.ptm_type,
        ptm_subtype=file.ptm_subtype,
        ptm_evidence_terms=file.ptm_evidence_terms,
        ptm_enrichment_methods=file.ptm_enrichment_methods,
        semantic_metadata_confidence=file.semantic_metadata_confidence,
        semantic_interpretation_trace=file.semantic_interpretation_trace,
        modification_scope=file.modification_scope,
        immunopeptide_scope=file.immunopeptide_scope,
        hla_class=file.hla_class,
        hla_alleles=file.hla_alleles,
        immunopeptide_evidence_terms=file.immunopeptide_evidence_terms,
        immunopeptide_enrichment_methods=file.immunopeptide_enrichment_methods,
        immunopeptide_metadata_confidence=file.immunopeptide_metadata_confidence,
        labeling_strategy=file.labeling_strategy,
        validity_status=file.validity_status,
        task_readiness_status=file.task_readiness_status,
        candidate_tier=tier,
        recommended_entrypoint=entrypoint,
        recommended_run_mode=run_mode,
        next_pipeline_steps=file.next_pipeline_steps or profile.next_pipeline_steps,
        ai_ready_target_schema=file.ai_ready_target_schema or profile.ai_ready_target_schema,
        required_labels=profile.required_labels,
        missing_task_requirements=file.missing_task_requirements,
        task_build_reasons=reasons,
        trust_score=file.trust_score,
        file_score=file.file_score,
        evidence_level=file.evidence_level,
        sdrf_match_status=file.sdrf_match_status,
        tissue=file.tissue,
        enzyme=file.enzyme,
        instrument_names=file.instrument_names,
        instrument_vendor=file.instrument_vendor or _instrument_vendor(file.instrument_names),
        instrument_families=file.instrument_families,
        fragmentation_methods=file.fragmentation_methods,
        acquisition_mode=file.acquisition_mode,
        isolation_window=file.isolation_window,
        resolution=file.resolution,
        collision_energy=file.collision_energy,
        scan_range=file.scan_range,
        lc_gradient_minutes=file.lc_gradient_minutes,
    )


def _task_build_decision(file: DiscoveredFile, profile: TaskProfile) -> tuple[TaskCandidateTier, str, str, list[str]]:
    reasons: list[str] = []
    can_run_upstream = (
        file.validity_status in USABLE_VALIDITY
        and not file.needs_review
        and file.file_role in HANDOFF_FILE_ROLES
    )
    if file.validity_status == "exclude":
        return "not_candidate", "not_available", "skip", ["excluded_by_discovery_validity"]
    if file.validity_status == "needs_review" or file.needs_review:
        return "review_before_use", "manual_review", "review", ["discovery_needs_review"]
    if file.file_role not in HANDOFF_FILE_ROLES:
        return "not_candidate", "not_available", "skip", [f"not_raw_or_peaklist:{file.file_role}"]

    if not can_run_upstream:
        reasons.append(f"validity_not_usable:{file.validity_status}")
        return "not_candidate", "not_available", "skip", reasons

    if profile.implementation_status != "active":
        reasons.append("task_exporter_not_implemented_yet")
        if file.task_readiness_status == "not_ready":
            reasons.extend(file.task_readiness_reasons)
        return "future_task_candidate", "batch_parameters", "parameters", _dedupe(reasons)

    if file.task_readiness_status not in PIPELINE_ELIGIBLE_STATUSES:
        reasons.append(f"task_not_ready:{file.task_readiness_status or 'not_evaluated'}")
        reasons.extend(file.task_readiness_reasons)
        return "not_candidate", "not_available", "skip", _dedupe(reasons)

    if file.task_readiness_status == "weak_ready":
        reasons.append("requires_downstream_label_generation")
    if file.validity_status == "weak_keep":
        reasons.append("weak_keep_should_be_limited_until_outcome_validated")
    if not file.download_url:
        reasons.append("download_url_missing_for_later_prepare")
    return "label_generation_candidate", "batch_parameters", "parameters", _dedupe(reasons)


def _summary(
    manifest: DatasetManifest,
    *,
    profile: TaskProfile,
    rows: list[TaskBuildFile],
    selection: TaskBuildSelection,
    resolved: TaskBuildSelection,
) -> dict[str, Any]:
    tier_counts = Counter(row.candidate_tier for row in rows)
    entrypoint_counts = Counter(row.recommended_entrypoint for row in rows)
    readiness_counts = Counter(row.task_readiness_status or "not_set" for row in rows)
    missing_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    project_counts = Counter(row.project_accession for row in rows)
    for row in rows:
        missing_counts.update(row.missing_task_requirements)
        reason_counts.update(row.task_build_reasons)

    candidate_count = tier_counts.get("label_generation_candidate", 0) + tier_counts.get("future_task_candidate", 0)
    if profile.implementation_status != "active":
        next_step = "implement_task_exporter_before_large_scale"
    elif candidate_count:
        next_step = "run_batch_parameters_then_downstream_label_export"
    else:
        next_step = "refine_discovery_or_review_candidates"

    return {
        "run_id": manifest.run_id,
        "task_type": profile.task_type,
        "task_profile": profile.display_name,
        "implementation_status": profile.implementation_status,
        "selection": selection,
        "resolved_selection": resolved,
        "manifest_file_count": len(manifest.files),
        "selected_files": len(rows),
        "candidate_files": candidate_count,
        "project_count": len(project_counts),
        "candidate_tier_counts": dict(sorted(tier_counts.items())),
        "recommended_entrypoint_counts": dict(sorted(entrypoint_counts.items())),
        "task_readiness_status_counts": dict(sorted(readiness_counts.items())),
        "missing_requirement_counts": dict(sorted(missing_counts.items())),
        "task_build_reason_counts": dict(sorted(reason_counts.items())),
        "target_schema": profile.ai_ready_target_schema,
        "required_labels": profile.required_labels,
        "required_metadata": profile.required_metadata,
        "next_pipeline_steps": profile.next_pipeline_steps,
        "quality_gate": profile.quality_gate,
        "next_step": next_step,
        "notes": [
            "Task build plan does not download files or run prepare/full.",
            "Discovery-stage readiness is a candidate judgment, not a completed AI-ready dataset.",
            "Final training data still requires downstream search/export and quality gates.",
        ],
    }


def _write_task_build_csv(path: Path, rows: list[TaskBuildFile]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TASK_BUILD_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(_row_to_csv(row))


def _write_candidate_file_names(path: Path, rows: list[TaskBuildFile]) -> None:
    seen: set[str] = set()
    lines: list[str] = []
    for row in rows:
        if row.candidate_tier not in {"label_generation_candidate", "future_task_candidate"}:
            continue
        if row.file_name in seen:
            continue
        seen.add(row.file_name)
        lines.append(row.file_name)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _row_to_csv(row: TaskBuildFile) -> dict[str, Any]:
    return {
        "run_id": row.run_id or "",
        "task_type": row.task_type,
        "task_profile": row.task_profile,
        "implementation_status": row.implementation_status,
        "repository": row.repository,
        "project_accession": row.project_accession,
        "project_title": row.project_title or "",
        "file_name": row.file_name,
        "download_url": row.download_url or "",
        "file_type": row.file_type,
        "file_role": row.file_role,
        "species": _join(row.species),
        "species_policy": row.species_policy,
        "canonical_species": _join(row.canonical_species),
        "organism_taxon_id": _join(row.organism_taxon_id),
        "ptm_type": row.ptm_type or "",
        "ptm_subtype": row.ptm_subtype or "",
        "ptm_evidence_terms": _join(row.ptm_evidence_terms),
        "ptm_enrichment_methods": _join(row.ptm_enrichment_methods),
        "semantic_metadata_confidence": row.semantic_metadata_confidence,
        "semantic_interpretation_trace": _join(row.semantic_interpretation_trace),
        "modification_scope": row.modification_scope or "",
        "immunopeptide_scope": row.immunopeptide_scope or "",
        "hla_class": _join(row.hla_class),
        "hla_alleles": _join(row.hla_alleles),
        "immunopeptide_evidence_terms": _join(row.immunopeptide_evidence_terms),
        "immunopeptide_enrichment_methods": _join(row.immunopeptide_enrichment_methods),
        "immunopeptide_metadata_confidence": row.immunopeptide_metadata_confidence,
        "labeling_strategy": row.labeling_strategy or "",
        "validity_status": row.validity_status,
        "task_readiness_status": row.task_readiness_status or "",
        "candidate_tier": row.candidate_tier,
        "recommended_entrypoint": row.recommended_entrypoint,
        "recommended_run_mode": row.recommended_run_mode,
        "next_pipeline_steps": _join(row.next_pipeline_steps),
        "ai_ready_target_schema": row.ai_ready_target_schema,
        "required_labels": _join(row.required_labels),
        "missing_task_requirements": _join(row.missing_task_requirements),
        "task_build_reasons": _join(row.task_build_reasons),
        "trust_score": row.trust_score,
        "file_score": row.file_score,
        "evidence_level": row.evidence_level,
        "sdrf_match_status": row.sdrf_match_status,
        "tissue": row.tissue or "",
        "enzyme": row.enzyme or "",
        "instrument_names": _join(row.instrument_names),
        "instrument_vendor": row.instrument_vendor or "",
        "instrument_families": _join(row.instrument_families),
        "fragmentation_methods": _join(row.fragmentation_methods),
        "acquisition_mode": row.acquisition_mode or "",
        "isolation_window": row.isolation_window if row.isolation_window is not None else "",
        "resolution": row.resolution if row.resolution is not None else "",
        "collision_energy": row.collision_energy if row.collision_energy is not None else "",
        "scan_range": row.scan_range or "",
        "lc_gradient_minutes": row.lc_gradient_minutes if row.lc_gradient_minutes is not None else "",
    }


def _join(values: list[str]) -> str:
    return ";".join(str(value) for value in values if str(value).strip())


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result
