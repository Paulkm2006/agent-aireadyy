from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator

from agent.models import JsonModel


class ObservationRecord(JsonModel):
    """One task-level model observation with its upstream leakage identities."""

    observation_id: str
    task_type: str
    project_id: str
    source_file_id: str
    file_family_id: str
    source_artifact_uri: str
    source_row_number: int
    spectrum_id: str
    scan_number: str = ""
    precursor_mz: float | None = None
    spectrum_mz: list[float] = Field(default_factory=list)
    spectrum_intensity: list[float] = Field(default_factory=list)
    peak_count: int | None = None
    total_ion_current: float | None = None
    fragment_coverage: float | None = None
    psm_score: float | None = None
    sample_id: str = ""
    subject_id: str = ""
    tissue: str = ""
    technical_replicate_id: str = ""
    fraction_id: str = ""
    tmt_plex_id: str = ""
    lab_id: str = ""
    instrument_id: str = ""
    instrument_vendor: str = ""
    organism_id: str = ""
    acquisition_id: str = ""
    fragmentation_method: str = ""
    isolation_window: float | None = None
    resolution: float | None = None
    collision_energy: float | None = None
    scan_range: str = ""
    gradient_id: str = ""
    lc_gradient_minutes: float | None = None
    enzyme: str = ""
    search_workflow_id: str = ""
    search_engines: list[str] = Field(default_factory=list)
    engine_q_values: dict[str, float] = Field(default_factory=dict)
    peptide: str = ""
    modified_peptide: str = ""
    protein_ids: list[str] = Field(default_factory=list)
    protein_family_ids: list[str] = Field(default_factory=list)
    modification_classes: list[str] = Field(default_factory=list)
    modification_sites: list[str] = Field(default_factory=list)
    peptide_frequency: int = 1
    representative_rank: int = 1
    charge: int | None = None
    q_value: float | None = None
    psm_probability: float | None = None
    label_type: str = ""
    label_payload: dict[str, Any] = Field(default_factory=dict)
    label_source: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetCatalog(JsonModel):
    """Materialized catalog produced without changing an existing Batch run."""

    source_batch_dir: str
    observations: list[ObservationRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    curation_report: dict[str, Any] = Field(default_factory=dict)


class IdentityAssertion(JsonModel):
    observation_id: str
    dimension: str
    raw_values: list[str] = Field(default_factory=list)
    canonical_values: list[str] = Field(default_factory=list)
    status: Literal["present", "missing"]
    source_kind: Literal["batch_summary", "source_artifact", "derived", "unavailable"]
    source_uri: str
    confidence: Literal["reported", "derived", "unavailable"]
    missing_reason: str = ""


class IdentityDimensionSummary(JsonModel):
    dimension: str
    observation_count: int
    present_count: int
    missing_count: int
    unique_count: int
    coverage: float


class IdentityLedger(JsonModel):
    schema_version: str = "identity-ledger/v1"
    source_batch_dir: str
    observation_count: int
    assertions: list[IdentityAssertion] = Field(default_factory=list)
    dimensions: list[IdentityDimensionSummary] = Field(default_factory=list)


class SplitAllocation(JsonModel):
    observation_id: str
    component_id: str
    split: str


class SplitPolicy(JsonModel):
    version: str = "1.0"
    peptide_identity_mode: Literal["exact", "il_equivalent"] = "il_equivalent"
    modification_identity_mode: Literal["class", "peptidoform"] = "class"
    instrument_identity_level: Literal["model"] = "model"
    organism_identity_level: Literal["taxon"] = "taxon"
    acquisition_identity_level: Literal["profile"] = "profile"


class SplitPlan(JsonModel):
    requested_protocol: str
    resolved_protocol: str
    status: str
    holdout_identity: str
    allocations: list[SplitAllocation] = Field(default_factory=list)
    group_count: int = 0
    missing_identity_count: int = 0
    reasons: list[str] = Field(default_factory=list)
    solver: str = ""
    solver_status: str = "not_run"
    target_ratios: dict[str, float] = Field(default_factory=dict)
    actual_counts: dict[str, int] = Field(default_factory=dict)
    actual_ratios: dict[str, float] = Field(default_factory=dict)
    identity_policy: dict[str, Any] = Field(default_factory=dict)


class SplitSuite(JsonModel):
    ratios: tuple[float, float, float]
    seed: int
    protocols: dict[str, SplitPlan]
    policy: SplitPolicy = Field(default_factory=SplitPolicy)


class LeakageFinding(JsonModel):
    dimension: str
    requirement: str
    status: str
    overlap_count: int = 0
    missing_count: int = 0
    affected_identities: list[str] = Field(default_factory=list)
    affected_observation_ids: list[str] = Field(default_factory=list)
    severity: str = "info"


class LeakageAudit(JsonModel):
    protocol: str
    status: str
    findings: list[LeakageFinding] = Field(default_factory=list)


class DatasetReleaseResult(JsonModel):
    release_id: str
    status: str
    protocol_statuses: dict[str, str]
    files: dict[str, str]


class DatasetConstructionJobSpec(JsonModel):
    """One validated, durable request shared by API, UI, and SDK tools."""

    batch_dir: str
    output_dir: str
    release_id: str = Field(min_length=1, max_length=160)
    task_spec: dict[str, Any]
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)
    seed: int = 42
    policy: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str = Field(default="", max_length=160)

    @field_validator("ratios")
    @classmethod
    def valid_ratios(
        cls,
        value: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        if any(part <= 0 for part in value) or abs(sum(value) - 1.0) > 1e-9:
            raise ValueError("ratios_must_be_three_positive_values_summing_to_one")
        return value

    @field_validator("task_spec")
    @classmethod
    def task_spec_requires_task_type(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not str(value.get("task_type") or "").strip():
            raise ValueError("task_spec_requires_task_type")
        return value
