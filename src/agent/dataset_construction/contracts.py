from __future__ import annotations

from typing import Any

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

from agent.dataset_construction.models import DatasetCatalog
from agent.dataset_construction.ingestion import canonical_task_type
from agent.dataset_construction.benchmark import (
    benchmark_coverage_report,
    denovo_benchmark_policy,
)


class DatasetContractError(ValueError):
    """Raised when a catalog cannot safely enter split construction."""


CATALOG_SCHEMA = pa.DataFrameSchema(
    {
        "observation_id": pa.Column(
            str,
            checks=pa.Check.str_length(min_value=1),
            nullable=False,
            unique=True,
        ),
        "task_type": pa.Column(str, checks=pa.Check.str_length(min_value=1)),
        "project_id": pa.Column(str, checks=pa.Check.str_length(min_value=1)),
        "source_file_id": pa.Column(str, checks=pa.Check.str_length(min_value=1)),
        "file_family_id": pa.Column(str, checks=pa.Check.str_length(min_value=1)),
        "source_artifact_uri": pa.Column(str, checks=pa.Check.str_length(min_value=1)),
        "source_row_number": pa.Column(int, checks=pa.Check.ge(0)),
        "spectrum_id": pa.Column(str, checks=pa.Check.str_length(min_value=1)),
    },
    strict=False,
    coerce=False,
    name="dataset_construction_catalog_v1",
)


_PEPTIDE_LABEL_TASKS = {
    "denovo",
    "ptm_denovo",
    "fragment_intensity_prediction",
    "rt_prediction",
}


def validate_catalog(
    catalog: DatasetCatalog,
    *,
    task_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate release-blocking invariants and return a compact evidence record."""

    if not catalog.observations:
        detail = catalog.curation_report.get("filter_counts")
        suffix = f": benchmark filters={detail}" if detail else ""
        raise DatasetContractError(f"catalog contains no model observations{suffix}")
    frame = pd.DataFrame(
        [observation.model_dump(mode="python") for observation in catalog.observations]
    )
    try:
        validated = CATALOG_SCHEMA.validate(frame, lazy=True)
    except SchemaErrors as exc:
        failures = exc.failure_cases.to_dict(orient="records")
        raise DatasetContractError(
            f"catalog violates dataset contract: {failures}"
        ) from exc
    task_spec = task_spec or {}
    task_type = canonical_task_type(task_spec.get("task_type"))
    if task_spec and not task_type:
        raise DatasetContractError("task_spec.task_type is required")
    label_policy = task_spec.get("label_policy") or {}
    if not isinstance(label_policy, dict):
        raise DatasetContractError("task_spec.label_policy must be a JSON object")
    require_peptide = bool(
        label_policy.get("require_peptide", task_type in _PEPTIDE_LABEL_TASKS)
    )
    require_q_value = bool(
        label_policy.get("require_q_value", task_type == "denovo")
    )
    require_confidence = bool(
        label_policy.get(
            "require_confidence",
            task_type in _PEPTIDE_LABEL_TASKS | {"fragment_intensity_prediction"},
        )
    )
    max_q_value_raw = label_policy.get(
        "max_q_value",
        0.01 if task_type in _PEPTIDE_LABEL_TASKS else None,
    )
    max_q_value = (
        float(max_q_value_raw) if max_q_value_raw is not None else None
    )
    benchmark_policy = denovo_benchmark_policy(task_spec)
    violations: list[str] = []
    for observation in catalog.observations:
        if task_type and canonical_task_type(observation.task_type) != task_type:
            violations.append(
                f"wrong_task_type:{observation.observation_id}:{observation.task_type}"
            )
        if require_peptide and not observation.peptide.strip():
            violations.append(f"missing_peptide:{observation.observation_id}")
        if (
            require_confidence
            and observation.q_value is None
            and observation.psm_probability is None
        ):
            violations.append(f"missing_confidence:{observation.observation_id}")
        if require_q_value and observation.q_value is None:
            violations.append(f"missing_q_value:{observation.observation_id}")
        if (
            max_q_value is not None
            and observation.q_value is not None
            and observation.q_value > max_q_value
        ):
            violations.append(
                f"q_value_above_threshold:{observation.observation_id}:{observation.q_value}"
            )
        label = observation.label_payload
        if task_type == "rt_prediction":
            value = label.get("retention_time")
            if not isinstance(value, (int, float)):
                violations.append(f"missing_retention_time:{observation.observation_id}")
            if not str(label.get("unit") or "").strip():
                violations.append(f"missing_retention_time_unit:{observation.observation_id}")
        elif task_type == "fragment_intensity_prediction":
            if int(label.get("target_count") or 0) <= 0:
                violations.append(f"missing_fragment_targets:{observation.observation_id}")
        elif task_type == "psm_scoring":
            target_decoy = str(label.get("target_decoy") or "").strip().casefold()
            if target_decoy not in {"target", "decoy", "true", "false", "0", "1"}:
                violations.append(f"missing_target_decoy_label:{observation.observation_id}")
        elif task_type == "ptm_denovo":
            if not observation.modified_peptide.strip() or not label.get("modification_tokens"):
                violations.append(f"missing_modified_peptide_label:{observation.observation_id}")
        if benchmark_policy is not None:
            engines = {value.strip().casefold() for value in observation.search_engines}
            required_engines = {
                str(value).strip().casefold()
                for value in benchmark_policy["required_search_engines"]
            }
            if "dda" not in observation.acquisition_id.casefold():
                violations.append(f"benchmark_requires_dda:{observation.observation_id}")
            if not required_engines.issubset(engines):
                violations.append(f"benchmark_missing_engine_consensus:{observation.observation_id}")
            if observation.peak_count is None:
                violations.append(f"benchmark_missing_peak_count:{observation.observation_id}")
            if not observation.spectrum_mz or not observation.spectrum_intensity:
                violations.append(f"benchmark_missing_msms_peaks:{observation.observation_id}")
            elif len(observation.spectrum_mz) != len(observation.spectrum_intensity):
                violations.append(f"benchmark_mismatched_msms_peaks:{observation.observation_id}")
            elif observation.peak_count != len(observation.spectrum_mz):
                violations.append(f"benchmark_peak_count_mismatch:{observation.observation_id}")
            required_metadata = {
                "scan_number": observation.scan_number,
                "precursor_mz": observation.precursor_mz,
                "charge": observation.charge,
                "species": observation.organism_id,
                "tissue": observation.tissue,
                "instrument": observation.instrument_id,
                "fragmentation": observation.fragmentation_method,
                "isolation_window": observation.isolation_window,
                "resolution": observation.resolution,
                "collision_energy": observation.collision_energy,
                "scan_range": observation.scan_range,
                "lc_gradient_minutes": observation.lc_gradient_minutes,
                "enzyme": observation.enzyme,
            }
            for field, value in required_metadata.items():
                if value is None or (isinstance(value, str) and not value.strip()):
                    violations.append(
                        f"benchmark_missing_metadata:{observation.observation_id}:{field}"
                    )
            modified = {
                value.strip().casefold()
                for value in observation.modification_classes
                if value.strip()
            } - {"none", "unmodified", "no modification"}
            if modified and not observation.modification_sites:
                violations.append(
                    f"benchmark_missing_modification_sites:{observation.observation_id}"
                )
            for engine in sorted(required_engines):
                q_value = observation.engine_q_values.get(engine)
                if q_value is None:
                    violations.append(
                        f"benchmark_missing_engine_q_value:{observation.observation_id}:{engine}"
                    )
                elif q_value > float(benchmark_policy["max_q_value"]):
                    violations.append(
                        f"benchmark_engine_q_value_above_threshold:{observation.observation_id}:{engine}:{q_value}"
                    )
    benchmark_report: dict[str, Any] = {}
    if benchmark_policy is not None:
        peptide_counts = frame.assign(
            _peptidoform=frame["modified_peptide"].replace("", pd.NA).fillna(frame["peptide"])
        )["_peptidoform"].value_counts()
        cap = int(benchmark_policy["max_spectra_per_modified_peptide"])
        if not peptide_counts.empty and int(peptide_counts.max()) > cap:
            violations.append(
                f"benchmark_peptidoform_frequency_above_cap:{int(peptide_counts.max())}>{cap}"
            )
        benchmark_report = benchmark_coverage_report(
            catalog.observations,
            policy=benchmark_policy,
        )
        violations.extend(
            f"benchmark_coverage:{issue}"
            for issue in benchmark_report["blocking_issues"]
        )
    if violations:
        raise DatasetContractError(
            f"catalog violates task label policy: {violations}"
        )
    return {
        "contract": CATALOG_SCHEMA.name,
        "status": "pass",
        "observation_count": int(len(validated)),
        "unique_observation_count": int(validated["observation_id"].nunique()),
        "source_artifact_count": int(validated["source_artifact_uri"].nunique()),
        "file_family_count": int(validated["file_family_id"].nunique()),
        "label_policy": {
            "task_type": task_type,
            "require_peptide": require_peptide,
            "require_q_value": require_q_value,
            "require_confidence": require_confidence,
            "max_q_value": max_q_value,
            "status": "pass",
        },
        "benchmark_profile": benchmark_report,
    }
