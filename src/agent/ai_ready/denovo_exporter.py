from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import Field

from agent.ai_ready.fragment_intensity_exporter import (
    _load_peaklists,
    _match_spectrum,
    _match_theoretical_ions,
)
from agent.ai_ready.rt_exporter import (
    CHARGE_COLUMNS,
    MODIFIED_COLUMNS,
    PEPTIDE_COLUMNS,
    PROBABILITY_COLUMNS,
    Q_VALUE_COLUMNS,
    SEARCH_ENGINE_COLUMNS,
    SOURCE_FILE_COLUMNS,
    SPECTRUM_COLUMNS,
    _clean_text,
    _find_column,
    _first_nonempty,
    _guess_search_engine,
    _load_task_build_metadata,
    _metadata_by_source,
    _optional_series,
    _series,
    _value_distribution,
)
from agent.models import JsonModel
from agent.utils import write_json
from agent.ai_ready.release_predicates import exporter_status_from_rows


DENOVO_SCHEMA_VERSION = "denovo_train_v1"
DENOVO_COLUMNS = [
    "project_accession",
    "source_file",
    "search_result_path",
    "peaklist_path",
    "spectrum_id",
    "scan_number",
    "peptide_sequence",
    "modified_sequence",
    "charge",
    "precursor_mz",
    "peak_count",
    "total_ion_current",
    "fragment_coverage",
    "psm_score",
    "q_value",
    "psm_probability",
    "search_engine",
    "search_engines_json",
    "engine_q_values_json",
    "species",
    "canonical_species",
    "organism_taxon_id",
    "tissue",
    "instrument_name",
    "instrument_vendor",
    "instrument_family",
    "fragmentation_method",
    "acquisition_mode",
    "isolation_window",
    "resolution",
    "collision_energy",
    "scan_range",
    "lc_gradient_minutes",
    "lc_gradient_bucket",
    "enzyme",
    "ptm_type",
    "ptm_sites",
    "modification_scope",
    "labeling_strategy",
    "peptide_frequency",
    "representative_rank",
    "spectrum_mz_json",
    "spectrum_intensity_json",
    "label_source",
]


class DenovoInput(JsonModel):
    path: str
    rows_in: int = 0
    rows_out: int = 0
    column_map: dict[str, str | None] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    filter_counts: dict[str, int] = Field(default_factory=dict)


class DenovoExportResult(JsonModel):
    status: str
    schema_version: str = DENOVO_SCHEMA_VERSION
    output_parquet: str
    preview_csv: str
    report_json: str
    schema_json_path: str
    rows_in: int = 0
    rows_out: int = 0
    filter_counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    inputs: list[DenovoInput] = Field(default_factory=list)


def export_denovo_ai_ready(
    search_results: list[str | Path],
    peaklists: list[str | Path],
    output_dir: str | Path,
    *,
    project_accession: str | None = None,
    source_file: str | None = None,
    task_build_plan: str | Path | None = None,
    q_value_threshold: float = 0.01,
    probability_threshold: float = 0.9,
    search_engine: str | None = None,
) -> DenovoExportResult:
    if not search_results:
        raise ValueError("At least one --search-result is required.")
    if not peaklists:
        raise ValueError("At least one --peaklist MGF path is required.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spectra, peaklist_warnings = _load_peaklists([Path(path) for path in peaklists])
    metadata = _load_task_build_metadata(task_build_plan)

    frames: list[pd.DataFrame] = []
    input_reports: list[DenovoInput] = []
    warnings: list[str] = list(peaklist_warnings)
    total_filter_counts: Counter[str] = Counter()
    for search_result in search_results:
        frame, report = _load_one_result(
            Path(search_result),
            spectra=spectra,
            metadata=metadata,
            project_accession=project_accession,
            source_file=source_file,
            q_value_threshold=q_value_threshold,
            probability_threshold=probability_threshold,
            search_engine=search_engine,
        )
        input_reports.append(report)
        warnings.extend(report.warnings)
        total_filter_counts.update(report.filter_counts)
        if not frame.empty:
            frames.append(frame)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=DENOVO_COLUMNS)
    combined, consensus_report = _collapse_dual_engine_consensus(combined)
    combined = combined.loc[:, DENOVO_COLUMNS]

    output_parquet = output_dir / "denovo_train.parquet"
    preview_csv = output_dir / "denovo.preview.csv"
    report_json = output_dir / "denovo_export_report.json"
    schema_json = output_dir / "denovo_schema.json"
    combined.to_parquet(output_parquet, index=False)
    combined.head(100).to_csv(preview_csv, index=False)
    write_json(schema_json, _schema_payload())

    rows_in = sum(item.rows_in for item in input_reports)
    rows_out = int(len(combined))
    export_status = exporter_status_from_rows(rows_out)
    report = {
        "status": export_status,
        "schema_version": DENOVO_SCHEMA_VERSION,
        "rows_in": rows_in,
        "rows_out": rows_out,
        "rows_filtered": rows_in - rows_out,
        "filter_counts": dict(sorted(total_filter_counts.items())),
        "warnings": sorted(set(warnings)),
        "spectrum_count": len(spectra),
        "charge_distribution": _value_distribution(combined.get("charge")),
        "search_result_count": len(input_reports),
        "psm_consensus": consensus_report,
        "inputs": [item.model_dump(mode="json") for item in input_reports],
        "outputs": {
            "denovo_train_parquet": str(output_parquet),
            "preview_csv": str(preview_csv),
            "schema_json": str(schema_json),
        },
    }
    write_json(report_json, report)
    return DenovoExportResult(
        status=export_status,
        output_parquet=str(output_parquet),
        preview_csv=str(preview_csv),
        report_json=str(report_json),
        schema_json_path=str(schema_json),
        rows_in=rows_in,
        rows_out=rows_out,
        filter_counts=dict(sorted(total_filter_counts.items())),
        warnings=sorted(set(warnings)),
        inputs=input_reports,
    )


def _load_one_result(
    path: Path,
    *,
    spectra: dict[str, Any],
    metadata: dict[str, dict[str, Any]],
    project_accession: str | None,
    source_file: str | None,
    q_value_threshold: float,
    probability_threshold: float,
    search_engine: str | None,
) -> tuple[pd.DataFrame, DenovoInput]:
    if not path.exists():
        raise ValueError(f"Search result does not exist: {path}")
    frame = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    column_map = _detect_columns(frame.columns)
    missing = [field for field in ["peptide_sequence", "charge", "spectrum_id"] if column_map.get(field) is None]
    if missing:
        report = DenovoInput(
            path=str(path),
            rows_in=int(len(frame)),
            rows_out=0,
            column_map=column_map,
            warnings=[f"missing_required_column:{field}" for field in missing],
            filter_counts={"missing_required_columns": int(len(frame))},
        )
        return pd.DataFrame(columns=DENOVO_COLUMNS), report

    output = pd.DataFrame()
    output["peptide_sequence"] = _series(frame, column_map["peptide_sequence"]).map(_clean_sequence)
    modified_column = column_map.get("modified_sequence")
    output["modified_sequence"] = _series(frame, modified_column).map(_clean_text) if modified_column else output["peptide_sequence"]
    output["charge"] = pd.to_numeric(_series(frame, column_map["charge"]), errors="coerce")
    output["spectrum_id"] = _series(frame, column_map["spectrum_id"]).map(_clean_text)
    output["q_value"] = pd.to_numeric(_optional_series(frame, column_map.get("q_value")), errors="coerce")
    output["psm_probability"] = pd.to_numeric(_optional_series(frame, column_map.get("psm_probability")), errors="coerce")
    output["psm_score"] = pd.to_numeric(_optional_series(frame, column_map.get("psm_score")), errors="coerce")

    source_series = _optional_series(frame, column_map.get("source_file")).map(_clean_text)
    if source_file:
        output["source_file"] = source_file
    elif column_map.get("source_file") is not None:
        output["source_file"] = source_series.replace("", path.stem)
    else:
        output["source_file"] = path.stem
    source_metadata = _metadata_by_source(metadata, output["source_file"], path.stem)
    output["project_accession"] = [
        project_accession or source_metadata[str(source)][0].get("project_accession") or ""
        for source in output["source_file"]
    ]
    output["search_result_path"] = str(path)
    output["search_engine"] = search_engine or _first_nonempty(_optional_series(frame, column_map.get("search_engine"))) or _guess_search_engine(path)
    output["search_engines_json"] = output["search_engine"].map(
        lambda value: json.dumps([_canonical_search_engine(value)], ensure_ascii=False)
    )
    output["engine_q_values_json"] = [
        json.dumps(
            {_canonical_search_engine(engine): float(q_value)},
            ensure_ascii=False,
            sort_keys=True,
        )
        if _canonical_search_engine(engine) != "unknown" and not pd.isna(q_value)
        else "{}"
        for engine, q_value in zip(output["search_engine"], output["q_value"], strict=True)
    ]
    output["species"] = [source_metadata[str(source)][0].get("species") or "" for source in output["source_file"]]
    output["canonical_species"] = [
        source_metadata[str(source)][0].get("canonical_species") or "" for source in output["source_file"]
    ]
    output["organism_taxon_id"] = [
        source_metadata[str(source)][0].get("organism_taxon_id") or "" for source in output["source_file"]
    ]
    output["tissue"] = [source_metadata[str(source)][0].get("tissue") or "" for source in output["source_file"]]
    output["instrument_name"] = [
        source_metadata[str(source)][0].get("instrument_name") or "" for source in output["source_file"]
    ]
    output["instrument_vendor"] = [
        source_metadata[str(source)][0].get("instrument_vendor") or "" for source in output["source_file"]
    ]
    output["instrument_family"] = [
        source_metadata[str(source)][0].get("instrument_family") or "" for source in output["source_file"]
    ]
    output["fragmentation_method"] = [
        source_metadata[str(source)][0].get("fragmentation_method") or source_metadata[str(source)][0].get("fragmentation_methods") or ""
        for source in output["source_file"]
    ]
    output["acquisition_mode"] = [
        source_metadata[str(source)][0].get("acquisition_mode") or "" for source in output["source_file"]
    ]
    for field in ("isolation_window", "resolution", "collision_energy", "scan_range", "lc_gradient_minutes", "enzyme", "ptm_sites"):
        output[field] = [source_metadata[str(source)][0].get(field) or "" for source in output["source_file"]]
    output["lc_gradient_bucket"] = output["lc_gradient_minutes"].map(_lc_gradient_bucket)
    output["ptm_type"] = [source_metadata[str(source)][0].get("ptm_type") or "" for source in output["source_file"]]
    output["modification_scope"] = [
        source_metadata[str(source)][0].get("modification_scope") or "" for source in output["source_file"]
    ]
    output["labeling_strategy"] = [
        source_metadata[str(source)][0].get("labeling_strategy") or "" for source in output["source_file"]
    ]
    output["ptm_sites"] = output["modified_sequence"].map(_modification_sites_json)
    output["peptide_frequency"] = 1
    output["representative_rank"] = 1

    filter_counts: Counter[str] = Counter()
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    for row in output.to_dict(orient="records"):
        reason = _row_filter_reason(row, q_value_threshold=q_value_threshold, probability_threshold=probability_threshold)
        if reason:
            filter_counts[reason] += 1
            continue
        spectrum = _match_spectrum(str(row["spectrum_id"]), spectra)
        if spectrum is None:
            filter_counts["spectrum_not_matched"] += 1
            warnings.append("spectrum_not_matched")
            continue
        rows.append(
            {
                **row,
                "peaklist_path": spectrum.path,
                "precursor_mz": spectrum.precursor_mz,
                "scan_number": spectrum.scan or _scan_number(str(row["spectrum_id"])),
                "peak_count": len(spectrum.mz),
                "total_ion_current": float(sum(spectrum.intensity)),
                "fragment_coverage": _fragment_coverage(
                    str(row["peptide_sequence"]), spectrum
                ),
                "fragmentation_method": row.get("fragmentation_method") or spectrum.fragmentation_method,
                "spectrum_mz_json": json.dumps(spectrum.mz, ensure_ascii=False),
                "spectrum_intensity_json": json.dumps(spectrum.intensity, ensure_ascii=False),
                "label_source": "high_confidence_psm_tsv_plus_mgf",
            }
        )

    filtered = pd.DataFrame(rows, columns=DENOVO_COLUMNS)
    report = DenovoInput(
        path=str(path),
        rows_in=int(len(frame)),
        rows_out=int(len(filtered)),
        column_map=column_map,
        warnings=sorted(set(warnings)),
        filter_counts={key: value for key, value in sorted(filter_counts.items()) if value},
    )
    return filtered, report


def _detect_columns(columns: pd.Index) -> dict[str, str | None]:
    available = list(columns)
    return {
        "peptide_sequence": _find_column(available, PEPTIDE_COLUMNS),
        "modified_sequence": _find_column(available, MODIFIED_COLUMNS),
        "charge": _find_column(available, CHARGE_COLUMNS),
        "spectrum_id": _find_column(available, SPECTRUM_COLUMNS),
        "source_file": _find_column(available, SOURCE_FILE_COLUMNS),
        "q_value": _find_column(available, Q_VALUE_COLUMNS),
        "psm_probability": _find_column(available, PROBABILITY_COLUMNS),
        "psm_score": _find_column(
            available,
            ["hyperscore", "score", "spectral angle"],
        ),
        "search_engine": _find_column(available, SEARCH_ENGINE_COLUMNS),
    }


def _canonical_search_engine(value: Any) -> str:
    text = _clean_text(value).casefold()
    if "sage" in text:
        return "sage"
    if "fragpipe" in text or "msfragger" in text:
        return "fragpipe"
    return text or "unknown"


def _consensus_key(row: pd.Series) -> tuple[str, str, str, str]:
    charge = row.get("charge")
    charge_text = "" if pd.isna(charge) else str(int(float(charge)))
    return (
        _clean_text(row.get("source_file")).casefold(),
        _clean_text(row.get("spectrum_id")).casefold(),
        charge_text,
        _clean_text(row.get("modified_sequence") or row.get("peptide_sequence")).casefold(),
    )


def _collapse_dual_engine_consensus(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Collapse matching FragPipe/Sage PSMs while preserving legacy single-engine rows."""

    if frame.empty:
        return frame, {"status": "not_available", "consensus_rows": 0}
    observed = {_canonical_search_engine(value) for value in frame["search_engine"]}
    required = {"fragpipe", "sage"}
    if not required.issubset(observed):
        return frame, {
            "status": "single_engine",
            "observed_engines": sorted(observed),
            "consensus_rows": 0,
        }

    rows: list[dict[str, Any]] = []
    consensus_rows = 0
    for _key, group in frame.groupby(frame.apply(_consensus_key, axis=1), sort=True):
        engines = {_canonical_search_engine(value) for value in group["search_engine"]}
        if not required.issubset(engines):
            rows.extend(group.to_dict(orient="records"))
            continue
        representative = group.sort_values(
            ["q_value", "psm_probability", "search_result_path"],
            ascending=[True, False, True],
            na_position="last",
            kind="stable",
        ).iloc[0].to_dict()
        engine_q_values: dict[str, float] = {}
        for engine, engine_rows in group.groupby(
            group["search_engine"].map(_canonical_search_engine), sort=True
        ):
            q_values = pd.to_numeric(engine_rows["q_value"], errors="coerce").dropna()
            if not q_values.empty:
                engine_q_values[str(engine)] = float(q_values.min())
        representative.update(
            {
                "q_value": max(engine_q_values.values()) if engine_q_values else representative.get("q_value"),
                "search_engine": "FragPipe∩Sage",
                "search_engines_json": json.dumps(sorted(engines), ensure_ascii=False),
                "engine_q_values_json": json.dumps(engine_q_values, ensure_ascii=False, sort_keys=True),
                "label_source": "fragpipe_sage_1pct_fdr_consensus_plus_mgf",
            }
        )
        rows.append(representative)
        consensus_rows += 1
    return pd.DataFrame(rows, columns=frame.columns), {
        "status": "available",
        "observed_engines": sorted(observed),
        "consensus_rows": consensus_rows,
        "non_consensus_rows": len(rows) - consensus_rows,
    }


def _scan_number(value: str) -> str:
    import re

    match = re.search(r"(?:scan[:=]\s*)?(\d+)", value, flags=re.IGNORECASE)
    return match.group(1) if match else value


def _fragment_coverage(peptide: str, spectrum: Any) -> float:
    denominator = max(0, 2 * (len(peptide) - 1))
    if not denominator:
        return 0.0
    return len(_match_theoretical_ions(peptide, spectrum, tolerance_da=0.5)) / denominator


def _lc_gradient_bucket(value: Any) -> str:
    try:
        minutes = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if minutes <= 45:
        return "short"
    if minutes >= 90:
        return "long"
    return "medium"


def _modification_sites_json(value: Any) -> str:
    import re

    sequence = _clean_text(value)
    sites: list[str] = []
    residue_index = 0
    index = 0
    while index < len(sequence):
        char = sequence[index]
        if char.isalpha():
            residue_index += 1
            index += 1
            continue
        if char in "[(":
            closing = "]" if char == "[" else ")"
            end = sequence.find(closing, index + 1)
            if end >= 0:
                token = sequence[index + 1 : end].strip()
                sites.append(f"{residue_index}:{token}" if residue_index else f"N-term:{token}")
                index = end + 1
                continue
        index += 1
    return json.dumps(sites, ensure_ascii=False)


def _row_filter_reason(row: dict[str, Any], *, q_value_threshold: float, probability_threshold: float) -> str | None:
    if not row.get("peptide_sequence"):
        return "missing_peptide_sequence"
    if pd.isna(row.get("charge")):
        return "missing_charge"
    if not row.get("spectrum_id"):
        return "missing_spectrum_id"
    q_value = row.get("q_value")
    if not pd.isna(q_value) and float(q_value) > q_value_threshold:
        return "q_value_above_threshold"
    probability = row.get("psm_probability")
    if not pd.isna(probability) and float(probability) < probability_threshold:
        return "probability_below_threshold"
    return None


def _clean_sequence(value: Any) -> str:
    text = _clean_text(value).upper()
    return "".join(char for char in text if char.isalpha())


def _schema_payload() -> dict[str, Any]:
    return {
        "schema_version": DENOVO_SCHEMA_VERSION,
        "target_schema": "denovo_train.parquet",
        "columns": {
            "peptide_sequence": "Unmodified peptide sequence label from a high-confidence PSM/search result row.",
            "modified_sequence": "Exact modified peptide identity used by consensus and leakage controls.",
            "spectrum_mz_json": "JSON array of MGF peak m/z values.",
            "spectrum_intensity_json": "JSON array of MGF peak intensities.",
            "search_engines_json": "JSON list of engines supporting this exact spectrum/peptidoform assignment.",
            "engine_q_values_json": "Per-engine q-values used for the 1% FDR consensus gate.",
            "fragment_coverage": "Matched unmodified b/y ions divided by the possible b/y ion count.",
            "peptide_frequency": "Number of observations for this modified peptide before representative-scan capping.",
            "representative_rank": "Deterministic quality rank within the modified-peptide group.",
            "label_source": "FragPipe/Sage consensus when both engine results are supplied; otherwise the legacy high-confidence PSM source.",
        },
        "notes": [
            "v1 creates supervised spectrum-sequence pairs from existing search results; it does not perform de novo inference.",
            "v1 reads MGF only and does not parse mzML directly.",
            "When both engines are supplied, only exact raw-file, scan, charge and modified-peptide matches are marked as FragPipe/Sage consensus.",
        ],
    }
