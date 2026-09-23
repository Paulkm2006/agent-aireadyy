from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import Field

from agent.models import JsonModel
from agent.utils import write_json
from agent.ai_ready.release_predicates import exporter_status_from_rows


RT_SCHEMA_VERSION = "rt_train_v0"
RT_TRAIN_COLUMNS = [
    "project_accession",
    "source_file",
    "search_result_path",
    "peptide_sequence",
    "modified_sequence",
    "charge",
    "retention_time",
    "retention_time_unit",
    "rt_unit_source",
    "spectrum_id",
    "q_value",
    "psm_probability",
    "search_engine",
    "species",
    "canonical_species",
    "organism_taxon_id",
    "instrument_family",
    "lc_gradient_minutes",
    "ptm_type",
    "modification_scope",
    "labeling_strategy",
    "label_source",
]
RT_PEPTIDE_COLUMNS = [
    "project_accession",
    "source_file",
    "peptide_sequence",
    "modified_sequence",
    "charge",
    "retention_time_median",
    "retention_time_mean",
    "retention_time_unit",
    "rt_unit_source",
    "psm_count",
    "best_q_value",
    "best_psm_probability",
    "search_result_paths",
    "search_engine",
    "species",
    "canonical_species",
    "organism_taxon_id",
    "instrument_family",
    "lc_gradient_minutes",
    "ptm_type",
    "modification_scope",
    "labeling_strategy",
    "label_source",
]

PEPTIDE_COLUMNS = [
    "peptide",
    "peptide sequence",
    "sequence",
    "stripped sequence",
    "stripped peptide",
    "base sequence",
]
MODIFIED_COLUMNS = [
    "modified peptide",
    "modified sequence",
    "assigned modifications",
    "peptide modified sequence",
    "modifications",
]
CHARGE_COLUMNS = ["charge", "precursor charge", "z"]
RT_COLUMNS = [
    "retention",
    "retention time",
    "retention time (min)",
    "retention time (sec)",
    "retention time (s)",
    "rt",
    "rt observed",
    "observed rt",
    "peptideprophet retention time",
]
Q_VALUE_COLUMNS = [
    "q value",
    "q-value",
    "qvalue",
    "spectrum q",
    "spectrum q-value",
    "spectrum_q",
    "peptide q",
    "peptide q-value",
    "peptide_q",
    "protein q",
    "protein q-value",
    "protein_q",
    "psm q-value",
    "expectation",
    "e-value",
]
PROBABILITY_COLUMNS = [
    "probability",
    "peptideprophet probability",
    "psm probability",
    "hyperscore probability",
]
SPECTRUM_COLUMNS = [
    "scannr",
    "spectrum",
    "spectrum id",
    "spectrum title",
    "scan",
    "scan number",
    "spectrum file",
    "psm id",
    "psmid",
    "psm_id",
    "filename",
]
SOURCE_FILE_COLUMNS = ["source file", "raw file", "spectrum file", "filename", "file name"]
SEARCH_ENGINE_COLUMNS = ["search engine", "engine"]


class RtExportOptions(JsonModel):
    q_value_threshold: float = 0.01
    probability_threshold: float = 0.9
    require_confidence: bool = False
    retention_time_unit: str | None = None
    retention_time_unit_source: str = "unknown"


class RtSearchInput(JsonModel):
    path: str
    rows_in: int = 0
    rows_out: int = 0
    column_map: dict[str, str | None] = Field(default_factory=dict)
    missing_required_columns: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    filter_counts: dict[str, int] = Field(default_factory=dict)
    task_build_plan_matched_sources: list[str] = Field(default_factory=list)
    task_build_plan_unmatched_sources: list[str] = Field(default_factory=list)


class RtExportResult(JsonModel):
    status: str
    schema_version: str = RT_SCHEMA_VERSION
    output_parquet: str
    preview_csv: str
    peptide_parquet: str
    peptide_preview_csv: str
    peptide_report_json: str
    report_json: str
    validation_report_json: str
    schema_json_path: str
    rows_in: int = 0
    rows_out: int = 0
    peptide_rows_out: int = 0
    filter_counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    inputs: list[RtSearchInput] = Field(default_factory=list)


def export_rt_ai_ready(
    search_results: list[str | Path],
    output_dir: str | Path,
    *,
    project_accession: str | None = None,
    source_file: str | None = None,
    task_build_plan: str | Path | None = None,
    q_value_threshold: float = 0.01,
    probability_threshold: float = 0.9,
    require_confidence: bool = False,
    retention_time_unit: str | None = None,
    search_engine: str | None = None,
) -> RtExportResult:
    if not search_results:
        raise ValueError("At least one --search-result is required.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    unit_source = "user_supplied" if retention_time_unit else "unknown"
    options = RtExportOptions(
        q_value_threshold=q_value_threshold,
        probability_threshold=probability_threshold,
        require_confidence=require_confidence,
        retention_time_unit=retention_time_unit,
        retention_time_unit_source=unit_source,
    )
    metadata = _load_task_build_metadata(task_build_plan)

    frames: list[pd.DataFrame] = []
    input_reports: list[RtSearchInput] = []
    global_warnings: list[str] = []
    total_filter_counts: Counter[str] = Counter()
    for search_result in search_results:
        frame, report = _load_one_result(
            Path(search_result),
            options=options,
            metadata=metadata,
            project_accession=project_accession,
            source_file=source_file,
            search_engine=search_engine,
        )
        input_reports.append(report)
        global_warnings.extend(report.warnings)
        total_filter_counts.update(report.filter_counts)
        if not frame.empty:
            frames.append(frame)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=RT_TRAIN_COLUMNS)
    combined = combined.loc[:, RT_TRAIN_COLUMNS]
    peptide_frame = _aggregate_peptide_level(combined)

    output_parquet = output_dir / "rt_train.parquet"
    preview_csv = output_dir / "rt_train.preview.csv"
    peptide_parquet = output_dir / "rt_train_peptide.parquet"
    peptide_preview_csv = output_dir / "rt_train_peptide.preview.csv"
    peptide_report_json = output_dir / "rt_peptide_aggregation_report.json"
    report_json = output_dir / "rt_export_report.json"
    validation_report_json = output_dir / "rt_validation_report.json"
    schema_json = output_dir / "rt_schema.json"

    combined.to_parquet(output_parquet, index=False)
    combined.head(100).to_csv(preview_csv, index=False)
    peptide_frame.to_parquet(peptide_parquet, index=False)
    peptide_frame.head(100).to_csv(peptide_preview_csv, index=False)
    schema = _schema_payload()
    write_json(schema_json, schema)

    rows_in = sum(item.rows_in for item in input_reports)
    rows_out = int(len(combined))
    export_status = exporter_status_from_rows(rows_out)
    rt_unit_sources = (
        sorted({str(v) for v in combined["rt_unit_source"].dropna().astype(str)})
        if rows_out and "rt_unit_source" in combined.columns
        else []
    )
    report = {
        "status": export_status,
        "schema_version": RT_SCHEMA_VERSION,
        "rows_in": rows_in,
        "rows_out": rows_out,
        "rt_unit_source": rt_unit_sources[0] if len(rt_unit_sources) == 1 else ("mixed" if rt_unit_sources else "unknown"),
        "has_confidence_column": any(
            item.column_map.get("q_value") is not None or item.column_map.get("psm_probability") is not None
            for item in input_reports
        ),
        "rows_filtered": rows_in - rows_out,
        "filter_counts": dict(sorted(total_filter_counts.items())),
        "warnings": sorted(set(global_warnings)),
        "retention_time": _numeric_distribution(combined.get("retention_time")),
        "charge_distribution": _value_distribution(combined.get("charge")),
        "project_count": int(combined["project_accession"].replace("", pd.NA).dropna().nunique()) if rows_out else 0,
        "source_file_count": int(combined["source_file"].replace("", pd.NA).dropna().nunique()) if rows_out else 0,
        "search_result_count": len(input_reports),
        "inputs": [item.model_dump(mode="json") for item in input_reports],
        "outputs": {
            "rt_train_parquet": str(output_parquet),
            "preview_csv": str(preview_csv),
            "rt_train_peptide_parquet": str(peptide_parquet),
            "peptide_preview_csv": str(peptide_preview_csv),
            "peptide_report_json": str(peptide_report_json),
            "validation_report_json": str(validation_report_json),
            "schema_json": str(schema_json),
        },
    }
    peptide_report = _peptide_aggregation_report(combined, peptide_frame)
    validation_report = _validation_report(
        combined=combined,
        peptide_frame=peptide_frame,
        input_reports=input_reports,
        filter_counts=total_filter_counts,
        warnings=global_warnings,
        task_build_plan_supplied=task_build_plan is not None,
    )
    write_json(report_json, report)
    write_json(peptide_report_json, peptide_report)
    write_json(validation_report_json, validation_report)
    return RtExportResult(
        status=export_status,
        output_parquet=str(output_parquet),
        preview_csv=str(preview_csv),
        peptide_parquet=str(peptide_parquet),
        peptide_preview_csv=str(peptide_preview_csv),
        peptide_report_json=str(peptide_report_json),
        report_json=str(report_json),
        validation_report_json=str(validation_report_json),
        schema_json_path=str(schema_json),
        rows_in=rows_in,
        rows_out=rows_out,
        peptide_rows_out=int(len(peptide_frame)),
        filter_counts=dict(sorted(total_filter_counts.items())),
        warnings=sorted(set(global_warnings)),
        inputs=input_reports,
    )


def _load_one_result(
    path: Path,
    *,
    options: RtExportOptions,
    metadata: dict[str, dict[str, Any]],
    project_accession: str | None,
    source_file: str | None,
    search_engine: str | None,
) -> tuple[pd.DataFrame, RtSearchInput]:
    if not path.exists():
        raise ValueError(f"Search result does not exist: {path}")
    frame = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    column_map = _detect_columns(frame.columns)
    missing = [
        field
        for field in ["peptide_sequence", "charge", "retention_time"]
        if column_map.get(field) is None
    ]
    warnings: list[str] = []
    if missing:
        report = RtSearchInput(
            path=str(path),
            rows_in=int(len(frame)),
            rows_out=0,
            column_map=column_map,
            missing_required_columns=missing,
            warnings=[f"missing_required_column:{field}" for field in missing],
            filter_counts={"missing_required_columns": int(len(frame))},
        )
        return pd.DataFrame(columns=RT_TRAIN_COLUMNS), report

    if column_map.get("q_value") is None and column_map.get("psm_probability") is None:
        message = "confidence_column_missing"
        if options.require_confidence:
            raise ValueError(f"{message}: {path}")
        warnings.append(message)

    output = pd.DataFrame()
    output["peptide_sequence"] = _series(frame, column_map["peptide_sequence"]).map(_clean_text)
    modified_column = column_map.get("modified_sequence")
    output["modified_sequence"] = (
        _series(frame, modified_column).map(_clean_text)
        if modified_column
        else output["peptide_sequence"]
    )
    output["charge"] = pd.to_numeric(_series(frame, column_map["charge"]), errors="coerce")
    output["retention_time"] = pd.to_numeric(_series(frame, column_map["retention_time"]), errors="coerce")
    if options.retention_time_unit:
        rt_unit = str(options.retention_time_unit)
        rt_unit_source = "user_supplied"
    else:
        rt_unit, rt_unit_source = _retention_time_unit_with_source(str(column_map["retention_time"]))
        if rt_unit_source != "column_explicit":
            warnings.append("rt_unit_unknown")
    output["retention_time_unit"] = rt_unit
    output["rt_unit_source"] = rt_unit_source
    output["spectrum_id"] = _optional_series(frame, column_map.get("spectrum_id"))
    output["q_value"] = pd.to_numeric(_optional_series(frame, column_map.get("q_value")), errors="coerce")
    output["psm_probability"] = pd.to_numeric(_optional_series(frame, column_map.get("psm_probability")), errors="coerce")

    source_series = _optional_series(frame, column_map.get("source_file")).map(_clean_text)
    if source_file:
        output["source_file"] = source_file
    elif column_map.get("source_file") is not None:
        output["source_file"] = source_series.replace("", path.stem)
    else:
        output["source_file"] = path.stem
    source_metadata = _metadata_by_source(metadata, output["source_file"], path.stem)
    matched_sources = sorted({source for source, (_, matched) in source_metadata.items() if matched})
    unmatched_sources = sorted({source for source, (_, matched) in source_metadata.items() if not matched})
    if metadata and unmatched_sources:
        warnings.extend(f"task_build_plan_source_unmatched:{source}" for source in unmatched_sources)
    output["project_accession"] = [
        project_accession or source_metadata[str(source)][0].get("project_accession") or ""
        for source in output["source_file"]
    ]
    output["search_result_path"] = str(path)
    output["search_engine"] = search_engine or _first_nonempty(_optional_series(frame, column_map.get("search_engine"))) or _guess_search_engine(path)
    output["species"] = [source_metadata[str(source)][0].get("species") or "" for source in output["source_file"]]
    output["canonical_species"] = [
        source_metadata[str(source)][0].get("canonical_species") or "" for source in output["source_file"]
    ]
    output["organism_taxon_id"] = [
        source_metadata[str(source)][0].get("organism_taxon_id") or "" for source in output["source_file"]
    ]
    output["instrument_family"] = [
        source_metadata[str(source)][0].get("instrument_family") or "" for source in output["source_file"]
    ]
    output["lc_gradient_minutes"] = [
        source_metadata[str(source)][0].get("lc_gradient_minutes") for source in output["source_file"]
    ]
    output["ptm_type"] = [source_metadata[str(source)][0].get("ptm_type") or "" for source in output["source_file"]]
    output["modification_scope"] = [
        source_metadata[str(source)][0].get("modification_scope") or "" for source in output["source_file"]
    ]
    output["labeling_strategy"] = [
        source_metadata[str(source)][0].get("labeling_strategy") or "" for source in output["source_file"]
    ]
    output["label_source"] = "search_result_tsv"

    keep = pd.Series(True, index=output.index)
    filter_counts: Counter[str] = Counter()
    for column, reason in [
        ("peptide_sequence", "missing_peptide_sequence"),
        ("charge", "missing_charge"),
        ("retention_time", "missing_retention_time"),
    ]:
        mask = output[column].isna() | (output[column].astype(str).str.strip() == "")
        filter_counts[reason] = int(mask.sum())
        keep &= ~mask
    if column_map.get("q_value") is not None:
        mask = output["q_value"].notna() & (output["q_value"] > options.q_value_threshold)
        filter_counts["q_value_above_threshold"] = int(mask.sum())
        keep &= ~mask
    if column_map.get("psm_probability") is not None:
        mask = output["psm_probability"].notna() & (output["psm_probability"] < options.probability_threshold)
        filter_counts["probability_below_threshold"] = int(mask.sum())
        keep &= ~mask

    filtered = output.loc[keep].copy()
    filtered["charge"] = filtered["charge"].astype("Int64")
    report = RtSearchInput(
        path=str(path),
        rows_in=int(len(frame)),
        rows_out=int(len(filtered)),
        column_map=column_map,
        warnings=warnings,
        filter_counts={key: value for key, value in sorted(filter_counts.items()) if value},
        task_build_plan_matched_sources=matched_sources,
        task_build_plan_unmatched_sources=unmatched_sources if metadata else [],
    )
    return filtered, report


def _detect_columns(columns: pd.Index) -> dict[str, str | None]:
    available = list(columns)
    return {
        "peptide_sequence": _find_column(available, PEPTIDE_COLUMNS),
        "modified_sequence": _find_column(available, MODIFIED_COLUMNS),
        "charge": _find_column(available, CHARGE_COLUMNS),
        "retention_time": _find_column(available, RT_COLUMNS),
        "q_value": _find_column(available, Q_VALUE_COLUMNS),
        "psm_probability": _find_column(available, PROBABILITY_COLUMNS),
        "spectrum_id": _find_column(available, SPECTRUM_COLUMNS),
        "source_file": _find_column(available, SOURCE_FILE_COLUMNS),
        "search_engine": _find_column(available, SEARCH_ENGINE_COLUMNS),
    }


def _find_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized = {_normalize_column(column): column for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    for candidate in candidates:
        for key, original in normalized.items():
            if candidate in key:
                return original
    return None


def _normalize_column(value: str) -> str:
    return " ".join(str(value or "").strip().casefold().replace("_", " ").replace("-", " ").split())


def _series(frame: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None:
        return pd.Series([pd.NA] * len(frame), index=frame.index)
    return frame[column]


def _optional_series(frame: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None:
        return pd.Series([""] * len(frame), index=frame.index)
    return frame[column].fillna("")


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.casefold() in {"nan", "none", "null"}:
        return ""
    return text


def _first_nonempty(values: pd.Series) -> str | None:
    for value in values:
        text = _clean_text(value)
        if text:
            return text
    return None


def _retention_time_unit(column: str) -> str:
    unit, _source = _retention_time_unit_with_source(column)
    return unit


def _retention_time_unit_with_source(column: str) -> tuple[str, str]:
    """Infer RT unit from column name.

    Display may still use minute when evidence is missing, but source is
    inferred_default/unknown so release science fails closed.
    """
    text = _normalize_column(column)
    if "sec" in text or text.endswith(" s") or "(s)" in text or text.endswith("_s"):
        return "second", "column_explicit"
    if "min" in text or "(min)" in text or text.endswith("_min"):
        return "minute", "column_explicit"
    return "minute", "inferred_default"


def _guess_search_engine(path: Path) -> str:
    name = path.name.casefold()
    if "sage" in name:
        return "sage"
    if "msfragger" in name or "fragpipe" in name:
        return "fragpipe"
    if "comet" in name:
        return "comet"
    return "unknown"


def _load_task_build_metadata(task_build_plan: str | Path | None) -> dict[str, dict[str, Any]]:
    if task_build_plan is None:
        return {}
    payload = json.loads(Path(task_build_plan).read_text(encoding="utf-8"))
    records: dict[str, dict[str, Any]] = {}
    for item in payload.get("files", []):
        source_file = str(item.get("file_name") or "").strip()
        if not source_file:
            continue
        records[source_file.casefold()] = {
            "project_accession": item.get("project_accession"),
            "source_file": source_file,
            "species": _join_values(item.get("species") or []),
            "canonical_species": _join_values(item.get("canonical_species") or []),
            "organism_taxon_id": _join_values(item.get("organism_taxon_id") or []),
            "instrument_family": _join_values(item.get("instrument_families") or []),
            "instrument_name": _join_values(item.get("instrument_names") or []),
            "instrument_vendor": item.get("instrument_vendor"),
            "fragmentation_method": _join_values(item.get("fragmentation_methods") or []),
            "fragmentation_methods": _join_values(item.get("fragmentation_methods") or []),
            "acquisition_mode": item.get("acquisition_mode"),
            "isolation_window": item.get("isolation_window"),
            "resolution": item.get("resolution"),
            "collision_energy": item.get("collision_energy"),
            "scan_range": item.get("scan_range"),
            "lc_gradient_minutes": item.get("lc_gradient_minutes"),
            "tissue": item.get("tissue"),
            "enzyme": item.get("enzyme"),
            "ptm_type": item.get("ptm_type"),
            "ptm_sites": item.get("ptm_sites"),
            "modification_scope": item.get("modification_scope"),
            "labeling_strategy": item.get("labeling_strategy"),
        }
    return records


def _metadata_for_source(metadata: dict[str, dict[str, Any]], source_file: str, stem: str) -> dict[str, Any]:
    if not metadata:
        return {}
    keys = [source_file.casefold(), Path(source_file).name.casefold(), stem.casefold()]
    for key in keys:
        if key in metadata:
            return metadata[key]
    for key, value in metadata.items():
        if stem.casefold() and stem.casefold() in key:
            return value
    return {}


def _metadata_by_source(
    metadata: dict[str, dict[str, Any]],
    sources: pd.Series,
    stem: str,
) -> dict[str, tuple[dict[str, Any], bool]]:
    result: dict[str, tuple[dict[str, Any], bool]] = {}
    for source in sources.astype(str).fillna(""):
        text = _clean_text(source) or stem
        if text in result:
            continue
        stem_fallback = stem if text == stem else ""
        matched = _metadata_for_source(metadata, text, stem_fallback)
        result[text] = (matched, bool(matched))
    return result


def _aggregate_peptide_level(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=RT_PEPTIDE_COLUMNS)
    working = frame.copy()
    working["retention_time"] = pd.to_numeric(working["retention_time"], errors="coerce")
    working["q_value"] = pd.to_numeric(working["q_value"], errors="coerce")
    working["psm_probability"] = pd.to_numeric(working["psm_probability"], errors="coerce")
    working["lc_gradient_minutes"] = pd.to_numeric(working["lc_gradient_minutes"], errors="coerce")
    grouped_rows: list[dict[str, Any]] = []
    group_columns = ["modified_sequence", "charge", "source_file"]
    for (modified_sequence, charge, source_file), group in working.groupby(group_columns, dropna=False):
        grouped_rows.append(
            {
                "project_accession": _first_value(group["project_accession"]),
                "source_file": source_file,
                "peptide_sequence": _first_value(group["peptide_sequence"]),
                "modified_sequence": modified_sequence,
                "charge": charge,
                "retention_time_median": _float_or_none(group["retention_time"].median()),
                "retention_time_mean": _float_or_none(group["retention_time"].mean()),
                "retention_time_unit": _first_value(group["retention_time_unit"]),
                "rt_unit_source": _first_value(group["rt_unit_source"]) if "rt_unit_source" in group.columns else "unknown",
                "psm_count": int(len(group)),
                "best_q_value": _float_or_none(group["q_value"].min(skipna=True)),
                "best_psm_probability": _float_or_none(group["psm_probability"].max(skipna=True)),
                "search_result_paths": ";".join(sorted(set(group["search_result_path"].dropna().astype(str)))),
                "search_engine": _first_value(group["search_engine"]),
                "species": _first_value(group["species"]),
                "canonical_species": _first_value(group["canonical_species"]),
                "organism_taxon_id": _first_value(group["organism_taxon_id"]),
                "instrument_family": _first_value(group["instrument_family"]),
                "lc_gradient_minutes": _float_or_none(group["lc_gradient_minutes"].dropna().iloc[0])
                if not group["lc_gradient_minutes"].dropna().empty
                else None,
                "ptm_type": _first_value(group["ptm_type"]),
                "modification_scope": _first_value(group["modification_scope"]),
                "labeling_strategy": _first_value(group["labeling_strategy"]),
                "label_source": "psm_level_rt_aggregation",
            }
        )
    output = pd.DataFrame(grouped_rows)
    return output.loc[:, RT_PEPTIDE_COLUMNS] if not output.empty else pd.DataFrame(columns=RT_PEPTIDE_COLUMNS)


def _peptide_aggregation_report(psm_frame: pd.DataFrame, peptide_frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "status": exporter_status_from_rows(len(peptide_frame)),
        "schema_version": RT_SCHEMA_VERSION,
        "psm_rows": int(len(psm_frame)),
        "peptide_rows": int(len(peptide_frame)),
        "aggregation_key": ["modified_sequence", "charge", "source_file"],
        "retention_time_median": _numeric_distribution(peptide_frame.get("retention_time_median")),
        "retention_time_mean": _numeric_distribution(peptide_frame.get("retention_time_mean")),
        "psm_count_distribution": _numeric_distribution(peptide_frame.get("psm_count")),
        "source_file_count": int(peptide_frame["source_file"].replace("", pd.NA).dropna().nunique())
        if not peptide_frame.empty
        else 0,
    }


def _validation_report(
    *,
    combined: pd.DataFrame,
    peptide_frame: pd.DataFrame,
    input_reports: list[RtSearchInput],
    filter_counts: Counter[str],
    warnings: list[str],
    task_build_plan_supplied: bool,
) -> dict[str, Any]:
    confidence_inputs = [
        item
        for item in input_reports
        if item.column_map.get("q_value") is not None or item.column_map.get("psm_probability") is not None
    ]
    missing_columns: Counter[str] = Counter()
    matched_sources: set[str] = set()
    unmatched_sources: set[str] = set()
    for item in input_reports:
        missing_columns.update(item.missing_required_columns)
        matched_sources.update(item.task_build_plan_matched_sources)
        unmatched_sources.update(item.task_build_plan_unmatched_sources)
    unit_sources = []
    if not combined.empty and "rt_unit_source" in combined.columns:
        unit_sources = sorted({str(v) for v in combined["rt_unit_source"].dropna().astype(str)})
    rt_unit_source = unit_sources[0] if len(unit_sources) == 1 else ("mixed" if unit_sources else "unknown")
    has_confidence = any(
        item.column_map.get("q_value") is not None or item.column_map.get("psm_probability") is not None
        for item in input_reports
    )
    return {
        "status": exporter_status_from_rows(len(combined)),
        "schema_version": RT_SCHEMA_VERSION,
        "input_files": len(input_reports),
        "rt_unit_source": rt_unit_source,
        "has_confidence_column": has_confidence,
        "rows_in": sum(item.rows_in for item in input_reports),
        "psm_rows_out": int(len(combined)),
        "peptide_rows_out": int(len(peptide_frame)),
        "filter_counts": dict(sorted(filter_counts.items())),
        "missing_required_column_counts": dict(sorted(missing_columns.items())),
        "warnings": sorted(set(warnings)),
        "retention_time": _numeric_distribution(combined.get("retention_time")),
        "charge_distribution": _value_distribution(combined.get("charge")),
        "project_count": int(combined["project_accession"].replace("", pd.NA).dropna().nunique())
        if not combined.empty
        else 0,
        "source_file_count": int(combined["source_file"].replace("", pd.NA).dropna().nunique())
        if not combined.empty
        else 0,
        "source_files": sorted(set(combined["source_file"].dropna().astype(str))) if not combined.empty else [],
        "confidence_column": {
            "inputs_with_confidence": len(confidence_inputs),
            "inputs_without_confidence": len(input_reports) - len(confidence_inputs),
        },
        "task_build_plan": {
            "supplied": task_build_plan_supplied,
            "matched_sources": sorted(matched_sources),
            "unmatched_sources": sorted(unmatched_sources),
        },
    }


def _join_values(values: list[Any]) -> str:
    return ";".join(str(value) for value in values if str(value or "").strip())


def _numeric_distribution(series: pd.Series | None) -> dict[str, float | int | None]:
    if series is None:
        return {"count": 0, "min": None, "median": None, "max": None}
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {
        "count": int(len(numeric)),
        "min": float(numeric.min()),
        "median": float(numeric.median()),
        "max": float(numeric.max()),
    }


def _value_distribution(series: pd.Series | None) -> dict[str, int]:
    if series is None:
        return {}
    return {
        str(key): int(value)
        for key, value in sorted(Counter(series.dropna().astype(str)).items())
    }


def _first_value(series: pd.Series) -> Any:
    for value in series:
        text = _clean_text(value)
        if text:
            return value
    return ""


def _float_or_none(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _schema_payload() -> dict[str, Any]:
    return {
        "schema_version": RT_SCHEMA_VERSION,
        "target_schema": "rt_train.parquet",
        "peptide_target_schema": "rt_train_peptide.parquet",
        "columns": {
            "project_accession": "PRIDE or repository project accession.",
            "source_file": "Original raw/mzML/peaklist file name if available.",
            "search_result_path": "Input search result TSV path.",
            "peptide_sequence": "Unmodified peptide sequence.",
            "modified_sequence": "Modified peptide string when available, otherwise peptide_sequence.",
            "charge": "Precursor charge.",
            "retention_time": "Observed retention time.",
            "retention_time_unit": "minute, second, or unknown when column/user evidence exists.",
            "rt_unit_source": "column_explicit | user_supplied | unknown provenance for release gates.",
            "spectrum_id": "Spectrum/scan identifier when available.",
            "q_value": "PSM or peptide q-value when available.",
            "psm_probability": "PSM probability when available.",
            "search_engine": "Search engine inferred or supplied.",
            "species": "Species from task build metadata when available.",
            "instrument_family": "Instrument family from discovery/task metadata when available.",
            "lc_gradient_minutes": "LC gradient minutes from discovery/task metadata when available.",
            "ptm_type": "PTM type from discovery/task metadata when available.",
            "label_source": "Source of the RT label.",
        },
        "peptide_columns": {
            "project_accession": "PRIDE or repository project accession.",
            "source_file": "Original raw/mzML/peaklist file name.",
            "peptide_sequence": "Unmodified peptide sequence.",
            "modified_sequence": "Modified peptide string.",
            "charge": "Precursor charge.",
            "retention_time_median": "Median observed retention time for the peptide/source/charge group.",
            "retention_time_mean": "Mean observed retention time for the peptide/source/charge group.",
            "retention_time_unit": "minute or second.",
            "psm_count": "Number of PSM-level rows in the aggregate.",
            "best_q_value": "Minimum q-value across grouped PSM rows.",
            "best_psm_probability": "Maximum probability across grouped PSM rows.",
            "search_result_paths": "Semicolon-joined input search result paths represented in the group.",
            "search_engine": "Search engine inferred or supplied.",
            "species": "Species from task build metadata when available.",
            "instrument_family": "Instrument family from discovery/task metadata when available.",
            "lc_gradient_minutes": "LC gradient minutes from discovery/task metadata when available.",
            "ptm_type": "PTM type from discovery/task metadata when available.",
            "label_source": "Source of the peptide-level RT label.",
        },
    }
