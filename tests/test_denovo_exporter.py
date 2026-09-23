from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from agent.ai_ready.denovo_exporter import export_denovo_ai_ready
from agent.cli import app


def _write_tsv(path: Path, rows: list[dict[str, object]]) -> Path:
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return path


def _write_mgf(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "BEGIN IONS",
                "TITLE=scan=101",
                "SCANS=101",
                "PEPMASS=500.2",
                "CHARGE=2+",
                "100.1 1000",
                "200.2 500",
                "END IONS",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_denovo_exporter_exports_parquet_from_psm_and_mgf(tmp_path: Path):
    search_result = _write_tsv(
        tmp_path / "psm.tsv",
        [
            {
                "Peptide": "PEPTIDEK",
                "Modified Peptide": "PEPTIDEK",
                "Charge": 2,
                "Spectrum": "scan=101",
                "Spectrum File": "sample.raw",
                "PSM Q-Value": 0.001,
                "PeptideProphet Probability": 0.95,
            }
        ],
    )
    peaklist = _write_mgf(tmp_path / "spectra.mgf")

    result = export_denovo_ai_ready(
        [search_result],
        [peaklist],
        tmp_path / "denovo",
        project_accession="PXD000001",
    )

    assert result.status == "completed"
    assert result.rows_out == 1
    frame = pd.read_parquet(result.output_parquet)
    assert frame.loc[0, "project_accession"] == "PXD000001"
    assert frame.loc[0, "peptide_sequence"] == "PEPTIDEK"
    assert frame.loc[0, "modified_sequence"] == "PEPTIDEK"
    assert frame.loc[0, "label_source"] == "high_confidence_psm_tsv_plus_mgf"
    assert json.loads(frame.loc[0, "spectrum_mz_json"]) == [100.1, 200.2]
    assert Path(result.preview_csv).exists()
    assert Path(result.report_json).exists()
    assert Path(result.schema_json_path).exists()


def test_denovo_exporter_filters_threshold_and_unmatched_spectra(tmp_path: Path):
    search_result = _write_tsv(
        tmp_path / "psm.tsv",
        [
            {"Peptide": "KEEPK", "Charge": 2, "Spectrum": "scan=101", "PSM Q-Value": 0.001},
            {"Peptide": "DROPQ", "Charge": 2, "Spectrum": "scan=101", "PSM Q-Value": 0.2},
            {"Peptide": "MISSK", "Charge": 2, "Spectrum": "scan=999", "PSM Q-Value": 0.001},
        ],
    )
    peaklist = _write_mgf(tmp_path / "spectra.mgf")

    result = export_denovo_ai_ready([search_result], [peaklist], tmp_path / "denovo")

    assert result.rows_in == 3
    assert result.rows_out == 1
    assert result.filter_counts["q_value_above_threshold"] == 1
    assert result.filter_counts["spectrum_not_matched"] == 1
    assert "spectrum_not_matched" in result.warnings


def test_denovo_exporter_keeps_modified_sequence_but_uses_clean_main_label(tmp_path: Path):
    search_result = _write_tsv(
        tmp_path / "psm.tsv",
        [
            {
                "Peptide": "PEPTIDEK",
                "Modified Peptide": "PEP[+80]TIDEK",
                "Charge": 2,
                "Spectrum": "scan=101",
                "PSM Q-Value": 0.001,
            }
        ],
    )
    peaklist = _write_mgf(tmp_path / "spectra.mgf")

    result = export_denovo_ai_ready([search_result], [peaklist], tmp_path / "denovo")

    frame = pd.read_parquet(result.output_parquet)
    assert frame.loc[0, "peptide_sequence"] == "PEPTIDEK"
    assert frame.loc[0, "modified_sequence"] == "PEP[+80]TIDEK"


def test_denovo_exporter_uses_task_build_plan_metadata(tmp_path: Path):
    search_result = _write_tsv(
        tmp_path / "psm.tsv",
        [{"Peptide": "PEPTIDEK", "Charge": 2, "Spectrum": "scan=101", "Spectrum File": "sample.raw", "PSM Q-Value": 0.001}],
    )
    peaklist = _write_mgf(tmp_path / "spectra.mgf")
    task_build_plan = tmp_path / "discovery_task_build_plan.json"
    task_build_plan.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "project_accession": "PXD000002",
                        "file_name": "sample.raw",
                        "species": ["human"],
                        "tissue": "liver",
                        "enzyme": "Trypsin",
                        "instrument_names": ["Q Exactive HF"],
                        "instrument_vendor": "Thermo Fisher Scientific",
                        "instrument_families": ["orbitrap"],
                        "fragmentation_methods": ["HCD"],
                        "acquisition_mode": "dda",
                        "isolation_window": 1.6,
                        "resolution": 30000,
                        "collision_energy": 30,
                        "scan_range": "100-1600 m/z",
                        "lc_gradient_minutes": 90,
                        "ptm_type": "phospho",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = export_denovo_ai_ready([search_result], [peaklist], tmp_path / "denovo", task_build_plan=task_build_plan)

    frame = pd.read_parquet(result.output_parquet)
    assert frame.loc[0, "project_accession"] == "PXD000002"
    assert frame.loc[0, "species"] == "human"
    assert frame.loc[0, "instrument_family"] == "orbitrap"
    assert frame.loc[0, "instrument_name"] == "Q Exactive HF"
    assert frame.loc[0, "instrument_vendor"] == "Thermo Fisher Scientific"
    assert frame.loc[0, "fragmentation_method"] == "HCD"
    assert frame.loc[0, "acquisition_mode"] == "dda"
    assert frame.loc[0, "isolation_window"] == 1.6
    assert frame.loc[0, "resolution"] == 30000
    assert frame.loc[0, "collision_energy"] == 30
    assert frame.loc[0, "scan_range"] == "100-1600 m/z"
    assert frame.loc[0, "lc_gradient_minutes"] == 90
    assert frame.loc[0, "tissue"] == "liver"
    assert frame.loc[0, "enzyme"] == "Trypsin"
    assert frame.loc[0, "ptm_type"] == "phospho"


def test_denovo_exporter_collapses_exact_fragpipe_sage_consensus(tmp_path: Path):
    fragpipe = _write_tsv(
        tmp_path / "fragpipe.tsv",
        [{
            "Peptide": "PEPTIDEK", "Modified Peptide": "PEP[+80]TIDEK",
            "Charge": 2, "Spectrum": "scan=101", "Spectrum File": "sample.raw",
            "PSM Q-Value": 0.003, "Search Engine": "MSFragger",
        }],
    )
    sage = _write_tsv(
        tmp_path / "sage.tsv",
        [{
            "Peptide": "PEPTIDEK", "Modified Peptide": "PEP[+80]TIDEK",
            "Charge": 2, "Spectrum": "scan=101", "Spectrum File": "sample.raw",
            "PSM Q-Value": 0.007, "Search Engine": "Sage",
        }],
    )

    result = export_denovo_ai_ready(
        [fragpipe, sage], [_write_mgf(tmp_path / "spectra.mgf")], tmp_path / "denovo"
    )

    frame = pd.read_parquet(result.output_parquet)
    assert len(frame) == 1
    assert frame.loc[0, "search_engine"] == "FragPipe∩Sage"
    assert json.loads(frame.loc[0, "search_engines_json"]) == ["fragpipe", "sage"]
    assert json.loads(frame.loc[0, "engine_q_values_json"]) == {
        "fragpipe": 0.003,
        "sage": 0.007,
    }
    assert frame.loc[0, "q_value"] == 0.007


def test_denovo_exporter_cli_writes_outputs(tmp_path: Path):
    search_result = _write_tsv(
        tmp_path / "psm.tsv",
        [{"Peptide": "PEPTIDEK", "Charge": 2, "Spectrum": "scan=101", "PSM Q-Value": 0.001}],
    )
    peaklist = _write_mgf(tmp_path / "spectra.mgf")
    output_dir = tmp_path / "denovo"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "export-denovo-ai-ready",
            "--search-result",
            str(search_result),
            "--peaklist",
            str(peaklist),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["rows_out"] == 1
    assert (output_dir / "denovo_train.parquet").exists()
    assert (output_dir / "denovo.preview.csv").exists()
    assert (output_dir / "denovo_export_report.json").exists()
    assert (output_dir / "denovo_schema.json").exists()
