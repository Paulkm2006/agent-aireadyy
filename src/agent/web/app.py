from __future__ import annotations

import asyncio
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import sqlite3
import threading
import time
import uuid
import zipfile
from collections import Counter, deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import monotonic
from types import SimpleNamespace
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from pydantic import BaseModel, ConfigDict
from fastapi import BackgroundTasks, FastAPI, Request, WebSocket, WebSocketDisconnect
from starlette.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from agent.agent_core.harness import run_agent_harness
from agent.ai_ready.agent_run_bridge import build_ai_ready_from_agent_run
from agent.ai_ready.agent_run_locator import locate_agent_run_inputs
from agent.ai_ready.input_locator import locate_ai_ready_inputs, select_ai_ready_inputs
from agent.ai_ready.input_profile import profile_ai_ready_inputs
from agent.ai_ready.data_scientist_loop import run_data_scientist_agent_loop
from agent.ai_ready.data_scientist_report import make_data_scientist_agent_report
from agent.ai_ready.curation_memory import apply_curation_decisions_to_memory
from agent.ai_ready.dataset_recipe import make_dataset_recipe
from agent.ai_ready.guidance_alignment import make_guidance_alignment_report
from agent.ai_ready.mini_e2e import validate_agent_run_ai_ready_mini
from agent.ai_ready.mini_e2e_batch import validate_agent_runs_ai_ready_batch
from agent.ai_ready.model_informed_discovery import (
    discovery_payload_from_model_request as build_model_informed_discovery_payload,
    model_informed_repository_plan as build_model_informed_repository_plan,
    model_request_records as model_informed_request_records,
)
from agent.ai_ready.model_loop import run_dataset_model_loop
from agent.ai_ready.real_smoke import run_ai_ready_real_smoke
from agent.ai_ready.validation import validate_ai_ready_build
from agent.control_plane.models import (
    AgentBudget,
    AgentEvent,
    DiscoveryQualityAudit,
    DynamicBudgetLimits,
    RuntimeProvenance,
    recommended_inspection_rounds,
)
from agent.discovery.agenda import agenda_for_manager
from agent.discovery.runner import run_agents_discovery
from agent.discovery.agentic import AgenticDiscoveryPlanner, OpenAICompatibleDiscoveryLLM, default_agentic_discovery_planner, default_discovery_llm_client
from agent.discovery.agentic_runner import run_agentic_discovery
from agent.discovery.features import extract_file_features, extract_project_features
from agent.discovery.constraints import (
    ScientificConstraint,
    constraint_slug,
    normalize_scientific_constraints,
    normalize_scientific_constraints_result,
)
from agent.discovery.manifest import write_dataset_manifest
from agent.discovery.memory import (
    VALID_REVIEW_DECISIONS,
    VALID_REVIEW_REASONS,
    DiscoveryMemory,
    DiscoveryReviewDecision,
    build_run_record,
    generate_discovery_run_id,
    load_dataset_manifest,
    now_utc_iso,
)
from agent.discovery.models import DatasetManifest, DatasetRequest, DiscoveredFile, DiscoveredProject, DiscoveryEvidence
from agent.discovery.portfolio import infer_portfolio_spec
from agent.discovery.ontology import (
    SPECIES_TERMS,
    general_query_terms_from_text,
    interpret_immunopeptide_metadata,
    is_immunopeptidomics_goal,
    normalize_labeling_strategy,
    normalize_ptm_type,
    normalize_species_values,
    species_from_text,
)
from agent.discovery.pride_discovery import discover_pride_dataset
from agent.discovery.publication import (
    BusinessCompletionDecision,
    business_completion_allows_success,
)
from agent.discovery.search_environment import PrideDiscoverySearchEnvironment
from agent.discovery.repository_discovery import discover_repository_dataset
from agent.discovery.scoring import build_discovered_project, classify_file_role, score_file, score_project
from agent.discovery.task_readiness import annotate_manifest_task_readiness, normalize_task_type
from agent.discovery.task_profiles import get_task_profile
from agent.discovery.validity import assess_file_validity, assess_project_validity
from agent.input.normalizer import safe_output_stem
from agent.metadata.context import detect_sdrf_file, load_sdrf_rows, select_sdrf_rows_for_file
from agent.oneclick.preflight import normalize_resource_policy, normalize_run_mode, run_preflight
from agent.progress import render_download_progress
from agent.pride.client import (
    PrideClient,
    PridePaginationState,
    list_project_files_paginated_with_state,
)
from agent.repositories.smoke import run_repository_smoke
from agent.repositories.iprox_adapter import IproxAdapter, refresh_public_iprox_index
from agent.repositories.massive_adapter import MassiveAdapter
from agent.repositories.pride_adapter import PrideAdapter
from agent.repositories.registry import RepositoryRegistry
from agent.runtime.system_metrics import collect_system_metrics
from agent.utils import write_json
from agent.operations.api import router as operations_router
from agent.operations.legacy import (
    import_legacy_discovery_summaries,
    import_legacy_history_index,
    is_material_legacy_history_record,
)
from agent.operations.queue import enqueue_discovery_job, revoke_discovery_job
from agent.operations.runtime import (
    close_operations_repository,
    get_operations_repository,
)
from agent.web.history import history_timestamp, merge_project_history_records, with_history_identity
from agent.web.storage_lifecycle import (
    clean_item_source_assets,
    delete_managed_tree,
    managed_child,
    path_size_bytes,
)
from agent.web.expert_review import ExpertPoolRegistry, expert_review_enabled, expert_review_root
from agent.web.expert_review.grading import (
    append_human_grade,
    apply_human_grades_for_export,
    effective_grade,
    queue_bucket,
)
from agent.web.expert_review.impact import compute_impact, load_json
from agent.web.expert_review.jobs import MAX_EXPERT_JOB_WORKERS, ExpertJudgeJobManager, reset_jobs_for_tests
from agent.web.expert_review.build_projection import attach_review_progress

try:
    from agents import RunContextWrapper
except ImportError:  # pragma: no cover - the direct adapter remains an optional fallback
    RunContextWrapper = Any  # type: ignore[assignment,misc]
from agent.web.expert_review.pool_builds import ExpertPoolBuildManager
from agent.web.expert_review.task_semantics import calibration_task_identity, interpret_review_task
from agent.web.expert_review.workspace_archive import (
    MAX_WORKSPACE_ARCHIVE_BYTES,
    WorkspaceArchiveError,
    export_workspace_archive,
    import_workspace_archive,
)
from agent.web.llm_config_store import LLMConfigStore


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _runs_dir.mkdir(exist_ok=True)
    operations = get_operations_repository()
    await asyncio.to_thread(
        import_legacy_discovery_summaries,
        operations,
        _runs_dir,
    )
    _sync_history_index_from_disk()
    await asyncio.to_thread(
        import_legacy_history_index,
        operations,
        _runs_dir,
    )
    _repair_interrupted_history_index()
    _start_result_cleanup_worker()
    try:
        yield
    finally:
        close_operations_repository()


app = FastAPI(title="PRIDE AI-ready Agent", version="0.3.1", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=5)
app.include_router(operations_router)

_benchmark_review_next_dir = Path(__file__).parent / "static" / "benchmark-review-next"
app.mount(
    "/benchmark-review-next/assets",
    StaticFiles(directory=_benchmark_review_next_dir / "assets", check_dir=False),
    name="benchmark-review-next-assets",
)
app.mount(
    "/benchmark-review/assets",
    StaticFiles(directory=_benchmark_review_next_dir / "assets", check_dir=False),
    name="benchmark-review-assets",
)

_tasks: dict[str, dict[str, Any]] = {}
_tasks_lock = threading.Lock()
_batches: dict[str, dict[str, Any]] = {}
_batches_lock = threading.Lock()
_batch_history_cache: dict[str, Any] = {
    "ts": 0.0,
    "root": "",
    "generation": 0,
    "records": [],
}
_batch_history_cache_lock = threading.Lock()
_history_index_lock = threading.RLock()
_history_delete_confirmations: dict[str, dict[str, Any]] = {}
_history_delete_confirmations_lock = threading.Lock()
_discovery_jobs: dict[str, dict[str, Any]] = {}
_discovery_jobs_lock = threading.RLock()
_runs_dir = Path(os.getenv("AGENT_RUNS_DIR", "runs")).resolve()
_templates_dir = Path(__file__).parent / "templates"
_ACTIVE_STATUSES = {"queued", "running"}
_TERMINAL_STATUSES = {"completed", "failed", "blocked", "cancelled", "interrupted"}
_cleanup_thread_started = False
_PUBLIC_HISTORY_FILE = "task_history.json"
_HISTORY_INDEX_FILE = "project_history.json"
_DOWNLOAD_CACHE_DIR = ".download_cache"
_DOWNLOAD_ZIP_NAME = "results-compressed.zip"
_BATCHES_DIR_NAME = "_batches"
_BATCH_MANIFEST_FILE = "batch_manifest.json"
_BATCH_EXCEL_FILE = "benchmark_results.xlsx"
_BATCH_AUDIT_ZIP_NAME = "batch_parameter_audit.zip"
_DISCOVERY_DIR_NAME = "discovery"
_DISCOVERY_MEMORY_DIR_NAME = "discovery_memory"
_AI_READY_BUILDS_DIR_NAME = "ai_ready_builds"
_DOWNLOAD_RESULT_DIRS = {"ai_ready", "msdt", "rawspectrum", "logs"}
_DOWNLOAD_ROOT_SUFFIXES = {".json", ".txt", ".log", ".tsv", ".csv"}
_DOWNLOAD_FRAGPIPE_PARAMETER_FILES = {"fragger.params", "msbooster_params.txt"}
_DISCOVERY_DOWNLOAD_FILES = {
    "dataset_request": ("dataset_request.json", "application/json"),
    "candidate_projects": ("candidate_projects.json", "application/json"),
    "dataset_manifest_json": ("dataset_manifest.json", "application/json"),
    "dataset_manifest_csv": ("dataset_manifest.csv", "text/csv"),
    "dataset_manifest_valid_csv": ("dataset_manifest_valid.csv", "text/csv"),
    "dataset_manifest_usable_csv": ("dataset_manifest_usable.csv", "text/csv"),
    "dataset_manifest_task_ready_csv": ("dataset_manifest_task_ready.csv", "text/csv"),
    "file_judgments_jsonl": ("file_judgments.jsonl", "application/x-ndjson"),
    "selected_files_xlsx": (
        "selected_files.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ),
    "files_parquet": ("files.parquet", "application/vnd.apache.parquet"),
    "file_judgments_parquet": (
        "file_judgments.parquet",
        "application/vnd.apache.parquet",
    ),
    "file_evidence_parquet": (
        "file_evidence.parquet",
        "application/vnd.apache.parquet",
    ),
    "batch_inputs": ("batch_inputs.txt", "text/plain"),
    "batch_inputs_valid": ("batch_inputs_valid.txt", "text/plain"),
    "batch_inputs_usable": ("batch_inputs_usable.txt", "text/plain"),
    "batch_inputs_task_ready": ("batch_inputs_task_ready.txt", "text/plain"),
    "discovery_summary": ("discovery_summary.json", "application/json"),
    "quality_report": ("quality_report.json", "application/json"),
    "repository_audit_json": ("repository_audit.json", "application/json"),
    "repository_audit_csv": ("repository_audit.csv", "text/csv"),
    "repository_audit_md": ("repository_audit.md", "text/markdown"),
    "task_ai_readiness_matrix_json": ("task_ai_readiness_matrix.json", "application/json"),
    "task_ai_readiness_matrix_csv": ("task_ai_readiness_matrix.csv", "text/csv"),
    "data_value_ranking_json": ("data_value_ranking.json", "application/json"),
    "data_value_ranking_csv": ("data_value_ranking.csv", "text/csv"),
    "data_value_report_md": ("data_value_report.md", "text/markdown"),
    "data_value_strategy_eval_json": ("data_value_strategy_eval.json", "application/json"),
    "data_value_strategy_eval_csv": ("data_value_strategy_eval.csv", "text/csv"),
    "data_value_strategy_eval_md": ("data_value_strategy_eval.md", "text/markdown"),
    "agentic_plan": ("agentic_plan.json", "application/json"),
    "agentic_rounds": ("agentic_rounds.json", "application/json"),
    "agents_discovery_summary_json": ("agents_discovery_summary.json", "application/json"),
    "agents_discovery_events_json": ("agents_discovery_events.json", "application/json"),
    "agents_discovery_report_md": ("agents_discovery_report.md", "text/markdown"),
    "agents_discovery_budget_json": ("agents_discovery_budget.json", "application/json"),
    "discovery_run_bundle_zip": ("discovery_run_bundle.zip", "application/zip"),
    "discovery_job_log_jsonl": ("logs/discovery_job.jsonl", "application/x-ndjson"),
    "project_judgments_table_csv": ("project_judgments_table.csv", "text/csv"),
    "selected_projects_review_csv": ("selected_projects_review.csv", "text/csv"),
    "selected_projects_review_json": ("selected_projects_review.json", "application/json"),
    "project_judgments_json": ("project_judgments.json", "application/json"),
}
_AI_READY_DOWNLOAD_FILES = {
    "input_profile_json": ("ai_ready_input_profile.json", "application/json"),
    "input_profile_csv": ("ai_ready_input_profile.csv", "text/csv"),
    "validation_report_json": ("ai_ready_validation_report.json", "application/json"),
    "validation_report_csv": ("ai_ready_validation_report.csv", "text/csv"),
    "build_summary_json": ("ai_ready_build_summary.json", "application/json"),
    "build_report_md": ("ai_ready_build_report.md", "text/markdown"),
    "builder_summary_json": ("agentic_dataset_build_summary.json", "application/json"),
    "builder_report_md": ("agentic_dataset_build_report.md", "text/markdown"),
    "builder_recommendations_json": ("agentic_dataset_build_recommendations.json", "application/json"),
    "input_locations_json": ("ai_ready_input_locations.json", "application/json"),
    "input_locations_csv": ("ai_ready_input_locations.csv", "text/csv"),
    "agent_run_input_locations_json": ("agent_run_input_locations.json", "application/json"),
    "agent_run_input_locations_csv": ("agent_run_input_locations.csv", "text/csv"),
    "agent_run_build_summary_json": ("agent_run_build_summary.json", "application/json"),
    "agent_run_build_report_md": ("agent_run_build_report.md", "text/markdown"),
    "mini_e2e_summary_json": ("mini_e2e_summary.json", "application/json"),
    "mini_e2e_report_md": ("mini_e2e_report.md", "text/markdown"),
    "mini_e2e_upstream_recovery_json": ("upstream_recovery/agent_recovery_report.json", "application/json"),
    "mini_e2e_upstream_recovery_md": ("upstream_recovery/agent_recovery_report.md", "text/markdown"),
    "mini_e2e_batch_summary_json": ("mini_e2e_batch_summary.json", "application/json"),
    "mini_e2e_batch_summary_csv": ("mini_e2e_batch_summary.csv", "text/csv"),
    "mini_e2e_batch_report_md": ("mini_e2e_batch_report.md", "text/markdown"),
    "real_smoke_summary_json": ("real_smoke_summary.json", "application/json"),
    "real_smoke_report_md": ("real_smoke_report.md", "text/markdown"),
    "discovery_feedback_preview_json": ("discovery_feedback_preview.json", "application/json"),
    "rt_report_json": ("rt_ai_ready/rt_export_report.json", "application/json"),
    "fragment_report_json": ("fragment_intensity_ai_ready/fragment_intensity_export_report.json", "application/json"),
    "psm_report_json": ("psm_scoring_ai_ready/psm_scoring_export_report.json", "application/json"),
    "denovo_report_json": ("denovo_ai_ready/denovo_export_report.json", "application/json"),
    "ptm_denovo_report_json": ("ptm_denovo_ai_ready/ptm_denovo_export_report.json", "application/json"),
    "chimeric_report_json": ("chimeric_ai_ready/chimeric_export_report.json", "application/json"),
    "chimeric_train_parquet": ("chimeric_ai_ready/chimeric_train.parquet", "application/octet-stream"),
    "repository_smoke_summary_json": ("repository_smoke_summary.json", "application/json"),
    "repository_smoke_summary_csv": ("repository_smoke_summary.csv", "text/csv"),
    "repository_smoke_report_md": ("repository_smoke_report.md", "text/markdown"),
    "repository_resolution_json": ("repository_resolution.json", "application/json"),
    "repository_context_json": ("repository_context.json", "application/json"),
    "repository_asset_json": ("repository_asset.json", "application/json"),
    "repository_audit_json": ("repository_audit.json", "application/json"),
    "repository_audit_csv": ("repository_audit.csv", "text/csv"),
    "repository_audit_md": ("repository_audit.md", "text/markdown"),
    "iprox_index_summary_json": ("iprox_index_summary.json", "application/json"),
    "iprox_project_index_jsonl": ("iprox_project_index.jsonl", "application/x-jsonlines"),
    "iprox_file_index_jsonl": ("iprox_file_index.jsonl", "application/x-jsonlines"),
    "agent_harness_summary_json": ("agent_harness_summary.json", "application/json"),
    "agent_harness_summary_csv": ("agent_harness_summary.csv", "text/csv"),
    "agent_harness_report_md": ("agent_harness_report.md", "text/markdown"),
    "agent_decision_trace_json": ("agent_decision_trace.json", "application/json"),
    "dataset_recipe_json": ("dataset_recipe.json", "application/json"),
    "dataset_recipe_md": ("dataset_recipe.md", "text/markdown"),
    "selected_files_csv": ("selected_files.csv", "text/csv"),
    "excluded_files_csv": ("excluded_files.csv", "text/csv"),
    "dataset_split_plan_json": ("dataset_split_plan.json", "application/json"),
    "dataset_split_plan_csv": ("dataset_split_plan.csv", "text/csv"),
    "split_rationale_md": ("split_rationale.md", "text/markdown"),
    "leakage_risk_report_json": ("leakage_risk_report.json", "application/json"),
    "leakage_risk_report_md": ("leakage_risk_report.md", "text/markdown"),
    "split_baseline_evaluation_json": ("split_baseline_evaluation.json", "application/json"),
    "split_baseline_evaluation_csv": ("split_baseline_evaluation.csv", "text/csv"),
    "split_baseline_evaluation_md": ("split_baseline_evaluation.md", "text/markdown"),
    "hard_benchmark_manifest_json": ("hard_benchmark_manifest.json", "application/json"),
    "hard_benchmark_manifest_csv": ("hard_benchmark_manifest.csv", "text/csv"),
    "counterfactual_benchmark_manifest_json": ("counterfactual_benchmark_manifest.json", "application/json"),
    "counterfactual_benchmark_manifest_csv": ("counterfactual_benchmark_manifest.csv", "text/csv"),
    "counterfactual_benchmark_report_md": ("counterfactual_benchmark_report.md", "text/markdown"),
    "coverage_gap_report_json": ("coverage_gap_report.json", "application/json"),
    "coverage_gap_report_md": ("coverage_gap_report.md", "text/markdown"),
    "agent_expansion_plan_json": ("agent_expansion_plan.json", "application/json"),
    "evidence_graph_json": ("evidence_graph.json", "application/json"),
    "evidence_graph_summary_md": ("evidence_graph_summary.md", "text/markdown"),
    "curation_queue_csv": ("curation_queue.csv", "text/csv"),
    "curation_queue_json": ("curation_queue.json", "application/json"),
    "curation_efficiency_report_json": ("curation_efficiency_report.json", "application/json"),
    "curation_efficiency_report_csv": ("curation_efficiency_report.csv", "text/csv"),
    "curation_efficiency_report_md": ("curation_efficiency_report.md", "text/markdown"),
    "curation_memory_update_json": ("curation_memory_update.json", "application/json"),
    "curation_memory_update_csv": ("curation_memory_update.csv", "text/csv"),
    "curation_memory_update_md": ("curation_memory_update.md", "text/markdown"),
    "model_eval_summary_json": ("model_eval_summary.json", "application/json"),
    "model_adapter_contract_json": ("model_adapter_contract.json", "application/json"),
    "model_adapter_contract_md": ("model_adapter_contract.md", "text/markdown"),
    "model_adapter_input_manifest_json": ("model_adapter_input_manifest.json", "application/json"),
    "model_adapter_input_manifest_csv": ("model_adapter_input_manifest.csv", "text/csv"),
    "external_model_metrics_json": ("external_model_metrics.json", "application/json"),
    "model_adapter_log": ("model_adapter.log", "text/plain"),
    "model_failure_modes_json": ("model_failure_modes.json", "application/json"),
    "model_loop_report_md": ("model_loop_report.md", "text/markdown"),
    "model_informed_gap_report_json": ("model_informed_gap_report.json", "application/json"),
    "model_informed_gap_report_md": ("model_informed_gap_report.md", "text/markdown"),
    "model_informed_expansion_plan_json": ("model_informed_expansion_plan.json", "application/json"),
    "model_informed_discovery_requests_json": ("model_informed_discovery_requests.json", "application/json"),
    "model_informed_discovery_requests_csv": ("model_informed_discovery_requests.csv", "text/csv"),
    "model_informed_discovery_requests_md": ("model_informed_discovery_requests.md", "text/markdown"),
    "model_informed_discovery_payloads_json": ("model_informed_discovery_payloads.json", "application/json"),
    "model_informed_discovery_payloads_csv": ("model_informed_discovery_payloads.csv", "text/csv"),
    "model_informed_discovery_payloads_md": ("model_informed_discovery_payloads.md", "text/markdown"),
    "model_informed_discovery_payload_queue_json": ("model_informed_discovery_payload_queue.json", "application/json"),
    "model_informed_discovery_payload_queue_csv": ("model_informed_discovery_payload_queue.csv", "text/csv"),
    "model_informed_discovery_payload_queue_md": ("model_informed_discovery_payload_queue.md", "text/markdown"),
    "model_informed_curation_queue_json": ("model_informed_curation_queue.json", "application/json"),
    "model_informed_curation_queue_csv": ("model_informed_curation_queue.csv", "text/csv"),
    "model_informed_curation_queue_md": ("model_informed_curation_queue.md", "text/markdown"),
    "model_strategy_comparison_json": ("model_strategy_comparison.json", "application/json"),
    "model_strategy_comparison_csv": ("model_strategy_comparison.csv", "text/csv"),
    "model_strategy_comparison_md": ("model_strategy_comparison.md", "text/markdown"),
    "real_data_scientist_agent_report_md": ("real_data_scientist_agent_report.md", "text/markdown"),
    "real_data_scientist_agent_summary_json": ("real_data_scientist_agent_summary.json", "application/json"),
    "guidance_alignment_report_json": ("guidance_alignment_report.json", "application/json"),
    "guidance_alignment_report_md": ("guidance_alignment_report.md", "text/markdown"),
    "data_scientist_agent_loop_summary_json": ("data_scientist_agent_loop_summary.json", "application/json"),
    "data_scientist_agent_loop_report_md": ("data_scientist_agent_loop_report.md", "text/markdown"),
}
_MAX_PERSISTED_LOGS = 2000
_DISCOVERY_SUMMARY_RECENT_LOGS = 240
_DISCOVERY_LOG_STRING_LIMIT = 240
_INTERRUPTED_HISTORY_MESSAGE = "服务重启或任务被手动停止，任务已中断。"
_RUN_MODE_FULL = "full"
_RUN_MODE_PREPARE = "prepare"
_RUN_MODE_PARAMETERS = "parameters"
_RUN_MODES = {_RUN_MODE_FULL, _RUN_MODE_PREPARE, _RUN_MODE_PARAMETERS}
_UI_LANGUAGES = {"en", "zh"}

# 默认配置（不从 .env 加载，由用户在页面填写）
_DEFAULT_CONFIG = {
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-v4-pro",
    "timeout": "1200",
}
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\[\d{1,3}(?:;\d{1,3})*m")
_CJK_RE = re.compile(r"[\u3400-\u9fff\u3000-\u303f\uff00-\uffef]")
def _app_timezone():
    timezone_name = os.getenv("TZ", "Asia/Shanghai")
    try:
        return ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        return timezone(timedelta(hours=8), "CST")


_APP_TZ = _app_timezone()


def _now() -> datetime:
    return datetime.now(_APP_TZ)


def _now_iso() -> str:
    return _now().isoformat()


def _now_time() -> str:
    return _now().strftime("%H:%M:%S")


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _full_workflow_enabled() -> bool:
    if _env_flag("AGENT_DISABLE_FULL_WORKFLOW", default=False):
        return False
    return _env_flag("AGENT_WEB_FULL_WORKFLOW_ENABLED", default=False)


def _clean_submitter(value: Any) -> str:
    submitter = _clean_text(value)
    submitter = re.sub(r"[\x00-\x1f\x7f]+", " ", submitter).strip()
    if not submitter:
        return "未填写"
    return submitter[:80]


def _clean_run_mode(value: Any) -> str:
    default = _RUN_MODE_FULL if _full_workflow_enabled() else _RUN_MODE_PREPARE
    mode = normalize_run_mode(value, default=default)
    if mode == _RUN_MODE_FULL and not _full_workflow_enabled():
        return _RUN_MODE_PREPARE
    return mode


def _clean_batch_run_mode(value: Any) -> str:
    return normalize_run_mode(value, default=_RUN_MODE_PARAMETERS)


def _clean_resource_policy(value: Any) -> str:
    return normalize_resource_policy(value)


def _clean_reviewed_fasta(value: Any) -> tuple[str | None, str | None]:
    fasta = _clean_text(value)
    if not fasta:
        return None, None
    if re.match(r"(?i)^(https?|ftp)://", fasta):
        return None, fasta
    return fasta, None


def _manifest_path_candidates(value: Any) -> list[Path]:
    text = _clean_text(value).replace("\\", "/")
    if not text:
        return []
    candidates: list[Path] = []
    path = Path(text)
    candidates.append(path if path.is_absolute() else Path.cwd() / path)
    marker = "data/local_mzml_samples/"
    if marker in text:
        candidates.append(Path.cwd() / text[text.index(marker) :])
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve(strict=False))
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _same_resolved_path(left: Path, right: Path) -> bool:
    return left.resolve(strict=False) == right.resolve(strict=False)


def _is_reusable_context_dir(path: Path) -> bool:
    return (path / "metadata.json").exists() and (path / "attributes.json").exists()


def _local_sample_context_dir_from_runs(project_accession: str, file_name: str) -> str | None:
    runs_dir = Path.cwd() / "runs"
    if not runs_dir.exists():
        return None
    stem = Path(file_name).stem
    candidates = [path for path in runs_dir.glob(f"{stem}*") if path.is_dir()]
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for candidate in candidates:
        if not _is_reusable_context_dir(candidate):
            continue
        task_state_path = candidate / "task_state.json"
        if task_state_path.exists():
            try:
                task_state = json.loads(task_state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                task_state = {}
            if (
                isinstance(task_state, dict)
                and _clean_text(task_state.get("project_accession"))
                and _clean_text(task_state.get("project_accession")).upper() != project_accession.upper()
            ):
                continue
        return str(candidate)
    return None


def _local_sample_context_dir_for_path(source_path: Path, project_accession: str, file_name: str) -> str | None:
    manifest_path = Path.cwd() / "data" / "local_mzml_samples" / "local_mzml_samples_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        records = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(records, list):
        return None
    for record in records:
        if not isinstance(record, dict):
            continue
        if _clean_text(record.get("project_accession")).upper() != project_accession.upper():
            continue
        record_file_name = _clean_text(record.get("file_name"))
        local_paths = [
            *_manifest_path_candidates(record.get("local_path")),
            *_manifest_path_candidates(record.get("prepared_mzml_path")),
        ]
        path_matches = any(_same_resolved_path(source_path, candidate) for candidate in local_paths)
        if not path_matches and record_file_name != file_name:
            continue
        for context_dir in _manifest_path_candidates(record.get("web_full_run_dir")):
            if _is_reusable_context_dir(context_dir):
                return str(context_dir)
        return _local_sample_context_dir_from_runs(project_accession, file_name)
    return None


def _known_local_source_from_input(value: Any) -> dict[str, str] | None:
    text = _clean_text(value)
    if not text or re.match(r"(?i)^(https?|ftp)://", text):
        return None
    path = Path(text)
    if not path.exists() or not path.is_file():
        return None
    lower = path.name.lower()
    if not lower.endswith((".raw", ".raw.zip", ".mzml", ".mzml.gz", ".mzxml", ".mzxml.gz")):
        return None
    project_accession = None
    for part in reversed(path.parts):
        if re.fullmatch(r"(?i)PXD\d{6,}", part):
            project_accession = part.upper()
            break
    if not project_accession:
        return None
    source = {
        "source_path": str(path),
        "project_accession": project_accession,
        "matched_file": path.name,
    }
    context_dir = _local_sample_context_dir_for_path(path, project_accession, path.name)
    if context_dir:
        source["context_dir"] = context_dir
    return source


def _run_mode_label(value: Any) -> str:
    mode = _clean_run_mode(value)
    if mode == _RUN_MODE_PARAMETERS:
        return "Parameters only"
    if mode == _RUN_MODE_PREPARE:
        return "Prepare input package"
    return "Full workflow"


def _clean_repository(value: Any, default: str = "pride") -> str:
    repository = _clean_text(value).lower().replace("-", "_")
    if repository in {"auto", "all"}:
        return "auto"
    if repository in {"pride", "px", "proteomexchange"}:
        return "pride"
    if repository in {"massive", "massive_ucsd", "msv", "gnps"}:
        return "massive"
    if repository in {"iprox", "ipx"}:
        return "iprox"
    if repository in {"local", "local_dir", "local_directory"}:
        return "local"
    return default


def _clean_ui_language(value: Any) -> str:
    language = _clean_text(value).lower()
    if language in {"zh", "zh_cn", "zh-cn", "cn", "chinese"}:
        return "zh"
    if language in {"en", "en_us", "en-us", "english"}:
        return "en"
    return "en"


def _strip_ansi(value: Any) -> str:
    return _ANSI_RE.sub("", str(value)).replace("\r", "")


def _redact_secrets(value: Any) -> str:
    text = _strip_ansi(value)
    text = re.sub(r"(?i)(api[_ -]?key\s*[:=]\s*)\S+", r"\1[redacted]", text)
    text = re.sub(r"sk-[A-Za-z0-9_\-]{6,}", "[redacted-api-key]", text)
    return text


def _contains_cjk(value: Any) -> bool:
    return bool(_CJK_RE.search(str(value)))


def _task_ui_language(task_id: str) -> str:
    task = _tasks.get(task_id)
    if not task:
        return "en"
    return _clean_ui_language(task.get("ui_language"))


def _english_punctuation(text: str) -> str:
    replacements = {
        "：": ": ",
        "；": "; ",
        "，": ", ",
        "。": ".",
        "（": " (",
        "）": ") ",
        "、": ", ",
        "…": "...",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "！": "!",
        "？": "?",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


_EN_LOG_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^\[(\d)/5\] 正在根据文件名解析 PRIDE 项目：(.+)$"), r"[\1/5] Resolving PRIDE project from file name: \2"),
    (re.compile(r"^\[(\d)/5\] 项目上下文已准备完成。SDRF 行数：(\d+)$"), r"[\1/5] Project context prepared. SDRF rows: \2"),
    (re.compile(r"^\[(\d)/5\] 已解析数据文件：(.+) （类型=(.+)，是否需要转换=(.+)）$"), r"[\1/5] Data file resolved: \2 (type=\3, requires_conversion=\4)"),
    (re.compile(r"^\[(\d)/5\] 文件属性推断完成。采集模式=(.+)$"), r"[\1/5] File attribute inference completed. Acquisition mode=\2"),
    (re.compile(r"^\[(\d)/5\] DDA 执行计划已生成。workflow=(.+)$"), r"[\1/5] DDA execution plan generated. workflow=\2"),
    (re.compile(r"^任务开始：(.+)$"), r"Task started: \1"),
    (re.compile(r"^输出目录：(.+)$"), r"Output directory: \1"),
    (re.compile(r"^运行模式：仅搜参数$"), "Run mode: parameter planning only"),
    (re.compile(r"^运行模式：完整流程$"), "Run mode: full workflow"),
    (re.compile(r"^任务已进入队列，当前位置 (\d+)/(\d+)。$"), r"Task queued. Position \1/\2."),
    (re.compile(r"^任务已从队列启动。$"), "Task started from queue."),
    (re.compile(r"^输入规范化：(.+)$"), r"Input normalized: \1"),
    (re.compile(r"^已选择主项目：(.+)$"), r"Selected primary project: \1"),
    (re.compile(r"^解析原因：(.+)$"), r"Resolution reason: \1"),
    (re.compile(r"^FASTA 下载源：(.+)$"), r"FASTA download source: \1"),
    (re.compile(r"^推荐 workflow：(.+)$"), r"Recommended workflow: \1"),
    (re.compile(r"^推荐 FASTA：(.+)$"), r"Recommended FASTA: \1"),
    (re.compile(r"^数据文件已就绪：(.+)$"), r"Data file ready: \1"),
    (re.compile(r"^输入包已生成：(.+)$"), r"Input bundle generated: \1"),
    (re.compile(r"^转换完成：(.+)$"), r"Conversion completed: \1"),
    (re.compile(r"^下载完成：(.+)$"), r"Download completed: \1"),
    (re.compile(r"^正在下载：(.+)$"), r"Downloading: \1"),
    (re.compile(r"^正在运行命令：(.+)$"), r"Running command: \1"),
    (re.compile(r"^\[阻断\]\s*(.+)$"), r"[blocked] \1"),
)

_EN_LOG_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("解析 PRIDE 项目", "Resolve PRIDE project"),
    ("下载 PRIDE 数据文件", "Download PRIDE data file"),
    ("生成 MSDT-Converter 输入包", "Generate MSDT-Converter input bundle"),
    ("运行 MSDT-Converter Docker", "Run MSDT-Converter Docker"),
    ("处理结果", "Process results"),
    ("正在初始化 AgentService…", "Initializing AgentService..."),
    ("AgentService 初始化完成", "AgentService initialized"),
    ("正在查询 PRIDE API 并调用大模型推断参数…", "Querying PRIDE API and inferring parameters with the LLM..."),
    ("正在查询 PRIDE Archive API 并匹配项目/文件…", "Querying PRIDE Archive API and matching project/file..."),
    ("PRIDE 查询完成。", "PRIDE query completed."),
    ("项目解析摘要：", "Project resolution summary: "),
    ("项目元数据摘要：", "Project metadata summary: "),
    ("文件资产判断：", "File asset decision: "),
    ("未找到 SDRF 行；将结合 PRIDE 项目描述、协议、文件名和参数/FASTA 文件线索推断搜库参数。", "No matching SDRF row was found; PRIDE metadata, protocols, file name, parameter files, and FASTA clues will be used to infer search parameters."),
    ("正在调用大模型确认文件属性和搜库参数。", "Calling the LLM to confirm file attributes and search parameters."),
    ("大模型正在阅读 PRIDE 元数据并生成搜库参数…", "The LLM is reading PRIDE metadata and generating search parameters..."),
    ("大模型确认结果已合并到属性推断中。", "LLM confirmation was merged into attribute inference."),
    ("PRIDE 查询和大模型推断完成", "PRIDE query and LLM inference completed"),
    ("属性判断：", "Attribute decision: "),
    ("搜库参数判断：", "Search parameter decision: "),
    ("数据适配提示：", "Data compatibility hint: "),
    ("执行计划：", "Execution plan: "),
    ("预期输出：", "Expected outputs: "),
    ("采集模式", "acquisition mode"),
    ("物种", "species"),
    ("仪器", "instrument"),
    ("酶", "enzyme"),
    ("项目", "project"),
    ("匹配文件", "matched file"),
    ("匹配类型", "match type"),
    ("匹配分数", "match score"),
    ("解析置信度", "resolution confidence"),
    ("置信度", "confidence"),
    ("实验类型", "experiment type"),
    ("解析类型", "resolved type"),
    ("是否需要转换", "requires conversion"),
    ("资产置信度", "asset confidence"),
    ("参数", "parameters"),
    ("固定修饰", "fixed modifications"),
    ("可变修饰", "variable modifications"),
    ("数据类型", "data type"),
    ("无", "none"),
    ("线程数", "threads"),
    ("原始数据类型", "raw data type"),
    ("正在下载数据文件", "Downloading data file"),
    ("下载完成", "Download complete"),
    ("已硬链接缓存的 PRIDE 文件", "Hard-linked cached PRIDE file"),
    ("已复制缓存的 PRIDE 文件", "Copied cached PRIDE file"),
    ("复用已下载的数据文件", "Reusing downloaded data file"),
    ("复用项目缓存中的 PRIDE 文件", "Reusing PRIDE project cache file"),
    ("数据文件需要格式转换", "Data file requires format conversion"),
    ("正在使用本地 msconvert 转换质谱文件", "Converting mass spectrometry file with local msconvert"),
    ("正在使用 Docker ProteoWizard 转换质谱文件", "Converting mass spectrometry file with Docker ProteoWizard"),
    ("主转换器失败", "Primary converter failed"),
    ("正在切换到备用转换器", "Switching to fallback converter"),
    ("数据文件已可直接用于执行", "Data file can be used directly"),
    ("正在解压", "Extracting"),
    ("解压完成", "Extraction completed"),
    ("已写入 Docker MSDT-Converter 配置", "Docker MSDT-Converter config written"),
    ("正在启动 MSDT-Converter Docker 镜像", "Starting MSDT-Converter Docker image"),
    ("MSDT-Converter 内部步骤失败，任务已标记为失败，不打包下载 ZIP。", "An MSDT-Converter internal step failed; the task was marked as failed and no ZIP will be packaged."),
    ("全部运行完成！", "Full workflow completed."),
    ("开始压缩打包结果 ZIP，打包完成后才会显示下载按钮。", "Compressing result ZIP; the download button will appear after packaging finishes."),
    ("结果 ZIP 已压缩打包完成，可以下载。", "Result ZIP is ready to download."),
    ("结果 ZIP 已存在，复用缓存", "Result ZIP already exists; reusing cache"),
    ("开始打包下载 ZIP", "Packaging download ZIP"),
    ("ZIP 打包进度", "ZIP packaging progress"),
    ("结果 ZIP 打包完成", "Result ZIP packaging completed"),
    ("仅搜参数模式", "Parameter-only mode"),
    ("已完成 PRIDE 项目解析、文件属性推断、workflow/FASTA/搜库参数计划生成。", "PRIDE project resolution, file attribute inference, and workflow/FASTA/search-parameter planning are complete."),
    ("参数推断完成", "Parameter inference completed"),
    ("人工已确认搜库参数；继续处理剩余步骤。", "Manual search-parameter review confirmed; continuing."),
    ("检测到项目级多个仪器；先准备/转换 mzML，并尝试从 mzML 解析文件级仪器。", "Multiple project-level instruments detected; preparing/converting mzML and reading file-level instrument metadata."),
    ("已从 mzML 解析文件级仪器", "File-level instrument parsed from mzML"),
    ("当前计划需要人工复核，暂不下载或准备数据文件。原因", "The current plan needs manual review; data download/preparation is paused. Reason"),
    ("未找到匹配的 SDRF 行，且项目包含多个物种；无法确定文件级物种信息。", "No matching SDRF row was found, and the project contains multiple species; file-level species cannot be determined."),
    ("未找到匹配的 SDRF 行，且项目包含多个仪器；无法确定文件级仪器信息。", "No matching SDRF row was found, and the project contains multiple instruments; file-level instrument cannot be determined."),
    ("当前 bottom-up MSDT 搜库流程不支持 Top-down 蛋白质组学项目。", "The current bottom-up MSDT search workflow does not support top-down proteomics projects."),
    ("大模型推荐的 workflow", "The LLM-recommended workflow"),
    ("不存在于 profiles/fragpipe/ 目录中。请检查 workflow 名称是否正确。", "does not exist in profiles/fragpipe/. Check the workflow name."),
    ("大模型未推荐 workflow。必须配置 LLM API 并确保大模型能正确推荐 workflow。请检查 AGENT_LLM_API_KEY 配置。", "The LLM did not recommend a workflow. Configure the LLM API and ensure it can recommend a workflow."),
    ("任务运行失败。", "Task execution failed."),
    ("网络连接失败。", "Network connection failed."),
    ("Docker 服务不可用。", "Docker service is unavailable."),
    ("内存不足导致任务失败。", "The task failed because memory was insufficient."),
    ("外部命令执行失败。", "An external command failed."),
    ("任务需要人工复核。", "The task needs manual review."),
    ("运行出错", "Run failed"),
    ("错误", "error"),
    ("失败", "failed"),
    ("成功", "succeeded"),
    ("完成", "completed"),
    ("正在", "in progress"),
    ("已", ""),
)


def _ascii_fallback(text: str, level: str = "") -> str:
    ascii_text = _CJK_RE.sub(" ", _english_punctuation(text))
    ascii_text = re.sub(r"[^A-Za-z0-9_./:;=+\-()[\]{}|,@#%&?\\\s]", " ", ascii_text)
    ascii_text = re.sub(r"\s{2,}", " ", ascii_text).strip(" ;,")
    if ascii_text and re.search(r"[A-Za-z0-9]", ascii_text):
        return f"Backend message: {ascii_text}"
    if str(level).lower() == "llm":
        return "LLM reasoning output was not shown in English logs; structured parameters were saved in the audit files."
    return "Backend message omitted in English mode because it was not localized."


def _english_fasta_review_message(text: str) -> str | None:
    if "UniProt" not in text or "FASTA" not in text:
        return None
    if "proteome ID" not in text and "占位" not in text and "鍗犱綅" not in text:
        return None
    species_match = re.search(r"environmental samples(?:\s*<[^>]+>)?", text, re.IGNORECASE)
    if species_match:
        species = species_match.group(0).strip()
    else:
        species = "the selected sample"
    prefix = "[blocked] " if "[阻断]" in text or "[blocked]" in text else ""
    return (
        f"{prefix}No real UniProt FASTA could be selected for species: {species}. "
        "Provide a reviewed FASTA URL or local FASTA path before running the full workflow."
    )


def _to_english_log_message(message: Any, level: str = "") -> str:
    text = _redact_secrets(message).strip()
    if not text:
        return ""
    for pattern, replacement in _EN_LOG_PATTERNS:
        text = pattern.sub(replacement, text)
    for old, new in sorted(_EN_LOG_REPLACEMENTS, key=lambda item: len(item[0]), reverse=True):
        text = text.replace(old, new)
    text = _english_punctuation(text)
    fasta_review = _english_fasta_review_message(text)
    if fasta_review:
        return fasta_review
    if not _contains_cjk(text):
        return text
    if str(level).lower() == "llm":
        return "LLM reasoning output was not shown in English logs; structured parameters were saved in the audit files."
    return _ascii_fallback(text, level=level)


def _localize_public_message(message: Any, language: str, level: str = "") -> str:
    text = _redact_secrets(message).strip()
    if _clean_ui_language(language) == "en":
        return _to_english_log_message(text, level=level)
    return text


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)
    return value


def _sanitize_log_entry(entry: Any) -> dict[str, Any] | None:
    if not isinstance(entry, dict):
        return None
    allowed = ("type", "ts", "level", "message", "key", "replace", "step", "status", "summary")
    sanitized: dict[str, Any] = {}
    for key in allowed:
        if key not in entry:
            continue
        value = entry[key]
        if key == "message":
            value = _redact_secrets(value).strip()
        sanitized[key] = _json_safe(value)
    if not sanitized:
        return None
    return sanitized


def _public_logs_from_task(task: dict[str, Any]) -> list[dict[str, Any]]:
    logs = list(task.get("logs") or [])
    ui_language = _clean_ui_language(task.get("ui_language"))
    public_logs: list[dict[str, Any]] = []
    for entry in logs[-_MAX_PERSISTED_LOGS:]:
        sanitized = _sanitize_log_entry(entry)
        if sanitized:
            if "message" in sanitized:
                sanitized["message"] = _localize_public_message(
                    sanitized["message"],
                    ui_language,
                    level=str(sanitized.get("level") or sanitized.get("type") or "info"),
                )
            public_logs.append(sanitized)
    return public_logs


def _parse_history_timestamp(value: Any) -> float | None:
    if not value:
        return None
    raw = str(value).strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_APP_TZ)
    return dt.timestamp()


def _history_retention_start(history: dict[str, Any], fallback_mtime: float = 0.0) -> float:
    for key in ("finished_at", "started_at", "created_at", "updated_at"):
        parsed = _parse_history_timestamp(history.get(key))
        if parsed is not None:
            return parsed
    return fallback_mtime


_HISTORY_DISPLAY_TIME_FIELDS = ("started_at", "created_at", "finished_at", "updated_at")


def _history_basename(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return Path(text).name


def _history_display_name(item: dict[str, Any]) -> str:
    input_name = _history_basename(item.get("input_value"))
    if input_name:
        return input_name
    for field in ("output_dir", "run_id", "result_id", "name", "history_id", "task_id"):
        value = _history_basename(item.get(field))
        if value:
            return value
    return ""


def _history_run_label(item: dict[str, Any]) -> str:
    for field in ("output_dir", "run_id", "result_id", "name", "history_id", "project_key"):
        value = _history_basename(item.get(field))
        if value:
            return value
    return _history_display_name(item)


def _history_time_label(item: dict[str, Any]) -> str:
    for field in _HISTORY_DISPLAY_TIME_FIELDS:
        if item.get(field):
            return field
    return "history_time" if item.get("history_time") else ""


def _history_duration_seconds(item: dict[str, Any]) -> int | None:
    started = _parse_history_timestamp(item.get("started_at") or item.get("created_at"))
    finished = _parse_history_timestamp(item.get("finished_at") or item.get("updated_at"))
    if started is None or finished is None or finished < started:
        return None
    return int(finished - started)


def _history_status_group(status: Any) -> str:
    normalized = str(status or "").strip().lower()
    if normalized in _ACTIVE_STATUSES:
        return "active"
    if normalized == "completed":
        return "success"
    if normalized == "blocked":
        return "blocked"
    if normalized == "failed":
        return "failed"
    return "unknown"


def _history_primary_action(item: dict[str, Any]) -> str:
    status = str(item.get("status") or "").strip().lower()
    if status in _ACTIVE_STATUSES:
        return "watch"
    if item.get("can_download"):
        return "download"
    if status in {"failed", "blocked"} or item.get("blocking_issues"):
        return "inspect"
    return "view"


def _decorate_history_item(record: dict[str, Any]) -> dict[str, Any]:
    item = with_history_identity(record)
    run_label = _history_run_label(item)
    if run_label:
        item.setdefault("run_id", run_label)
        item.setdefault("result_id", run_label)
    item["display_name"] = _history_display_name(item) or run_label
    item["run_label"] = run_label or item["display_name"]
    item["time_label"] = _history_time_label(item)
    item["duration_seconds"] = _history_duration_seconds(item)
    item["status_group"] = _history_status_group(item.get("status"))
    if item.get("usable_partial_outputs"):
        item["status_group"] = "partial"
    item["primary_action"] = _history_primary_action(item)
    kind = str(item.get("kind") or "task")
    status = str(item.get("status") or "").lower()
    if kind == "discovery":
        item["open_kind"] = "discovery_job" if item.get("job_id") else "discovery_run"
        item["open_id"] = item.get("job_id") or item.get("discovery_id") or item.get("run_id")
        discovery_id = _clean_text(item.get("discovery_id") or item.get("run_id"))
        run_dir = _safe_discovery_dir(discovery_id)
        result_available = bool(run_dir and run_dir.exists())
    elif kind == "batch":
        item["open_kind"] = "batch"
        item["open_id"] = item.get("batch_id") or item.get("result_id")
        batch_id = _clean_text(item.get("batch_id") or item.get("result_id"))
        result_available = bool(
            batch_id
            and safe_output_stem(batch_id) == batch_id
            and _batch_dir(batch_id).exists()
        )
    else:
        item["open_kind"] = "task"
        item["open_id"] = item.get("task_id") or item.get("result_id")
        result_available = True
    if kind in {"discovery", "batch"}:
        item["result_available"] = result_available
        item["open_available"] = status in _ACTIVE_STATUSES or result_available
        item["can_download"] = bool(item.get("can_download") and result_available)
        if not result_available:
            item["recorded_size_bytes"] = int(
                item.get("recorded_size_bytes") or item.get("size_bytes") or 0
            )
            item["size_bytes"] = 0
    item["deletable"] = kind in {"discovery", "batch"} and status not in _ACTIVE_STATUSES
    item["delete_block_reason"] = (
        "请先停止运行中的任务。" if status in _ACTIVE_STATUSES else ""
    )
    return item


def _history_summary(active_tasks: list[dict[str, Any]], results: list[dict[str, Any]]) -> dict[str, Any]:
    items = [*active_tasks, *results]
    status_counts = Counter(str(item.get("status") or "unknown") for item in items)
    status_group_counts = Counter(str(item.get("status_group") or _history_status_group(item.get("status"))) for item in items)
    storage_bytes = 0
    for item in items:
        try:
            storage_bytes += int(item.get("size_bytes") or 0)
        except (TypeError, ValueError):
            continue
    return {
        "total": len(items),
        "active": len(active_tasks),
        "results": len(results),
        "downloadable": sum(1 for item in items if item.get("can_download")),
        "storage_bytes": storage_bytes,
        "failed": status_counts.get("failed", 0),
        "blocked": status_counts.get("blocked", 0),
        "interrupted": sum(1 for item in items if item.get("interrupted")),
        "status_counts": dict(status_counts),
        "status_group_counts": dict(status_group_counts),
    }


def _positive_float(value: str, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _max_concurrent_tasks() -> int:
    raw = os.getenv("AGENT_MAX_CONCURRENT_TASKS", "1")
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def _result_retention_seconds() -> int:
    raw = os.getenv("AGENT_RESULT_RETENTION_SECONDS", "1800")
    try:
        parsed = int(float(raw))
    except (TypeError, ValueError):
        return 1800
    return max(1, parsed)


def _max_result_projects() -> int:
    raw = os.getenv("AGENT_MAX_RESULT_PROJECTS", "4")
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        return 4
    return max(1, parsed)


def _protected_result_dir_names() -> set[str]:
    defaults = {
        "baseline_validation",
        "real_smoke",
        "multi_mini_validation",
        "local_validation",
        "ai_ready_builds",
    }
    raw = os.getenv("AGENT_PROTECTED_RESULT_DIRS", "")
    extra = {item.strip() for item in raw.split(",") if item.strip()}
    return defaults | extra


def _zip_compress_level() -> int:
    raw = os.getenv("AGENT_ZIP_COMPRESS_LEVEL", "6")
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        return 6
    return min(9, max(1, parsed))


def _max_batch_items() -> int:
    """Max lines accepted by POST /api/batches/parameters.

    Default is intentionally high so L1 discovery lists are not truncated at 100.
    Set AGENT_MAX_BATCH_ITEMS=0 (or negative) for no practical product cap
    (still clamped to a process safety ceiling to avoid OOM).
    """
    raw = os.getenv("AGENT_MAX_BATCH_ITEMS", "10000")
    try:
        parsed = int(str(raw).strip())
    except (TypeError, ValueError):
        return 10000
    # 0 / negative => effectively uncapped for product use.
    if parsed <= 0:
        return 100_000
    return max(1, min(parsed, 100_000))


def _max_batch_jobs() -> int:
    raw = os.getenv("AGENT_MAX_BATCH_JOBS", "4")
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        return 4
    return max(1, parsed)


def _batch_root_dir() -> Path:
    return _runs_dir / _BATCHES_DIR_NAME


def _batch_dir(batch_id: str) -> Path:
    return _batch_root_dir() / safe_output_stem(batch_id)


def _discovery_root_dir() -> Path:
    return _runs_dir / _DISCOVERY_DIR_NAME


def _discovery_memory_dir() -> Path:
    return _runs_dir / _DISCOVERY_MEMORY_DIR_NAME


def _ai_ready_root_dir() -> Path:
    return _runs_dir / _AI_READY_BUILDS_DIR_NAME


def _safe_discovery_dir(discovery_id: str) -> Path | None:
    if not discovery_id or safe_output_stem(discovery_id) != discovery_id:
        return None
    root = _discovery_root_dir().resolve()
    candidate = (_discovery_root_dir() / discovery_id).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _safe_ai_ready_dir(build_id: str) -> Path | None:
    if not build_id or safe_output_stem(build_id) != build_id:
        return None
    root = _ai_ready_root_dir().resolve()
    candidate = (_ai_ready_root_dir() / build_id).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _batch_manifest_path(batch_id: str) -> Path:
    return _batch_dir(batch_id) / _BATCH_MANIFEST_FILE


def _clean_batch_inputs(body: dict[str, Any]) -> list[str]:
    raw_inputs = body.get("inputs")
    inputs: list[str] = []
    if isinstance(raw_inputs, list):
        inputs.extend(_clean_text(item) for item in raw_inputs)
    else:
        text = _clean_text(body.get("input_text") or body.get("batch_input") or body.get("input_value"))
        inputs.extend(line.strip() for line in text.splitlines())
    return [item for item in inputs if item and not item.startswith("#")]


def _clean_batch_input_records(body: dict[str, Any], inputs: list[str]) -> list[dict[str, Any] | None]:
    raw_records = body.get("input_records")
    if not isinstance(raw_records, list):
        return [None for _ in inputs]
    cleaned: list[dict[str, Any]] = []
    allowed_keys = {
        "input",
        "repository",
        "project_accession",
        "project_title",
        "file_name",
        "download_url",
        "file_type",
        "file_role",
        "acquisition_mode",
        "species_policy",
        "canonical_species",
        "organism_taxon_id",
        "ptm_type",
        "ptm_subtype",
        "ptm_evidence_terms",
        "ptm_enrichment_methods",
        "semantic_metadata_confidence",
        "modification_scope",
        "immunopeptide_scope",
        "hla_class",
        "hla_alleles",
        "immunopeptide_evidence_terms",
        "immunopeptide_enrichment_methods",
        "immunopeptide_metadata_confidence",
        "labeling_strategy",
        "validity_status",
        "task_type",
        "task_readiness_status",
        "evidence_level",
        "sdrf_match_status",
        "instrument_families",
        "fragmentation_methods",
        "lc_gradient_minutes",
        "source_discovery_job_id",
        "source_discovery_id",
        "source_batch_index",
        "source_file_identifier",
    }
    for raw in raw_records:
        if not isinstance(raw, dict):
            continue
        file_name = _clean_text(raw.get("file_name") or raw.get("input") or raw.get("input_value"))
        if not file_name:
            continue
        record: dict[str, Any] = {"file_name": file_name, "input": _clean_text(raw.get("input")) or file_name}
        for key in allowed_keys:
            if key in {"input", "file_name"} or key not in raw:
                continue
            value = raw.get(key)
            if isinstance(value, list):
                record[key] = [_clean_text(item) for item in value if _clean_text(item)]
            elif key == "lc_gradient_minutes":
                try:
                    record[key] = float(value) if value is not None and str(value).strip() else None
                except (TypeError, ValueError):
                    record[key] = None
            elif key == "source_batch_index":
                try:
                    record[key] = int(value)
                except (TypeError, ValueError):
                    continue
            else:
                text = _clean_text(value)
                if text:
                    record[key] = text
        cleaned.append(record)
    if len(cleaned) == len(inputs) and all(
        record.get("input") == inputs[index] or record.get("file_name") == inputs[index]
        for index, record in enumerate(cleaned)
    ):
        return cleaned
    by_name: dict[str, dict[str, Any]] = {}
    for record in cleaned:
        by_name.setdefault(str(record.get("file_name") or ""), record)
        by_name.setdefault(str(record.get("input") or ""), record)
    return [by_name.get(input_value) for input_value in inputs]


def _batch_jobs(value: Any, item_count: int) -> int:
    try:
        requested = int(value)
    except (TypeError, ValueError):
        requested = 3
    return max(1, min(requested, item_count, _max_batch_jobs()))


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def _agent_discovery_configuration(
    body: dict[str, Any],
) -> tuple[str, AgentBudget, DynamicBudgetLimits]:
    # Agent-autonomous default: no Budget-Agent grant chain. Explicit multi_agent remains
    # available via body.discovery_mode / AGENT_DISCOVERY_MODE for legacy cost control.
    requested_mode = _clean_text(
        body.get("discovery_mode")
        or body.get("agent_mode")
        or os.getenv("AGENT_DISCOVERY_MODE")
        or "single_agent"
    ).lower()
    mode = requested_mode if requested_mode in {"single_agent", "multi_agent"} else "single_agent"
    target_projects = _bounded_int(
        body.get("max_projects"),
        default=100,
        minimum=1,
        maximum=1000,
    )
    target_inspection_rounds = recommended_inspection_rounds(target_projects)
    budget = AgentBudget(
        max_turns=_bounded_int(os.getenv("AGENT_MAX_MODEL_TURNS"), default=80, minimum=1, maximum=200),
        max_tool_calls=_bounded_int(os.getenv("AGENT_MAX_TOOL_CALLS"), default=250, minimum=1, maximum=500),
        max_discovery_rounds=_bounded_int(
            os.getenv("AGENT_MAX_DISCOVERY_ROUNDS"),
            default=max(target_inspection_rounds, 8),
            minimum=1,
            maximum=30,
        ),
    )
    # Hard safety ceilings only. Sized so multi-page PRIDE search + full inspection
    # of large candidate pools does not hit hard_repository_request_limit casually.
    limits = DynamicBudgetLimits(
        initial_query_units=_bounded_int(
            os.getenv("AGENT_INITIAL_QUERY_UNITS"), default=200, minimum=1, maximum=10000
        ),
        expanded_query_units=_bounded_int(
            os.getenv("AGENT_EXPANDED_QUERY_UNITS"), default=800, minimum=1, maximum=10000
        ),
        max_query_units=_bounded_int(os.getenv("AGENT_MAX_QUERY_UNITS"), default=2500, minimum=1, maximum=10000),
        initial_repository_requests=_bounded_int(
            os.getenv("AGENT_INITIAL_REPOSITORY_REQUESTS"), default=5000, minimum=1, maximum=100000
        ),
        expanded_repository_requests=_bounded_int(
            os.getenv("AGENT_EXPANDED_REPOSITORY_REQUESTS"), default=15000, minimum=1, maximum=100000
        ),
        max_repository_requests=_bounded_int(
            os.getenv("AGENT_MAX_REPOSITORY_REQUESTS"), default=30000, minimum=1, maximum=100000
        ),
        max_elapsed_seconds=_bounded_int(
            os.getenv("AGENT_MAX_ELAPSED_SECONDS"), default=14400, minimum=30, maximum=172800
        ),
        budget_agent_max_turns=_bounded_int(
            os.getenv("AGENT_BUDGET_AGENT_MAX_TURNS"), default=3, minimum=2, maximum=10
        ),
    )
    time_preference = _clean_text(
        body.get("time_budget_preference") or body.get("time_budget") or "multi_round"
    ).lower()
    if time_preference == "fast":
        # A user-selected fast run is an operational budget, not a claim that
        # quality gates can be skipped. The closing audit still reports any
        # target or evidence shortfall explicitly.
        budget = budget.model_copy(
            update={
                "max_turns": min(budget.max_turns, 20),
                "max_tool_calls": min(budget.max_tool_calls, 80),
                "max_discovery_rounds": min(budget.max_discovery_rounds, 3),
            }
        )
        limits = limits.model_copy(
            update={
                "initial_query_units": min(limits.initial_query_units, 80),
                "expanded_query_units": min(limits.expanded_query_units, 160),
                "max_query_units": min(limits.max_query_units, 300),
                "initial_repository_requests": min(
                    limits.initial_repository_requests, 1_000
                ),
                "expanded_repository_requests": min(
                    limits.expanded_repository_requests, 2_000
                ),
                "max_repository_requests": min(
                    limits.max_repository_requests, 3_000
                ),
                "max_elapsed_seconds": min(limits.max_elapsed_seconds, 900),
            }
        )
    project_plan = body.get("project_plan")
    hard_ceilings = (
        project_plan.get("hard_ceilings")
        if isinstance(project_plan, dict)
        and isinstance(project_plan.get("hard_ceilings"), dict)
        else {}
    )
    if hard_ceilings:
        # A persisted project plan may narrow server safety budgets, but it can
        # never expand them. Keep this generic and fail-safe at the execution
        # boundary so a resumed plan cannot bypass its approved ceilings.
        ceiling_turns = _bounded_int(
            hard_ceilings.get("max_model_turns"),
            default=budget.max_turns,
            minimum=1,
            maximum=budget.max_turns,
        )
        ceiling_tool_calls = _bounded_int(
            hard_ceilings.get("max_tool_calls"),
            default=budget.max_tool_calls,
            minimum=1,
            maximum=budget.max_tool_calls,
        )
        ceiling_rounds = _bounded_int(
            hard_ceilings.get("max_discovery_rounds"),
            default=budget.max_discovery_rounds,
            minimum=1,
            maximum=budget.max_discovery_rounds,
        )
        ceiling_runtime_minutes = _bounded_int(
            hard_ceilings.get("max_runtime_minutes"),
            default=max(1, limits.max_elapsed_seconds // 60),
            minimum=1,
            maximum=max(1, limits.max_elapsed_seconds // 60),
        )
        budget = budget.model_copy(
            update={
                "max_turns": min(budget.max_turns, ceiling_turns),
                "max_tool_calls": min(budget.max_tool_calls, ceiling_tool_calls),
                "max_discovery_rounds": min(
                    budget.max_discovery_rounds, ceiling_rounds
                ),
            }
        )
        limits = limits.model_copy(
            update={
                "max_elapsed_seconds": min(
                    limits.max_elapsed_seconds, ceiling_runtime_minutes * 60
                )
            }
        )
    return mode, budget, limits


def _clean_discovery_species(value: Any, *, default: list[str] | None = None) -> list[str]:
    if isinstance(value, list):
        values = [_clean_text(item) for item in value]
    else:
        values = re.split(r"[,;\n]+", _clean_text(value))
    species = [item for item in values if item]
    return species or list(default or [])


def _clean_discovery_ptm_types(value: Any, *, default: list[str] | None = None) -> list[str]:
    if isinstance(value, list):
        values = [_clean_text(item) for item in value]
    else:
        values = re.split(r"[,;\n]+", _clean_text(value)) if value is not None else []
    normalized: list[str] = []
    for item in values:
        ptm = normalize_ptm_type(item)
        if ptm and ptm not in normalized:
            normalized.append(ptm)
    return normalized or list(default or [])


_FIXED_DISCOVERY_RUN_HORIZON = "candidates_reviewed"


def _candidate_delivery_satisfies_run_horizon(
    body: Mapping[str, Any],
    progress: Mapping[str, Any],
) -> bool:
    """Separate reviewed-candidate delivery from stricter modeling readiness."""

    requested_project_target = _bounded_int(
        body.get("max_projects"),
        default=20,
        minimum=1,
        maximum=1_000_000,
    )
    portfolio_spec = body.get("portfolio_spec") if isinstance(body.get("portfolio_spec"), Mapping) else {}
    portfolio_project_target = _bounded_int(
        portfolio_spec.get("target_projects"),
        default=requested_project_target,
        minimum=1,
        maximum=1_000_000,
    )
    portfolio_file_target = _bounded_int(
        portfolio_spec.get("target_files"),
        default=0,
        minimum=0,
        maximum=1_000_000,
    )
    requested_quota = _clean_text(body.get("quota_flexibility")).lower()
    common = (
        _clean_text(body.get("run_horizon")) == _FIXED_DISCOVERY_RUN_HORIZON
        and requested_quota != "open_ended"
        and _clean_text(body.get("coverage_mode")).lower()
        not in {"exhaustive", "maximize", "all"}
    )
    if not common or int(progress.get("qualified_count") or 0) < portfolio_project_target:
        return False
    if int(progress.get("batch_count") or 0) > 0:
        return True
    # A strict portfolio request can finish with an audited candidate manifest
    # before the incremental publisher emits a batch. Treat the requested
    # portfolio itself as the delivery unit; legacy non-portfolio runs retain
    # the historical batch requirement.
    return bool(
        portfolio_file_target > 0
        and int(progress.get("usable_file_count") or 0) >= portfolio_file_target
    )


def _has_explicit_exhaustive_discovery_intent(value: Any) -> bool:
    """Recognize user language that means portfolio-wide discovery, not a quota."""

    text = _clean_text(value)
    if not text:
        return False
    return bool(
        re.search(
            r"("
            r"越多越好|尽可能多|尽量多|尽量搜全|搜全|覆盖全|不设上限|无上限|全量|"
            r"(?:所有|全部).{0,24}(?:数据|项目|文件|候选)|"
            r"(?:数据|项目|文件|候选).{0,16}(?:所有|全部)|"
            r"as\s+many\s+as\s+possible|all\s+(?:relevant\s+)?(?:data|datasets|projects|files)|"
            r"no\s+(?:candidate\s+pool\s+)?limit|open[-\s]?ended|exhaustive|comprehensive"
            r")",
            text,
            re.IGNORECASE,
        )
    )


def _discovery_exhaustive_payload_conflict(body: Mapping[str, Any]) -> bool:
    """Fail closed when old clients turn an explicit exhaustive goal into 20/80."""

    prompt = _clean_text(
        body.get("prompt")
        or body.get("visible_prompt")
        or body.get("objective")
        or body.get("goal")
        or ""
    )
    if not _has_explicit_exhaustive_discovery_intent(prompt):
        return False
    if _clean_text(body.get("quota_flexibility")).lower() == "fixed":
        return False
    return not (
        body.get("continuous_discovery") is True
        and _clean_text(body.get("scale_mode")).lower() == "exhaustive"
        and _clean_text(body.get("quota_flexibility")).lower() == "open_ended"
        and _clean_text(body.get("quantity_scope")).lower() == "portfolio"
        and _clean_text(body.get("portfolio_size_preference")).lower().startswith("maximize")
    )


def _clean_dataset_request(body: dict[str, Any]) -> DatasetRequest:
    prompt_text = _clean_text(body.get("prompt") or body.get("visible_prompt") or body.get("goal") or "")
    explicit_exhaustive = _has_explicit_exhaustive_discovery_intent(prompt_text)
    incoming_provenance = body.get("constraint_provenance")
    incoming_provenance = (
        {str(key): str(value) for key, value in incoming_provenance.items()}
        if isinstance(incoming_provenance, dict)
        else {}
    )
    explicit_hard = body.get("hard_constraint_fields")
    explicit_hard_contract = isinstance(explicit_hard, list)
    explicit_hard_fields = {
        str(item).strip()
        for item in (explicit_hard if isinstance(explicit_hard, list) else [])
        if str(item).strip()
    }

    def _user_hard(field: str, *, supplied: bool) -> bool:
        # Only user-explicit or caller-whitelisted fields become hard constraints.
        # Parser/LLM defaults ("inferred"/"default"/"user_or_parsed") stay soft.
        if explicit_hard_contract:
            return supplied and field in explicit_hard_fields
        provenance = str(incoming_provenance.get(field) or "").strip().lower()
        if provenance in {"inferred", "default", "parser", "llm", "user_or_parsed"}:
            return False
        if provenance == "user" and supplied:
            return True
        return supplied and not provenance

    acquisition_raw = body.get("acquisition_mode") if "acquisition_mode" in body else body.get("acquisition")
    labeling_raw = body.get("labeling_strategy") if "labeling_strategy" in body else body.get("labeling")
    acquisition_supplied = bool(_clean_text(acquisition_raw))
    labeling_supplied = bool(_clean_text(labeling_raw))
    acquisition = _clean_text(acquisition_raw or "unknown").lower() or "unknown"
    if acquisition in {"", "any", "auto"}:
        acquisition = "unknown"
    repository = _clean_repository(body.get("repository") or "pride")
    mixed_acquisition_policy = _clean_text(
        body.get("mixed_acquisition_policy") or "review_mixed"
    ).lower()
    if mixed_acquisition_policy not in {"reject_mixed", "review_mixed", "allow"}:
        mixed_acquisition_policy = "review_mixed"
    # Product invariant: every repository-discovery task finds and reviews
    # candidates. Older clients may still send another horizon, but execution
    # must never downgrade to plan-only or candidate-only behavior.
    run_horizon = _FIXED_DISCOVERY_RUN_HORIZON
    quota_flexibility = _clean_text(
        body.get("quota_flexibility") or "recommended"
    ).lower()
    if quota_flexibility not in {"fixed", "recommended", "open_ended"}:
        quota_flexibility = "recommended"
    if explicit_exhaustive and quota_flexibility != "fixed":
        quota_flexibility = "open_ended"
    raw_project_target = body.get("max_projects")
    has_positive_project_target = (
        raw_project_target is not None
        and _bounded_int(raw_project_target, default=0, minimum=0, maximum=5000) > 0
    )
    if (
        has_positive_project_target
        and not explicit_exhaustive
        and quota_flexibility == "recommended"
    ):
        # Product semantics now have only two execution modes. Older browser
        # sessions can still submit the retired curated/recommended fields
        # together with an exact numeric target. Treat that combination as the
        # user's fixed target so it cannot fall back to the autonomous SDK path
        # and continue reviewing after the requested count has been reached.
        quota_flexibility = "fixed"
    time_budget_preference = _clean_text(
        body.get("time_budget_preference") or body.get("time_budget") or "multi_round"
    ).lower()
    if time_budget_preference not in {"fast", "multi_round"}:
        time_budget_preference = "multi_round"
    on_safety_ceiling = _clean_text(body.get("on_safety_ceiling") or "ask").lower()
    if on_safety_ceiling not in {"ask", "auto_continue_within_safety", "stop"}:
        on_safety_ceiling = "ask"

    goal_supplied = "goal" in body and bool(_clean_text(body.get("goal")))
    raw_goal = _clean_text(body.get("goal") or "general")
    goal = raw_goal.lower()
    broad_proteomics = bool(
        re.search(
            r"(proteomics|proteome|peptidomics|shotgun|bottom[\s-]?up|蛋白质组|肽组|蛋白肽|质谱蛋白)",
            f"{raw_goal} {prompt_text}",
            re.IGNORECASE,
        )
    )
    # Prevent "human protein/peptide data" from being collapsed into immunopeptidomics
    # unless the user explicitly asked for HLA/MHC/immunopeptidomics.
    if is_immunopeptidomics_goal(goal) or is_immunopeptidomics_goal(prompt_text):
        if broad_proteomics and not re.search(
            r"(immunopeptidom|hla\b|mhc\b|ligandome|免疫肽)",
            f"{raw_goal} {prompt_text}",
            re.IGNORECASE,
        ):
            goal = "general"
        else:
            goal = "immunopeptidomics"
    elif goal not in {"general", "ptm", "immunopeptidomics"}:
        # Free-form scientific goals stay open; keep distinctive terms as query seeds.
        goal = "general"

    if goal == "general":
        ptm_type = "unknown_ptm"
        ptm_types: list[str] = []
    else:
        default_ptm = "unknown_ptm" if is_immunopeptidomics_goal(goal) else "phospho"
        ptm_types = _clean_discovery_ptm_types(body.get("ptm_types"), default=[])
        if not ptm_types:
            ptm_types = _clean_discovery_ptm_types(
                body.get("ptm_type") or body.get("ptm"),
                default=[default_ptm] if _user_hard("ptm_type", supplied=True) else [],
            )
        ptm_type = ptm_types[0] if ptm_types else "unknown_ptm"

    raw_query_terms = body.get("query_terms")
    if isinstance(raw_query_terms, list):
        query_terms = [_clean_text(item) for item in raw_query_terms if _clean_text(item)]
    else:
        query_terms = []
    if goal == "general" and not query_terms:
        query_terms = [*query_terms, *general_query_terms_from_text(prompt_text or raw_goal)]
    raw_species = body.get("species")
    species = _clean_discovery_species(raw_species, default=[])
    species_supplied = "species" in body and bool(species)
    # Free-text human preference becomes a soft species seed (not hard unless provenance=user).
    if not species and (
        re.search(r"\b(human|homo sapiens)\b", prompt_text, re.IGNORECASE)
        or re.search(r"(人类|人源|智人)", prompt_text)
    ):
        species = ["human"]
        species_supplied = True
        incoming_provenance.setdefault("species", "inferred")
    species_policy = _clean_text(body.get("species_policy") or "open").lower()
    if species_policy not in {"open", "include_only", "exclude"}:
        species_policy = "open"
    # Broad human proteomics / "越多越好" should hard-filter non-human projects.
    if species and {"human", "homo sapiens"} & {str(item).casefold() for item in species}:
        if re.search(
            r"(人类|人源|human|homo sapiens).{0,24}(蛋白|肽|proteom|peptid)|越多越好|尽可能多|as many as possible",
            prompt_text,
            re.IGNORECASE,
        ):
            species_policy = "include_only"
            incoming_provenance["species"] = str(incoming_provenance.get("species") or "user")
            incoming_provenance["species_policy"] = "user"
    if species and species_policy == "open" and str(incoming_provenance.get("species") or "") == "user":
        species_policy = "include_only"
    canonical_species, taxon_ids = normalize_species_values(species)
    immunopeptide = interpret_immunopeptide_metadata(
        " ".join([goal, prompt_text, _clean_text(body.get("immunopeptide_context"))])
    )

    hard_constraint_fields = ["repository"]
    constraint_provenance = {
        "repository": "user" if "repository" in body else "default",
    }
    if goal_supplied and _user_hard("goal", supplied=True) and goal in {"ptm", "immunopeptidomics"}:
        hard_constraint_fields.append("goal")
        constraint_provenance["goal"] = "user"
    elif goal_supplied:
        constraint_provenance["goal"] = str(incoming_provenance.get("goal") or "inferred")
    if species_supplied and _user_hard("species", supplied=True):
        hard_constraint_fields.extend(["species", "species_policy"])
        constraint_provenance["species"] = "user"
        constraint_provenance["species_policy"] = "user"
    elif species_supplied:
        constraint_provenance["species"] = str(incoming_provenance.get("species") or "inferred")
    if acquisition_supplied and _user_hard("acquisition_mode", supplied=True) and acquisition != "unknown":
        hard_constraint_fields.append("acquisition_mode")
        constraint_provenance["acquisition_mode"] = "user"
    elif acquisition_supplied:
        constraint_provenance["acquisition_mode"] = str(
            incoming_provenance.get("acquisition_mode") or "inferred"
        )
    labeling_strategy = normalize_labeling_strategy(labeling_raw or "unknown")
    if labeling_supplied and _user_hard("labeling_strategy", supplied=True) and labeling_strategy != "unknown":
        hard_constraint_fields.append("labeling_strategy")
        constraint_provenance["labeling_strategy"] = "user"
    elif labeling_supplied:
        constraint_provenance["labeling_strategy"] = str(
            incoming_provenance.get("labeling_strategy") or "inferred"
        )
    ptm_supplied = any(key in body for key in ("ptm", "ptm_type", "ptm_types"))
    if goal != "general" and ptm_supplied and _user_hard("ptm_type", supplied=True):
        hard_constraint_fields.extend(["ptm_type", "ptm_types"])
        constraint_provenance["ptm_type"] = "user"
    elif ptm_supplied:
        constraint_provenance["ptm_type"] = str(incoming_provenance.get("ptm_type") or "inferred")

    # Portfolio quantity language ("越多越好") is a soft success preference, never a hard cap.
    quantity_scope = _clean_text(body.get("quantity_scope") or "unspecified").lower()
    portfolio_size_preference = _clean_text(body.get("portfolio_size_preference") or "") or None
    if quantity_scope not in {"unspecified", "portfolio", "per_project"}:
        quantity_scope = "unspecified"
    if quota_flexibility == "open_ended":
        quantity_scope = "portfolio"
        portfolio_size_preference = (
            portfolio_size_preference or "maximize_qualified_projects"
        )
    if quantity_scope == "unspecified" and (
        explicit_exhaustive
        or re.search(
            r"(越多越好|尽可能多|as many as possible|maximize.*(project|dataset|sample)|越多越)",
            prompt_text,
            re.IGNORECASE,
        )
    ):
        quantity_scope = "portfolio"
        portfolio_size_preference = portfolio_size_preference or "maximize_qualified_projects"

    instrument_preference = _clean_text(body.get("instrument_preference") or "none").lower()
    if instrument_preference not in {
        "none",
        "newer",
        "classic",
        "newer_with_legacy_floor",
    }:
        instrument_preference = "none"
    legacy_floor_ratio_raw = body.get("legacy_floor_ratio")
    try:
        legacy_floor_ratio = (
            max(0.0, min(1.0, float(legacy_floor_ratio_raw)))
            if legacy_floor_ratio_raw is not None
            else None
        )
    except (TypeError, ValueError):
        legacy_floor_ratio = None

    def _bounded_text_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return list(
            dict.fromkeys(
                _clean_text(item)[:240]
                for item in value[:100]
                if _clean_text(item)
            )
        )

    exclude_rules = _bounded_text_list(body.get("exclude_rules"))
    success_criteria = _bounded_text_list(body.get("success_criteria"))
    constraint_normalize = normalize_scientific_constraints_result(body.get("scientific_constraints"))
    if constraint_normalize.rejected:
        # Fail-closed: do not silently accept a partial hard-constraint array.
        details = "; ".join(
            str(item.get("message") or item.get("error_code") or "invalid_constraint")
            for item in constraint_normalize.rejected[:5]
        )
        raise ValueError(
            "scientific_constraints contains invalid items and was rejected: "
            f"{details}"
        )
    scientific_constraints = list(constraint_normalize.accepted)
    portfolio_spec = infer_portfolio_spec(
        prompt_text,
        body.get("portfolio_spec"),
    )
    # Direct Discovery-job callers do not necessarily pass through the pool
    # builder. Keep the numeric portfolio contract authoritative there too:
    # otherwise “8 projects, 16 files” silently falls back to broad defaults.
    portfolio_target_projects = _bounded_int(
        portfolio_spec.get("target_projects"), default=0, minimum=0, maximum=5000
    )
    portfolio_target_files = _bounded_int(
        portfolio_spec.get("target_files"), default=0, minimum=0, maximum=200000
    )
    portfolio_min_files = _bounded_int(
        portfolio_spec.get("min_files_per_project"), default=0, minimum=0, maximum=5000
    )
    portfolio_max_files = _bounded_int(
        portfolio_spec.get("max_files_per_project"), default=0, minimum=0, maximum=5000
    )
    requested_max_projects = (
        _bounded_int(body.get("max_projects"), default=2000, minimum=1, maximum=5000)
        if body.get("max_projects") not in (None, "")
        else (portfolio_target_projects or 2000)
    )
    requested_max_files = (
        _bounded_int(body.get("max_files"), default=100000, minimum=1, maximum=200000)
        if body.get("max_files") not in (None, "")
        else (portfolio_target_files or 100000)
    )
    requested_max_files_per_project = (
        _bounded_int(body.get("max_files_per_project"), default=500, minimum=1, maximum=5000)
        if body.get("max_files_per_project") not in (None, "")
        else (portfolio_max_files or 500)
    )
    if portfolio_min_files and "files_per_project" in set(
        portfolio_spec.get("hard_dimensions") or []
    ):
        hard_constraint_fields.append("per_project_min_files")
    existing_constraint_ids = {item.id.casefold() for item in scientific_constraints}
    if instrument_preference != "none" and "builtin.instrument-era" not in existing_constraint_ids:
        scientific_constraints.append(
            ScientificConstraint(
                id="builtin.instrument-era",
                label="仪器代际偏好",
                dimension="instrument_generation",
                operator=("prefer_newer" if instrument_preference != "classic" else "prefer_classic"),
                value={
                    "preference": instrument_preference,
                    "legacy_floor_ratio": legacy_floor_ratio,
                },
                strength="soft",
                scope="project",
                evidence_required=True,
                rationale="Use observed instrument models, not publication date, for ranking.",
                source="user" if "instrument_preference" in body else "inferred",
            )
        )

    def _exclusion_identity(value: Any) -> str:
        normalized = _clean_text(value).casefold()
        normalized = re.sub(
            r"^(?:排除|不要|不含|剔除|exclude(?:d)?|without|omit|remove|no)\s*[:：-]?\s*",
            "",
            normalized,
            flags=re.IGNORECASE,
        )
        return re.sub(r"[^\w\u4e00-\u9fff]+", "", normalized)

    for index, rule in enumerate(exclude_rules, start=1):
        constraint_id = f"exclude.{index}"
        if constraint_id.casefold() in existing_constraint_ids:
            continue
        normalized_rule = _exclusion_identity(rule)
        if any(
            normalized_rule
            and normalized_rule
            in {
                _exclusion_identity(constraint.label),
                _exclusion_identity(constraint.value)
                if isinstance(constraint.value, str)
                else "",
            }
            for constraint in scientific_constraints
        ):
            # The semantic verifier may preserve the same user exclusion in
            # both the first-class exclusion list and a structured constraint.
            # One evidence-backed assessment is sufficient; do not create a
            # second synthetic constraint for the same requirement.
            continue
        scientific_constraints.append(
            ScientificConstraint(
                id=constraint_id,
                label=rule,
                dimension="user_exclusion",
                operator="exclude_if_matches",
                value=rule,
                strength="hard",
                scope="project",
                evidence_required=True,
                rationale="User-specified exclusion rule.",
                source="user",
            )
        )
    for index, criterion in enumerate(success_criteria, start=1):
        constraint_id = f"success.{index}"
        if constraint_id.casefold() in existing_constraint_ids:
            continue
        scientific_constraints.append(
            ScientificConstraint(
                id=constraint_id,
                label=criterion,
                dimension="success_criterion",
                operator="satisfies",
                value=criterion,
                strength="soft",
                scope="portfolio",
                evidence_required=True,
                rationale="User-defined success criterion.",
                source="user",
            )
        )
    if scientific_constraints:
        for constraint in scientific_constraints:
            constraint_provenance[f"constraint:{constraint.id}"] = constraint.source
            if constraint.strength == "hard":
                hard_constraint_fields.append(f"constraint:{constraint.id}")

    return DatasetRequest(
        repository=repository,
        goal=goal,
        ptm_type=ptm_type,
        ptm_types=ptm_types,
        query_terms=list(dict.fromkeys(query_terms)),
        species=species,
        species_policy=species_policy,  # type: ignore[arg-type]
        canonical_species=canonical_species,
        organism_taxon_id=taxon_ids,
        modification_scope=(
            ";".join(ptm_types)
            if goal == "ptm" and ptm_types
            else None
        ),
        immunopeptide_scope=immunopeptide.scope if goal == "immunopeptidomics" else None,
        hla_class=list(immunopeptide.hla_classes) if goal == "immunopeptidomics" else [],
        hla_alleles=list(immunopeptide.hla_alleles) if goal == "immunopeptidomics" else [],
        immunopeptide_evidence_terms=list(immunopeptide.evidence_terms) if goal == "immunopeptidomics" else [],
        immunopeptide_enrichment_methods=(
            list(immunopeptide.enrichment_methods) if goal == "immunopeptidomics" else []
        ),
        immunopeptide_metadata_confidence=(
            immunopeptide.confidence if goal == "immunopeptidomics" else 0.0
        ),
        labeling_strategy=labeling_strategy,
        labeling_hard="labeling_strategy" in hard_constraint_fields,
        acquisition_mode=acquisition,
        mixed_acquisition_policy=mixed_acquisition_policy,  # type: ignore[arg-type]
        instrument_preference=instrument_preference,  # type: ignore[arg-type]
        legacy_floor_ratio=legacy_floor_ratio,
        exclude_rules=exclude_rules,
        success_criteria=success_criteria,
        scientific_constraints=scientific_constraints,
        max_projects=(
            max(
                2000,
                requested_max_projects,
            )
            if explicit_exhaustive and quota_flexibility != "fixed"
            else requested_max_projects
        ),
        max_files=(
            max(100000, requested_max_files)
            if explicit_exhaustive and quota_flexibility != "fixed"
            else requested_max_files
        ),
        max_candidate_projects=(
            max(
                20000,
                _bounded_int(
                    body.get("max_candidate_projects"),
                    default=20000,
                    minimum=1,
                    maximum=20000,
                ),
            )
            if explicit_exhaustive and quota_flexibility != "fixed"
            else _bounded_int(
                body.get("max_candidate_projects"),
                default=5000,
                minimum=1,
                maximum=20000,
            )
        ),
        max_files_per_project=requested_max_files_per_project,
        partial_delivery_batch_size=_bounded_int(
            body.get("partial_delivery_batch_size"), default=500, minimum=1, maximum=5000
        ),
        inspection_batch_size=_bounded_int(
            body.get("inspection_batch_size"), default=30, minimum=1, maximum=100
        ),
        continuous_discovery=(
            True
            if quota_flexibility in {"fixed", "open_ended"}
            else bool(body.get("continuous_discovery", quantity_scope == "portfolio"))
        ),
        quantity_scope=quantity_scope,  # type: ignore[arg-type]
        portfolio_size_preference=portfolio_size_preference,
        quota_flexibility=quota_flexibility,  # type: ignore[arg-type]
        run_horizon=run_horizon,  # type: ignore[arg-type]
        time_budget_preference=time_budget_preference,  # type: ignore[arg-type]
        on_safety_ceiling=on_safety_ceiling,  # type: ignore[arg-type]
        harvest_all_qualified=quantity_scope == "portfolio" and bool(portfolio_size_preference),
        hard_constraint_fields=list(dict.fromkeys(hard_constraint_fields)),
        constraint_provenance=constraint_provenance,
        portfolio_spec=portfolio_spec,
        per_project_min_files=portfolio_min_files or None,
    )


def _model_request_records(payload: Any) -> list[dict[str, Any]]:
    return model_informed_request_records(payload)


def _find_model_informed_discovery_request(body: dict[str, Any]) -> dict[str, Any]:
    direct = body.get("request")
    if isinstance(direct, dict):
        records = _model_request_records(direct)
        if records:
            return records[0]

    build_id = _clean_text(body.get("build_id") or body.get("ai_ready_build_id"))
    if build_id:
        output_dir = _safe_ai_ready_dir(build_id)
        if output_dir is None or not output_dir.exists():
            raise ValueError("AI-ready build not found.")
        payload = _read_json_if_exists(output_dir / "model_informed_discovery_requests.json")
        records = _model_request_records(payload)
        if not records:
            raise ValueError("No model-informed discovery requests found for this build.")

        request_id = _clean_text(body.get("request_id") or body.get("model_request_id"))
        if not request_id:
            if len(records) == 1:
                return records[0]
            raise ValueError("request_id is required when a build has multiple discovery requests.")
        for record in records:
            if _clean_text(record.get("request_id")) == request_id:
                return record
        raise ValueError(f"Model-informed discovery request not found: {request_id}")

    if any(key in body for key in ("request_id", "query", "constraints", "dimension", "target")):
        records = _model_request_records(body)
        if records:
            return records[0]
    raise ValueError("Provide either request or build_id.")


def _first_clean_request_value(*values: Any) -> str:
    for value in values:
        if isinstance(value, list):
            for item in value:
                text = _clean_text(item)
                if text:
                    return text
            continue
        text = _clean_text(value)
        if text:
            return text
    return ""


def _species_from_model_request(request: dict[str, Any], constraints: dict[str, Any]) -> list[str]:
    raw = (
        constraints.get("species")
        or constraints.get("organism")
        or constraints.get("organism_preference")
        or request.get("species")
        or request.get("organism")
    )
    if raw:
        return _clean_discovery_species(raw)
    return ["human"]


def _repository_from_model_request(request: dict[str, Any]) -> str:
    repository = _clean_text(request.get("repository"))
    if repository:
        return _clean_repository(repository, default="auto")
    repositories = request.get("repositories")
    if isinstance(repositories, list):
        cleaned = [_clean_repository(item, default="") for item in repositories]
        cleaned = [item for item in cleaned if item]
        if len(set(cleaned)) == 1:
            return cleaned[0]
        if cleaned:
            return "auto"
    return "auto"


def _ptm_from_model_request(request: dict[str, Any], constraints: dict[str, Any]) -> str:
    raw = _first_clean_request_value(
        constraints.get("modification_scope"),
        constraints.get("ptm_type"),
        constraints.get("ptm"),
        request.get("ptm_type"),
        request.get("modification_scope"),
    )
    if not raw and _clean_text(request.get("dimension")).lower() in {"ptm", "modification", "modification_scope"}:
        raw = _clean_text(request.get("target"))
    if raw in {"any_ptm", "modified_peptides", "modified_peptide"}:
        return "unknown_ptm"
    return normalize_ptm_type(raw or "unknown_ptm")


def _query_from_model_request(request: dict[str, Any], payload: dict[str, Any]) -> str:
    query = _clean_text(request.get("query") or payload.get("query") or payload.get("prompt"))
    if query:
        return query
    pieces = [
        _clean_text(request.get("task_type")),
        _clean_text(request.get("target")),
        _clean_text(request.get("dimension")),
        _clean_text(request.get("reason")),
    ]
    return " ".join(piece for piece in pieces if piece).strip()


def _discovery_payload_from_model_request(
    request: dict[str, Any],
    *,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return build_model_informed_discovery_payload(request, overrides=overrides)


def _run_model_informed_discovery_payload(body: dict[str, Any]) -> dict[str, Any]:
    request = _find_model_informed_discovery_request(body)
    payload = _discovery_payload_from_model_request(request, overrides=body)
    return {
        "status": "ready",
        "request_id": payload.get("model_informed_request_id"),
        "requires_user_confirmation": payload.get("requires_user_confirmation", True),
        "payload": payload,
        "request": request,
    }


_DISCOVERY_TASK_TYPES = {
    "",
    "rt_prediction",
    "fragment_intensity_prediction",
    "psm_scoring",
    "denovo",
    "ptm_denovo",
    "chimeric_interpretation",
}
_DISCOVERY_DIVERSITY_STRATEGIES = {"balanced", "high", "off"}
_DISCOVERY_REPOSITORIES = {"pride", "massive", "iprox", "auto", "local"}
_DISCOVERY_GOALS = {"general", "ptm", "immunopeptidomics"}
_POOL_BUILD_SCALE_PRESETS: dict[str, dict[str, int]] = {
    "quick": {
        "max_projects": 20,
        "max_candidate_projects": 80,
        "max_files": 2000,
        "max_files_per_project": 150,
    },
    "curated": {
        "max_projects": 50,
        "max_candidate_projects": 250,
        "max_files": 2000,
        "max_files_per_project": 150,
    },
    "balanced": {
        "max_projects": 200,
        "max_candidate_projects": 800,
        "max_files": 10000,
        "max_files_per_project": 250,
    },
    "exhaustive": {
        # Soft per-run safety thresholds only. Portfolio maximize keeps every
        # qualified project across resumable rounds; these are not business caps.
        "max_projects": 2000,
        "max_candidate_projects": 20000,
        "max_files": 100000,
        "max_files_per_project": 500,
    },
}


def _normalise_pool_build_language(value: Any) -> str:
    return "zh-CN" if _clean_ui_language(value) == "zh" else "en"


def _normalise_pool_build_scale(value: Any, *, prompt: str = "", allow_auto: bool = True) -> str:
    raw = _clean_text(value).casefold().replace("-", "_")
    aliases = {
        "quick": "quick",
        "fast": "quick",
        "快速": "quick",
        "selected": "curated",
        "select": "curated",
        "curated": "curated",
        "pilot": "curated",
        "精选": "curated",
        "balanced": "balanced",
        "balance": "balanced",
        "均衡": "balanced",
        "exhaustive": "exhaustive",
        "comprehensive": "exhaustive",
        "complete": "exhaustive",
        "尽量搜全": "exhaustive",
        "搜全": "exhaustive",
        "auto": "auto",
        "automatic": "auto",
        "自动": "auto",
    }
    normalized = aliases.get(raw)
    if normalized and (normalized != "auto" or allow_auto):
        return normalized
    text = _clean_text(prompt).casefold()
    if _has_explicit_exhaustive_discovery_intent(text):
        return "exhaustive"
    if any(marker in text for marker in ("快速", "quick", "find enough", "达到即停")):
        return "quick"
    if any(marker in text for marker in ("精选", "少量", "先验证", "pilot", "curated", "small set")):
        return "curated"
    if any(marker in text for marker in ("均衡", "balanced")):
        return "balanced"
    return "auto" if allow_auto else "balanced"


def _english_discovery_query_terms(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(
        dict.fromkeys(
            term
            for item in value
            if (term := _clean_text(item))
            and term.isascii()
            and re.search(r"[A-Za-z]", term)
        )
    )


def _pool_build_scale_warning(scale_mode: str, output_language: str) -> str | None:
    if scale_mode != "exhaustive":
        return None
    if output_language == "zh-CN":
        return "“尽量搜全”不设业务候选池上限；安全上限只用于分轮运行与续跑，不会把结果截断为固定项目数。"
    return "Exhaustive mode has no business candidate-pool cap; safety thresholds only split resumable rounds and do not truncate the result to a fixed project count."


def _localize_prompt_parse_warning(value: Any, output_language: str) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    if output_language == "zh-CN":
        if _contains_cjk(text):
            return text
        patterns = (
            (r"^Unsupported task_type '(.+)' was ignored\.$", r"不支持任务类型“\1”，已忽略。"),
            (r"^Unsupported repository '(.+)' was ignored; using PRIDE\.$", r"不支持数据仓库“\1”，已改用 PRIDE。"),
            (r"^Unsupported discovery target '(.+)' was ignored; using general\.$", r"不支持发现目标“\1”，已改用通用发现。"),
            (r"^Acquisition '(.+)' is not supported in this DDA-first workflow; using dda\.$", r"当前 DDA 优先流程不支持采集模式“\1”，已改用 DDA。"),
        )
        for pattern, replacement in patterns:
            if re.match(pattern, text):
                return re.sub(pattern, replacement, text)
        return "请求解析器报告了需要注意的条件。"
    if _contains_cjk(text):
        return "The prompt parser reported a condition that requires attention."
    return text


def _localize_prompt_parse_reasoning(value: Any, output_language: str) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    if output_language == "zh-CN" and not _contains_cjk(text):
        return "请求已解析，并生成英文仓库检索词。"
    if output_language == "en" and _contains_cjk(text):
        return "The request was parsed and English repository search terms were generated."
    return text


def _explicit_discovery_goal_overrides(prompt: str) -> dict[str, Any]:
    text = _clean_text(prompt).lower()
    overrides: dict[str, Any] = {}
    round_match = re.search(
        r"(?:agentic\s*)?rounds?\s*[:=]?\s*([12])|\b([12])\s*(?:agentic\s*)?rounds?\b",
        text,
    )
    if round_match:
        round_value = next((item for item in round_match.groups() if item), "")
        if round_value:
            overrides["agentic_rounds"] = int(round_value)
    elif any(marker in text for marker in ("one round", "single round", "round 1")):
        overrides["agentic_rounds"] = 1
    elif any(marker in text for marker in ("two rounds", "second round", "round 2")):
        overrides["agentic_rounds"] = 2
    return overrides


def _normalise_discovery_goal_parse(raw: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    payload = raw.get("fields") if isinstance(raw.get("fields"), dict) else raw
    warnings = [str(item) for item in raw.get("warnings", []) if str(item).strip()] if isinstance(raw.get("warnings"), list) else []

    raw_goal = _clean_text(payload.get("goal") or current.get("goal") or "general").lower()
    species = payload.get("species", current.get("species") if "species" in current else [])
    species_values = _clean_discovery_species(species, default=[])
    species_policy = _clean_text(payload.get("species_policy") or current.get("species_policy") or "open").lower()
    if species_policy not in {"open", "include_only", "exclude"}:
        species_policy = "open"
    task_type = _clean_text(payload.get("task_type") or current.get("task_type")).lower()
    if task_type not in _DISCOVERY_TASK_TYPES:
        warnings.append(f"Unsupported task_type '{task_type}' was ignored.")
        task_type = _clean_text(current.get("task_type")).lower()
        task_type = task_type if task_type in _DISCOVERY_TASK_TYPES else ""

    diversity_strategy = _clean_text(payload.get("diversity_strategy") or current.get("diversity_strategy") or "balanced").lower()
    if diversity_strategy not in _DISCOVERY_DIVERSITY_STRATEGIES:
        diversity_strategy = "balanced"
    repository = _clean_repository(payload.get("repository") or current.get("repository") or "pride")
    if repository not in _DISCOVERY_REPOSITORIES:
        warnings.append(f"Unsupported repository '{repository}' was ignored; using PRIDE.")
        repository = "pride"

    goal = raw_goal
    prompt_text = _clean_text(current.get("prompt"))
    if goal not in _DISCOVERY_GOALS:
        if is_immunopeptidomics_goal(goal) and not re.search(
            r"(proteomics|proteome|peptidomics|蛋白质组|肽组|蛋白肽|shotgun|bottom[\s-]?up)",
            f"{raw_goal} {prompt_text}",
            re.IGNORECASE,
        ):
            goal = "immunopeptidomics"
        else:
            warnings.append(f"Unsupported discovery target '{goal}' was ignored; using general.")
            goal = "general"
    if goal == "immunopeptidomics" and re.search(
        r"(proteomics|proteome|peptidomics|蛋白质组|肽组|蛋白肽|shotgun|bottom[\s-]?up)",
        f"{raw_goal} {prompt_text}",
        re.IGNORECASE,
    ) and not re.search(
        r"(immunopeptidom|hla\b|mhc\b|ligandome|免疫肽)",
        f"{raw_goal} {prompt_text}",
        re.IGNORECASE,
    ):
        warnings.append("Broad proteomics/peptidomics request kept as general rather than immunopeptidomics.")
        goal = "general"
    if goal == "general":
        ptm_type = "unknown_ptm"
        ptm_types: list[str] = []
    else:
        default_ptm = "unknown_ptm" if goal == "immunopeptidomics" else "phospho"
        ptm_types = _clean_discovery_ptm_types(payload.get("ptm_types"), default=[])
        if not ptm_types:
            ptm_types = _clean_discovery_ptm_types(payload.get("ptm_type") or current.get("ptm_types") or current.get("ptm_type"), default=[default_ptm])
        ptm_type = ptm_types[0] if ptm_types else default_ptm
    labeling_strategy = normalize_labeling_strategy(
        payload.get("labeling_strategy") or current.get("labeling_strategy") or "unknown"
    )

    acquisition = _clean_text(payload.get("acquisition_mode") or current.get("acquisition_mode") or "unknown").lower()
    if acquisition in {"", "any", "auto"}:
        acquisition = "unknown"
    if acquisition not in {"dda", "dia", "unknown"}:
        warnings.append(f"Acquisition '{acquisition}' is not a supported open mode; using unknown.")
        acquisition = "unknown"
    if not species_values and re.search(r"(human|homo sapiens|人类|人源|智人)", prompt_text, re.IGNORECASE):
        species_values = ["human"]
    if species_values and species_policy == "open" and re.search(
        r"(only human|human only|strict human|只要人|仅人类|人类数据|人源数据|人类蛋白质组|人类肽组|人类蛋白肽)",
        prompt_text,
        re.IGNORECASE,
    ):
        species_policy = "include_only"
    canonical_species, taxon_ids = normalize_species_values(species_values)
    raw_query_terms = payload.get("query_terms") or current.get("query_terms") or []
    query_terms = _english_discovery_query_terms(raw_query_terms)
    if goal == "general" and not query_terms:
        query_terms = _english_discovery_query_terms(general_query_terms_from_text(prompt_text or raw_goal))
    scale_mode = _normalise_pool_build_scale(
        payload.get("scale_mode") or current.get("scale_mode"),
        prompt=prompt_text,
        allow_auto=True,
    )
    output_language = _normalise_pool_build_language(
        current.get("output_language") or payload.get("output_language")
    )

    return {
        "fields": {
            "repository": repository,
            "goal": goal,
            "ptm_type": ptm_type,
            "ptm_types": ptm_types,
            "query_terms": query_terms,
            "species": species_values,
            "species_policy": species_policy,
            "canonical_species": canonical_species,
            "organism_taxon_id": taxon_ids,
            "modification_scope": None if goal == "general" else ";".join(ptm_types or [ptm_type]),
            "labeling_strategy": labeling_strategy,
            "acquisition_mode": acquisition,
            "task_type": task_type,
            "max_projects": _bounded_int(payload.get("max_projects"), default=_bounded_int(current.get("max_projects"), default=50, minimum=1, maximum=300), minimum=1, maximum=300),
            "max_files": _bounded_int(payload.get("max_files"), default=_bounded_int(current.get("max_files"), default=2000, minimum=1, maximum=10000), minimum=1, maximum=10000),
            "max_candidate_projects": _bounded_int(payload.get("max_candidate_projects"), default=_bounded_int(current.get("max_candidate_projects"), default=300, minimum=1, maximum=20000), minimum=1, maximum=20000),
            "max_files_per_project": _bounded_int(payload.get("max_files_per_project"), default=_bounded_int(current.get("max_files_per_project"), default=100, minimum=1, maximum=200), minimum=1, maximum=200),
            "agentic_rounds": _bounded_int(payload.get("agentic_rounds"), default=_bounded_int(current.get("agentic_rounds"), default=1, minimum=1, maximum=2), minimum=1, maximum=2),
            "diversity_strategy": diversity_strategy,
            "agentic": current.get("agentic") is True,
            "scale_mode": scale_mode,
            "output_language": output_language,
        },
        "warnings": warnings,
        "reasoning": _clean_text(raw.get("reasoning") or raw.get("rationale")),
    }


def _discovery_goal_parse_system_prompt() -> str:
    return (
        "You parse proteomics dataset discovery goals into safe UI fields. "
        "Return JSON only. Do not search PRIDE and do not invent unsupported capabilities. "
        "Supported repository values: pride, massive, iprox, auto. "
        "Broad remote discovery is PRIDE-first; MassIVE/iProX v1 use repository-smoke for known accessions/files. "
        "Supported discovery target values: general, ptm, immunopeptidomics. Default to goal=general for broad or future dataset searches. "
        "Use goal=ptm only when the user explicitly asks for PTM-enriched data. Use goal=immunopeptidomics only when the user explicitly wants HLA/MHC ligandome/immunopeptidomics as the discovery target. "
        "For general HLA, drug-treatment, disease, perturbation, cell-line, or future task searches, keep goal=general and put useful search phrases in query_terms. "
        "For HLA/MHC ligandome, eluted ligand, neoantigen, or immunopeptidome goals set ptm_type=unknown_ptm and ptm_types=[] unless a modification is explicitly requested. "
        "Supported ptm_type values: phospho, acetyl, ubiquitin, glyco, methyl, unknown_ptm; when goal=ptm, return ptm_types as a list and allow multiple values. acquisition_mode=unknown unless the user explicitly requires DDA or DIA. "
        "Supported labeling_strategy values: label_free, TMT, iTRAQ, unknown. Prefer unknown unless the user explicitly requires a labeling strategy. "
        "Species policy defaults to open; use include_only when the user asks for human-only or only listed species, and exclude only when the user explicitly excludes species. "
        "For requests about human proteomics/peptidomics/protein-peptide data (including Chinese '人类蛋白质组/肽组/蛋白肽'), set goal=general, species=['human'], species_policy=include_only, and do NOT set goal=immunopeptidomics unless HLA/MHC/immunopeptidomics is explicit. "
        "PTM interpretation should normalize semantic terms and enrichment methods such as pSer/pThr/pTyr, kinase signaling, phosphosite localization, Ti/Fe/Ga/Ti4+-IMAC, MOAC, PolyMAC, Titansphere, GlyGly/K-GG, Kac, HILIC, lectin enrichment, Kme/Rme. "
        "Immunopeptidomics interpretation should normalize HLA/MHC ligandome, immunopeptidome, HLA/MHC eluted ligands, neoantigen, antigen presentation, HLA-IP/MHC-IP, W6/32, pan-HLA, HLA class I/II, MHC class I/II, and HLA alleles such as HLA-A*02:01. "
        "Supported task_type values: rt_prediction, fragment_intensity_prediction, psm_scoring, "
        "denovo, ptm_denovo, chimeric_interpretation, or empty string. "
        "Supported diversity_strategy values: balanced, high, off. "
        "Supported scale_mode values: quick, balanced, exhaustive (curated is a legacy alias). "
        "Use quick with quota_flexibility=fixed when the user asks for a concrete number of usable projects; infer exhaustive when the user asks for as many relevant projects as possible / 越多越好 / 尽可能多. "
        "All query_terms must be concise English phrases suitable for repository search even when the request is written in another language. For broad human proteomics, include terms such as human proteomics, shotgun proteomics, label free quantitation, TMT, DIA, phosphoproteomics, affinity purification mass spectrometry, plasma proteomics. "
        "Warnings and reasoning must use the requested output language. "
        "Do not enable agentic discovery; agentic execution is controlled only by explicit advanced settings. "
        "If the user asks for DIA/PRM/SRM/MRM, keep acquisition_mode open/unknown unless they force DDA, and add a warning only when a hard DDA constraint was requested."
    )


def _run_discovery_goal_parse(body: dict[str, Any]) -> dict[str, Any]:
    prompt = _clean_text(body.get("prompt"))
    if not prompt:
        raise ValueError("Please enter a discovery request.")
    llm_config = body.get("llm_config") if isinstance(body.get("llm_config"), dict) else {}
    client = _discovery_llm_client(
        llm_config,
        allow_server_default=body.get("allow_server_default") is not False,
    )
    if client is None:
        raise ValueError("No discovery LLM API key found. Fill API Configuration or set DEEPSEEK_API_KEY.")
    current = body.get("current") if isinstance(body.get("current"), dict) else {}
    output_language = _normalise_pool_build_language(current.get("output_language") or body.get("output_language"))
    explanation_language = "Chinese" if output_language == "zh-CN" else "English"
    user_prompt = (
        "Parse this discovery request into a JSON object with fields, warnings, and reasoning.\n\n"
        f"Discovery request:\n{prompt}\n\n"
        f"Current UI fields:\n{json.dumps(current, ensure_ascii=False, indent=2)}\n\n"
        f"Return warnings and reasoning in {explanation_language}. Always return query_terms in English.\n\n"
        "Expected JSON shape:\n"
        "{\n"
        '  "fields": {\n'
        '    "repository": "pride",\n'
        '    "goal": "general",\n'
        '    "ptm_type": "phospho",\n'
        '    "ptm_types": ["phospho", "acetyl"],\n'
        '    "query_terms": ["drug treatment DDA proteomics"],\n'
        '    "scale_mode": "balanced",\n'
        '    "species": ["human"],\n'
        '    "species_policy": "open",\n'
        '    "labeling_strategy": "label_free",\n'
        '    "acquisition_mode": "dda",\n'
        '    "task_type": "rt_prediction",\n'
        '    "agentic_rounds": 1,\n'
        '    "diversity_strategy": "high"\n'
        "  },\n"
        '  "warnings": [],\n'
        '  "reasoning": "short explanation"\n'
        "}"
    )
    raw = client.complete_json(system_prompt=_discovery_goal_parse_system_prompt(), user_prompt=user_prompt)
    parsed = _normalise_discovery_goal_parse(raw, {**current, "prompt": prompt})
    explicit_overrides = _explicit_discovery_goal_overrides(prompt)
    if explicit_overrides:
        parsed["fields"].update(explicit_overrides)
    return {
        "status": "completed",
        "parser": "llm",
        "prompt": prompt,
        **parsed,
    }


_DISCOVERY_AGENT_GUIDANCE_PATH = (
    Path(__file__).resolve().parents[3] / "docs" / "discovery-agent-guidance.md"
)
_DISCOVERY_AGENT_GUIDANCE_FALLBACK = """Scientific discovery guidance:
- Choose the highest-value next decision from the current science context; do not follow a fixed questionnaire.
- Treat gap reports and legacy pending questions as guidance, not a command.
- For exploratory immunopeptide/HLA-ligandome discovery, do not default to PTM de novo. Keep it human-prioritized and offer either quick fixed-target discovery (20 usable projects by default) or exhaustive open-ended coverage.
- De novo sequencing, PSM scoring, and RT prediction are optional later tasks after the exploratory corpus is understood.
"""
_DISCOVERY_TURN_ACTIONS = {
    "chat",
    "advise",
    "clarify",
    "update_strategy",
    "ready_to_confirm",
    "confirm_strategy",
    "refuse_search",
}
_DISCOVERY_STRATEGY_FIRST_CLASS_FIELDS = {
    "objective",
    "task_type",
    "run_horizon",
    "species",
    "species_policy",
    "species_coverage",
    "acquisition_mode",
    "mixed_acquisition_policy",
    "ptm_types",
    "special_themes",
    "selected_search_terms",
    "labeling_strategy",
    "labeling_hard",
    "coverage_mode",
    "target_project_count",
    "max_candidate_projects",
    "quota_flexibility",
    "time_budget",
    "on_safety_ceiling",
    "instrument_preference",
    "legacy_floor_ratio",
    "exclude_rules",
    "success_criteria",
    "scientific_constraints",
    "notes",
    "open_risks",
    "repository",
}
_DISCOVERY_STRATEGY_PATCH_FIELDS = set(_DISCOVERY_STRATEGY_FIRST_CLASS_FIELDS)
# ``notes`` is explicitly context-only; actionable requirements must live in a
# first-class field or scientific_constraints.  A repair verifier may remove a
# duplicated model-authored note without invalidating the atomic strategy
# update, while every execution-affecting field remains required.
_DISCOVERY_NON_ATOMIC_CONTEXT_FIELDS = {"notes"}
# Low-risk first-class fields may skip the independent semantic verifier when
# the Manager emits a typed update_strategy patch limited to this set (single
# field or a small compound update). scientific_constraints still forces critic.
_DISCOVERY_LOW_RISK_SINGLE_FIELD_VERIFIER_SKIP = frozenset(
    {
        "objective",
        "task_type",
        "species",
        "species_policy",
        "species_coverage",
        "acquisition_mode",
        "mixed_acquisition_policy",
        "coverage_mode",
        "target_project_count",
        "quota_flexibility",
        "labeling_strategy",
        "labeling_hard",
        "instrument_preference",
        "run_horizon",
        "special_themes",
        "selected_search_terms",
        "notes",
    }
)
# Max fields in a compound low-risk skip (compound natural-language dumps).
# A complete structured recap currently normalizes to 13 low-risk fields once
# open-ended quota, mixed-acquisition, and labeling hardness are made explicit.
# Leave one slot for other first-class preferences while preserving an over-max
# guard; scientific_constraints and every non-whitelist field still require
# the independent verifier.
_DISCOVERY_LOW_RISK_COMPOUND_MAX_FIELDS = 15
# On hard verifier reject, retain only soft theme/context fields from the
# primary update_strategy tool call instead of wiping the whole card.
_DISCOVERY_SOFT_REJECT_KEEP_FIELDS = frozenset(
    {"objective", "special_themes", "notes", "task_type"}
)
_DISCOVERY_STRATEGY_RESERVED_RUNTIME_FIELDS = {
    "query_terms",
    "diversity_strategy",
    "constraints_enabled",
    "hard_constraint_fields",
    "constraint_provenance",
    "agentic_rounds",
    "max_files",
    "max_files_per_project",
    "original_prompt",
    "runtime",
    "llm_config",
    "grill_confirmed",
    "strategy_fingerprint",
    "strategy_fingerprint_payload",
}
_DISCOVERY_STRATEGY_PATCH_ALIASES = {
    "goal": "objective",
    "goal_summary": "objective",
    "goalSummary": "objective",
    "taskType": "task_type",
    "runHorizon": "run_horizon",
    "speciesPolicy": "species_policy",
    "speciesCoverage": "species_coverage",
    "acquisitionMode": "acquisition_mode",
    "mixedAcquisitionPolicy": "mixed_acquisition_policy",
    "ptmTypes": "ptm_types",
    "specialThemes": "special_themes",
    "selectedSearchTerms": "selected_search_terms",
    "labelingStrategy": "labeling_strategy",
    "labelingHard": "labeling_hard",
    "coverageMode": "coverage_mode",
    "scale_mode": "coverage_mode",
    "scaleMode": "coverage_mode",
    "targetProjectCount": "target_project_count",
    "max_projects": "target_project_count",
    "maxProjects": "target_project_count",
    "maxCandidateProjects": "max_candidate_projects",
    "quotaFlexibility": "quota_flexibility",
    "timeBudget": "time_budget",
    "time_budget_preference": "time_budget",
    "timeBudgetPreference": "time_budget",
    "onSafetyCeiling": "on_safety_ceiling",
    "instrumentPreference": "instrument_preference",
    "legacyFloorRatio": "legacy_floor_ratio",
    "excludeRules": "exclude_rules",
    "successCriteria": "success_criteria",
    "scientificConstraints": "scientific_constraints",
    "openRisks": "open_risks",
}
_DISCOVERY_STRATEGY_PATCH_CONTRACT = {
    "$clear": (
        "Every first-class strategy field accepts null to reset to the empty-strategy default; "
        "array fields also accept [] to clear. Query/runtime extension fields use omission, not null."
    ),
    "$limits": {
        "objective_chars": 120,
        "notes_chars": 4000,
        "array_items": 100,
        "array_item_chars": 240,
        "target_project_count": 5000,
        "max_candidate_projects": 20000,
    },
    "objective": "string",
    "task_type": [
        "rt_prediction",
        "fragment_intensity_prediction",
        "psm_scoring",
        "denovo",
        "ptm_denovo",
        "chimeric_interpretation",
        "browse_only",
        "other",
    ],
    "run_horizon": ["candidates_reviewed"],
    "species": "array[string], [] clears",
    "species_policy": ["open", "include_only", "prefer", "exclude"],
    "species_coverage": ["none", "prefer_listed", "broaden"],
    "acquisition_mode": ["dda", "dia", "unknown"],
    "mixed_acquisition_policy": ["reject_mixed", "review_mixed", "allow"],
    "ptm_types": "array[string], [] clears",
    "special_themes": "array[string], [] clears",
    "selected_search_terms": "ordered array[string], [] clears",
    "labeling_strategy": [
        "label_free",
        "tmt",
        "itraq",
        "silac",
        "dimethyl",
        "unknown",
        "any",
    ],
    "labeling_hard": "boolean or null",
    "coverage_mode (scale_mode accepted as input alias)": ["quick", "curated", "balanced", "exhaustive"],
    "target_project_count (max_projects accepted as input alias)": "positive integer or null",
    "max_candidate_projects": "positive integer or null",
    "quota_flexibility": ["fixed", "recommended", "open_ended"],
    "time_budget": ["fast", "multi_round"],
    "on_safety_ceiling": ["ask", "auto_continue_within_safety", "stop"],
    "instrument_preference": ["none", "newer", "classic", "newer_with_legacy_floor"],
    "legacy_floor_ratio": "number from 0 to 1 or null",
    "exclude_rules/success_criteria/open_risks": "array[string], [] clears",
    "scientific_constraints": (
        "array[{id,label,dimension,operator,value,strength,scope,evidence_required,"
        "rationale,source}], [] clears"
    ),
    "notes": "non-empty string or null",
    "repository": ["pride", "massive", "iprox", "auto"],
}
_DISCOVERY_STRATEGY_FIELD_SEMANTICS = {
    "objective": "The user's scientific/data-discovery objective; do not use it as a dump for other fields.",
    "task_type": (
        "The downstream analytical use of the data. browse_only is a deliberate user choice, "
        "not a default. Finding or reviewing candidates describes run_horizon and never by itself "
        "authorizes browse_only."
    ),
    "run_horizon": (
        "Fixed system invariant: every discovery run finds candidates and reviews them. "
        "It is always candidates_reviewed and must never be offered as a user choice."
    ),
    "species": "Organism names or taxa chosen by the user.",
    "species_policy": "Whether species are open, mandatory inclusions, preferences, or exclusions.",
    "species_coverage": "Whether to keep species neutral, prefer listed organisms, or broaden coverage.",
    "acquisition_mode": "Mass-spectrometry acquisition mode; unknown means intentionally unrestricted/open.",
    "mixed_acquisition_policy": "How mixed-acquisition projects are handled.",
    "ptm_types": "Post-translational modifications only; immunopeptidomics itself is not a PTM.",
    "special_themes": "研究主题/biological study themes. Never use this field for 标记方式, labeling chemistry, acquisition, or run horizon.",
    "selected_search_terms": (
        "Exact ordered PRIDE repository query phrases explicitly selected by the user. "
        "Preserve their order because discovery searches core terms before broad fallback terms."
    ),
    "labeling_strategy": "标记方式/chemical or isotope labeling strategy; any means intentionally unrestricted/open. A request that labeling is open belongs here, never in special_themes.",
    "labeling_hard": "Whether the labeling choice is a hard filter.",
    "coverage_mode": "Quick fixed-target versus exhaustive-coverage execution mode, distinct from the exact project target.",
    "target_project_count": "Desired selected-project count, not the candidate-pool size.",
    "max_candidate_projects": (
        "Bounded-mode retention limit. Compatibility hint only in continuous/maximize mode, "
        "where candidate count is not capped."
    ),
    "quota_flexibility": (
        "Whether the project target is fixed, recommended, or open-ended. A user-stated numeric "
        "project target is fixed and must stop new search/review scheduling once reached."
    ),
    "time_budget": "Fast single-pass preference versus a multi-round search.",
    "on_safety_ceiling": "What to do at server safety limits; it cannot remove those limits.",
    "instrument_preference": "Instrument-era preference, not an observed repository fact.",
    "legacy_floor_ratio": "Minimum legacy share when a mixed instrument-era preference is used.",
    "exclude_rules": "Explicit exclusions.",
    "success_criteria": "User-defined criteria for a successful discovery result.",
    "scientific_constraints": (
        "Open-ended structured scientific requirements. Use this whenever a meaningful user "
        "requirement has no first-class field; do not reduce it to prose-only notes."
    ),
    "notes": "Meaningful constraints without a first-class field.",
    "open_risks": "Evidence checks or unresolved scientific risks retained for later review.",
    "repository": "Repository scope.",
}
_DISCOVERY_STRATEGY_FIELD_LABELS_ZH = {
    "objective": "目标",
    "task_type": "下游任务",
    "run_horizon": "交付终点",
    "species": "物种",
    "species_policy": "物种策略",
    "species_coverage": "物种覆盖",
    "acquisition_mode": "采集方式",
    "mixed_acquisition_policy": "混合采集处理",
    "ptm_types": "PTM",
    "special_themes": "研究主题",
    "selected_search_terms": "仓库检索主题词",
    "labeling_strategy": "标记方式",
    "labeling_hard": "标记硬限制",
    "coverage_mode": "覆盖模式",
    "target_project_count": "目标项目数",
    "max_candidate_projects": "普通模式候选保留上限",
    "quota_flexibility": "数量弹性",
    "time_budget": "时间偏好",
    "on_safety_ceiling": "触顶策略",
    "instrument_preference": "仪器偏好",
    "legacy_floor_ratio": "经典仪器占比下限",
    "exclude_rules": "排除条件",
    "success_criteria": "成功标准",
    "scientific_constraints": "科学约束",
    "notes": "备注约束",
    "open_risks": "待核验证据",
    "repository": "仓库",
}

_DISCOVERY_EXPLICIT_ENUM_HINTS: dict[str, dict[str, tuple[str, ...]]] = {
    "task_type": {
        "browse_only": ("browse", "浏览", "先看看", "只找数据", "任务未定", "摸清"),
        "rt_prediction": ("rt", "保留时间"),
        "fragment_intensity_prediction": ("fragment intensity", "碎片强度"),
        "psm_scoring": ("psm", "打分", "评分"),
        "denovo": ("de novo", "denovo", "从头测序"),
        "ptm_denovo": ("ptm de novo", "ptm denovo", "修饰从头测序"),
        "chimeric_interpretation": ("chimeric", "嵌合谱"),
        "other": ("other", "其它任务", "其他任务"),
    },
    "run_horizon": {
        "plan_only": ("plan only", "只做计划", "先做计划"),
        "candidates_only": ("candidates only", "候选即停", "找到候选", "只要候选"),
        "candidates_reviewed": ("review candidates", "审查候选", "复核候选"),
        "ai_ready_table": ("ai-ready", "ai ready", "训练表", "数据表"),
        "pre_release": ("pre-release", "预发布"),
        "full_release": ("full release", "完整发布"),
    },
    "species_policy": {
        "open": ("open", "开放", "不限物种", "不限制物种"),
        "include_only": ("only", "只要", "仅限", "必须是"),
        "prefer": ("prefer", "优先"),
        "exclude": ("exclude", "排除", "不要"),
    },
    "species_coverage": {
        "none": ("neutral", "不扩物种", "无需扩展"),
        "prefer_listed": ("prefer listed", "优先列出"),
        "broaden": ("broaden", "覆盖更多物种", "扩展物种"),
    },
    "acquisition_mode": {
        "dda": ("dda",),
        "dia": ("dia",),
        "unknown": ("不限采集", "采集方式开放", "采集方式未知"),
    },
    "mixed_acquisition_policy": {
        "reject_mixed": ("reject mixed", "排除混合", "不要混合"),
        "review_mixed": ("review mixed", "审查混合", "文件级审查"),
        "allow": ("allow mixed", "允许混合", "混合也可以"),
    },
    "labeling_strategy": {
        "label_free": ("label-free", "label free", "无标记"),
        "tmt": ("tmt",),
        "itraq": ("itraq",),
        "silac": ("silac",),
        "dimethyl": ("dimethyl", "二甲基"),
        "unknown": ("标记未知",),
        "any": ("标记开放", "不限标记", "任何标记"),
    },
    "coverage_mode": {
        "quick": ("quick", "快速", "达到即停", "找够"),
        "curated": ("curated", "精选", "少量高质量"),
        "balanced": ("balanced", "均衡", "平衡"),
        "exhaustive": ("exhaustive", "尽量搜全", "搜全", "最大覆盖"),
    },
    "quota_flexibility": {
        "fixed": ("fixed", "固定数量", "必须达到"),
        "recommended": ("recommended", "about", "around", "approximately", "大约", "约", "左右"),
        "open_ended": ("open ended", "数量不限", "越多越好"),
    },
    "time_budget": {
        "fast": ("fast", "尽快", "快速"),
        "multi_round": ("multi-round", "multi round", "多轮"),
    },
    "on_safety_ceiling": {
        "ask": ("撞顶询问", "到上限问我"),
        "auto_continue_within_safety": ("安全范围自动继续",),
        "stop": ("撞顶停止", "到上限停止"),
    },
    "instrument_preference": {
        "none": ("不限仪器", "无仪器偏好"),
        "newer": ("newer", "新仪器", "较新仪器"),
        "classic": ("classic", "经典仪器", "老仪器"),
        "newer_with_legacy_floor": ("legacy floor", "保留经典仪器", "新旧兼顾"),
    },
    "repository": {
        "pride": ("pride",),
        "massive": ("massive",),
        "iprox": ("iprox",),
        "auto": ("自动选仓库", "多仓库", "all repositories"),
    },
}


def _filter_discovery_unaccepted_recommendations(
    patch: Mapping[str, Any],
    *,
    user_message: str,
    selected_decision: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    """Fail closed when an SDK tool tries to write its own recommended defaults.

    The Agent remains responsible for semantic interpretation. This boundary
    only requires surface evidence for schema fields that commonly carry
    defaults; free-form scientific constraints continue through the generic
    validated patch contract.
    """

    text = user_message.casefold().replace("_", " ")
    if re.search(
        r"按.{0,8}(推荐|建议|默认)|采用.{0,8}(上述|这套|方案)|use .{0,8}(recommended |suggested )?defaults?",
        text,
        flags=re.IGNORECASE,
    ):
        return dict(patch), []

    selected_fields: set[str] = set()
    if isinstance(selected_decision, Mapping):
        selected_fields.update(
            _normalise_discovery_decision_target_fields(
                selected_decision.get("target_fields"),
                focus=_clean_text(selected_decision.get("focus")),
                option_ids=[
                    _clean_text((selected_decision.get("option") or {}).get("id"))
                ]
                if isinstance(selected_decision.get("option"), Mapping)
                else [],
            )
        )

    free_form_fields = {
        "objective",
        "ptm_types",
        "special_themes",
        "exclude_rules",
        "success_criteria",
        "scientific_constraints",
        "notes",
        "open_risks",
    }
    numeric_tokens = set(re.findall(r"\d+(?:\.\d+)?", text))
    detected_species, _taxa = species_from_text(user_message)
    normalized_detected_species = {
        value.casefold() for value in detected_species
    }

    accepted: dict[str, Any] = {}
    dropped: list[str] = []
    for field, value in patch.items():
        if field in free_form_fields or field in selected_fields:
            accepted[field] = value
            continue
        supported = False
        if field == "species" and isinstance(value, list):
            normalized_values, _ids = normalize_species_values(value)
            supported = bool(normalized_values) and all(
                item.casefold() in normalized_detected_species
                or item.casefold() in text
                for item in normalized_values
            )
        elif field in {"target_project_count", "max_candidate_projects"}:
            supported = str(value) in numeric_tokens
        elif field == "labeling_hard":
            supported = bool(re.search(r"只要|必须|硬限制|hard constraint|required", text))
        elif field == "legacy_floor_ratio":
            supported = bool(numeric_tokens) and bool(
                re.search(r"经典|老仪器|legacy|占比|比例", text)
            )
        else:
            value_text = _clean_text(value).casefold()
            hints = _DISCOVERY_EXPLICIT_ENUM_HINTS.get(field, {}).get(value_text, ())
            supported = bool(value_text) and (
                value_text.replace("_", " ") in text
                or any(hint.casefold() in text for hint in hints)
            )
        if supported:
            accepted[field] = value
        else:
            dropped.append(field)
    return accepted, dropped

_DISCOVERY_STRATEGY_ENUM_FIELDS: dict[str, set[str]] = {
    "task_type": {
        "rt_prediction",
        "fragment_intensity_prediction",
        "psm_scoring",
        "denovo",
        "ptm_denovo",
        "chimeric_interpretation",
        "browse_only",
        "other",
    },
    "run_horizon": {
        "plan_only",
        "candidates_only",
        "candidates_reviewed",
        "ai_ready_table",
        "pre_release",
        "full_release",
    },
    "species_policy": {"open", "include_only", "prefer", "exclude"},
    "species_coverage": {"none", "prefer_listed", "broaden"},
    "acquisition_mode": {"dda", "dia", "unknown"},
    "mixed_acquisition_policy": {"reject_mixed", "review_mixed", "allow"},
    "labeling_strategy": {
        "label_free",
        "tmt",
        "itraq",
        "silac",
        "dimethyl",
        "unknown",
        "any",
    },
    "coverage_mode": {"quick", "curated", "balanced", "exhaustive"},
    "quota_flexibility": {"fixed", "recommended", "open_ended"},
    "time_budget": {"fast", "multi_round"},
    "on_safety_ceiling": {"ask", "auto_continue_within_safety", "stop"},
    "instrument_preference": {"none", "newer", "classic", "newer_with_legacy_floor"},
}
_DISCOVERY_STRATEGY_STRING_FIELDS = {
    "objective",
    "notes",
}
_DISCOVERY_STRATEGY_TEXT_LIMITS = {"objective": 120, "notes": 4000}
_DISCOVERY_STRATEGY_ARRAY_MAX_ITEMS = 100
_DISCOVERY_STRATEGY_ARRAY_ITEM_MAX_CHARS = 240
_DISCOVERY_STRATEGY_ARRAY_FIELDS = {
    "species",
    "ptm_types",
    "special_themes",
    "selected_search_terms",
    "exclude_rules",
    "success_criteria",
    "open_risks",
}
_DISCOVERY_STRATEGY_BOOLEAN_FIELDS = {"labeling_hard"}
_DISCOVERY_STRATEGY_INTEGER_LIMITS = {
    # Public strategy safety thresholds. Open-ended mode continues across
    # resumable rounds and therefore never treats these as business caps.
    "target_project_count": 5000,
    "max_candidate_projects": 20000,
}
# ``null`` has one cross-layer meaning for every first-class IntentSpec field:
# reset that field to createEmptyIntent's safe default. Query/runtime extension
# fields deliberately do not accept null; omission keeps their server defaults.
_DISCOVERY_STRATEGY_NULLABLE_FIELDS = _DISCOVERY_STRATEGY_FIRST_CLASS_FIELDS


def _normalise_strategy_species_values(values: list[str]) -> list[str]:
    """Canonicalize structured taxa using exact aliases only.

    Natural-language species detection intentionally supports fuzzy matching,
    but this function receives an already structured strategy array.  Applying
    substring matching here can turn ``non-human primate`` into ``human`` or
    collapse ``human and mouse`` to a single taxon, reversing user intent.
    Unknown structured values therefore survive verbatim for the discovery
    Agent to resolve with evidence later.
    """

    exact_terms: dict[str, str] = {}
    for term in SPECIES_TERMS:
        aliases = {
            term.canonical,
            term.scientific_name,
            term.taxon_id,
            *term.aliases,
        }
        for alias in aliases:
            exact_terms[_clean_text(alias).casefold()] = term.canonical

    normalized = [
        exact_terms.get(_clean_text(item).casefold(), _clean_text(item))
        for item in values
        if _clean_text(item)
    ]
    cleaned = list(
        dict.fromkeys(_clean_text(item) for item in normalized if _clean_text(item))
    )
    folded = {item.casefold() for item in cleaned}

    def _group_key(item: str) -> str:
        value = item.casefold()
        candidates: list[str] = []
        if value.endswith("ies") and len(value) > 4:
            candidates.append(value[:-3] + "y")
        if value.endswith("es") and len(value) > 3:
            candidates.append(value[:-2])
        if value.endswith("s") and len(value) > 3:
            candidates.append(value[:-1])
        return next((candidate for candidate in candidates if candidate in folded), value)

    grouped: dict[str, list[tuple[int, str]]] = {}
    for index, item in enumerate(cleaned):
        grouped.setdefault(_group_key(item), []).append((index, item))
    representatives = [
        min(items, key=lambda pair: (len(pair[1]), pair[0]))
        for items in grouped.values()
    ]
    return [item for _index, item in sorted(representatives)]


def _discovery_agent_guidance() -> str:
    """Load repository guidance when packaged, otherwise keep D1 usable."""
    try:
        guidance = _DISCOVERY_AGENT_GUIDANCE_PATH.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        guidance = ""
    return guidance or _DISCOVERY_AGENT_GUIDANCE_FALLBACK.strip()


def _json_object(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    try:
        normalized = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError):
        return {}
    return normalized if isinstance(normalized, dict) else {}


def _validate_discovery_strategy_patch(
    raw_patch: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Validate one generic strategy delta at the model boundary.

    Unknown scientific constraints are preserved as reviewable notes. A known
    field with an invalid shape/value is different: it invalidates the complete
    tool event so a malformed model envelope can never produce a partial write.
    """
    patch: dict[str, Any] = {}
    errors: list[str] = []
    unknown: list[tuple[str, Any]] = []
    repository_risks: list[str] = []

    def _store(key: str, value: Any, raw_key: str) -> None:
        if key in patch and patch[key] != value:
            errors.append(f"conflicting values for {key} (including alias {raw_key})")
            return
        patch[key] = value

    for raw_key_value, value in raw_patch.items():
        raw_key = str(raw_key_value)
        key = _DISCOVERY_STRATEGY_PATCH_ALIASES.get(raw_key, raw_key)
        if key not in _DISCOVERY_STRATEGY_PATCH_FIELDS:
            # Execution transport fields are not scientific dimensions. The
            # strict SDK tool cannot emit them; direct/fallback adapters ignore
            # them rather than polluting the live strategy card.
            if raw_key in _DISCOVERY_STRATEGY_RESERVED_RUNTIME_FIELDS:
                continue
            try:
                rendered = json.dumps(
                    value,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
            except (TypeError, ValueError):
                errors.append(f"{raw_key} is not JSON serializable")
            else:
                unknown.append((raw_key, value))
            continue

        if key == "run_horizon":
            # Keep accepting the legacy wire field, but normalize every model
            # proposal (including null and old enum values) to the fixed task
            # contract. This prevents a dialogue turn from reintroducing Q2.
            _store(key, _FIXED_DISCOVERY_RUN_HORIZON, raw_key)
            continue

        if value is None:
            if key in _DISCOVERY_STRATEGY_NULLABLE_FIELDS:
                _store(key, None, raw_key)
            else:
                errors.append(f"{key} does not allow null")
            continue

        if key in _DISCOVERY_STRATEGY_ENUM_FIELDS:
            if not isinstance(value, str):
                errors.append(f"{key} must be a string enum")
                continue
            normalized = value.strip().lower()
            if normalized not in _DISCOVERY_STRATEGY_ENUM_FIELDS[key]:
                errors.append(f"{key} has unsupported value {value!r}")
                continue
            _store(key, normalized, raw_key)
            continue

        if key in _DISCOVERY_STRATEGY_STRING_FIELDS:
            if not isinstance(value, str):
                errors.append(f"{key} must be a string")
                continue
            normalized = value.strip()
            if not normalized:
                errors.append(f"{key} must not be empty")
                continue
            text_limit = _DISCOVERY_STRATEGY_TEXT_LIMITS[key]
            if len(normalized) > text_limit:
                errors.append(f"{key} must be at most {text_limit} characters")
                continue
            _store(key, normalized, raw_key)
            continue

        if key == "scientific_constraints":
            if not isinstance(value, list):
                errors.append("scientific_constraints must be an array")
                continue
            if len(value) > _DISCOVERY_STRATEGY_ARRAY_MAX_ITEMS:
                errors.append(
                    f"scientific_constraints must contain at most "
                    f"{_DISCOVERY_STRATEGY_ARRAY_MAX_ITEMS} items"
                )
                continue
            normalize_result = normalize_scientific_constraints_result(
                list(value),
                max_items=_DISCOVERY_STRATEGY_ARRAY_MAX_ITEMS,
            )
            if normalize_result.rejected:
                # Atomic fail-closed: one invalid item invalidates the whole write.
                errors.append("scientific_constraints contains an invalid constraint")
                continue
            constraints = [item.model_dump(mode="json") for item in normalize_result.accepted]
            _store(key, constraints, raw_key)
            continue

        if key == "species":
            if not isinstance(value, list):
                errors.append("species must be an array of strings")
                continue
            if len(value) > _DISCOVERY_STRATEGY_ARRAY_MAX_ITEMS:
                errors.append(
                    f"species must contain at most {_DISCOVERY_STRATEGY_ARRAY_MAX_ITEMS} items"
                )
                continue
            if any(
                not isinstance(item, str)
                or not item.strip()
                or len(item.strip()) > _DISCOVERY_STRATEGY_ARRAY_ITEM_MAX_CHARS
                for item in value
            ):
                errors.append(
                    "species must contain only non-empty strings of at most "
                    f"{_DISCOVERY_STRATEGY_ARRAY_ITEM_MAX_CHARS} characters"
                )
                continue
            _store(
                key,
                _normalise_strategy_species_values([item.strip() for item in value]),
                raw_key,
            )
            continue

        if key in _DISCOVERY_STRATEGY_ARRAY_FIELDS:
            if not isinstance(value, list):
                errors.append(f"{key} must be an array of strings")
                continue
            if len(value) > _DISCOVERY_STRATEGY_ARRAY_MAX_ITEMS:
                errors.append(
                    f"{key} must contain at most {_DISCOVERY_STRATEGY_ARRAY_MAX_ITEMS} items"
                )
                continue
            normalized_items: list[str] = []
            invalid_item = False
            for item in value:
                if (
                    not isinstance(item, str)
                    or not item.strip()
                    or len(item.strip()) > _DISCOVERY_STRATEGY_ARRAY_ITEM_MAX_CHARS
                ):
                    invalid_item = True
                    break
                cleaned = item.strip()
                if cleaned not in normalized_items:
                    normalized_items.append(cleaned)
            if invalid_item:
                errors.append(
                    f"{key} must contain only non-empty strings of at most "
                    f"{_DISCOVERY_STRATEGY_ARRAY_ITEM_MAX_CHARS} characters"
                )
                continue
            _store(key, normalized_items, raw_key)
            continue

        if key in _DISCOVERY_STRATEGY_BOOLEAN_FIELDS:
            if type(value) is not bool:
                errors.append(f"{key} must be boolean")
                continue
            _store(key, value, raw_key)
            continue

        if key in _DISCOVERY_STRATEGY_INTEGER_LIMITS:
            limit = _DISCOVERY_STRATEGY_INTEGER_LIMITS[key]
            if type(value) is not int or not 1 <= value <= limit:
                errors.append(f"{key} must be an integer from 1 to {limit}")
                continue
            _store(key, value, raw_key)
            continue

        if key == "legacy_floor_ratio":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append("legacy_floor_ratio must be a number from 0 to 1")
                continue
            normalized_ratio = float(value)
            if not math.isfinite(normalized_ratio) or not 0 <= normalized_ratio <= 1:
                errors.append("legacy_floor_ratio must be a number from 0 to 1")
                continue
            _store(key, normalized_ratio, raw_key)
            continue

        if key == "constraint_provenance":
            if not isinstance(value, Mapping):
                errors.append("constraint_provenance must be an object of strings")
                continue
            normalized_provenance: dict[str, str] = {}
            invalid_entry = False
            for item_key, item_value in value.items():
                if not isinstance(item_key, str) or not isinstance(item_value, str):
                    invalid_entry = True
                    break
                clean_key = item_key.strip()
                clean_value = item_value.strip()
                if not clean_key or not clean_value:
                    invalid_entry = True
                    break
                normalized_provenance[clean_key] = clean_value
            if invalid_entry:
                errors.append("constraint_provenance must contain non-empty string pairs")
                continue
            _store(key, normalized_provenance, raw_key)
            continue

        if key == "repository":
            if not isinstance(value, str) or not value.strip():
                errors.append("repository must be a non-empty string")
                continue
            requested_repository = value.strip()
            repository = _clean_repository(requested_repository, default="")
            if repository:
                _store(key, repository, raw_key)
            else:
                repository_risks.append(
                    f"Unsupported repository requires review: {requested_repository}"
                )
            continue

        # Every canonical field must have a validator above. This branch is a
        # schema-maintenance failure and therefore fails closed.
        errors.append(f"{key} has no strategy validator")

    if unknown:
        existing_constraints = patch.get("scientific_constraints")
        if not isinstance(existing_constraints, list):
            existing_constraints = []
        generated: list[dict[str, Any]] = []
        for index, (raw_key, raw_value) in enumerate(unknown, start=1):
            generated.append(
                ScientificConstraint(
                    id=f"unmapped.{index}.{constraint_slug(raw_key)}"[:96],
                    label=f"未映射约束：{raw_key}",
                    dimension=raw_key[:120],
                    operator="matches",
                    value=raw_value,
                    strength="soft",
                    scope="project",
                    evidence_required=True,
                    rationale=(
                        "Preserved losslessly because the strategy schema has no dedicated field."
                    ),
                    source="user",
                ).model_dump(mode="json")
            )
        patch["scientific_constraints"] = [*existing_constraints, *generated]
    if repository_risks:
        existing_risks = patch.get("open_risks")
        if not isinstance(existing_risks, list):
            existing_risks = []
        patch["open_risks"] = list(
            dict.fromkeys([*existing_risks, *repository_risks])
        )
    notes = patch.get("notes")
    if isinstance(notes, str) and len(notes) > _DISCOVERY_STRATEGY_TEXT_LIMITS["notes"]:
        errors.append(
            f"notes must be at most {_DISCOVERY_STRATEGY_TEXT_LIMITS['notes']} characters"
        )
    risks = patch.get("open_risks")
    if isinstance(risks, list) and (
        len(risks) > _DISCOVERY_STRATEGY_ARRAY_MAX_ITEMS
        or any(
            not isinstance(item, str)
            or not item
            or len(item) > _DISCOVERY_STRATEGY_ARRAY_ITEM_MAX_CHARS
            for item in risks
        )
    ):
        errors.append("generated open_risks exceed the public string-array limits")
    return patch, errors


def _normalise_discovery_strategy_patch(raw_patch: Mapping[str, Any]) -> dict[str, Any]:
    patch, errors = _validate_discovery_strategy_patch(raw_patch)
    return {} if errors else patch


def _drop_unchanged_discovery_patch_fields(
    patch: Mapping[str, Any],
    intent_snapshot: Mapping[str, Any],
    *,
    preserve_fields: set[str] | None = None,
) -> dict[str, Any]:
    """Keep true value deltas plus explicitly authorized resolution deltas.

    A selected Agent option may intentionally resolve a field to the value that
    already represents the empty/open default.  ``preserve_fields`` is supplied
    only after a real SDK ``update_strategy`` call explicitly names an active
    decision target, so those keys carry new decision state even when their
    values are unchanged.  Ordinary model echoes are still removed.
    """

    preserved = preserve_fields or set()
    return {
        key: value
        for key, value in patch.items()
        if (
            key in preserved
            or value is None
            or key not in intent_snapshot
            or intent_snapshot.get(key) != value
        )
    }


def _discovery_turn_commitment_patch(
    raw: Mapping[str, Any],
    *,
    user_message: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Decode the model's clause-level self-audit with source grounding.

    ``None`` means an older response omitted the additive audit contract. An
    object (including ``{}``) means the model supplied it. The audit never
    authorizes mutation by itself: callers still require one explicit
    ``update_strategy`` tool event. It only makes that event complete and
    prevents tool fields that the model could not ground in the latest turn.
    """

    interpretation = raw.get("turn_interpretation")
    if not isinstance(interpretation, Mapping):
        return None, []
    commitments = interpretation.get("commitments")
    if not isinstance(commitments, list):
        return None, ["turn_interpretation.commitments must be an array"]
    if len(commitments) > 50:
        return None, ["turn_interpretation contains too many commitments"]

    # A clause-level decision list is a redundant completeness channel from the
    # same model call. It is grounded by the server-provided clause id, so a
    # smaller model can omit an item from the summary list without losing an
    # explicit user choice.
    audit_items: list[Mapping[str, Any]] = [
        item for item in commitments if isinstance(item, Mapping)
    ]
    clause_text_by_id = {
        item["id"]: item["text"]
        for item in _discovery_latest_message_clauses(user_message)
    }
    clause_audit = interpretation.get("clause_audit")
    if isinstance(clause_audit, list):
        for clause in clause_audit[:30]:
            if not isinstance(clause, Mapping):
                continue
            clause_id = _clean_text(clause.get("clause_id"))
            clause_text = clause_text_by_id.get(clause_id)
            decisions = clause.get("decisions")
            if not clause_text or not isinstance(decisions, list):
                continue
            for decision in decisions[:20]:
                if not isinstance(decision, Mapping):
                    continue
                audit_items.append(
                    {
                        "field": decision.get("field"),
                        "value": decision.get("value"),
                        "source": clause_text,
                    }
                )

    # The clause audit is additive.  Some OpenAI-compatible providers omit the
    # nested response_json audit even though they executed the typed
    # ``update_strategy`` function correctly.  Treating an explicitly empty
    # audit as an authoritative empty patch made every such tool call collapse
    # into "no changed fields".  In that compatibility case the real SDK tool
    # call remains the mutation authority and the normal schema validator below
    # still applies to it.
    if not audit_items:
        return None, []

    message_evidence = re.sub(r"\s+", "", user_message).casefold()
    raw_patch: dict[str, Any] = {}
    for index, item in enumerate(audit_items):
        raw_field = _clean_text(item.get("field"))
        source = _clean_text(item.get("source"))
        if not raw_field or "value" not in item or not source:
            # This is an optional redundant audit channel.  One malformed item
            # must not veto a valid SDK function call or the other grounded
            # commitments from the same turn.
            continue
        source_evidence = re.sub(r"\s+", "", source).casefold()
        if not source_evidence or source_evidence not in message_evidence:
            # The audit is a per-field authorization filter. A model may echo a
            # snapshot-derived recommendation alongside genuine commitments;
            # ignore that ungrounded field without sacrificing other grounded
            # clauses from the same explicit tool event.
            continue
        canonical_field = _DISCOVERY_STRATEGY_PATCH_ALIASES.get(raw_field, raw_field)
        value = item.get("value")
        if canonical_field in raw_patch and raw_patch[canonical_field] != value:
            # Keep the first grounded interpretation; the explicit typed tool
            # value resolves overlaps later.
            continue
        candidate, candidate_errors = _validate_discovery_strategy_patch(
            {canonical_field: value}
        )
        if candidate_errors:
            continue
        for candidate_field, candidate_value in candidate.items():
            if candidate_field not in raw_patch:
                raw_patch[candidate_field] = candidate_value
    if not raw_patch:
        # The model attempted an audit but grounded none of its claims in the
        # latest message.  This is materially different from an omitted/empty
        # compatibility audit and must fail the proposed mutation closed.
        return {}, []
    patch, validation_errors = _validate_discovery_strategy_patch(raw_patch)
    if validation_errors:
        return None, []
    return patch, []


def _discovery_turn_patch(
    raw: Mapping[str, Any],
    *,
    user_message: str,
    intent_snapshot: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Decode the sole authorized mutation path: action + typed tool patch.

    Legacy ``extra_fields`` and lexical parsing are intentionally not inputs.
    They remain response projections only, so a missing/malformed D1 envelope
    cannot mutate the strategy card.
    """
    if _clean_text(raw.get("action") or raw.get("mode")).lower() != "update_strategy":
        return {}, []
    raw_calls = raw.get("tool_calls")
    if not isinstance(raw_calls, list):
        return {}, ["update_strategy requires a tool_calls array"]
    update_calls = [
        call
        for call in raw_calls
        if isinstance(call, Mapping)
        and _clean_text(call.get("name") or call.get("tool")) == "update_strategy"
    ]
    if len(update_calls) != 1:
        return {}, ["update_strategy requires exactly one update_strategy tool call"]
    arguments = update_calls[0].get("arguments")
    if not isinstance(arguments, Mapping) or "patch" not in arguments:
        return {}, ["update_strategy tool arguments require an explicit patch object"]
    raw_patch = arguments.get("patch")
    if not isinstance(raw_patch, Mapping):
        return {}, ["update_strategy patch must be an object"]
    serializable_patch = _json_object(dict(raw_patch))
    if raw_patch and not serializable_patch:
        return {}, ["update_strategy patch must be valid JSON"]
    patch, errors = _validate_discovery_strategy_patch(serializable_patch)
    if errors:
        return {}, errors
    commitment_patch, commitment_errors = _discovery_turn_commitment_patch(
        raw,
        user_message=user_message,
    )
    if commitment_errors:
        return {}, commitment_errors
    if commitment_patch is not None:
        # A non-empty grounded audit is the per-field authorization filter: it
        # fills omissions and drops tool-only defaults.  Free-text paraphrases
        # may legitimately differ between the audit and typed tool, so the
        # typed value wins only for string fields.  Conflicting enum/list/
        # numeric commitments remain unsafe and fail closed.
        conflicts = {
            key
            for key in set(patch).intersection(commitment_patch)
            if patch[key] != commitment_patch[key]
        }
        unsafe_conflicts = conflicts.difference(_DISCOVERY_STRATEGY_STRING_FIELDS)
        if unsafe_conflicts:
            return {}, [
                "update_strategy patch conflicts with grounded commitments: "
                + ", ".join(sorted(unsafe_conflicts))
            ]
        reconciled_patch = dict(commitment_patch)
        for key in conflicts:
            reconciled_patch[key] = patch[key]
        patch = reconciled_patch
    patch = _drop_unchanged_discovery_patch_fields(patch, intent_snapshot)
    if not patch:
        return {}, ["update_strategy patch contains no changed fields"]
    return patch, []


def _discovery_explicit_tool_patch(
    raw: Mapping[str, Any],
    *,
    intent_snapshot: Mapping[str, Any],
    include_unchanged: bool = False,
) -> dict[str, Any]:
    """Best-effort canonical view of exactly what the raw tool patch claimed."""

    calls = raw.get("tool_calls")
    if not isinstance(calls, list):
        return {}
    updates = [
        call
        for call in calls
        if isinstance(call, Mapping)
        and _clean_text(call.get("name") or call.get("tool")) == "update_strategy"
    ]
    if len(updates) != 1:
        return {}
    arguments = updates[0].get("arguments")
    if not isinstance(arguments, Mapping):
        return {}
    candidate = arguments.get("patch")
    if not isinstance(candidate, Mapping):
        return {}
    patch = _normalise_discovery_strategy_patch(_json_object(dict(candidate)))
    if include_unchanged:
        return patch
    return _drop_unchanged_discovery_patch_fields(patch, intent_snapshot)


def _drop_uncommitted_discovery_null_placeholders(
    patch: Mapping[str, Any],
    commitment_patch: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Separate provider-expanded optional nulls from explicit clear actions.

    Some OpenAI-compatible tool callers serialize every omitted optional field
    as ``null``. A null is a real clear operation only when the model's
    independent turn interpretation also records that field as a commitment.
    With no interpretation, one isolated null remains backward-compatible; a
    mass of null optional fields fails closed instead of wiping the live card.
    """

    candidate = dict(patch)
    null_fields = {key for key, value in candidate.items() if value is None}
    if not null_fields:
        return candidate
    committed_nulls = {
        key
        for key, value in (commitment_patch or {}).items()
        if value is None
    }
    if commitment_patch is None and len(null_fields) == 1:
        return candidate
    return {
        key: value
        for key, value in candidate.items()
        if value is not None or key in committed_nulls
    }


def _format_discovery_strategy_patch_summary(patch: Mapping[str, Any]) -> str:
    """Render applied strategy fields for user-facing Chinese copy."""

    rendered: list[str] = []
    priority = [
        "objective",
        "task_type",
        "run_horizon",
        "species",
        "acquisition_mode",
        "special_themes",
        "target_project_count",
        "instrument_preference",
        "exclude_rules",
        "scientific_constraints",
    ]
    ordered_keys = [key for key in priority if key in patch]
    ordered_keys.extend(key for key in patch if key not in ordered_keys)
    value_labels = {
        ("task_type", "browse_only"): "先浏览探索",
        ("run_horizon", "candidates_only"): "找到候选即停",
        ("run_horizon", "candidates_reviewed"): "找到并审查候选",
        ("instrument_preference", "newer"): "新仪器优先",
        ("instrument_preference", "classic"): "经典仪器优先",
        ("acquisition_mode", "dda"): "DDA",
        ("acquisition_mode", "dia"): "DIA",
    }
    for key in ordered_keys:
        value = patch[key]
        label = _DISCOVERY_STRATEGY_FIELD_LABELS_ZH.get(key, key)
        if value is None or value == [] or value == "":
            value_text = "已清空"
        elif isinstance(value, list):
            list_values: list[str] = []
            for item in value:
                if isinstance(item, Mapping):
                    list_values.append(
                        _clean_text(item.get("label") or item.get("id"))
                        or "结构化约束"
                    )
                else:
                    list_values.append(str(item))
            value_text = "、".join(list_values)
        elif isinstance(value, Mapping):
            value_text = _clean_text(value.get("label") or value.get("id")) or "结构化设置"
        elif isinstance(value, bool):
            value_text = "是" if value else "否"
        else:
            value_text = value_labels.get((key, str(value)), str(value))
        rendered.append(f"{label}={value_text}")
    visible = rendered[:10]
    remainder = len(rendered) - len(visible)
    summary = "；".join(visible) or "无"
    if remainder > 0:
        summary += f"；另有 {remainder} 项结构化设置"
    return summary


def _format_discovery_field_names_zh(fields: Any) -> str:
    """Join strategy field ids into Chinese labels for residual-missing copy."""

    names: list[str] = []
    if isinstance(fields, (list, tuple, set)):
        for field in fields:
            key = _clean_text(field)
            if not key:
                continue
            label = _DISCOVERY_STRATEGY_FIELD_LABELS_ZH.get(key, key)
            if label not in names:
                names.append(label)
    return "、".join(names)


def _format_discovery_reconciled_update_message(patch: Mapping[str, Any]) -> str:
    """Tell the user the card truth when model prose and validated delta diverge."""

    summary = _format_discovery_strategy_patch_summary(patch)
    return (
        f"已按你这轮明确的要求更新策略：{summary}。"
        "没有列出的现有设置保持不变；其它科学建议只作为讨论，不会悄悄写入策略。"
    )


def _discovery_compound_commitment_hints(user_message: str) -> dict[str, Any]:
    """Deterministic, generic extractions for packed multi-commitment turns.

    Only fills fields clearly supported by the latest message. Does not invent
    modeling tasks (e.g. RT) unless the text states them. Used to complete an
    under-specified Manager tool patch so local soft-reject cannot blank
    species / acquisition / scale / horizon that the user already said.
    """

    text = _clean_text(user_message)
    if not text:
        return {}
    lower = text.casefold()
    hints: dict[str, Any] = {}
    recap_labels = (
        "科学目标",
        "研究主题",
        "物种",
        "采集模式",
        "下游任务",
        "交付终点",
        "规模",
        "标记方式",
    )
    structured_recap = sum(label in text for label in recap_labels) >= 4
    if structured_recap:
        objective_match = re.search(
            r"科学目标\s*[：:]\s*(.+?)(?=研究主题\s*[：:]|物种\s*[：:]|采集模式\s*[：:]|下游任务\s*[：:]|交付终点\s*[：:]|规模\s*[：:]|标记方式\s*[：:]|$)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if objective_match and _clean_text(objective_match.group(1)):
            hints["objective"] = _clean_text(objective_match.group(1))[:120]

    # Species (generic bilingual cues)
    if re.search(
        r"人源|人类|智人|\bhuman\b|homo\s*sapiens",
        text,
        flags=re.IGNORECASE,
    ):
        hints["species"] = ["human"]
        if re.search(r"仅|只要|限定|strict|only", text, flags=re.IGNORECASE):
            hints["species_policy"] = "include_only"
        else:
            hints["species_policy"] = "prefer"
            hints["species_coverage"] = "prefer_listed"
    elif re.search(r"小鼠|老鼠|\bmouse\b|mus\s*musculus", text, flags=re.IGNORECASE):
        hints["species"] = ["mouse"]
        hints["species_policy"] = "prefer"

    # Downstream task — only when explicit
    if re.search(
        r"\brt\b|保留时间|retention\s*time|rt\s*预测|rt预测",
        text,
        flags=re.IGNORECASE,
    ):
        hints["task_type"] = "rt_prediction"
    elif re.search(r"denovo|de\s*novo|从头测序", text, flags=re.IGNORECASE):
        hints["task_type"] = "denovo"
    elif re.search(r"psm\s*评分|psm\s*score", text, flags=re.IGNORECASE):
        hints["task_type"] = "psm_scoring"
    elif re.search(r"碎片强度|fragment\s*intensity", text, flags=re.IGNORECASE):
        hints["task_type"] = "fragment_intensity"
    elif structured_recap and re.search(
        r"下游任务\s*[：:].{0,40}(?:browse_only|纯浏览|浏览探索)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        hints["task_type"] = "browse_only"

    # Scale / quota
    fixed_count_match = re.search(
        r"(?:只要|需要|目标|找够|达到|搜|找)\s*(\d{1,5})\s*个"
        r"(?:\s*(?:可用)?项目)?"
        r"|(\d{1,5})\s*个\s*(?:可用)?项目"
        r"|^\s*(\d{1,5})\s*个\s*$",
        text,
        flags=re.IGNORECASE,
    )
    if fixed_count_match:
        count_text = next(
            group for group in fixed_count_match.groups() if group is not None
        )
        hints["coverage_mode"] = "quick"
        hints["quota_flexibility"] = "fixed"
        hints["target_project_count"] = min(5000, max(1, int(count_text)))
    elif re.search(
        r"越多越好|尽可能多|尽量多|搜全|覆盖全|不设上限|无上限|全量|"
        r"(?:所有|全部).{0,24}(?:数据|项目|文件|候选)|"
        r"(?:数据|项目|文件|候选).{0,16}(?:所有|全部)|"
        r"exhaustive|as\s*many|all\s+(?:relevant\s+)?(?:data|datasets|projects|files)|"
        r"open[-\s]?ended|不限(数量|规模)?",
        text,
        flags=re.IGNORECASE,
    ):
        hints["quota_flexibility"] = "open_ended"
        hints["coverage_mode"] = "exhaustive"
        hints["target_project_count"] = None
    elif re.search(r"快速|quick|精选|curated", text, flags=re.IGNORECASE):
        hints["coverage_mode"] = "quick"
        hints["quota_flexibility"] = "fixed"
        hints["target_project_count"] = 20

    # Acquisition
    if re.search(r"\bdda\b|data[-\s]?dependent|仅\s*dda|只要\s*dda", text, flags=re.IGNORECASE):
        hints["acquisition_mode"] = "dda"
        hints["mixed_acquisition_policy"] = "reject_mixed"
    elif re.search(r"\bdia\b|data[-\s]?independent", text, flags=re.IGNORECASE):
        hints["acquisition_mode"] = "dia"

    # Run horizon
    if re.search(
        r"审查候选|候选\+?审查|candidates?_reviewed|找到并审查",
        text,
        flags=re.IGNORECASE,
    ):
        hints["run_horizon"] = "candidates_reviewed"
    elif re.search(r"仅候选|只要候选|candidates?_only", text, flags=re.IGNORECASE):
        hints["run_horizon"] = "candidates_only"
    elif re.search(r"只做计划|仅规划|plan_only", text, flags=re.IGNORECASE):
        hints["run_horizon"] = "plan_only"

    # Domain theme (generic immunopeptide / HLA cues — not a full ontology)
    if re.search(
        r"免疫肽|免疫肽组|hla|mhc|ligandome|immunopeptid",
        text,
        flags=re.IGNORECASE,
    ):
        hints["special_themes"] = ["immunopeptidomics"]
        if "objective" not in hints and len(text) <= 80:
            hints["objective"] = text.strip()[:200]

    if structured_recap and re.search(
        r"标记方式\s*[：:].{0,30}(?:不限|保持开放|\bany\b)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        hints["labeling_strategy"] = "any"
        hints["labeling_hard"] = False

    return hints


def _merge_discovery_compound_commitment_hints(
    patch: Mapping[str, Any] | None,
    user_message: str,
    *,
    intent_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Union Manager patch with deterministic compound hints (hints fill gaps only)."""

    base = dict(patch) if isinstance(patch, Mapping) else {}
    hints = _discovery_compound_commitment_hints(user_message)
    if not hints:
        return base
    # Only help packed multi-commitment turns when the Manager under-wrote
    # structural fields (soft-only dumps: objective/themes/task alone).
    # Never expand an already multi-structural Manager patch — that pollutes
    # critic/partial-grounding paths and can incorrectly flip skip.
    structural_hint_keys = {
        "species",
        "task_type",
        "acquisition_mode",
        "run_horizon",
        "quota_flexibility",
        "coverage_mode",
        "target_project_count",
    }
    structural_hits = len(structural_hint_keys.intersection(hints))
    base_structural = len(structural_hint_keys.intersection(base))
    recap_labels = (
        "科学目标",
        "研究主题",
        "物种",
        "采集模式",
        "下游任务",
        "交付终点",
        "规模",
        "标记方式",
    )
    structured_recap = sum(label in user_message for label in recap_labels) >= 4
    if base_structural >= 2 and not structured_recap:
        # Manager already committed multiple structural fields.
        return base
    if len(base) < 2 and structural_hits < 2:
        return base
    if len(base) < 2 and structural_hits < 3 and len(hints) < 4:
        # Need a clearly packed sentence (e.g. species+task+acquisition).
        return base
    # Soft-only / thin Manager dump: require a clearly packed user sentence.
    if base_structural < 2 and structural_hits < 3 and len(hints) < 4:
        return base
    merged = dict(base)
    snapshot = intent_snapshot if isinstance(intent_snapshot, Mapping) else {}
    for field, value in hints.items():
        if field in merged:
            continue
        # Keep skip-path pure: never inject keys outside the low-risk whitelist.
        if field not in _DISCOVERY_LOW_RISK_SINGLE_FIELD_VERIFIER_SKIP:
            continue
        # Do not overwrite an intentional empty clear already on the card via patch
        if field in snapshot and field not in base and value is None:
            continue
        merged[field] = value
    if merged == base:
        return base
    normalized, errors = _validate_discovery_strategy_patch(merged)
    if errors or not normalized:
        return base
    # Preserve non-whitelist keys from the original Manager patch so they still
    # force the semantic critic (e.g. ptm_types / scientific_constraints).
    manager_hard = {
        field: value
        for field, value in base.items()
        if field not in _DISCOVERY_LOW_RISK_SINGLE_FIELD_VERIFIER_SKIP
    }
    if manager_hard:
        out = dict(normalized)
        out.update(manager_hard)
        revalidated, re_errors = _validate_discovery_strategy_patch(out)
        return revalidated if not re_errors and revalidated else base
    # Pure low-risk compound: keep only whitelist keys for a clean skip path.
    return {
        field: value
        for field, value in normalized.items()
        if field in _DISCOVERY_LOW_RISK_SINGLE_FIELD_VERIFIER_SKIP
    }


def _discovery_low_risk_single_field_verifier_skip(
    patch: Mapping[str, Any],
    *,
    tool_interpretation_difference: bool,
    provider_compatibility_recovery: Mapping[str, Any] | None = None,
) -> bool:
    """True when a typed tool patch may skip the second verifier.

    Supports one field or a small compound dump of first-class low-risk fields
    (natural-language multi-commitment turns). scientific_constraints and
    provider plain-text recovery still require the critic. A single-field
    tool/interpretation gap still forces the critic; multi-field pure-whitelist
    compounds may skip even when the reconciled patch differs from the tool
    dump so soft-reject cannot blank hard whitelist fields (species, DDA,
    horizon, quota).
    """

    recovery = (
        provider_compatibility_recovery
        if isinstance(provider_compatibility_recovery, Mapping)
        else {}
    )
    if not isinstance(patch, Mapping) or not patch:
        return False
    keys = set(patch)
    if "scientific_constraints" in keys:
        return False
    if not keys.issubset(_DISCOVERY_LOW_RISK_SINGLE_FIELD_VERIFIER_SKIP):
        return False
    if len(keys) > _DISCOVERY_LOW_RISK_COMPOUND_MAX_FIELDS:
        return False
    if tool_interpretation_difference and len(keys) < 2:
        # Single-field interpretation gaps still need the critic.
        return False
    if (
        _clean_text(recovery.get("mode"))
        == "json_action_contract_after_plain_text"
    ):
        return False
    return True


def _discovery_soft_reject_critic_rejected_fields(
    semantic_verification: Mapping[str, Any] | None,
) -> set[str]:
    """Field ids the critic explicitly failed, vetoed, or left ungrounded.

    An empty set means a global reject with no field-level signal (soft-only keep).
    """

    if not isinstance(semantic_verification, Mapping):
        return set()
    rejected: set[str] = set()

    def _add_field(raw: Any) -> None:
        key = _clean_text(raw)
        if not key:
            return
        field_root = re.split(r"[.\\[]", key, maxsplit=1)[0]
        field = _DISCOVERY_STRATEGY_PATCH_ALIASES.get(field_root, field_root)
        if field:
            rejected.add(field)

    for key in (
        "missing_fields",
        "rejected_fields",
        "ungrounded_fields",
        "field_errors",
        "critic_rejected_fields",
        "removed_uncommitted_fields",
    ):
        raw = semantic_verification.get(key)
        if isinstance(raw, Mapping):
            for field, value in raw.items():
                _add_field(field)
                if isinstance(value, Mapping):
                    _add_field(value.get("field"))
                elif isinstance(value, str) and key == "field_errors":
                    # Mapping of field -> reason.
                    continue
        elif isinstance(raw, (list, tuple, set)):
            for item in raw:
                if isinstance(item, Mapping):
                    _add_field(item.get("field") or item.get("name"))
                else:
                    _add_field(item)

    raw_evidence = semantic_verification.get("evidence")
    if isinstance(raw_evidence, list):
        for item in raw_evidence[:100]:
            if not isinstance(item, Mapping):
                continue
            status = _clean_text(
                item.get("status") or item.get("verdict") or item.get("result")
            ).casefold()
            if status in {"reject", "rejected", "fail", "failed", "ungrounded", "missing"}:
                _add_field(item.get("field"))
    return rejected


def _discovery_soft_reject_kept_patch(
    explicit_tool_patch: Mapping[str, Any],
    *,
    semantic_verification: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Subset of a primary tool patch retained after verifier reject (soft-reject v2).

    Authority is always the explicit SDK tool patch — never critic-authored values.
    - Field-level critic errors: keep low-risk whitelist tool keys not named by critic.
    - Global reject (no field list): keep only soft theme/context keys.
    """

    if not isinstance(explicit_tool_patch, Mapping) or not explicit_tool_patch:
        return {}
    critic_rejected = _discovery_soft_reject_critic_rejected_fields(
        semantic_verification
    )
    if critic_rejected:
        # Field-level mode: low-risk whitelist minus critic vetoes (hard keys may
        # survive when the critic did not name them).
        eligible = {
            field
            for field in explicit_tool_patch
            if (
                field in _DISCOVERY_LOW_RISK_SINGLE_FIELD_VERIFIER_SKIP
                and field not in critic_rejected
            )
        }
    else:
        # Global reject: soft set only — never auto-keep species/DDA/horizon.
        eligible = {
            field
            for field in explicit_tool_patch
            if field in _DISCOVERY_SOFT_REJECT_KEEP_FIELDS
        }
    # scientific_constraints are never soft-kept; they force critic and fail closed.
    eligible.discard("scientific_constraints")
    candidate = {
        field: value
        for field, value in explicit_tool_patch.items()
        if field in eligible
    }
    if not candidate:
        return {}
    normalized, errors = _validate_discovery_strategy_patch(candidate)
    if errors or not normalized:
        return {}
    return normalized


def _discovery_soft_reject_dropped_fields(
    explicit_tool_patch: Mapping[str, Any],
    kept_patch: Mapping[str, Any],
) -> list[str]:
    """Tool keys that did not survive soft-reject (sorted for stable audits)."""

    if not isinstance(explicit_tool_patch, Mapping) or not explicit_tool_patch:
        return []
    kept = set(kept_patch) if isinstance(kept_patch, Mapping) else set()
    return sorted(set(explicit_tool_patch).difference(kept))


def _format_discovery_soft_reject_message(
    patch: Mapping[str, Any],
    *,
    dropped_fields: Any = None,
) -> str:
    """Chinese copy when verifier rejects some fields but a keep subset remains."""

    summary = _format_discovery_strategy_patch_summary(patch)
    residual = _format_discovery_field_names_zh(dropped_fields)
    if residual:
        return (
            f"已写入可核验部分：{summary}。"
            f"未写入：{residual}（语义校验未通过或字段不在可软保留集合）。"
            "本轮未整包清空策略。"
        )
    return (
        f"已写入可核验部分：{summary}。"
        "其余字段待确认后再写入；本轮未整包清空策略。"
    )


def _format_discovery_partial_grounding_message(
    patch: Mapping[str, Any],
    missing_fields: Any = None,
) -> str:
    """Explain partial semantic apply: what landed vs what remains open."""

    applied = _format_discovery_strategy_patch_summary(patch)
    residual = _format_discovery_field_names_zh(missing_fields)
    if residual:
        return (
            f"已根据可核验部分更新策略：{applied}。"
            f"以下字段未能可靠对齐，未写入本轮策略：{residual}。"
            "你可以补充这些信息，或继续基于已应用部分推进。"
        )
    return (
        f"已根据可核验部分更新策略：{applied}。"
        "未能完全核验的字段未写入；没有列出的现有设置保持不变。"
    )


_DISCOVERY_DECISION_MAX_OPTIONS = 8


def _normalise_discovery_decision_target_fields(
    raw: Any,
    *,
    focus: str,
    option_ids: list[str],
) -> list[str]:
    fields: list[str] = []
    if isinstance(raw, list):
        for value in raw:
            name = _clean_text(value)
            canonical = _DISCOVERY_STRATEGY_PATCH_ALIASES.get(name, name)
            if canonical in _DISCOVERY_STRATEGY_PATCH_FIELDS and canonical not in fields:
                fields.append(canonical)
    if fields:
        return fields

    canonical_focus = _DISCOVERY_STRATEGY_PATCH_ALIASES.get(focus, focus)
    if canonical_focus in _DISCOVERY_STRATEGY_PATCH_FIELDS:
        return [canonical_focus]

    ids = {value.casefold() for value in option_ids if value}
    enum_matches = [
        field
        for field, allowed in _DISCOVERY_STRATEGY_ENUM_FIELDS.items()
        if len(ids.intersection(allowed)) >= 2
    ]
    return enum_matches if len(enum_matches) == 1 else []


_DISCOVERY_CONTRACT_NOISE_MARKERS: tuple[str, ...] = (
    "选项菜单不完整",
    "下一问菜单不完整",
    "下一问结构不完整",
    "动作契约不完整",
    "契约不一致",
    "不把它当作有效提问",
    "系统生成的下一问",
    "策略修改已按工具事件处理",
    "next_decision requires",
    "missing or unsupported D1 action",
    "contract_errors",
)


def _discovery_user_asks_clarification(user_message: str) -> bool:
    """True when the user is asking what the previous system notice meant."""

    text = _clean_text(user_message)
    if not text:
        return False
    markers = (
        "什么意思",
        "啥意思",
        "什么意思啊",
        "这是什么意思",
        "这啥意思",
        "什么意思？",
        "啥意思？",
        "what does that mean",
        "what do you mean",
        "what does this mean",
    )
    folded = text.casefold()
    return any(marker.casefold() in folded for marker in markers)


def _discovery_history_has_contract_noise(
    dialogue_history: list[Mapping[str, Any]] | None,
) -> bool:
    """Detect recent assistant contract/repair copy the user may be asking about."""

    if not dialogue_history:
        return False
    for item in reversed(list(dialogue_history)[-12:]):
        if not isinstance(item, Mapping):
            continue
        role = _clean_text(item.get("role")).lower()
        if role != "assistant":
            continue
        content = _clean_text(item.get("content"))
        if any(marker in content for marker in _DISCOVERY_CONTRACT_NOISE_MARKERS):
            return True
    return False


def _discovery_contract_noise_clarification_hint() -> str:
    """Inject into the Manager user prompt after contract noise + 什么意思."""

    return (
        "server_hint (authoritative for this turn): The user is asking what a previous "
        "system/contract notice meant (e.g. incomplete next_decision menu or action "
        "contract). Explain in plain Chinese that the strategy card may already have "
        "been updated, but a follow-up option menu was dropped as incomplete—not that "
        "their science goal is invalid. Do not echo internal English contract_errors. "
        "If critical_decision_agenda still has unresolved critical items, emit a "
        "schema-complete next_decision for the highest-priority remaining item "
        "(question + recommendation.reason + 2-8 options each with strategy_patch) "
        "or ask one clear free-text question. Prefer action=clarify when presenting "
        "that follow-up; keep action=advise only if no menu is needed.\n\n"
    )


def _format_discovery_incomplete_next_decision_message() -> str:
    """User-facing copy when next_decision fails schema and is not repaired yet."""

    return (
        "刚才准备的下一问选项不完整，我先不把它当作有效提问。"
        "策略若已更新会保留在卡上；请直接用自然语言说明你的选择，或再说一次数据目标。"
    )


def _format_discovery_contract_noise_clarification_message() -> str:
    """Explain prior contract noise when the user asks 什么意思."""

    return (
        "刚才的提示是说：策略可能已写入，但系统生成的下一问选项不完整，所以没有展示菜单。"
        "这不是说你的目标无效。下面补上当前最关键的待确认问题；你也可以直接用自然语言回答。"
    )


def _synthesize_discovery_next_decision_from_agenda(
    remaining_critical: list[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Build a schema-valid next_decision when the model omitted or broke one.

    Keeps grilling alive after a successful update_strategy so users are not
    left with only an internal contract error string.
    """

    if not remaining_critical:
        return None
    item = remaining_critical[0]
    item_id = _clean_text(item.get("id"))
    target_fields = [
        _clean_text(field)
        for field in item.get("target_fields") or item.get("decision_variables") or []
        if _clean_text(field)
    ]
    reason = _clean_text(item.get("reason")) or "该选择会影响搜索范围与科学可用性。"

    templates: dict[str, dict[str, Any]] = {
        "generalization_scope": {
            "focus": "物种范围",
            "question": "这次搜索的物种范围如何定？",
            "options": [
                {
                    "id": "human_only",
                    "label": "只要人源",
                    "reason": "免疫肽/训练场景下人源数据最丰富、注释更全",
                    "strategy_patch": {
                        "species": ["human"],
                        "species_policy": "include_only",
                        "species_coverage": "none",
                    },
                },
                {
                    "id": "human_prefer",
                    "label": "人源优先，其他不排除",
                    "reason": "先覆盖人源，同时保留其它物种以观察分布",
                    "strategy_patch": {
                        "species": ["human"],
                        "species_policy": "prefer",
                        "species_coverage": "prefer_listed",
                    },
                },
                {
                    "id": "species_open",
                    "label": "保持开放，先看数据分布",
                    "reason": "由检索结果再决定是否收紧物种",
                    "strategy_patch": {
                        "species": [],
                        "species_policy": "open",
                        "species_coverage": "broaden",
                    },
                },
            ],
        },
        "acquisition_compatibility": {
            "focus": "采集方式",
            "question": "采集方式如何限制？",
            "options": [
                {
                    "id": "dda_only",
                    "label": "只要 DDA",
                    "reason": "谱图与标签对应更直接，适合多数训练任务",
                    "strategy_patch": {
                        "acquisition_mode": "dda",
                        "mixed_acquisition_policy": "reject_mixed",
                    },
                },
                {
                    "id": "dda_prefer",
                    "label": "DDA 优先，混合项目审查",
                    "reason": "扩大候选池，混合项目单独审查",
                    "strategy_patch": {
                        "acquisition_mode": "dda",
                        "mixed_acquisition_policy": "review_mixed",
                    },
                },
                {
                    "id": "acq_open",
                    "label": "暂不限制",
                    "reason": "先看仓库分布再决定",
                    "strategy_patch": {
                        "acquisition_mode": "unknown",
                        "mixed_acquisition_policy": "review_mixed",
                    },
                },
            ],
        },
        "search_scale": {
            "focus": "搜索规模",
            "question": "选择数量模式：快速找够指定数量，还是尽可能搜全？",
            "options": [
                {
                    "id": "quick_20",
                    "label": "快速找够 20 个可用项目",
                    "reason": "核心词优先分页检索并审查，达到 20 个立即停止",
                    "strategy_patch": {
                        "coverage_mode": "quick",
                        "target_project_count": 20,
                        "quota_flexibility": "fixed",
                    },
                },
                {
                    "id": "open_ended",
                    "label": "尽可能搜全（不设候选池上限）",
                    "reason": "按顺序拉完全部确认检索词并审查",
                    "strategy_patch": {
                        "coverage_mode": "exhaustive",
                        "target_project_count": None,
                        "quota_flexibility": "open_ended",
                    },
                },
            ],
        },
        "delivery_horizon": {
            "focus": "本次终点",
            "question": "这次运行做到哪一步？",
            "options": [
                {
                    "id": "candidates_reviewed",
                    "label": "候选项目 + 审查",
                    "reason": "可审计后再进入下游",
                    "strategy_patch": {"run_horizon": "candidates_reviewed"},
                },
                {
                    "id": "candidates_only",
                    "label": "仅候选列表",
                    "reason": "先看有哪些项目",
                    "strategy_patch": {"run_horizon": "candidates_only"},
                },
                {
                    "id": "plan_only",
                    "label": "仅规划",
                    "reason": "不访问仓库，只定策略",
                    "strategy_patch": {"run_horizon": "plan_only"},
                },
            ],
        },
        "downstream_task": {
            "focus": "下游任务",
            "question": "这些数据主要准备用于什么分析？",
            "options": [
                {
                    "id": "browse_only",
                    "label": "浏览探索",
                    "reason": "先摸清仓库分布",
                    "strategy_patch": {"task_type": "browse_only"},
                },
                {
                    "id": "denovo",
                    "label": "从头测序",
                    "reason": "需要高质量 MS/MS",
                    "strategy_patch": {"task_type": "denovo"},
                },
                {
                    "id": "rt_prediction",
                    "label": "保留时间预测",
                    "reason": "需要 RT 与肽段配对线索",
                    "strategy_patch": {"task_type": "rt_prediction"},
                },
            ],
        },
        "scientific_objective": {
            "focus": "科学目标",
            "question": "用一句话描述你想找的数据目标？",
            "options": [
                {
                    "id": "immuno",
                    "label": "免疫肽 / HLA 配体方向",
                    "reason": "主题聚焦免疫肽组",
                    "strategy_patch": {
                        "objective": "免疫肽组学数据发现",
                        "special_themes": ["immunopeptidomics"],
                    },
                },
                {
                    "id": "general_proteomics",
                    "label": "通用蛋白质组数据",
                    "reason": "不限定免疫肽主题",
                    "strategy_patch": {"objective": "蛋白质组学数据发现"},
                },
                {
                    "id": "free_text",
                    "label": "我用自然语言说明",
                    "reason": "目标较特殊，稍后文字补充",
                    "strategy_patch": {"objective": "待用户补充的科学目标"},
                },
            ],
        },
    }

    template = templates.get(item_id)
    if template is None and target_fields:
        # Generic fallback: open vs keep asking via free-ish options on first field.
        field = target_fields[0]
        template = {
            "focus": field,
            "question": f"还需要确认「{field}」如何设定？",
            "options": [
                {
                    "id": f"{field}_recommend",
                    "label": "按常见稳妥默认",
                    "reason": reason,
                    "strategy_patch": {},
                },
                {
                    "id": f"{field}_open",
                    "label": "先保持开放",
                    "reason": "由检索结果再收紧",
                    "strategy_patch": {},
                },
                {
                    "id": f"{field}_describe",
                    "label": "我文字说明具体要求",
                    "reason": "该维度较特殊",
                    "strategy_patch": {},
                },
            ],
        }

    if template is None:
        return None

    options = template["options"]
    recommendation = {
        **options[0],
        "reason": _clean_text(options[0].get("reason")) or reason,
    }
    raw_decision = {
        "focus": template["focus"],
        "target_fields": target_fields or list((options[0].get("strategy_patch") or {}).keys()),
        "question": template["question"],
        "recommendation": recommendation,
        "options": options,
        "option_mode": "focused",
        "allow_free_text": True,
        "revisit_existing": False,
    }
    return _normalise_discovery_next_decision(raw_decision)


def _normalise_discovery_next_decision(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping):
        return None
    focus = _clean_text(raw.get("focus"))
    question = _clean_text(raw.get("question"))
    if not focus or not question:
        return None

    options: list[dict[str, Any]] = []
    option_patch_presence: list[bool] = []
    raw_options = raw.get("options") if isinstance(raw.get("options"), list) else []
    for raw_option in raw_options:
        if not isinstance(raw_option, Mapping):
            continue
        option_id = _clean_text(raw_option.get("id"))
        label = _clean_text(raw_option.get("label"))
        if not option_id or not label:
            continue
        option: dict[str, Any] = {"id": option_id, "label": label}
        reason = _clean_text(raw_option.get("reason"))
        if reason:
            option["reason"] = reason
        raw_option_patch = raw_option.get("strategy_patch")
        if raw_option_patch is None:
            raw_option_patch = raw_option.get("strategyPatch")
        option_patch_presence.append(isinstance(raw_option_patch, Mapping))
        if isinstance(raw_option_patch, Mapping):
            option_patch, option_patch_errors = _validate_discovery_strategy_patch(
                _json_object(dict(raw_option_patch))
            )
            # A rendered option is executable UI, not explanatory prose.  Its
            # mutation meaning must therefore be complete and schema-valid at
            # the time the Manager creates the option; a later turn may not
            # invent extra fields for a short/numeric selection.
            if option_patch_errors or not option_patch:
                return None
            option["strategy_patch"] = option_patch
        if not any(existing["id"] == option_id for existing in options):
            options.append(option)

    # Mixed executable/non-executable menus are ambiguous: the same numeric UI
    # gesture would have different authority depending on which row it hits.
    # Keep legacy all-unpatched decisions readable during rollout, but reject a
    # partial migration.  Newly prompted decisions are required to predeclare
    # every option patch below.
    if any(option_patch_presence) and not all(option_patch_presence):
        return None

    recommendation: dict[str, Any] | None = None
    raw_recommendation = raw.get("recommendation")
    if isinstance(raw_recommendation, Mapping):
        rec_id = _clean_text(raw_recommendation.get("id"))
        rec_label = _clean_text(raw_recommendation.get("label"))
        if rec_id and rec_label:
            recommendation = {"id": rec_id, "label": rec_label}
            rec_reason = _clean_text(raw_recommendation.get("reason"))
            if rec_reason:
                recommendation["reason"] = rec_reason
    else:
        rec_id = _clean_text(raw_recommendation)
        recommendation = next((dict(option) for option in options if option["id"] == rec_id), None)
    if recommendation is None and options:
        recommendation = dict(options[0])
    if recommendation is None or not _clean_text(recommendation.get("reason")):
        return None
    if recommendation is not None and not any(
        option.get("id") == recommendation.get("id") for option in options
    ):
        options.insert(0, dict(recommendation))
    option_mode = (
        "expanded" if _clean_text(raw.get("option_mode")).lower() == "expanded" else "focused"
    )
    option_limit = _DISCOVERY_DECISION_MAX_OPTIONS if option_mode == "expanded" else 5
    options = options[:option_limit]
    if len(options) < 2:
        return None

    target_fields = _normalise_discovery_decision_target_fields(
        raw.get("target_fields"),
        focus=focus,
        option_ids=[option["id"] for option in options],
    )
    executable_options = [
        option for option in options if isinstance(option.get("strategy_patch"), Mapping)
    ]
    if executable_options:
        # target_fields is a display/memory projection only.  Derive it from
        # the already validated executable contracts instead of trusting a
        # model-authored scope list that could smuggle an unrelated field into
        # the next turn (for example build_training -> plan_only).
        target_fields = list(
            dict.fromkeys(
                field
                for option in executable_options
                for field in option["strategy_patch"]
            )
        )
        recommendation_option = next(
            (
                option
                for option in executable_options
                if option.get("id") == recommendation.get("id")
            ),
            None,
        )
        if recommendation_option is None:
            return None
        recommendation = {
            **recommendation_option,
            **recommendation,
            "strategy_patch": dict(recommendation_option["strategy_patch"]),
        }

    return {
        "focus": focus,
        "target_fields": target_fields,
        "question": question,
        "recommendation": recommendation,
        "options": options,
        "option_mode": option_mode,
        "revisit_existing": raw.get("revisit_existing") is True,
        "allow_free_text": True,
        "option_patch_contract": (
            "predeclared_v1" if executable_options else "legacy_unbound"
        ),
    }


def _scope_discovery_next_decision_to_unresolved_fields(
    decision: Mapping[str, Any] | None,
    resolved_fields: set[str],
) -> dict[str, Any] | None:
    """Prevent a new menu from silently reopening settled strategy fields.

    A Manager-authored option is an executable mutation contract.  When the
    user has already resolved search scale (for example exhaustive/open-ended),
    a later downstream-task recommendation must not bundle a conflicting
    curated quota into the same numeric choice.  Explicit reconsideration
    remains available through ``revisit_existing=true``.
    """

    if not isinstance(decision, Mapping) or decision.get("revisit_existing") is True:
        return dict(decision) if isinstance(decision, Mapping) else None
    if decision.get("option_patch_contract") != "predeclared_v1":
        # Legacy menus have no executable option patch to narrow. Their
        # selected value is grounded later by the validated strategy tool call.
        return dict(decision)
    protected_fields = {
        _DISCOVERY_STRATEGY_PATCH_ALIASES.get(field, field)
        for field in resolved_fields
        if _clean_text(field)
    }
    if not protected_fields:
        return dict(decision)

    scoped_options: list[dict[str, Any]] = []
    for raw_option in decision.get("options") or []:
        if not isinstance(raw_option, Mapping):
            continue
        option = dict(raw_option)
        raw_patch = option.get("strategy_patch")
        if isinstance(raw_patch, Mapping):
            scoped_patch = {
                field: value
                for field, value in raw_patch.items()
                if field not in protected_fields
            }
            # A numeric choice with no remaining mutation meaning is not an
            # executable option. Reject the whole menu instead of presenting a
            # row that appears to change strategy but cannot do so.
            if not scoped_patch:
                return None
            option["strategy_patch"] = scoped_patch
        scoped_options.append(option)
    if len(scoped_options) < 2:
        return None

    scoped = dict(decision)
    scoped["options"] = scoped_options
    scoped["target_fields"] = list(
        dict.fromkeys(
            field
            for option in scoped_options
            for field in (
                option.get("strategy_patch")
                if isinstance(option.get("strategy_patch"), Mapping)
                else {}
            )
        )
    )
    recommendation = scoped.get("recommendation")
    recommendation_id = (
        _clean_text(recommendation.get("id"))
        if isinstance(recommendation, Mapping)
        else ""
    )
    recommendation_option = next(
        (
            option
            for option in scoped_options
            if _clean_text(option.get("id")) == recommendation_id
        ),
        None,
    )
    if recommendation_option is None:
        return None
    scoped["recommendation"] = {
        **recommendation_option,
        **dict(recommendation),
        "strategy_patch": dict(recommendation_option.get("strategy_patch") or {}),
    }
    return scoped


_DISCOVERY_ENUM_OPTION_ORDER: dict[str, list[str]] = {
    "labeling_strategy": [
        "label_free",
        "tmt",
        "itraq",
        "silac",
        "dimethyl",
        "any",
        "unknown",
    ],
}
_DISCOVERY_ENUM_OPTION_LABELS: dict[str, dict[str, str]] = {
    "labeling_strategy": {
        "label_free": "无标记（label-free）",
        "tmt": "TMT（多重等重标签）",
        "itraq": "iTRAQ（等重标签）",
        "silac": "SILAC（细胞代谢标记）",
        "dimethyl": "二甲基标记（化学同位素）",
        "any": "保持开放（不限制标记方式）",
        "unknown": "标记信息未知",
    }
}


def _discovery_enum_option_values(field: str) -> list[str]:
    allowed = _DISCOVERY_STRATEGY_ENUM_FIELDS.get(field, set())
    preferred = _DISCOVERY_ENUM_OPTION_ORDER.get(field, [])
    ordered = [value for value in preferred if value in allowed]
    ordered.extend(sorted(value for value in allowed if value not in ordered))
    if "any" in ordered and "unknown" in ordered:
        ordered.remove("unknown")
    return ordered[:_DISCOVERY_DECISION_MAX_OPTIONS]


def _expand_discovery_enum_decision_options(
    decision: Mapping[str, Any] | None,
    assistant_message: str,
) -> dict[str, Any] | None:
    if not isinstance(decision, Mapping):
        return None
    target_fields = decision.get("target_fields")
    if not isinstance(target_fields, list) or len(target_fields) != 1:
        return dict(decision)
    field = _clean_text(target_fields[0])
    allowed_values = _discovery_enum_option_values(field)
    if len(allowed_values) < 2:
        return dict(decision)

    options = [
        dict(option)
        for option in decision.get("options") or []
        if isinstance(option, Mapping)
    ]
    existing_ids = {
        _clean_text(option.get("id")).casefold()
        for option in options
        if _clean_text(option.get("id"))
    }
    missing_values = [
        value for value in allowed_values if value.casefold() not in existing_ids
    ]
    if not missing_values:
        return dict(decision)

    expanded = decision.get("option_mode") == "expanded"
    if not expanded:
        folded_message = assistant_message.casefold().replace("_", " ")
        labels = _DISCOVERY_ENUM_OPTION_LABELS.get(field, {})
        for value in missing_values:
            normalized_value = value.casefold().replace("_", " ")
            normalized_label = _clean_text(labels.get(value)).casefold()
            if (
                (len(value) >= 3 and normalized_value in folded_message)
                or (bool(normalized_label) and normalized_label in folded_message)
            ):
                expanded = True
                break
    if not expanded:
        return dict(decision)

    labels = _DISCOVERY_ENUM_OPTION_LABELS.get(field, {})
    by_id = {
        _clean_text(option.get("id")).casefold(): option
        for option in options
        if _clean_text(option.get("id"))
    }
    expanded_options: list[dict[str, Any]] = []
    for value in allowed_values:
        existing = by_id.get(value.casefold())
        if existing is not None:
            option = dict(existing)
            option.setdefault("strategy_patch", {field: value})
            expanded_options.append(option)
        else:
            expanded_options.append(
                {
                    "id": value,
                    "label": labels.get(value) or value.replace("_", " "),
                    "strategy_patch": {field: value},
                }
            )
    result = dict(decision)
    result["options"] = expanded_options
    result["option_mode"] = "expanded"
    result["option_patch_contract"] = "predeclared_v1"
    recommendation = result.get("recommendation")
    if isinstance(recommendation, Mapping):
        recommendation_id = _clean_text(recommendation.get("id")).casefold()
        recommended_option = next(
            (
                option
                for option in expanded_options
                if _clean_text(option.get("id")).casefold() == recommendation_id
            ),
            None,
        )
        if recommended_option is not None:
            result["recommendation"] = {
                **recommended_option,
                **dict(recommendation),
                "strategy_patch": dict(recommended_option["strategy_patch"]),
            }
    return result


def _normalise_discovery_gap_report(raw: Any) -> dict[str, Any]:
    report = raw if isinstance(raw, Mapping) else {}

    def _slots(key: str) -> list[str]:
        values = report.get(key) if isinstance(report.get(key), list) else []
        cleaned = [_clean_text(value) for value in values]
        return list(
            dict.fromkeys(
                value
                for value in cleaned
                if value and value not in {"horizon", "run_horizon", "delivery_horizon"}
            )
        )

    return {
        "required_missing": _slots("required_missing"),
        "optional_missing": _slots("optional_missing"),
        "ready_for_confirm": report.get("ready_for_confirm") is True,
    }


def _discovery_critical_decision_agenda(
    intent_snapshot: Mapping[str, Any],
    gap_report: Mapping[str, Any],
    resolved_fields: set[str],
) -> list[dict[str, Any]]:
    """Prioritize unresolved user decisions without prescribing a questionnaire.

    This is a deterministic planning guard, not a turn-order engine.  It tells
    the Manager which unresolved choices materially affect feasibility, search
    cost, or scientific validity.  The Manager may still chat, answer a user
    question, accept a compound update, or choose a more relevant personalized
    question; it must not declare readiness while a critical item remains.
    Repository facts are deliberately excluded because the Agent should fetch
    those during Discovery rather than ask the user to guess them.
    """
    fixed_snapshot = {
        **dict(intent_snapshot),
        "run_horizon": _FIXED_DISCOVERY_RUN_HORIZON,
    }
    agenda = agenda_for_manager(
        fixed_snapshot,
        gap_report=gap_report,
        resolved_fields={*resolved_fields, "run_horizon"},
    )
    return [
        item
        for item in agenda
        if "run_horizon" not in set(item.get("target_fields") or [])
        and _clean_text(item.get("focus")).casefold()
        not in {"horizon", "run_horizon", "delivery_horizon"}
    ]


def _normalise_discovery_dialogue_history(raw: Any) -> list[dict[str, str]]:
    """Keep a compact, role-safe conversation window for follow-up reasoning."""
    items = raw if isinstance(raw, list) else []
    history: list[dict[str, str]] = []
    for item in items[-40:]:
        if not isinstance(item, Mapping):
            continue
        role = _clean_text(item.get("role")).lower()
        content = _clean_text(item.get("content"))[:2000]
        if role in {"user", "assistant"} and content:
            history.append({"role": role, "content": content})
    return history


_DISCOVERY_SESSION_MEMORY_PREFIX = "[discovery-session-state]"


def _normalise_discovery_session_id(raw: Any) -> str:
    value = _clean_text(raw)[:160]
    if not value:
        return ""
    return re.sub(r"[^A-Za-z0-9_.:-]+", "-", value).strip("-.")


def _discovery_dialogue_session_db() -> Path:
    configured = _clean_text(os.getenv("AGENT_DIALOGUE_SESSION_DB"))
    path = Path(configured) if configured else _discovery_jobs_dir() / "dialogue_sessions.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _discovery_decision_memory_signature(record: Mapping[str, Any]) -> str:
    identity = {
        "focus": _clean_text(record.get("focus")).casefold(),
        "target_fields": sorted(
            _clean_text(value)
            for value in record.get("target_fields") or []
            if _clean_text(value)
        ),
        "option_ids": sorted(
            _clean_text(value).casefold()
            for value in record.get("option_ids") or []
            if _clean_text(value)
        ),
    }
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_discovery_persistent_decision_memory(
    session_id: str,
) -> list[dict[str, Any]]:
    if not session_id:
        return []
    try:
        with sqlite3.connect(_discovery_dialogue_session_db(), timeout=5.0) as database:
            database.execute(
                """
                CREATE TABLE IF NOT EXISTS discovery_decision_memory (
                    session_id TEXT NOT NULL,
                    decision_signature TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (session_id, decision_signature)
                )
                """
            )
            rows = database.execute(
                """
                SELECT record_json
                FROM discovery_decision_memory
                WHERE session_id = ?
                ORDER BY updated_at ASC
                LIMIT 50
                """,
                (session_id,),
            ).fetchall()
    except (OSError, sqlite3.Error):
        return []
    records: list[dict[str, Any]] = []
    for (raw_record,) in rows:
        try:
            record = json.loads(raw_record)
        except (TypeError, ValueError):
            continue
        if isinstance(record, Mapping):
            records.append(dict(record))
    return _normalise_discovery_decision_memory(records)


def _store_discovery_persistent_decision_memory(
    session_id: str,
    resolved_decision: Mapping[str, Any] | None,
) -> None:
    if not session_id or not isinstance(resolved_decision, Mapping):
        return
    normalized = _normalise_discovery_decision_memory([resolved_decision])
    if not normalized:
        return
    record = normalized[0]
    signature = _discovery_decision_memory_signature(record)
    try:
        with sqlite3.connect(_discovery_dialogue_session_db(), timeout=5.0) as database:
            database.execute(
                """
                CREATE TABLE IF NOT EXISTS discovery_decision_memory (
                    session_id TEXT NOT NULL,
                    decision_signature TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (session_id, decision_signature)
                )
                """
            )
            database.execute(
                """
                INSERT INTO discovery_decision_memory (
                    session_id, decision_signature, record_json, updated_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id, decision_signature) DO UPDATE SET
                    record_json = excluded.record_json,
                    updated_at = excluded.updated_at
                """,
                (
                    session_id,
                    signature,
                    json.dumps(record, ensure_ascii=False, separators=(",", ":")),
                    time.time(),
                ),
            )
            database.execute(
                """
                DELETE FROM discovery_decision_memory
                WHERE session_id = ?
                  AND decision_signature NOT IN (
                    SELECT decision_signature
                    FROM discovery_decision_memory
                    WHERE session_id = ?
                    ORDER BY updated_at DESC
                    LIMIT 50
                  )
                """,
                (session_id, session_id),
            )
    except (OSError, sqlite3.Error):
        return


def _discovery_session_item_text(item: Mapping[str, Any]) -> str:
    content = item.get("content")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for value in content:
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, Mapping):
            text_value = _clean_text(value.get("text") or value.get("content"))
            if text_value:
                parts.append(text_value)
    return "\n".join(parts).strip()


def _load_discovery_dialogue_session(
    session_id: str,
    fallback_history: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    if not session_id:
        return fallback_history, []
    persistent_memory = _load_discovery_persistent_decision_memory(session_id)
    try:
        from agents import SQLiteSession

        async def _load() -> list[dict[str, Any]]:
            session = SQLiteSession(session_id, _discovery_dialogue_session_db())
            try:
                return list(await session.get_items(limit=120))
            finally:
                session.close()

        items = asyncio.run(_load())
    except Exception:
        return fallback_history, persistent_memory

    history: list[dict[str, str]] = []
    remembered: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        role = _clean_text(item.get("role")).lower()
        content = _discovery_session_item_text(item)
        if _clean_text(item.get("type")).lower() == "function_call_output":
            tool_output = _clean_text(item.get("output"))
            try:
                response = _coerce_discovery_dialogue_json(tool_output)
            except (TypeError, ValueError, json.JSONDecodeError):
                response = {}
            assistant_text = _clean_text(response.get("assistant_message"))
            if assistant_text:
                role = "assistant"
                content = assistant_text
        if role not in {"user", "assistant"} or not content:
            continue
        history.append({"role": role, "content": content[:8000]})
        if role == "assistant" and _DISCOVERY_SESSION_MEMORY_PREFIX in content:
            raw_state = content.rsplit(_DISCOVERY_SESSION_MEMORY_PREFIX, 1)[-1].strip()
            try:
                state = json.loads(raw_state)
            except (TypeError, ValueError):
                continue
            record = state.get("resolved_decision") if isinstance(state, Mapping) else None
            if isinstance(record, Mapping):
                remembered.append(dict(record))
    normalized_history = _normalise_discovery_dialogue_history(history)
    return (
        normalized_history or fallback_history,
        _normalise_discovery_decision_memory([*persistent_memory, *remembered]),
    )


def _store_discovery_dialogue_session_turn(
    session_id: str,
    *,
    user_message: str,
    assistant_message: str,
    action: str,
    patch: Mapping[str, Any],
    next_decision: Mapping[str, Any] | None,
    resolved_decision: Mapping[str, Any] | None,
) -> None:
    if not session_id:
        return
    _store_discovery_persistent_decision_memory(session_id, resolved_decision)
    state = {
        "action": action,
        "strategy_patch": dict(patch),
        "next_decision": dict(next_decision) if isinstance(next_decision, Mapping) else None,
        "resolved_decision": (
            dict(resolved_decision) if isinstance(resolved_decision, Mapping) else None
        ),
    }
    memory_text = (
        f"{assistant_message.strip()}\n\n{_DISCOVERY_SESSION_MEMORY_PREFIX}"
        f"{json.dumps(state, ensure_ascii=False, separators=(',', ':'))}"
    )
    try:
        from agents import SQLiteSession

        async def _store() -> None:
            session = SQLiteSession(session_id, _discovery_dialogue_session_db())
            try:
                await session.add_items(
                    [
                        {"role": "user", "content": user_message},
                        {"role": "assistant", "content": memory_text},
                    ]
                )
            finally:
                session.close()

        asyncio.run(_store())
    except Exception:
        # Dialogue remains functional with the request-carried fallback history.
        return


def _normalise_discovery_turn_action(
    raw: Mapping[str, Any],
    *,
    legacy_intent: str,
    patch: Mapping[str, Any],
    next_decision: Mapping[str, Any] | None,
) -> str:
    requested = _clean_text(raw.get("action") or raw.get("mode")).lower()
    if requested in _DISCOVERY_TURN_ACTIONS:
        action = requested
    else:
        # Legacy outputs may still project conversational actions, but legacy
        # fields can never authorize a card write or a confirmation.
        action = {
            "chitchat": "chat",
            "explain": "advise",
            "clarify": "clarify",
            "request_defaults": "advise",
            "request_confirm": "ready_to_confirm",
            "refuse_search": "refuse_search",
        }.get(legacy_intent, "clarify" if next_decision is not None else "advise")
    if action == "update_strategy" and not patch:
        return "clarify" if next_decision is not None else "advise"
    return action


def _legacy_discovery_turn_intent(action: str, raw_intent: str) -> str:
    if action == "update_strategy" and raw_intent in {
        "answer_question",
        "multi_fill",
        "revise",
        "request_defaults",
    }:
        return raw_intent
    if action == "ready_to_confirm" and raw_intent == "request_confirm":
        return raw_intent
    return {
        "chat": "chitchat",
        "advise": "explain",
        "clarify": "clarify",
        "update_strategy": "revise",
        "ready_to_confirm": "explain",
        "confirm_strategy": "request_confirm",
        "refuse_search": "refuse_search",
    }[action]


_DISCOVERY_GRILL_DEFAULT_REQUEST_SECONDS = 60.0
_DISCOVERY_GRILL_MAX_REQUEST_SECONDS = 300.0
_DISCOVERY_SEMANTIC_VERIFIER_ATTEMPT_SECONDS = 15.0
_DISCOVERY_SEMANTIC_VERIFIER_MAX_ATTEMPTS = 2
_DISCOVERY_CONFIRMATION_VOLATILE_FIELDS = {
    "confirmed",
    "answered",
    "inferred",
    "parseWarnings",
    "parse_warnings",
    "parseReasoning",
    "parse_reasoning",
}


def _discovery_strategy_fingerprint(intent_snapshot: Mapping[str, Any]) -> str:
    snapshot = _json_object(dict(intent_snapshot))
    for key in _DISCOVERY_CONFIRMATION_VOLATILE_FIELDS:
        snapshot.pop(key, None)
    if not snapshot:
        return ""
    canonical = json.dumps(
        snapshot,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


_DISCOVERY_EXECUTION_FINGERPRINT_VOLATILE_FIELDS = {
    "strategy_fingerprint",
    "strategy_fingerprint_payload",
    "idempotency_key",
    # Added only after the HTTP confirmation check when the queued job is
    # persisted. It is server-owned routing state, not part of what the user
    # reviewed, so the execution-boundary recheck must ignore it too.
    "_execution_discovery_id",
    # Added only by the resume endpoint. It authorizes reuse of the existing
    # control-plane run and must not invalidate the user's original strategy
    # confirmation.
    "_resume_existing_discovery_run",
}


def _discovery_execution_snapshot(body: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = _json_object(dict(body))
    for key in _DISCOVERY_EXECUTION_FINGERPRINT_VOLATILE_FIELDS:
        snapshot.pop(key, None)
    return snapshot


def _discovery_execution_fingerprint(body: Mapping[str, Any]) -> str:
    """Bind an explicit confirmation to the exact repository-search payload.

    This Python-native projection remains the backward-compatible proof path.
    New browser clients also submit their exact canonical JSON text so valid
    numbers cannot diverge solely because Python and JavaScript format an
    exponent differently.
    """

    snapshot = _discovery_execution_snapshot(body)
    if not snapshot:
        return ""
    canonical = json.dumps(
        snapshot,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _json_values_equal(left: Any, right: Any) -> bool:
    """Type-safe JSON semantic equality (JSON has one numeric type)."""

    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isfinite(float(left)) and math.isfinite(float(right)) and left == right
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, str) or isinstance(right, str):
        return isinstance(left, str) and isinstance(right, str) and left == right
    if isinstance(left, list) or isinstance(right, list):
        return (
            isinstance(left, list)
            and isinstance(right, list)
            and len(left) == len(right)
            and all(_json_values_equal(a, b) for a, b in zip(left, right))
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_json_values_equal(left[key], right[key]) for key in left)
        )
    return False


def _discovery_confirmation_context(
    body: Mapping[str, Any],
    *,
    phase: str,
    intent_snapshot: Mapping[str, Any],
    gap_report: Mapping[str, Any],
) -> tuple[bool, str, str]:
    fingerprint = _discovery_strategy_fingerprint(intent_snapshot)
    if phase != "awaiting_confirm":
        return False, "phase is not awaiting_confirm", fingerprint
    if not fingerprint:
        return False, "current strategy snapshot is empty", fingerprint
    normalized_gap = _normalise_discovery_gap_report(gap_report)
    if normalized_gap["required_missing"] or not normalized_gap["ready_for_confirm"]:
        return False, "current strategy is not ready for confirmation", fingerprint
    resolved_fields = _normalise_discovery_resolved_fields(None, intent_snapshot)
    unresolved_critical = [
        item
        for item in _discovery_critical_decision_agenda(
            intent_snapshot,
            normalized_gap,
            resolved_fields,
        )
        if item.get("critical") is True
    ]
    if unresolved_critical:
        return (
            False,
            "critical strategy decisions remain unresolved: "
            + ", ".join(_clean_text(item.get("id")) for item in unresolved_critical),
            fingerprint,
        )

    expected_fingerprint = _clean_text(body.get("pending_strategy_fingerprint"))
    pending_snapshot = body.get("pending_strategy_snapshot")
    if isinstance(pending_snapshot, Mapping):
        expected_fingerprint = _discovery_strategy_fingerprint(pending_snapshot)
    if expected_fingerprint and not secrets.compare_digest(
        expected_fingerprint,
        fingerprint,
    ):
        return False, "current strategy no longer matches the pending snapshot", fingerprint
    return True, "", fingerprint


def _bind_discovery_turn_request_budget(client: Any, body: Mapping[str, Any]) -> float:
    try:
        client_timeout = float(
            getattr(client, "timeout", _DISCOVERY_GRILL_DEFAULT_REQUEST_SECONDS)
        )
    except (TypeError, ValueError):
        client_timeout = _DISCOVERY_GRILL_DEFAULT_REQUEST_SECONDS
    if not math.isfinite(client_timeout) or client_timeout <= 0:
        client_timeout = _DISCOVERY_GRILL_DEFAULT_REQUEST_SECONDS

    raw_budget = body.get("request_timeout_seconds")
    try:
        requested = float(raw_budget) if raw_budget is not None else client_timeout
    except (TypeError, ValueError):
        requested = client_timeout
    if not math.isfinite(requested) or requested <= 0:
        requested = client_timeout
    budget = min(_DISCOVERY_GRILL_MAX_REQUEST_SECONDS, max(1.0, requested))

    # The production OpenAI-compatible client exposes a per-request timeout.
    # Clamp it to this turn's single budget; custom test/adapter clients without
    # that attribute still receive exactly one call below.
    if hasattr(client, "timeout"):
        try:
            current = float(getattr(client, "timeout"))
        except (TypeError, ValueError):
            current = budget
        try:
            setattr(client, "timeout", min(current, budget) if current > 0 else budget)
        except (AttributeError, TypeError):
            pass
    return budget


class _DiscoveryStrategyPatchToolInput(BaseModel):
    """SDK tool input; deterministic validators remain the mutation authority."""

    model_config = ConfigDict(extra="forbid")

    objective: str | None = None
    task_type: str | None = None
    run_horizon: str | None = None
    species: list[str] | None = None
    species_policy: str | None = None
    species_coverage: str | None = None
    acquisition_mode: str | None = None
    mixed_acquisition_policy: str | None = None
    ptm_types: list[str] | None = None
    special_themes: list[str] | None = None
    labeling_strategy: str | None = None
    labeling_hard: bool | None = None
    coverage_mode: str | None = None
    target_project_count: int | None = None
    max_candidate_projects: int | None = None
    quota_flexibility: str | None = None
    time_budget: str | None = None
    on_safety_ceiling: str | None = None
    instrument_preference: str | None = None
    legacy_floor_ratio: float | None = None
    exclude_rules: list[str] | None = None
    success_criteria: list[str] | None = None
    scientific_constraints: list[ScientificConstraint] | None = None
    notes: str | None = None
    open_risks: list[str] | None = None
    repository: str | None = None


class _DiscoveryPatchEvidenceInput(BaseModel):
    """One verifier claim grounded in an exact latest-turn quote."""

    model_config = ConfigDict(extra="forbid")

    field: str
    source: str
    rationale: str = ""


class _DiscoveryPatchVerificationInput(BaseModel):
    """Bounded second-Agent review for multi-clause strategy mutations."""

    model_config = ConfigDict(extra="forbid")

    verdict: str
    patch: _DiscoveryStrategyPatchToolInput
    evidence: list[_DiscoveryPatchEvidenceInput]
    rationale: str = ""


class _DiscoveryScientificAdvisorInput(BaseModel):
    """Bounded question from the Dialogue Manager to its scientific specialist."""

    model_config = ConfigDict(extra="forbid")

    question: str
    decision_goal: str = "prioritize the next scientifically material user decision"


class _DiscoveryScientificAdvisorDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    priority: int
    target_fields: list[str]
    question: str
    recommendation: str
    reason: str


class _DiscoveryScientificAdvisorOutput(BaseModel):
    """Read-only specialist result consumed by the user-facing Manager."""

    model_config = ConfigDict(extra="forbid")

    analysis: str
    critical_decisions: list[_DiscoveryScientificAdvisorDecision]
    repository_evidence_to_fetch: list[str]
    scientific_risks: list[str]


async def _sdk_discovery_update_strategy(
    ctx: RunContextWrapper[SimpleNamespace],
    patch: _DiscoveryStrategyPatchToolInput,
    response_json: str,
) -> str:
    """Record one explicit strategy delta and return the complete D1 response JSON."""

    payload = patch.model_dump(mode="python", exclude_unset=True)
    validated, errors = _validate_discovery_strategy_patch(payload)
    if errors:
        return json.dumps(
            {
                "action": "advise",
                "assistant_message": (
                    "这轮策略修改没有通过结构校验，当前策略保持不变。"
                    "你可以继续用自然语言说明想改什么。"
                ),
                "tool_calls": [],
                "contract_errors": errors,
            },
            ensure_ascii=False,
        )
    calls = ctx.context.tool_calls
    if calls:
        return json.dumps(
            {
                "action": "advise",
                "assistant_message": "同一轮只能执行一个对话动作，当前策略保持不变。",
                "tool_calls": [],
            },
            ensure_ascii=False,
        )
    calls.append(
        {
            "name": "update_strategy",
            "arguments": {"patch": validated},
        }
    )
    try:
        response = _coerce_discovery_dialogue_json(response_json)
    except (TypeError, ValueError, json.JSONDecodeError):
        response = {
            "assistant_message": "已按你的明确选择更新策略。",
            "turn_interpretation": {"commitments": [], "consultations": []},
        }
    response["action"] = "update_strategy"
    return json.dumps(response, ensure_ascii=False)


async def _sdk_discovery_verify_strategy_patch(
    ctx: RunContextWrapper[SimpleNamespace],
    verification: _DiscoveryPatchVerificationInput,
) -> str:
    """Capture one auditable semantic review; this tool never writes the card."""

    verdict = _clean_text(verification.verdict).lower()
    if verdict not in {"accept", "repair", "reject"}:
        verdict = "reject"
    candidate = verification.patch.model_dump(mode="python", exclude_unset=True)
    # Some OpenAI-compatible models serialize unmentioned optional enum fields
    # as ``""``.  Empty strings are neither a valid enum nor the contract's
    # explicit clear operation (which is null), so treating them as omission is
    # the only non-mutating interpretation.  This avoids rejecting an otherwise
    # grounded multi-field correction because of model-generated placeholders.
    candidate = {
        key: value
        for key, value in candidate.items()
        if not (isinstance(value, str) and not value.strip())
    }
    validated, errors = _validate_discovery_strategy_patch(candidate)
    payload = {
        "verdict": "reject" if errors else verdict,
        "patch": {} if errors else validated,
        "evidence": [item.model_dump(mode="json") for item in verification.evidence],
        "rationale": _clean_text(verification.rationale)[:1200],
        "errors": errors,
    }
    ctx.context.verification = payload
    return json.dumps(payload, ensure_ascii=False)


async def _sdk_discovery_respond(
    ctx: RunContextWrapper[SimpleNamespace],
    response_json: str,
) -> str:
    """Return one non-mutating chat, advice, clarification, or readiness response."""

    try:
        response = _coerce_discovery_dialogue_json(response_json)
    except (TypeError, ValueError, json.JSONDecodeError):
        response = {
            "action": "chat",
            "assistant_message": "我在听。你可以继续说你的科学问题或数据需求。",
            "tool_calls": [],
        }
    if _clean_text(response.get("action")).lower() in {
        "update_strategy",
        "confirm_strategy",
    }:
        response["action"] = "advise"
        response["assistant_message"] = (
            "这轮只进行了讨论，策略没有修改。"
            + (" " + _clean_text(response.get("assistant_message")) if _clean_text(response.get("assistant_message")) else "")
        )
    return json.dumps(response, ensure_ascii=False)


async def _sdk_discovery_confirm_strategy(
    ctx: RunContextWrapper[SimpleNamespace],
    response_json: str,
) -> str:
    """Record explicit approval and return the complete D1 response JSON."""

    calls = ctx.context.tool_calls
    if calls:
        return json.dumps(
            {
                "action": "advise",
                "assistant_message": "同一轮不能既修改策略又确认策略；本轮未确认。",
                "tool_calls": [],
            },
            ensure_ascii=False,
        )
    calls.append({"name": "confirm_strategy", "arguments": {}})
    try:
        response = _coerce_discovery_dialogue_json(response_json)
    except (TypeError, ValueError, json.JSONDecodeError):
        response = {"assistant_message": "已确认当前策略，但尚未启动仓库搜索。"}
    response["action"] = "confirm_strategy"
    return json.dumps(response, ensure_ascii=False)


def _load_discovery_dialogue_agents_sdk() -> dict[str, Any]:
    try:
        from agents import (
            Agent,
            AsyncOpenAI,
            ModelSettings,
            OpenAIChatCompletionsModel,
            RunConfig,
            Runner,
            function_tool,
        )
    except ImportError as exc:  # pragma: no cover - deployment configuration failure
        raise RuntimeError(
            "OpenAI Agents SDK is required for discovery dialogue. "
            "Install the project agents-sdk dependency."
        ) from exc
    return {
        "Agent": Agent,
        "AsyncOpenAI": AsyncOpenAI,
        "ModelSettings": ModelSettings,
        "OpenAIChatCompletionsModel": OpenAIChatCompletionsModel,
        "RunConfig": RunConfig,
        "Runner": Runner,
        "function_tool": function_tool,
    }


def _coerce_discovery_dialogue_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        decoded = json.loads(match.group(0))
    if not isinstance(decoded, dict):
        raise ValueError("Discovery dialogue Agent output must be a JSON object.")
    return decoded


def _run_discovery_dialogue_json_compatibility(
    client: OpenAICompatibleDiscoveryLLM,
    *,
    system_prompt: str,
    dialogue_history: list[dict[str, str]],
    state_prompt: str,
) -> dict[str, Any]:
    """Recover a Manager action when a provider ignores required SDK tools.

    The completion is still model-owned and schema-generic.  This helper does
    not mutate state: the caller may synthesize one typed action event only
    after validating the returned action/tool envelope, and the normal
    commitment plus semantic gates run afterwards.
    """

    compatibility_prompt = (
        f"{state_prompt}\n\n"
        "PROVIDER JSON ACTION COMPATIBILITY (authoritative for this retry): The previous "
        "SDK attempt returned ordinary prose instead of invoking its required terminal "
        "function. Return exactly one JSON object matching the D1 contract above; do not "
        "return prose outside JSON. For action=update_strategy, include exactly one textual "
        "tool_calls entry named update_strategy with arguments.patch. The server will schema-"
        "validate that envelope and synthesize the equivalent typed event before the normal "
        "commitment and semantic gates. For a non-mutating turn use action=chat, advise, or "
        "clarify and tool_calls=[]. Never confirm unless the supplied confirmation_context is "
        "eligible. This compatibility retry grants no additional strategy authority."
    )
    raw = client.complete_json_messages(
        messages=[
            {"role": "system", "content": system_prompt},
            *dialogue_history,
            {"role": "user", "content": compatibility_prompt},
        ]
    )
    return dict(raw) if isinstance(raw, Mapping) else {}



def _discovery_tool_required_model_settings(
    sdk: Mapping[str, Any],
    *,
    model_name: str,
    temperature: float = 0,
) -> Any:
    """Build ModelSettings for forced tool use.

    DeepSeek thinking-mode models (e.g. deepseek-v4-pro default) reject
    tool_choice=required unless thinking is explicitly disabled.
    """
    kwargs: dict[str, Any] = {
        "temperature": temperature,
        "parallel_tool_calls": False,
        "tool_choice": "required",
    }
    name = str(model_name or "").casefold()
    if "deepseek" in name:
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    return sdk["ModelSettings"](**kwargs)


def _run_discovery_dialogue_agents_sdk(
    client: OpenAICompatibleDiscoveryLLM,
    *,
    system_prompt: str,
    dialogue_history: list[dict[str, str]],
    state_prompt: str,
    user_message: str,
    session_id: str,
    advisor_context: Mapping[str, Any] | None = None,
    model: Any | None = None,
) -> dict[str, Any]:
    """Run D1 as a real SDK Agent; persistence happens after server validation.

    The SDK session cannot be attached here because it would persist raw tool
    output before commitment reconciliation and semantic verification.  The
    caller supplies canonical history and stores exactly one normalized turn
    only after all server-side gates have finished.
    """

    sdk = _load_discovery_dialogue_agents_sdk()
    caller_supplied_model = model is not None
    owned_async_client: Any | None = None
    if model is None:
        owned_async_client = sdk["AsyncOpenAI"](
            api_key=client.api_key,
            base_url=client.base_url,
            timeout=client.timeout,
            max_retries=1,
        )
        model = sdk["OpenAIChatCompletionsModel"](
            model=client.model,
            openai_client=owned_async_client,
            buffer_streamed_tool_calls=True,
        )

    context = SimpleNamespace(tool_calls=[], advisor_calls=[])
    bounded_advisor_context = _json_object(dict(advisor_context or {}))
    advisor_instructions = (
        "You are a read-only scientific planning specialist for a proteomics data-discovery "
        "Dialogue Manager. Analyze the concrete scientific task rather than following a fixed "
        "questionnaire. Prioritize only decisions the user must make; list repository facts under "
        "repository_evidence_to_fetch instead of asking the user to guess them. Search scale is a "
        "material decision for every executable run unless quota is explicitly open-ended. For "
        "training tasks, examine downstream labels, acquisition compatibility, biological "
        "generalization, leakage/batch risks, and evidence requirements. Never mutate a strategy, "
        "confirm it, or claim PRIDE facts that have not been retrieved. Return concise structured "
        "analysis for the user-facing Manager.\n"
        "COMPOUND ANSWERS: when the latest user message already packs multiple explicit "
        "commitments (species, task, scale, acquisition, horizon, themes, etc.), treat those as "
        "settled. Do not invent a multi-step quiz or re-ask fields they stated. List at most the "
        "true remaining readiness blockers (usually zero or one). Never invent a downstream "
        "task_type the user did not state (e.g. immunopeptide topic alone is not rt_prediction).\n\n"
        "Authoritative bounded context:\n"
        + json.dumps(bounded_advisor_context, ensure_ascii=False, indent=2)
    )
    advisor_agent = sdk["Agent"][SimpleNamespace](
        name="Proteomics Scientific Planning Advisor",
        instructions=advisor_instructions,
        model=model,
        output_type=_DiscoveryScientificAdvisorOutput,
        model_settings=sdk["ModelSettings"](temperature=0),
    )

    async def _extract_advisor_output(run_result: Any) -> str:
        output = run_result.final_output
        if isinstance(output, BaseModel):
            payload = output.model_dump(mode="json")
        elif isinstance(output, Mapping):
            payload = dict(output)
        else:
            try:
                payload = _coerce_discovery_dialogue_json(str(output or ""))
            except (TypeError, ValueError, json.JSONDecodeError):
                payload = {"analysis": _clean_text(output)}
        context.advisor_calls.append(_json_safe(payload))
        return json.dumps(payload, ensure_ascii=False)

    advisor_tool = advisor_agent.as_tool(
        tool_name="consult_scientific_advisor",
        tool_description=(
            "Ask the read-only proteomics specialist to prioritize task-specific scientific "
            "decisions and distinguish user choices from repository evidence. Call at most once "
            "per turn, and only for open-ended scientific planning, a newly introduced complex "
            "task, or a nontrivial multi-field conflict. Never call for greetings, short topic "
            "commits, exact option selections, single-field edits, straightforward confirmations, "
            "compound multi-commitment dumps whose explicit fields are already clear, or any turn "
            "that already has a clear terminal action. After a compound answer, do not use the "
            "advisor to invent a multi-step quiz; finish with update_strategy for stated fields."
        ),
        parameters=_DiscoveryScientificAdvisorInput,
        include_input_schema=True,
        custom_output_extractor=_extract_advisor_output,
        max_turns=1,
    )
    tools = [
        advisor_tool,
        sdk["function_tool"](
            _sdk_discovery_respond,
            name_override="respond",
            description_override=(
                "Finish a non-mutating chat, advice, clarification, refusal, or ready-to-confirm "
                "turn. Pass the complete D1 contract object serialized in response_json."
            ),
            strict_mode=False,
        ),
        sdk["function_tool"](
            _sdk_discovery_update_strategy,
            name_override="update_strategy",
            description_override=(
                "Write only strategy choices the user has explicitly committed to in this "
                "turn (one field or a full multi-field compound patch when they packed several "
                "commitments into one message), and pass the complete D1 contract object "
                "serialized in response_json. Map every explicit commitment in ONE patch. "
                "Advice, comparisons, examples, hypothetical values, and your own recommended "
                "defaults must not be written unless the user explicitly accepts them. Do not "
                "invent task_type or other fields the user did not state."
            ),
            strict_mode=False,
        ),
        sdk["function_tool"](
            _sdk_discovery_confirm_strategy,
            name_override="confirm_strategy",
            description_override=(
                "Record unambiguous approval of the exact strategy currently awaiting confirmation. "
                "Pass the complete D1 contract object serialized in response_json. This does not "
                "start PRIDE search."
            ),
            strict_mode=False,
        ),
    ]
    instructions = (
        f"{system_prompt}\n\n"
        "SDK ACTION PROTOCOL (mandatory): finish by invoking exactly one function tool and never "
        "return assistant text directly. Use respond for a non-mutating turn, update_strategy for "
        "an explicit user commitment (including multi-field compound dumps in ONE patch), or "
        "confirm_strategy for eligible approval. Put the complete D1 response object, serialized "
        "as JSON text, in response_json. For update_strategy, also supply the canonical patch with "
        "every field the latest message committed—not a questionnaire. The selected function "
        "tool is the action authority; a textual tool_calls array is only a projection and cannot "
        "replace the function call. When selected_agent_option is present, it is an explicit "
        "commitment: invoke update_strategy, not respond. If the option intentionally keeps a "
        "field open/default, submit the relevant target field with its canonical open, empty, or "
        "default value so the decision itself is recorded. consult_scientific_advisor is a bounded "
        "read-only specialist: call it at most once, and only when scientific planning genuinely "
        "benefits. Skip it for greetings, short topic/theme commits, single-field edits, exact "
        "option selections, confirmation, and compound multi-commitment answers whose fields are "
        "already explicit. Prefer finishing in one Manager turn with respond/update_strategy/"
        "confirm_strategy whenever the user commitment is already clear. After any advisor call, "
        "continue as the same Manager and finish with exactly one of respond/update_strategy/"
        "confirm_strategy. Advisor never owns the user reply or strategy mutation; never use it "
        "to push a multi-step quiz after a compound answer.\n\n"
        "CURRENT TURN STATE AND OUTPUT CONTRACT (authoritative for this run):\n"
        f"{state_prompt}"
    )
    agent = sdk["Agent"][SimpleNamespace](
        name="Proteomics Discovery Dialogue Agent",
        instructions=instructions,
        model=model,
        tools=tools,
        model_settings=_discovery_tool_required_model_settings(
            sdk,
            model_name=client.model,
        ),
        tool_use_behavior={
            "stop_at_tool_names": ["respond", "update_strategy", "confirm_strategy"]
        },
    )

    runner_input: list[dict[str, str]] = [
        *dialogue_history,
        {"role": "user", "content": user_message},
    ]
    async def _execute() -> Any:
        run_config = sdk["RunConfig"](
            workflow_name="proteomics_discovery_dialogue_v1",
            group_id=session_id or None,
            trace_metadata={"workflow": "discovery_dialogue"},
            tracing_disabled=True,
            trace_include_sensitive_data=False,
        )
        try:
            # OpenAI-compatible gateways occasionally close a streamed response
            # before the final chunk arrives.  The SDK's own retry is not enough
            # for this failure because it can happen after a tool round starts.
            # Retry the whole bounded dialogue once, without persisting anything
            # between attempts; strategy persistence happens only after this
            # function returns and passes the server-side gates.
            for attempt in range(2):
                try:
                    return await sdk["Runner"].run(
                        starting_agent=agent,
                        input=runner_input,
                        context=context,
                        max_turns=2,
                        run_config=run_config,
                    )
                except Exception as exc:
                    message = _clean_text(str(exc)).casefold()
                    transient_markers = (
                        "incomplete chunked read",
                        "peer closed connection",
                        "server disconnected",
                        "connection reset",
                        "readtimeout",
                        "remoteprotocolerror",
                    )
                    if attempt != 0 or not any(
                        marker in message for marker in transient_markers
                    ):
                        raise
                    context.tool_calls.clear()
                    context.advisor_calls.clear()
                    await asyncio.sleep(0.8)
        finally:
            if owned_async_client is not None:
                await owned_async_client.close()

    # The budget applies to the complete Agent loop (including a possible
    # tool round-trip), not independently to every provider request.
    result = asyncio.run(
        asyncio.wait_for(_execute(), timeout=max(1.0, float(client.timeout)))
    )

    final_output = result.final_output
    if isinstance(final_output, Mapping):
        raw = dict(final_output)
    else:
        serialized_output = str(final_output or "")
        if not serialized_output.strip():
            item_types = [
                type(item).__name__
                for item in list(getattr(result, "new_items", []) or [])[:20]
            ]
            terminal_actions = [
                _clean_text(call.get("name"))
                for call in context.tool_calls
                if isinstance(call, Mapping) and _clean_text(call.get("name"))
            ]
            if terminal_actions:
                raise ValueError(
                    "Discovery dialogue SDK produced an empty final output after a terminal "
                    f"action (run_items={item_types or ['none']}, "
                    f"terminal_actions={terminal_actions}, "
                    f"advisor_calls={len(context.advisor_calls)})."
                )
            raw = {
                "action": "advise",
                "assistant_message": (
                    "模型这轮没有执行结构化对话工具，当前策略保持不变。"
                    "我会继续按你的原话检查是否遗漏了明确要求。"
                ),
                "turn_interpretation": {"commitments": [], "consultations": []},
                "tool_calls": [],
                "_provider_compatibility_recovery": {
                    "mode": "empty_output_as_non_mutating",
                    "run_items": item_types,
                    "advisor_calls": len(context.advisor_calls),
                },
            }
        else:
            try:
                raw = _coerce_discovery_dialogue_json(serialized_output)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                # Some OpenAI-compatible providers ignore tool_choice="required"
                # and return an ordinary assistant message.  Plain prose has no
                # mutation authority, but it is still usable as a non-mutating
                # response.  The outer omission auditor may then ask the same
                # Dialogue Manager to repair a missed commitment.  JSON-looking
                # malformed output and prose after a real terminal action still
                # fail closed.
                looks_like_json = serialized_output.lstrip().startswith(("{", "[", "```"))
                if context.tool_calls or looks_like_json:
                    raise ValueError(
                        "Discovery dialogue SDK produced a non-JSON final output "
                        f"(type={type(final_output).__name__}, chars={len(serialized_output)})."
                    ) from exc
                compatibility_raw: dict[str, Any] = {}
                if not caller_supplied_model:
                    try:
                        compatibility_client = OpenAICompatibleDiscoveryLLM(
                            api_key=client.api_key,
                            model=client.model,
                            base_url=client.base_url,
                            timeout=max(2.0, float(client.timeout)),
                        )
                        compatibility_raw = _run_discovery_dialogue_json_compatibility(
                            compatibility_client,
                            system_prompt=system_prompt,
                            dialogue_history=dialogue_history,
                            state_prompt=state_prompt,
                        )
                    except Exception:
                        compatibility_raw = {}
                compatibility_action = _clean_text(
                    compatibility_raw.get("action") or compatibility_raw.get("mode")
                ).lower()
                synthesized_calls: list[dict[str, Any]] = []
                if compatibility_action == "update_strategy":
                    candidate_calls = compatibility_raw.get("tool_calls")
                    update_calls = [
                        call
                        for call in candidate_calls
                        if isinstance(call, Mapping)
                        and _clean_text(call.get("name") or call.get("tool"))
                        == "update_strategy"
                    ] if isinstance(candidate_calls, list) else []
                    if len(update_calls) == 1:
                        arguments = update_calls[0].get("arguments")
                        candidate_patch = (
                            arguments.get("patch")
                            if isinstance(arguments, Mapping)
                            else None
                        )
                        if isinstance(candidate_patch, Mapping):
                            validated_patch, validation_errors = (
                                _validate_discovery_strategy_patch(
                                    _json_object(dict(candidate_patch))
                                )
                            )
                            if validated_patch and not validation_errors:
                                synthesized_calls = [
                                    {
                                        "name": "update_strategy",
                                        "arguments": {"patch": validated_patch},
                                    }
                                ]
                elif compatibility_action == "confirm_strategy":
                    synthesized_calls = [
                        {"name": "confirm_strategy", "arguments": {}}
                    ]

                if compatibility_raw:
                    raw = compatibility_raw
                    context.tool_calls = synthesized_calls
                    raw.setdefault("assistant_message", serialized_output.strip())
                    raw["_provider_compatibility_recovery"] = {
                        "mode": "json_action_contract_after_plain_text",
                        "chars": len(serialized_output),
                        "action": compatibility_action or "unknown",
                        "typed_event_synthesized": bool(synthesized_calls),
                        "advisor_calls": len(context.advisor_calls),
                    }
                else:
                    raw = {
                        "action": "advise",
                        "assistant_message": serialized_output.strip(),
                        "turn_interpretation": {
                            "commitments": [],
                            "consultations": [],
                        },
                        "tool_calls": [],
                        "_provider_compatibility_recovery": {
                            "mode": "plain_text_as_non_mutating",
                            "chars": len(serialized_output),
                            "advisor_calls": len(context.advisor_calls),
                        },
                    }
    # Only SDK-executed tools are mutation/confirmation authority. Any textual
    # tool_calls emitted in the final answer are replaced, never trusted.
    raw["tool_calls"] = list(context.tool_calls)
    raw["_agent_runtime"] = "openai_agents"
    raw["_sdk_session_managed"] = False
    if context.advisor_calls:
        raw["_advisor_calls"] = list(context.advisor_calls)
    return raw


def _ground_discovery_patch_verification(
    verification: Mapping[str, Any],
    *,
    user_message: str,
    intent_snapshot: Mapping[str, Any],
    proposed_patch: Mapping[str, Any],
    allow_commitment_recovery: bool = False,
    preserve_unchanged_fields: set[str] | None = None,
) -> dict[str, Any]:
    """Validate a verifier result without any vocabulary or ontology branches."""

    verdict = _clean_text(verification.get("verdict")).lower()
    canonical_proposed_patch, proposed_errors = _validate_discovery_strategy_patch(
        dict(proposed_patch)
    )
    if proposed_errors:
        return {}
    raw_patch = verification.get("patch")
    if verdict not in {"accept", "repair"} or not isinstance(raw_patch, Mapping):
        return {}
    patch, errors = _validate_discovery_strategy_patch(dict(raw_patch))
    if errors or not patch:
        return {}

    message_evidence = re.sub(r"\s+", "", user_message).casefold()
    grounded_fields: set[str] = set()
    evidence = verification.get("evidence")
    if isinstance(evidence, list):
        for item in evidence[:100]:
            if not isinstance(item, Mapping):
                continue
            raw_field = _clean_text(item.get("field"))
            # Models often cite one member of a structured array as
            # ``scientific_constraints[0]``.  The evidence still grounds the
            # canonical top-level field; indices are audit detail, not a new
            # strategy key.
            field_root = re.split(r"[.\[]", raw_field, maxsplit=1)[0]
            field = _DISCOVERY_STRATEGY_PATCH_ALIASES.get(field_root, field_root)
            source = re.sub(r"\s+", "", _clean_text(item.get("source"))).casefold()
            if field in patch and source and source in message_evidence:
                grounded_fields.add(field)
    # The verifier is allowed to repair only fields for which it supplied an
    # exact latest-message evidence span.  This keeps the second Agent from
    # smuggling its own recommended defaults into the card.
    # The critic may veto or narrow a proposal, never author a new value.  This
    # is the single-writer boundary recommended by manager-style orchestration:
    # only the user-facing Dialogue Manager can propose a card mutation.
    allowed_fields = set(canonical_proposed_patch)
    grounded_patch = {
        field: value
        for field, value in patch.items()
        if (
            field in grounded_fields
            and field in allowed_fields
            and _json_values_equal(value, canonical_proposed_patch.get(field))
        )
    }
    return _drop_unchanged_discovery_patch_fields(
        grounded_patch,
        intent_snapshot,
        preserve_fields=preserve_unchanged_fields,
    )


def _normalise_discovery_patch_verification_audit(
    verification: Mapping[str, Any],
    *,
    user_message: str,
    proposed_patch: Mapping[str, Any],
    grounded_patch: Mapping[str, Any],
) -> dict[str, Any]:
    """Make the public verifier audit describe the delta it can authorize.

    Compatible models occasionally label a no-op review as ``repair`` and
    explain changes that are merely retained in the current strategy.  The
    mutation boundary already trusts only the evidence-grounded delta; the
    audit projection must use that same source of truth rather than repeating
    a contradictory model narrative.
    """

    proposed, proposed_errors = _validate_discovery_strategy_patch(
        dict(proposed_patch)
    )
    if proposed_errors:
        proposed = {}
    effective_patch = dict(grounded_patch)
    effective_verdict = (
        "accept" if effective_patch == proposed else "repair"
    )

    compact_message = re.sub(r"\s+", "", user_message).casefold()
    evidence_rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    raw_evidence = verification.get("evidence")
    if isinstance(raw_evidence, list):
        for item in raw_evidence[:100]:
            if not isinstance(item, Mapping):
                continue
            raw_field = _clean_text(item.get("field"))
            field_root = re.split(r"[.\[]", raw_field, maxsplit=1)[0]
            field = _DISCOVERY_STRATEGY_PATCH_ALIASES.get(field_root, field_root)
            source = _clean_text(item.get("source"))
            if (
                field not in effective_patch
                or not source
                or re.sub(r"\s+", "", source).casefold() not in compact_message
            ):
                continue
            key = (field, source.casefold())
            if key in seen:
                continue
            seen.add(key)
            row = {"field": field, "source": source}
            rationale = _clean_text(item.get("rationale"))[:500]
            if rationale:
                row["rationale"] = rationale
            evidence_rows.append(row)

    model_rationale = _clean_text(verification.get("rationale"))[:1200]
    if effective_verdict == "accept":
        rationale = (
            "Independent semantic verifier confirmed the evidence-grounded "
            "primary strategy delta."
        )
    elif proposed:
        rationale = (
            "Independent semantic verifier produced an evidence-grounded "
            "correction to the primary strategy delta."
        )
    else:
        rationale = (
            "Independent semantic verifier recovered an evidence-grounded "
            "strategy delta omitted by the primary turn."
        )
    return {
        "verdict": effective_verdict,
        "evidence": evidence_rows,
        "rationale": rationale,
        "model_rationale": model_rationale,
    }


def _run_discovery_patch_verifier_json_fallback(
    client: OpenAICompatibleDiscoveryLLM,
    *,
    user_message: str,
    intent_snapshot: Mapping[str, Any],
    proposed_patch: Mapping[str, Any],
    selected_decision: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Provider-compatible structured fallback for the read-only critic.

    Some OpenAI-compatible providers ignore required function tools.  A critic
    has no action authority, so a bounded JSON-object completion is an
    appropriate compatibility path: its output still passes the same schema,
    evidence, scope, and single-writer gates as an SDK tool result.
    """

    omission_mode = not proposed_patch
    omission_rule = (
        "The Manager proposed no patch. Classify every punctuation-delimited clause "
        "independently. A declarative clause stating the user's intended goal, dataset "
        "use, research topic, or downstream task is a commitment even when another "
        "clause asks for advice. If any such clause exists, verdict must be repair. "
        "Report it only in candidate_findings as objects containing canonical field, "
        "canonical value, exact source quote, and optional rationale. candidate_findings "
        "are read-only observations for a Manager retry, not a strategy patch. Return no "
        "patch key in omission mode."
        if omission_mode
        else (
            "The Manager proposed a patch. You may accept it or narrow it, but patch may "
            "contain only identical proposed fields and values. Never add or change a field."
        )
    )
    output_contract = (
        "Return exactly one JSON object with keys verdict (repair|reject), "
        "candidate_findings (array), and rationale (string). Each candidate finding must "
        "contain field, value, source, and optional rationale."
        if omission_mode
        else (
            "Return exactly one JSON object with keys verdict (accept|repair|reject), patch "
            "(object), evidence (array of objects with field, source, and optional rationale), "
            "and rationale (string)."
        )
    )
    system_prompt = (
        "You are a read-only semantic critic for a proteomics data-discovery Dialogue "
        "Manager. You cannot update or confirm a strategy and cannot start search. "
        f"{omission_rule} Recommendations, examples, hypotheticals, questions, and unstated "
        "defaults are not commitments. A biological research topic belongs in "
        "special_themes; requirements without a first-class field belong in "
        "scientific_constraints. For every reported field include an exact source quote "
        f"from the latest message. {output_contract} Do not wrap it in another object and "
        "do not return prose outside JSON.\n\n"
        f"Canonical patch contract: {json.dumps(_DISCOVERY_STRATEGY_PATCH_CONTRACT, ensure_ascii=False)}\n"
        f"Field meanings: {json.dumps(_DISCOVERY_STRATEGY_FIELD_SEMANTICS, ensure_ascii=False)}"
    )
    user_prompt = (
        f"Latest user message: {user_message}\n"
        f"Latest clauses: {json.dumps(_discovery_latest_message_clauses(user_message), ensure_ascii=False)}\n"
        f"Current strategy: {json.dumps(dict(intent_snapshot), ensure_ascii=False)}\n"
        f"Active selected option: {json.dumps(dict(selected_decision), ensure_ascii=False) if isinstance(selected_decision, Mapping) else 'null'}\n"
        f"Proposed patch: {json.dumps(dict(proposed_patch), ensure_ascii=False)}\n"
        "Audit now."
    )
    raw = client.complete_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    if isinstance(raw.get("verification"), Mapping):
        raw = dict(raw["verification"])
    if omission_mode and isinstance(raw.get("candidate_findings"), list):
        candidate_patch: dict[str, Any] = {}
        evidence: list[dict[str, str]] = []
        for finding in raw.get("candidate_findings", [])[:100]:
            if not isinstance(finding, Mapping):
                continue
            raw_field = _clean_text(finding.get("field"))
            field_root = re.split(r"[.\[]", raw_field, maxsplit=1)[0]
            field = _DISCOVERY_STRATEGY_PATCH_ALIASES.get(field_root, field_root)
            source = _clean_text(finding.get("source"))
            if not field or "value" not in finding or not source:
                continue
            finding_value = finding.get("value")
            # A single finding is often serialized as a scalar even when the
            # canonical strategy field is an array.  Treat it as one reported
            # item, not as a different semantic value; the normal validator
            # still canonicalizes and bounds the resulting array.
            if (
                field in _DISCOVERY_STRATEGY_ARRAY_FIELDS
                and finding_value is not None
                and not isinstance(finding_value, list)
            ):
                finding_value = [finding_value]
            normalized, errors = _validate_discovery_strategy_patch(
                {field: finding_value}
            )
            if errors or field not in normalized:
                continue
            value = normalized[field]
            if field in candidate_patch and not _json_values_equal(
                candidate_patch[field], value
            ):
                continue
            candidate_patch[field] = value
            row = {"field": field, "source": source}
            rationale = _clean_text(finding.get("rationale"))[:500]
            if rationale:
                row["rationale"] = rationale
            evidence.append(row)
        raw = {
            **dict(raw),
            "verdict": "repair" if candidate_patch else "reject",
            "patch": candidate_patch,
            "evidence": evidence,
            "findings_contract": "candidate_findings_v1",
        }
    return dict(raw)


def _run_discovery_patch_verifier_agents_sdk(
    client: OpenAICompatibleDiscoveryLLM,
    *,
    user_message: str,
    intent_snapshot: Mapping[str, Any],
    proposed_patch: Mapping[str, Any],
    timeout_seconds: float,
    model: Any | None = None,
    allow_commitment_recovery: bool = False,
    use_update_strategy_tool: bool = False,
    required_fields: set[str] | None = None,
    preserve_unchanged_fields: set[str] | None = None,
    selected_decision: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Use a separate SDK Agent to audit high-risk multi-clause updates once."""

    sdk = _load_discovery_dialogue_agents_sdk()
    caller_supplied_model = model is not None
    owned_async_client: Any | None = None
    if model is None:
        owned_async_client = sdk["AsyncOpenAI"](
            api_key=client.api_key,
            base_url=client.base_url,
            timeout=max(2.0, timeout_seconds),
            max_retries=1,
        )
        model = sdk["OpenAIChatCompletionsModel"](
            model=client.model,
            openai_client=owned_async_client,
            buffer_streamed_tool_calls=True,
        )

    context = SimpleNamespace(verification=None)
    # A verifier is a critic, never a second strategy writer.  Keep the legacy
    # keyword arguments temporarily for call-site compatibility, but do not let
    # them rename this read-only tool or grant mutation authority.
    update_strategy_authority = False
    verifier_tool_name = "verify_strategy_patch"
    verifier_tool_description = (
        "Read-only omission audit of the latest user turn. The Manager proposed no "
        "strategy delta, so identify any explicit user commitments as evidence-grounded "
        "candidate findings. Classify punctuation-delimited clauses independently: a "
        "declarative clause stating the user's intended goal, dataset use, or downstream "
        "task is a commitment even when another clause asks for advice. In that case "
        "verdict=repair is mandatory. Candidate values are repair feedback only: this tool "
        "never writes the strategy, confirms it, or starts search."
        if not proposed_patch
        else (
            "Read-only audit of the Manager's proposed strategy delta. Return an "
            "evidence-grounded view of identical proposed fields only; never add or change "
            "a field, confirm a strategy, or act as update_strategy authority."
        )
    )
    verifier_tool = sdk["function_tool"](
        _sdk_discovery_verify_strategy_patch,
        name_override=verifier_tool_name,
        description_override=verifier_tool_description,
        strict_mode=False,
    )
    recovery_instructions = (
        "You have no write authority. If proposed_patch is non-empty, patch may contain only "
        "identical proposed fields. If proposed_patch is empty but the latest user message "
        "contains concrete commitments the Manager missed, use verdict=repair and place the "
        "evidence-grounded candidate values in patch only as non-authoritative critic findings; "
        "the server will discard those values and ask the user-facing Manager to propose again. "
        "Use verdict=reject with an empty patch only for a genuinely non-committing consultation.\n\n"
    )
    omission_audit_instructions = (
        "OMISSION-AUDIT PRECEDENCE (mandatory because proposed_patch is empty): First classify "
        "every supplied C-id independently; do not classify the whole message only by its final "
        "question or request. A declarative clause in which the user states what they want, plan, "
        "intend, or need to research/build/use the data for is a strategy commitment to objective "
        "and to every canonical downstream task explicitly contained in that clause. A separate "
        "clause asking you to analyze, recommend, explain, or choose the next decision is a "
        "consultation, but it cannot cancel or reclassify the preceding commitment. If at least "
        "one clause is a commitment, verdict=reject is forbidden: return verdict=repair, candidate "
        "values in patch, and an exact source quote per field. Do not add recommended defaults.\n\n"
        if not proposed_patch
        else ""
    )
    instructions = "".join(
        [
        omission_audit_instructions,
        "You are an independent scientific intent verifier for a proteomics discovery Agent. "
        "Review only the latest user message, its punctuation-delimited clauses, the current "
        f"strategy, and the proposed patch. Invoke {verifier_tool_name} exactly once and never "
        "return plain text. Return the complete corrected delta, not merely the changed part of "
        "the proposal. For a non-empty proposal, include no field that is absent from "
        "proposed_patch and do not change a proposed value. For an empty proposal, candidate "
        "patch values are findings only and require exact evidence. Include no recommendation, "
        "example, hypothetical value, or unstated default. A biological study topic belongs in "
        "special_themes rather than only objective. Requirements without a first-class field "
        "belong in scientific_constraints. Use one canonical English organism/taxon label per "
        "intended taxon; never list translations or synonyms as separate species. For every "
        "patch field provide an exact source quote copied from the latest message. A find/review/"
        "delivery instruction is run_horizon, not task_type; never infer browse_only unless the "
        "user explicitly chooses browsing/exploration as the downstream use. A numeric project "
        "target alone is recommended, not fixed, unless the user explicitly says exact/fixed/must. "
        "A user stating their own intended research goal, dataset purpose, or downstream task "
        "is a commitment even when the same sentence asks for analysis or a recommendation. "
        "Do not duplicate already structured fields into notes. Make patch and rationale agree. Use verdict "
        "accept when the proposal is already complete, repair when you correct it, and reject "
        "only when the latest message contains no strategy commitment.\n\n",
        recovery_instructions,
        f"Canonical patch contract: {json.dumps(_DISCOVERY_STRATEGY_PATCH_CONTRACT, ensure_ascii=False)}\n"
        f"Field meanings: {json.dumps(_DISCOVERY_STRATEGY_FIELD_SEMANTICS, ensure_ascii=False)}\n"
        f"Latest user message: {user_message}\n"
        "Active selected option (server-resolved interaction context; null when the latest "
        "message did not select an Agent option): "
        f"{json.dumps(dict(selected_decision), ensure_ascii=False) if isinstance(selected_decision, Mapping) else 'null'}\n"
        "When an active selected option is present, evaluate the proposed patch against that "
        "option's id, label, reason, and target_fields rather than treating a numeric/short "
        "selection_text as context-free. The selection is explicit user authorization for the "
        "option's meaning. Cite the exact selection_text from the latest message as evidence for "
        "each correctly mapped target field.\n"
        f"Latest clauses: {json.dumps(_discovery_latest_message_clauses(user_message), ensure_ascii=False)}\n"
        f"Current strategy: {json.dumps(dict(intent_snapshot), ensure_ascii=False)}\n"
        f"Proposed patch: {json.dumps(dict(proposed_patch), ensure_ascii=False)}",
        ]
    )
    agent = sdk["Agent"][SimpleNamespace](
        name="Discovery Strategy Semantic Verifier",
        instructions=instructions,
        model=model,
        tools=[verifier_tool],
        model_settings=_discovery_tool_required_model_settings(
            sdk,
            model_name=client.model,
        ),
        tool_use_behavior="stop_on_first_tool",
    )

    async def _execute() -> Any:
        try:
            return await sdk["Runner"].run(
                starting_agent=agent,
                input="Audit the latest strategy delta now.",
                context=context,
                max_turns=1,
                run_config=sdk["RunConfig"](
                    workflow_name="proteomics_discovery_strategy_verifier_v1",
                    trace_metadata={"workflow": "discovery_strategy_verifier"},
                    tracing_disabled=True,
                    trace_include_sensitive_data=False,
                ),
            )
        finally:
            if owned_async_client is not None:
                await owned_async_client.close()

    asyncio.run(
        asyncio.wait_for(_execute(), timeout=max(2.0, float(timeout_seconds)))
    )
    raw_verification = (
        context.verification if isinstance(context.verification, Mapping) else {}
    )
    raw_verdict = _clean_text(raw_verification.get("verdict")).lower()
    if (
        (not raw_verification or raw_verdict not in {"accept", "repair", "reject"})
        and not caller_supplied_model
    ):
        try:
            fallback_client = OpenAICompatibleDiscoveryLLM(
                api_key=client.api_key,
                model=client.model,
                base_url=client.base_url,
                timeout=max(2.0, float(timeout_seconds)),
            )
            raw_verification = _run_discovery_patch_verifier_json_fallback(
                fallback_client,
                user_message=user_message,
                intent_snapshot=intent_snapshot,
                proposed_patch=proposed_patch,
                selected_decision=selected_decision,
            )
            raw_verification["provider_compatibility_recovery"] = {
                "mode": "json_object_after_missing_sdk_tool",
                "critic_authority": "read_only",
            }
            raw_verdict = _clean_text(raw_verification.get("verdict")).lower()
        except Exception as exc:
            return {
                "verdict": "unavailable",
                "patch": {},
                "verified": False,
                "rationale": (
                    "Neither the SDK verifier tool nor its bounded JSON compatibility "
                    "fallback produced a valid result."
                ),
                "error": _redact_secrets(str(exc))[:500],
                "tool_authority": "verify_strategy_patch",
                "provider_compatibility_recovery": {
                    "mode": "json_object_after_missing_sdk_tool",
                    "critic_authority": "read_only",
                    "status": "failed",
                },
            }
    if not raw_verification or raw_verdict not in {"accept", "repair", "reject"}:
        # No executed verifier tool is an availability failure, not an
        # authoritative semantic rejection.  The independently validated
        # primary update may degrade through this state; an explicit reject or
        # an evidence-grounding failure below still fails closed.
        return {
            "verdict": "unavailable",
            "patch": {},
            "verified": False,
            "rationale": "The independent verifier did not produce a valid tool result.",
            "tool_authority": (
                "update_strategy"
                if update_strategy_authority
                else "verify_strategy_patch"
            ),
        }
    raw_verification_patch = raw_verification.get("patch")
    tool_authority = (
        "update_strategy" if update_strategy_authority else "verify_strategy_patch"
    )
    critic_suggested_fields: list[str] = []
    if isinstance(raw_verification_patch, Mapping):
        suggested_patch, suggested_errors = _validate_discovery_strategy_patch(
            dict(raw_verification_patch)
        )
        compact_message = re.sub(r"\s+", "", user_message).casefold()
        grounded_suggestion_fields: set[str] = set()
        raw_evidence = raw_verification.get("evidence")
        if isinstance(raw_evidence, list):
            for item in raw_evidence[:100]:
                if not isinstance(item, Mapping):
                    continue
                raw_field = _clean_text(item.get("field"))
                field_root = re.split(r"[.\[]", raw_field, maxsplit=1)[0]
                field = _DISCOVERY_STRATEGY_PATCH_ALIASES.get(field_root, field_root)
                source = re.sub(
                    r"\s+", "", _clean_text(item.get("source"))
                ).casefold()
                if field in suggested_patch and source and source in compact_message:
                    grounded_suggestion_fields.add(field)
        if not suggested_errors:
            critic_suggested_fields = sorted(grounded_suggestion_fields)
    if not proposed_patch and raw_verdict == "repair" and critic_suggested_fields:
        # Findings are audit/repair input only.  The candidate values never
        # cross the mutation boundary; the Dialogue Manager must issue a new
        # update_strategy proposal on its bounded retry.
        return {
            **dict(raw_verification),
            "verdict": "repair",
            "patch": {},
            "verified": False,
            "critic_suggested_fields": critic_suggested_fields,
            "tool_authority": "verify_strategy_patch",
            "rationale": (
                "Read-only critic found evidence-grounded commitments omitted by "
                "the Dialogue Manager; Manager retry required."
            ),
        }
    if (
        allow_commitment_recovery
        and not proposed_patch
        and raw_verdict == "reject"
        and isinstance(raw_verification_patch, Mapping)
        and not raw_verification_patch
    ):
        # In commitment-recovery mode, this is a positive semantic result: the
        # independent update-capable Agent executed its tool and confirmed that
        # the latest message contains no card commitment.
        return {
            **dict(raw_verification),
            "verdict": "accept",
            "patch": {},
            "verified": True,
            "no_commitment_confirmed": True,
            "tool_authority": tool_authority,
            "critic_suggested_fields": critic_suggested_fields,
        }
    grounded_patch = _ground_discovery_patch_verification(
        raw_verification,
        user_message=user_message,
        intent_snapshot=intent_snapshot,
        proposed_patch=proposed_patch,
        allow_commitment_recovery=allow_commitment_recovery,
        preserve_unchanged_fields=preserve_unchanged_fields,
    )
    # Atomic completeness applies to the primary tool delta. Compatible
    # providers sometimes materialize every omitted optional verifier field as
    # JSON null; those schema placeholders are neither user-authored clears nor
    # additional required commitments. Grounded, evidenced recovery fields may
    # still be added to ``grounded_patch``, but ungrounded verifier-only fields
    # must not turn an otherwise complete primary delta into a false rejection.
    atomic_fields = (
        set(proposed_patch)
        if required_fields is None
        else {
            field
            for field in required_fields
            if field in _DISCOVERY_STRATEGY_PATCH_FIELDS
        }
    )
    if raw_verdict == "repair":
        atomic_fields.difference_update(_DISCOVERY_NON_ATOMIC_CONTEXT_FIELDS)
    missing_fields = sorted(atomic_fields.difference(grounded_patch))
    if raw_verdict == "reject" and not grounded_patch:
        # Hard reject with no evidence-grounded fields: wipe the delta.
        return {
            **dict(raw_verification),
            "verdict": "reject",
            "patch": {},
            "verified": False,
            "missing_fields": missing_fields,
            "rationale": (
                _clean_text(raw_verification.get("rationale"))[:1200]
                or "Independent semantic verification rejected the strategy delta."
            ),
            "model_verdict": raw_verdict,
            "tool_authority": tool_authority,
        }
    if missing_fields and not grounded_patch:
        # Accept/repair verdict but nothing evidence-grounded: full incomplete reject.
        return {
            **dict(raw_verification),
            "verdict": "reject",
            "patch": {},
            "verified": False,
            "missing_fields": missing_fields,
            "rationale": (
                "Independent semantic verification did not ground the complete "
                "proposed strategy delta."
            ),
            "model_verdict": raw_verdict,
            "tool_authority": tool_authority,
        }
    if missing_fields and grounded_patch:
        # Partial grounding: apply the evidence-backed subset; surface residual
        # missing fields so the grill turn / UX can report what remains open.
        audit = _normalise_discovery_patch_verification_audit(
            raw_verification,
            user_message=user_message,
            proposed_patch=proposed_patch,
            grounded_patch=grounded_patch,
        )
        return {
            **dict(raw_verification),
            **audit,
            "verdict": "repair",
            "patch": grounded_patch,
            "verified": True,
            "missing_fields": missing_fields,
            "partial_grounding": True,
            "model_verdict": raw_verdict,
            "tool_authority": tool_authority,
            "rationale": (
                "Independent semantic verification grounded a subset of the "
                "primary strategy delta; ungrounded fields were omitted instead "
                "of rejecting the whole update. Missing: "
                + ", ".join(missing_fields)
            ),
        }
    audit = _normalise_discovery_patch_verification_audit(
        raw_verification,
        user_message=user_message,
        proposed_patch=proposed_patch,
        grounded_patch=grounded_patch,
    )
    return {
        **dict(raw_verification),
        **audit,
        "patch": grounded_patch,
        "verified": bool(grounded_patch),
        "tool_authority": tool_authority,
    }


def _complete_discovery_dialogue_json(
    client: Any,
    *,
    system_prompt: str,
    dialogue_history: list[dict[str, str]],
    user_prompt: str,
    user_message: str = "",
    session_id: str = "",
    advisor_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one bounded dialogue turn, preferring the Agents SDK in production."""
    runtime = _clean_text(os.getenv("AGENT_D1_RUNTIME") or "openai_agents").lower()
    if isinstance(client, OpenAICompatibleDiscoveryLLM) and runtime != "direct":
        return _run_discovery_dialogue_agents_sdk(
            client,
            system_prompt=system_prompt,
            dialogue_history=dialogue_history,
            state_prompt=user_prompt,
            user_message=user_message or user_prompt,
            session_id=session_id,
            advisor_context=advisor_context,
        )

    complete_messages = getattr(client, "complete_json_messages", None)
    if not callable(complete_messages):
        raw = client.complete_json(system_prompt=system_prompt, user_prompt=user_prompt)
        return raw if isinstance(raw, dict) else {}

    messages = [
        {"role": "system", "content": system_prompt},
        *dialogue_history,
        {"role": "user", "content": user_prompt},
    ]
    raw = complete_messages(messages=messages)
    return raw if isinstance(raw, dict) else {}


def _resolve_discovery_pending_selection(
    user_message: str,
    pending_decision: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Resolve a reference to an Agent-generated option without interpreting it.

    The decision itself is dynamic and model-owned.  This helper only keeps the
    UI/API interaction lossless: a bare 1-based index, exact option id, or exact
    option label is projected back to the corresponding structured option for
    the next model turn.  New options already carry their Manager-authored,
    schema-validated strategy patch; the later model may not reinterpret it.
    """

    if not isinstance(pending_decision, Mapping):
        return None
    options = pending_decision.get("options")
    if not isinstance(options, list) or not options:
        return None

    selection_text = _clean_text(user_message)
    if not selection_text:
        return None

    selected: Mapping[str, Any] | None = None
    if re.fullmatch(r"\d+", selection_text):
        try:
            option_index = int(selection_text) - 1
        except ValueError:
            option_index = -1
        if 0 <= option_index < len(options) and isinstance(options[option_index], Mapping):
            selected = options[option_index]
    else:
        folded_selection = selection_text.casefold()
        for option in options:
            if not isinstance(option, Mapping):
                continue
            option_id = _clean_text(option.get("id"))
            option_label = _clean_text(option.get("label"))
            if folded_selection in {option_id.casefold(), option_label.casefold()}:
                selected = option
                break

    if selected is None:
        return None

    selected_patch = selected.get("strategy_patch")
    selected_patch = (
        dict(selected_patch) if isinstance(selected_patch, Mapping) else None
    )
    selected_target_fields = (
        list(selected_patch)
        if selected_patch
        else list(pending_decision.get("target_fields") or [])
    )
    return {
        "focus": _clean_text(pending_decision.get("focus")),
        "target_fields": selected_target_fields,
        "question": _clean_text(pending_decision.get("question")),
        "option": dict(selected),
        **({"strategy_patch": selected_patch} if selected_patch else {}),
        "selection_text": selection_text,
        "explicit_acceptance": True,
        "instruction": (
            "The user explicitly accepted this Agent-generated option. "
            "Apply or acknowledge its meaning now and do not ask the same decision again."
        ),
    }


def _discovery_synthesize_option_patch_from_labels(
    selected_decision: Mapping[str, Any],
) -> dict[str, Any]:
    """Best-effort patch when an option omitted strategy_patch (model under-write).

    Only maps clearly labeled choices; never invents unrelated science fields.
    """
    option = selected_decision.get("option")
    if not isinstance(option, Mapping):
        return {}
    blob = " ".join(
        [
            _clean_text(option.get("id")),
            _clean_text(option.get("label")),
            _clean_text(option.get("reason")),
            _clean_text(selected_decision.get("focus")),
            " ".join(
                _clean_text(field)
                for field in (selected_decision.get("target_fields") or [])
                if _clean_text(field)
            ),
        ]
    ).casefold()
    patch: dict[str, Any] = {}
    if any(
        token in blob
        for token in ("label_free", "label-free", "label free", "无标记", "非标记")
    ):
        patch["labeling_strategy"] = "label_free"
        if any(
            token in blob
            for token in ("只要", "仅", "硬", "严格", "排除", "hard", "only")
        ):
            patch["labeling_hard"] = True
        else:
            patch["labeling_hard"] = False
    elif any(
        token in blob
        for token in (
            "不限标记",
            "不限制",
            "任意标记",
            "都可以",
            "open labeling",
            "any labeling",
        )
    ):
        patch["labeling_strategy"] = "unknown"
        patch["labeling_hard"] = False
    elif "tmt" in blob:
        patch["labeling_strategy"] = "TMT"
    elif "itraq" in blob:
        patch["labeling_strategy"] = "iTRAQ"
    if "仪器不限" in blob or (
        ("仪器" in blob or "instrument" in blob)
        and any(token in blob for token in ("不限", "none", "任意", "open"))
    ):
        patch["instrument_preference"] = "none"
    if not patch:
        return {}
    normalized, errors = _validate_discovery_strategy_patch(patch)
    return {} if errors else normalized


def _discovery_selected_option_strategy_patch(
    selected_decision: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return the immutable mutation contract authored with a UI option.

    The user may accept an option by index, exact id, or exact label.  Those
    short references contain no standalone field semantics, so the only safe
    authority is the schema-validated patch persisted with that option when the
    Manager created it.  A later model call may explain or plan the next step,
    but it cannot widen or reinterpret this patch.

    If the Manager omitted strategy_patch on the option (under-write), synthesize
    a minimal patch from clear option labels so bare "1" still updates the card.
    """

    if not isinstance(selected_decision, Mapping):
        return {}
    option = selected_decision.get("option")
    if not isinstance(option, Mapping):
        return {}
    raw_patch = option.get("strategy_patch")
    if not isinstance(raw_patch, Mapping):
        raw_patch = selected_decision.get("strategy_patch")
    if isinstance(raw_patch, Mapping) and raw_patch:
        patch, errors = _validate_discovery_strategy_patch(_json_object(dict(raw_patch)))
        if not errors and patch:
            return patch
    return _discovery_synthesize_option_patch_from_labels(selected_decision)


def _same_discovery_decision(
    left: Mapping[str, Any] | None,
    right: Mapping[str, Any] | None,
) -> bool:
    """Return whether two dynamic decisions have the same focus and option ids."""

    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return False
    left_focus = _clean_text(left.get("focus")).casefold()
    right_focus = _clean_text(right.get("focus")).casefold()

    def _option_ids(decision: Mapping[str, Any]) -> tuple[str, ...]:
        raw_options = decision.get("options")
        if not isinstance(raw_options, list):
            return ()
        return tuple(
            _clean_text(option.get("id")).casefold()
            for option in raw_options
            if isinstance(option, Mapping) and _clean_text(option.get("id"))
        )

    left_ids = _option_ids(left)
    right_ids = _option_ids(right)
    if not left_ids or not right_ids:
        return False

    same_options = frozenset(left_ids) == frozenset(right_ids)
    same_focus = bool(left_focus) and left_focus == right_focus
    left_fields = {
        _clean_text(field)
        for field in left.get("target_fields") or []
        if _clean_text(field)
    }
    right_fields = {
        _clean_text(field)
        for field in right.get("target_fields") or []
        if _clean_text(field)
    }
    if left_fields or right_fields:
        # target_fields is the semantic identity. This preserves harmless
        # focus paraphrases while preventing generic option ids from making two
        # unrelated card dimensions collide.
        return same_options and left_fields == right_fields
    return same_options and same_focus


def _normalise_discovery_decision_memory(raw: Any) -> list[dict[str, Any]]:
    items = raw if isinstance(raw, list) else []
    memory: list[dict[str, Any]] = []

    def _identity(
        record: Mapping[str, Any],
    ) -> tuple[tuple[str, ...], tuple[str, ...], str]:
        option_identity = tuple(
            sorted(
                _clean_text(value).casefold()
                for value in record.get("option_ids") or []
                if _clean_text(value)
            )
        )
        field_identity = tuple(
            sorted(
                _clean_text(value)
                for value in record.get("target_fields") or []
                if _clean_text(value)
            )
        )
        fallback_focus = (
            "" if field_identity else _clean_text(record.get("focus")).casefold()
        )
        return option_identity, field_identity, fallback_focus

    for item in items[-50:]:
        if not isinstance(item, Mapping):
            continue
        option_ids = item.get("option_ids")
        if not isinstance(option_ids, list):
            continue
        cleaned_ids = list(
            dict.fromkeys(
                _clean_text(value)
                for value in option_ids[:_DISCOVERY_DECISION_MAX_OPTIONS]
                if _clean_text(value)
            )
        )
        if len(cleaned_ids) < 2:
            continue
        focus = _clean_text(item.get("focus"))
        target_fields = _normalise_discovery_decision_target_fields(
            item.get("target_fields"),
            focus=focus,
            option_ids=cleaned_ids,
        )
        record = {
            "focus": focus,
            "target_fields": target_fields,
            "option_ids": cleaned_ids,
            "selected_option_id": _clean_text(item.get("selected_option_id")),
            "selected_option_label": _clean_text(item.get("selected_option_label")),
        }
        raw_selected_values = item.get("selected_values")
        if isinstance(raw_selected_values, Mapping):
            selected_values = {
                field: _json_safe(raw_selected_values.get(field))
                for field in target_fields
                if field in raw_selected_values
            }
            if selected_values:
                record["selected_values"] = selected_values
        signature = _identity(record)
        memory = [
            previous
            for previous in memory
            if _identity(previous) != signature
        ]
        memory.append(record)
    return memory


def _discovery_resolved_decision_record(
    pending_decision: Mapping[str, Any] | None,
    selected_decision: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(pending_decision, Mapping) or not isinstance(selected_decision, Mapping):
        return None
    option = selected_decision.get("option")
    options = pending_decision.get("options")
    if not isinstance(option, Mapping) or not isinstance(options, list):
        return None
    option_ids = [
        _clean_text(item.get("id"))
        for item in options
        if isinstance(item, Mapping) and _clean_text(item.get("id"))
    ]
    if len(option_ids) < 2:
        return None
    selected_patch = _discovery_selected_option_strategy_patch(selected_decision)
    return {
        "focus": _clean_text(pending_decision.get("focus")),
        "target_fields": (
            list(selected_patch)
            if selected_patch
            else list(pending_decision.get("target_fields") or [])
        ),
        "option_ids": option_ids,
        "selected_option_id": _clean_text(option.get("id")),
        "selected_option_label": _clean_text(option.get("label")),
    }


def _discovery_explicit_resolution_patch(
    selected_decision: Mapping[str, Any] | None,
    *,
    validated_tool_patch: Mapping[str, Any],
    commitment_patch: Mapping[str, Any] | None,
    intent_snapshot: Mapping[str, Any],
    requested_action: str,
    agent_runtime: str,
) -> dict[str, Any]:
    """Return unchanged target values that carry an explicit resolution.

    This is deliberately narrower than a normal strategy patch.  It requires
    the authorities that distinguish an intentional open/default choice from a
    model echo: an SDK-executed ``update_strategy`` call, a schema-valid field
    whose submitted value equals the live card, and either an active
    Agent-authored option selected by the user or a separately grounded
    clause-level commitment for that field.  The latter keeps arbitrary
    free-text answers first-class without introducing phrase-specific parsers.

    The returned keys are decision-state deltas and may be sent to the client
    so ``resolved_fields`` advances even though the scientific value itself is
    unchanged.
    """

    if (
        requested_action != "update_strategy"
        or agent_runtime != "openai_agents"
        or (
            not isinstance(selected_decision, Mapping)
            and not isinstance(commitment_patch, Mapping)
        )
    ):
        return {}
    target_fields = {
        _DISCOVERY_STRATEGY_PATCH_ALIASES.get(field, field)
        for raw_field in selected_decision.get("target_fields") or []
        if (field := _clean_text(raw_field))
    } if isinstance(selected_decision, Mapping) else set()
    return {
        field: validated_tool_patch[field]
        for field in validated_tool_patch
        if (
            field in intent_snapshot
            and validated_tool_patch[field] == intent_snapshot.get(field)
            and (
                field in target_fields
                or (
                    isinstance(commitment_patch, Mapping)
                    and field in commitment_patch
                    and commitment_patch[field] == validated_tool_patch[field]
                )
            )
        )
    }


def _discovery_selected_option_was_applied(
    selected_decision: Mapping[str, Any] | None,
    *,
    effective_patch: Mapping[str, Any],
    validated_tool_patch: Mapping[str, Any],
    intent_snapshot: Mapping[str, Any],
    requested_action: str,
    mutation_valid: bool,
) -> bool:
    """Return whether a referenced option actually resolved its card fields.

    A numeric selection only identifies an Agent-authored option. It does not
    resolve that decision until a validated update emits a target-field delta,
    or explicitly repeats a target value already equal to the live card.
    """

    if not isinstance(selected_decision, Mapping) or not mutation_valid:
        return False
    target_fields = {
        _clean_text(field)
        for field in selected_decision.get("target_fields") or []
        if _clean_text(field)
    }
    if not target_fields or requested_action != "update_strategy":
        return False
    if target_fields.intersection(effective_patch):
        return True
    return any(
        field in validated_tool_patch
        and field in intent_snapshot
        and validated_tool_patch[field] == intent_snapshot.get(field)
        for field in target_fields
    )


def _normalise_discovery_resolved_fields(
    raw: Any,
    intent_snapshot: Mapping[str, Any],
) -> set[str]:
    values = raw if isinstance(raw, list) else intent_snapshot.get("resolved_fields")
    values = values if isinstance(values, list) else []
    resolved: set[str] = set()
    for value in values:
        name = _clean_text(value)
        canonical = _DISCOVERY_STRATEGY_PATCH_ALIASES.get(name, name)
        if canonical in _DISCOVERY_STRATEGY_PATCH_FIELDS:
            resolved.add(canonical)
    return resolved


_DISCOVERY_UNSET_SENTINELS: dict[str, set[Any]] = {
    "species_policy": {"open"},
    "species_coverage": {"none"},
    "acquisition_mode": {"unknown"},
    "mixed_acquisition_policy": {"review_mixed"},
    "labeling_strategy": {"unknown", "any"},
    "quota_flexibility": {"recommended"},
    "on_safety_ceiling": {"ask"},
    "instrument_preference": {"none"},
    "repository": {"pride"},
}


def _discovery_strategy_field_is_resolved(
    field: str,
    intent_snapshot: Mapping[str, Any],
    resolved_fields: set[str],
) -> bool:
    if field in resolved_fields:
        return True
    if field not in intent_snapshot:
        return False
    value = intent_snapshot.get(field)
    if value is None or value == "" or value == [] or value == {}:
        return False
    try:
        if value in _DISCOVERY_UNSET_SENTINELS.get(field, set()):
            return False
    except TypeError:
        pass
    return True


def _discovery_decision_was_resolved(
    decision: Mapping[str, Any] | None,
    decision_memory: list[dict[str, Any]],
    *,
    intent_snapshot: Mapping[str, Any],
    resolved_fields: set[str],
) -> bool:
    if not isinstance(decision, Mapping) or decision.get("revisit_existing") is True:
        return False
    option_ids = {
        _clean_text(option.get("id")).casefold()
        for option in decision.get("options") or []
        if isinstance(option, Mapping) and _clean_text(option.get("id"))
    }
    target_fields = {
        _clean_text(field)
        for field in decision.get("target_fields") or []
        if _clean_text(field)
    }
    focus = _clean_text(decision.get("focus")).casefold()
    for previous in decision_memory:
        previous_ids = {
            _clean_text(value).casefold()
            for value in previous.get("option_ids") or []
            if _clean_text(value)
        }
        previous_fields = {
            _clean_text(value)
            for value in previous.get("target_fields") or []
            if _clean_text(value)
        }
        if target_fields and previous_fields and target_fields == previous_fields:
            selected_values = previous.get("selected_values")
            if isinstance(selected_values, Mapping) and target_fields.issubset(
                selected_values
            ):
                return all(
                    field in intent_snapshot
                    and intent_snapshot.get(field) == selected_values.get(field)
                    for field in target_fields
                )
            # Backward-compatible records written before selected values were
            # persisted are trusted only while every target field is still
            # materially set. Clearing a field reopens the decision.
            return all(
                _discovery_strategy_field_is_resolved(
                    field,
                    intent_snapshot,
                    resolved_fields,
                )
                for field in target_fields
            )
        previous_focus = _clean_text(previous.get("focus")).casefold()
        if (
            not target_fields
            and not previous_fields
            and option_ids
            and option_ids == previous_ids
            and focus
            and focus == previous_focus
        ):
            return True
    return bool(target_fields) and all(
        _discovery_strategy_field_is_resolved(
            field,
            intent_snapshot,
            resolved_fields,
        )
        for field in target_fields
    )


def _filter_discovery_decision_memory_for_snapshot(
    decision_memory: list[dict[str, Any]],
    *,
    intent_snapshot: Mapping[str, Any],
    resolved_fields: set[str],
) -> list[dict[str, Any]]:
    """Discard remembered choices whose target fields were later changed."""

    current: list[dict[str, Any]] = []
    for record in decision_memory:
        target_fields = {
            _clean_text(field)
            for field in record.get("target_fields") or []
            if _clean_text(field)
        }
        if not target_fields:
            current.append(record)
            continue
        selected_values = record.get("selected_values")
        if isinstance(selected_values, Mapping) and target_fields.issubset(
            selected_values
        ):
            if all(
                field in intent_snapshot
                and intent_snapshot.get(field) == selected_values.get(field)
                for field in target_fields
            ):
                current.append(record)
            continue
        if all(
            _discovery_strategy_field_is_resolved(
                field,
                intent_snapshot,
                resolved_fields,
            )
            for field in target_fields
        ):
            current.append(record)
    return current


def _repair_discovery_selected_enum_patch(
    raw: Mapping[str, Any],
    selected_decision: Mapping[str, Any] | None,
    intent_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Repair only the shape of an explicit model tool event for a UI choice.

    Some OpenAI-compatible models wrap a selected scalar as
    ``{"value": ..., "label": ...}``, which correctly fails the normal patch
    validator.  When (and only when) the model already requested exactly one
    ``update_strategy`` tool call, an accepted option id may still identify one
    unique enum field in the public strategy schema.  Reconstructing that one
    scalar delta is schema-driven; it does not infer vocabulary or authorize a
    card write without the model's explicit tool event.
    """

    if not isinstance(selected_decision, Mapping):
        return {}
    if _clean_text(raw.get("action") or raw.get("mode")).lower() != "update_strategy":
        return {}
    raw_calls = raw.get("tool_calls")
    if not isinstance(raw_calls, list):
        return {}
    update_calls = [
        call
        for call in raw_calls
        if isinstance(call, Mapping)
        and _clean_text(call.get("name") or call.get("tool")) == "update_strategy"
    ]
    if len(update_calls) != 1:
        return {}

    option = selected_decision.get("option")
    if not isinstance(option, Mapping):
        return {}
    option_id = _clean_text(option.get("id")).lower()
    if not option_id:
        return {}

    focus = _clean_text(selected_decision.get("focus"))
    canonical_focus = _DISCOVERY_STRATEGY_PATCH_ALIASES.get(focus, focus)
    candidate_fields: list[str] = []
    if (
        canonical_focus in _DISCOVERY_STRATEGY_ENUM_FIELDS
        and option_id in _DISCOVERY_STRATEGY_ENUM_FIELDS[canonical_focus]
    ):
        candidate_fields.append(canonical_focus)
    for field, allowed_values in _DISCOVERY_STRATEGY_ENUM_FIELDS.items():
        if option_id in allowed_values and field not in candidate_fields:
            candidate_fields.append(field)
    if len(candidate_fields) != 1:
        return {}

    repaired, errors = _validate_discovery_strategy_patch(
        {candidate_fields[0]: option_id}
    )
    if errors:
        return {}
    return _drop_unchanged_discovery_patch_fields(repaired, intent_snapshot)


def _discovery_latest_message_clauses(user_message: str) -> list[dict[str, str]]:
    """Expose generic punctuation-delimited clauses as a completeness aid.

    This does not interpret vocabulary or mutate state. It only prevents a
    long, coordinated user message from being treated as one opaque sentence
    by smaller chat models.
    """

    pieces = [
        piece.strip()
        for piece in re.split(r"(?<=[,，;；。.!?！？])\s*|[\r\n]+", user_message)
        if piece.strip()
    ]
    if not pieces:
        pieces = [user_message]
    return [
        {"id": f"C{index}", "text": piece}
        for index, piece in enumerate(pieces[:30], start=1)
    ]


def _discovery_grill_turn_system_prompt() -> str:
    return (
        "You are a proteomics notebook partner for PRIDE data discovery — a scientific "
        "collaborator who takes notes on a strategy card via tools, not a form wizard or "
        "questionnaire clerk. "
        "NOTEBOOK DEFAULT: read the full dialogue plus the latest user message; extract every "
        "explicit scientific commitment; when there is one or more commitments, call "
        "update_strategy once with all of them in a single patch; when there is none, use "
        "action=chat or advise and leave the card unchanged. "
        "Latest user intent wins: corrections overwrite earlier card fields. "
        "MULTI-COMMITMENT FIRST: compound Chinese/English messages that already state several "
        "concrete choices (species, task/downstream use, scale, acquisition, horizon, themes, "
        "etc.) MUST become one update_strategy patch — never a multi-step questionnaire, and "
        "never re-ask a value the user just gave. "
        "Ask only about true remaining gaps that still affect search/ranking/feasibility and "
        "that the user has not covered; prefer one natural-language follow-up sentence. "
        "A next_decision option menu is optional, useful only when discrete alternatives help; "
        "never emit a 2–8 option menu when free chat or a single free-text ask is enough; never "
        "turn a greeting or pure consultation into a menu. "
        "There is no fixed question order. gap_report, critical_decision_agenda, and pending "
        "questions are readiness guidance, not a script. Reason by meaning, not keyword lists. "
        "Only an explicit update_strategy tool event may change the card (SDK tool authority). "
        "Prose, extra_fields, and advice never write the card. Pure advice stays advice until "
        "accepted. Natural-language approval uses confirm_strategy only when "
        "phase=awaiting_confirm for the current snapshot. A generic ok/yes during grilling is "
        "not confirmation. You never start PRIDE discovery; grill_confirmed=true remains the "
        "server gate. capability-honest: this surface can plan_only (without repository access), "
        "candidates_only, candidates_reviewed; ai_ready_table, pre_release, and full_release "
        "need a reviewed result and a separate executor—never imply confirm silently runs those. "
        "Reply in natural Chinese and always return one JSON object matching the turn "
        "contract in the user message.\n\n"
        "Repository scientific guidance:\n"
        f"{_discovery_agent_guidance()}"
    )


def _deterministic_discovery_search_term_extension(
    user_message: str,
    intent_snapshot: Mapping[str, Any],
) -> list[str] | None:
    """Parse an explicit ordered repository-term extension without an LLM.

    This intentionally handles only high-confidence additive commands. Other
    edits (replacement, deletion, or ambiguous scientific prose) continue
    through the normal Agent boundary.
    """

    message = _clean_text(user_message)
    if not message or not re.search(r"(?:检索|搜索).{0,6}词", message):
        return None
    if not re.search(r"(?:扩充|新增|追加|添加|加入|补充)", message):
        return None
    if re.search(r"(?:替换|改为|只用|删除|移除)", message):
        return None

    sections: list[str] = []
    additions = re.search(
        r"(?:并\s*)?(?:新增|追加|添加|加入|补充)\s*[:：]\s*(.+?)"
        r"(?=(?:[。；;\n]\s*)?最后(?:再)?(?:使用|加入|添加)|$)",
        message,
        flags=re.DOTALL,
    )
    if additions:
        sections.append(additions.group(1))
    broad = re.search(
        r"最后(?:再)?(?:使用|加入|添加)\s*[:：]?\s*(.+)$",
        message,
        flags=re.DOTALL,
    )
    if broad:
        sections.append(broad.group(1))
    if not sections:
        return None

    extracted: list[str] = []
    for section in sections:
        for raw_term in re.split(r"[,，、;；\n]+", section):
            term = raw_term.strip().strip("`'\"“”‘’（）()[]{}。.!！?？")
            term = re.sub(
                r"\s*(?:进行|用于|作为)\s*(?:最后一层|宽泛)?(?:补漏|检索|搜索).*$",
                "",
                term,
                flags=re.IGNORECASE,
            ).strip()
            if not term or len(term) > _DISCOVERY_STRATEGY_ARRAY_ITEM_MAX_CHARS:
                continue
            extracted.append(term)
    if not extracted:
        return None

    current = intent_snapshot.get("selected_search_terms")
    current_terms = current if isinstance(current, list) else []
    ordered: list[str] = []
    seen: set[str] = set()
    for raw_term in [*current_terms, *extracted]:
        if not isinstance(raw_term, str):
            continue
        term = re.sub(r"\s+", " ", raw_term).strip()
        key = term.casefold()
        if not term or key in seen:
            continue
        seen.add(key)
        ordered.append(term)
    if len(ordered) > _DISCOVERY_STRATEGY_ARRAY_MAX_ITEMS:
        return None
    return ordered


def _deterministic_discovery_search_term_turn(
    body: Mapping[str, Any],
    *,
    user_message: str,
) -> dict[str, Any] | None:
    intent_snapshot = (
        _json_object(body.get("intent_snapshot"))
        if isinstance(body.get("intent_snapshot"), Mapping)
        else {}
    )
    terms = _deterministic_discovery_search_term_extension(
        user_message,
        intent_snapshot,
    )
    if terms is None:
        return None
    patch, errors = _validate_discovery_strategy_patch(
        {"selected_search_terms": terms}
    )
    if errors or not patch:
        return None

    current_terms = intent_snapshot.get("selected_search_terms")
    current_count = len(current_terms) if isinstance(current_terms, list) else 0
    added_count = max(0, len(terms) - current_count)
    assistant_message = (
        f"已按原顺序保留现有检索词，并追加 {added_count} 个检索词；"
        f"当前共 {len(terms)} 个。确认前仍可继续调整，尚未访问 PRIDE。"
    )
    session_id = _normalise_discovery_session_id(body.get("session_id"))
    decision_memory = _normalise_discovery_decision_memory(
        body.get("decision_memory")
    )
    result = {
        "action": "update_strategy",
        "mode": "update_strategy",
        "assistant_message": assistant_message,
        "tool_calls": [
            {"name": "update_strategy", "arguments": {"patch": patch}}
        ],
        "gap_report": _normalise_discovery_gap_report(body.get("gap_report")),
        "intent": "revise",
        "advance": True,
        "answer_text": "",
        "extra_fields": patch,
        "understanding": "Explicit ordered repository search terms were appended.",
        "next_focus": None,
        "ready_for_confirm": False,
        "phase": _clean_text(body.get("phase") or "grilling").lower() or "grilling",
        "pending_question_id": "",
        "strategy_fingerprint": "",
        "status": "completed",
        "parser": "deterministic_search_terms",
        "agent_runtime": "deterministic",
        "llm_used": False,
        "request_budget_seconds": 0.0,
        "decision_memory": decision_memory,
        "decision_agenda": [],
        "session_id": session_id,
    }
    _store_discovery_dialogue_session_turn(
        session_id,
        user_message=user_message,
        assistant_message=assistant_message,
        action="update_strategy",
        patch=patch,
        next_decision=None,
        resolved_decision=None,
    )
    return result


def _run_discovery_grill_turn(body: dict[str, Any]) -> dict[str, Any]:
    """Run one model-owned D1 turn; structural validation is deterministic."""
    turn_started_at = monotonic()
    try:
        manager_repair_attempt = max(0, int(body.get("_manager_repair_attempt") or 0))
    except (TypeError, ValueError):
        manager_repair_attempt = 0
    manager_repair_feedback = (
        _json_object(body.get("_manager_repair_feedback"))
        if isinstance(body.get("_manager_repair_feedback"), Mapping)
        else {}
    )
    user_message = _clean_text(body.get("user_message") or body.get("prompt"))
    if not user_message:
        raise ValueError("Please enter a message.")
    deterministic_turn = _deterministic_discovery_search_term_turn(
        body,
        user_message=user_message,
    )
    if deterministic_turn is not None:
        return deterministic_turn

    llm_config = body.get("llm_config") if isinstance(body.get("llm_config"), dict) else {}
    client = _discovery_llm_client(
        llm_config,
        allow_server_default=body.get("allow_server_default") is not False,
    )
    if client is None:
        raise ValueError(
            "No discovery LLM API key found. Fill API Configuration or set DEEPSEEK_API_KEY."
        )
    request_budget_seconds = _bind_discovery_turn_request_budget(client, body)

    phase = _clean_text(body.get("phase") or "grilling").lower() or "grilling"
    pending = body.get("pending_question") if isinstance(body.get("pending_question"), dict) else None
    raw_pending_decision = body.get("pending_decision")
    pending_decision = _normalise_discovery_next_decision(raw_pending_decision)
    intent_snapshot = (
        _json_object(body.get("intent_snapshot"))
        if isinstance(body.get("intent_snapshot"), dict)
        else {}
    )
    intent_snapshot["run_horizon"] = _FIXED_DISCOVERY_RUN_HORIZON
    decision_memory = _normalise_discovery_decision_memory(
        body.get("decision_memory")
    )
    resolved_fields = _normalise_discovery_resolved_fields(
        body.get("resolved_fields"),
        intent_snapshot,
    )
    resolved_fields.add("run_horizon")
    pending_decision = _scope_discovery_next_decision_to_unresolved_fields(
        pending_decision,
        resolved_fields,
    )
    selected_decision = _resolve_discovery_pending_selection(
        user_message,
        pending_decision,
    )
    candidate_resolved_decision = _discovery_resolved_decision_record(
        pending_decision,
        selected_decision,
    )
    resolved_decision: dict[str, Any] | None = None
    answered = body.get("answered") if isinstance(body.get("answered"), dict) else {}
    turn_kind = _clean_text(body.get("turn_kind") or "answer").lower() or "answer"
    local_summary = _clean_text(body.get("local_summary"))
    input_gap_report = (
        body.get("gap_report") if isinstance(body.get("gap_report"), dict) else {}
    )
    session_id = _normalise_discovery_session_id(body.get("session_id"))
    fallback_history = _normalise_discovery_dialogue_history(body.get("dialogue_history"))
    dialogue_history, session_decision_memory = _load_discovery_dialogue_session(
        session_id,
        fallback_history,
    )
    decision_memory = _normalise_discovery_decision_memory(
        [*session_decision_memory, *decision_memory]
    )
    decision_memory = _filter_discovery_decision_memory_for_snapshot(
        decision_memory,
        intent_snapshot=intent_snapshot,
        resolved_fields=resolved_fields,
    )
    critical_decision_agenda = _discovery_critical_decision_agenda(
        intent_snapshot,
        input_gap_report,
        resolved_fields,
    )
    confirmation_eligible, confirmation_reason, strategy_fingerprint = (
        _discovery_confirmation_context(
            body,
            phase=phase,
            intent_snapshot=intent_snapshot,
            gap_report=input_gap_report,
        )
    )

    pending_block = "(none - first message or no open question)"
    if pending:
        options = pending.get("options") if isinstance(pending.get("options"), list) else []
        option_lines: list[str] = []
        for index, option in enumerate(options, start=1):
            if not isinstance(option, Mapping):
                continue
            tag = " [recommended]" if option.get("recommended") else ""
            option_lines.append(
                f"{index}. id={_clean_text(option.get('id'))} "
                f"label={_clean_text(option.get('label'))}{tag}"
            )
        pending_block = (
            f"id: {_clean_text(pending.get('id'))}\n"
            f"prompt: {_clean_text(pending.get('prompt'))}\n"
            f"why: {_clean_text(pending.get('why'))}\n"
            "options:\n"
            + ("\n".join(option_lines) if option_lines else "(none)")
        )
    pending_decision_block = (
        json.dumps(pending_decision, ensure_ascii=False, indent=2)
        if pending_decision is not None
        else "(none)"
    )
    selected_decision_block = (
        json.dumps(selected_decision, ensure_ascii=False, indent=2)
        if selected_decision is not None
        else "(none)"
    )

    confirmation_context = {
        "eligible": confirmation_eligible,
        "reason_if_ineligible": confirmation_reason,
        "strategy_fingerprint": strategy_fingerprint,
        "rule": (
            "confirm_strategy is allowed only for an unambiguous approval of this exact "
            "snapshot while phase=awaiting_confirm; it never starts search itself"
        ),
    }
    field_semantics = json.dumps(
        _DISCOVERY_STRATEGY_FIELD_SEMANTICS,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    enum_catalog = json.dumps(
        {field: sorted(values) for field, values in _DISCOVERY_STRATEGY_ENUM_FIELDS.items()},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    latest_clauses = _discovery_latest_message_clauses(user_message)
    contract_noise_clarification = (
        _discovery_user_asks_clarification(user_message)
        and _discovery_history_has_contract_noise(dialogue_history)
    )
    contract_noise_hint = (
        _discovery_contract_noise_clarification_hint()
        if contract_noise_clarification
        else ""
    )
    user_prompt = (
        "Handle one notebook dialogue turn for proteomics data discovery (chat freely, extract commitments, tool-write the card, ask only real gaps).\n\n"
        f"turn_kind: {turn_kind}\n"
        f"phase: {phase}\n"
        f"user_message: {user_message}\n\n"
        f"{contract_noise_hint}"
        "latest_message_clauses (generic completeness aid; inspect every C-id):\n"
        f"{json.dumps(latest_clauses, ensure_ascii=False, indent=2)}\n\n"
        "Critical latest-turn scope rule: a 'discuss only / do not update' instruction inside an older history turn ended with that turn unless explicitly declared persistent. The latest user_message is a new turn; a latest acceptance, adoption, replacement, correction, open, or clear is actionable now.\n\n"
        "Critical clause rule: apply each concrete choice in the latest message even when another clause asks for explanation or says not to decide the remaining fields. Never ask the user to reconfirm a value they just chose. A keep-unchanged/as-is instruction for a dimension has priority over incidental values mentioned while asking why, comparing, or quoting prior advice; omit that dimension from the patch.\n\n"
        f"current_intent_snapshot:\n"
        f"{json.dumps(intent_snapshot, ensure_ascii=False, indent=2)}\n\n"
        f"answered_flags:\n{json.dumps(answered, ensure_ascii=False)}\n\n"
        f"pending_question:\n{pending_block}\n\n"
        f"active_agent_decision:\n{pending_decision_block}\n\n"
        f"selected_agent_option:\n{selected_decision_block}\n\n"
        "resolved_decision_memory (authoritative: do not ask these again unless the latest user explicitly reopens one):\n"
        f"{json.dumps(decision_memory, ensure_ascii=False, indent=2)}\n\n"
        "explicitly_resolved_strategy_fields (an open/unknown value may be an intentional answer):\n"
        f"{json.dumps(sorted(resolved_fields), ensure_ascii=False)}\n\n"
        "gap_report (guidance only; multi-fill / compound patch is preferred when the user packed multiple commitments):\n"
        f"{json.dumps(input_gap_report, ensure_ascii=False, indent=2)}\n\n"
        "critical_decision_agenda (apply any items already answered in latest_user_message via update_strategy; only ask about remaining blockers; not a fixed questionnaire):\n"
        f"{json.dumps(critical_decision_agenda, ensure_ascii=False, indent=2)}\n\n"
        "read_only_critic_feedback_from_previous_attempt (findings only; the Manager must independently re-propose any justified patch):\n"
        f"{json.dumps(manager_repair_feedback, ensure_ascii=False, indent=2) if manager_repair_feedback else '(none)'}\n\n"
        "confirmation_context (authoritative structural precondition):\n"
        f"{json.dumps(confirmation_context, ensure_ascii=False, indent=2)}\n\n"
        "local_rule_summary (optional helper, may be incomplete):\n"
        f"{local_summary or '(none)'}\n\n"
        "Return one JSON object with this primary shape:\n"
        "{\n"
        '  "action": "chat|advise|clarify|update_strategy|ready_to_confirm|'
        'confirm_strategy|refuse_search",\n'
        '  "assistant_message": "natural Chinese reply",\n'
        '  "turn_interpretation": {\n'
        '    "commitments": [{"field": "canonical field", "value": "canonical value", "source": "short exact quote from latest message"}],\n'
        '    "consultations": ["brief non-committing clause"],\n'
        '    "clause_audit": [{"clause_id": "C1", "classification": "commitment|consultation|procedural", "decisions": [{"field": "canonical field", "value": "canonical value"}]}],\n'
        '    "prior_turn_only_instructions_expired": true\n'
        "  },\n"
        '  "tool_calls": [{"name": "update_strategy", '
        '"arguments": {"patch": {"supported_field": "replacement value"}}}],\n'
        '  "next_decision": {\n'
        '    "focus": "one semantic focus",\n'
        '    "target_fields": ["one or more canonical strategy fields"],\n'
        '    "question": "one personalized scientific decision",\n'
        '    "recommendation": {"id": "id", "label": "label", "reason": "short reason"},\n'
        '    "options": [{"id": "id", "label": "label", "reason": "optional", '
        '"strategy_patch": {"canonical field": "the complete value authorized by this option"}}],\n'
        '    "option_mode": "focused|expanded",\n'
        '    "revisit_existing": false,\n'
        '    "allow_free_text": true\n'
        "  },\n"
        '  "gap_report": {"required_missing": [], "optional_missing": [], '
        '"ready_for_confirm": false},\n'
        '  "ready_for_confirm": false,\n'
        '  "intent": "optional legacy projection",\n'
        '  "answer_text": "optional legacy answer",\n'
        '  "understanding": "short current understanding"\n'
        "}\n\n"
        "Rules:\n"
        "- Choose the action from meaning and the current science context. There is no fixed question order.\n"
        "- Only action=update_strategy plus exactly one update_strategy tool call with an explicit object at arguments.patch may change the card. extra_fields, prose, numbers, ontologies, and lexical examples are never mutation authority.\n"
        "- When SDK tools are available, invoke update_strategy or confirm_strategy as a real function tool. Never simulate a tool call only by writing tool_calls in final JSON; the server injects the calls actually executed by the SDK.\n"
        "- The patch is a generic schema delta: include every and only choice the user establishes, accepts, replaces, opens, excludes, clears, or explicitly asks you to default.\n"
        "- A user commitment to one dimension authorizes only that dimension. For example, naming a research topic does not authorize your recommended task, species, quantity, coverage, acquisition, or labeling defaults. State such recommendations as advice and wait for acceptance. A multi-field patch is allowed only when the latest message establishes each field or explicitly accepts a previously presented bundle/default.\n"
        "- Before choosing the action, split the latest message into clauses. For each clause decide consultation versus commitment, resolve its subject and negation scope, and map every commitment to its canonical field. Check that every committed field appears exactly once in the patch and that no uncommitted field appears.\n"
        "- Populate turn_interpretation before deriving action and tool_calls. The set of canonical fields in commitments must equal the set of patch fields (one commitment may legitimately map to two tightly coupled fields). Every source must quote the latest user_message, not the snapshot or old history. If commitments is non-empty, action must be update_strategy.\n"
        "- clause_audit must contain every C-id from latest_message_clauses exactly once. A commitment clause must repeat every affected canonical field and value in decisions; a consultation/procedural clause uses decisions=[]. This deliberate redundancy is required. After clause_audit, scan the complete canonical field list once more before emitting the tool call.\n"
        "- A direct adoption, replacement, correction, opening, exclusion, or clear is already authorization. Do not ask the user to confirm that individual field again, even if you think another value would be scientifically better; apply it and put the caveat in the same reply.\n"
        "- Interpret elliptical answers in the active planning context. A short topic, organism, method, labeling family, quantity, or similar choice is a commitment when it naturally answers or advances the current plan, even with conversational hedges such as '吧' or '左右'; only a real question, hypothetical, quotation, or explicit uncertainty stays consultation.\n"
        "- A user stating their own intended research goal, dataset purpose, or downstream task (including first-person desire/plan language such as wanting, planning, or intending to do X) establishes objective/task commitments. A simultaneous request for your analysis, recommendation, or next-step advice does not cancel those concrete commitments: update them first and then advise.\n"
        "- Never say that you recorded, set, changed, or remembered a strategy value in assistant_message unless you actually invoke update_strategy for that value in the same turn.\n"
        "- Scope words locally. An instruction to explain or avoid deciding the remaining recommendations does not cancel a concrete choice made in another clause. Likewise, keep-one-field-unchanged does not block changes to other named fields.\n"
        "- A value mentioned inside a why/how/comparison question, a quotation, an example, or a reference to prior advice is not a commitment by itself. If the same message says that dimension stays unchanged, no patch for that dimension is allowed.\n"
        "- Do not silently repair a scientific inconsistency by clearing or rewriting fields the user did not change. Apply the requested delta, preserve the existing potentially conflicting field, and surface the conflict as one next decision or open risk. Scientific advice never creates mutation authority.\n"
        "- A previous-turn instruction to discuss only or not write the card expires after that turn unless the user explicitly made it persistent. A latest-turn acceptance such as choosing a referenced proposal is a new commitment.\n"
        "- A consultation, comparison, feasibility question, hypothetical, greeting, or capability question does not update strategy. Recommendations remain advice until accepted.\n"
        "- A pure social greeting uses action=chat and next_decision=null. Reply naturally and invite free-form discussion; do not turn a greeting into a structured menu.\n"
        "- Resolve references from native dialogue roles. Prior context disambiguates a commitment but does not authorize unrelated fields.\n"
        "- When selected_agent_option is present, the user has explicitly accepted that option from the active Agent-generated decision. You must use action=update_strategy and invoke update_strategy; do not use chat/advise/respond. Apply its meaning now. For an intentional open/no-change option, explicitly submit the relevant target field with its canonical open/default/empty value: this is a resolution delta even when the value already matches the card. Never repeat the same focus and option set in next_decision.\n"
        "- For an accepted option whose active focus is a canonical strategy field and whose option id is a valid field value, write the scalar delta as {focus: option.id}. Never wrap a patch value in an object containing value/label/reason. If the choice intentionally changes several fields, emit each canonical scalar/array value directly.\n"
        "- Defaults use action=update_strategy. If the resulting card is executable, also set ready_for_confirm=true; this offers confirmation but never confirms or starts search.\n"
        "- Every first-class strategy field may be null to clear it back to the empty-strategy default. Arrays may also use []. Query/runtime extension fields are cleared by omission, not null.\n"
        "- run_horizon is a fixed system invariant: it is always candidates_reviewed. Never ask the user where the run should stop, never offer plan_only/candidates_only/downstream horizons, and never clear or change this field.\n"
        f"- Canonical patch contract: "
        f"{json.dumps(_DISCOVERY_STRATEGY_PATCH_CONTRACT, ensure_ascii=False, separators=(',', ':'))}.\n"
        f"- Canonical field meanings: {field_semantics}.\n"
        f"- Canonical enum option catalog: {enum_catalog}. Use it as a capability catalog, not a fixed questionnaire; retain free text and notes for meaningful values outside first-class fields.\n"
        "- Preserve every meaningful requirement without a dedicated field in scientific_constraints. "
        "Choose a stable id, explicit hard/soft strength, project/file/sample/portfolio scope, operator, "
        "JSON value, and evidence_required=true. notes is context only and must never be the sole home of "
        "an actionable requirement. This rule is generic: do not wait for a vocabulary-specific branch.\n"
        "- next_decision must include target_fields (one or more canonical strategy fields) and revisit_existing. Set revisit_existing=true only when the latest user explicitly asks to reconsider an already-set choice.\n"
        "- Every next_decision option must include a non-empty, schema-valid strategy_patch that completely expresses only that option's mutation meaning. All options in one menu must use this contract. target_fields is derived by the server from these patches and is not mutation authority. A later numeric/id/label selection applies exactly the stored option patch; the later model must not add defaults or reinterpret it.\n"
        "- critical_decision_agenda lists readiness blockers, NOT a forced one-by-one quiz. If the latest user_message already resolves one or more agenda items, apply them via update_strategy in THIS turn and do not re-ask them.\n"
        "- COMPOUND UPDATES ARE THE DEFAULT when the user packs multiple commitments into one message (including Chinese separators · / ， / 、 / 和 / 以及 / 逗号). Map every explicit commitment into one update_strategy.patch. Example: '人源免疫肽，RT 预测，越多越好，DDA' should set species/human+prefer or include_only, special_themes or objective for immunopeptide, task_type=rt_prediction, quota open_ended or exhaustive coverage, and acquisition_mode=dda in ONE patch. The server supplies the fixed candidates_reviewed horizon.\n"
        "- Do NOT invent downstream tasks the user did not state (e.g. do not write task_type=rt_prediction when they only said immunopeptide data). Objective/theme may record the topic; task_type only when a modeling/use task is explicit or they accept a recommendation.\n"
        "- After applying all explicit commitments, if critical_decision_agenda still lists unresolved critical items (critical=true), do not go silent: surface the highest-priority remaining gap. Prefer one natural-language follow-up in assistant_message (and next_decision=null when free-text is enough). Emit a next_decision menu only when discrete alternatives genuinely help the user choose; if you do emit one, it must be schema-complete (question + recommendation.reason + 2-8 options each with strategy_patch). Species/generalization_scope remains a real readiness gap for training tasks (denovo/RT/etc.) when species is empty—ask it in notebook language; do not claim it is optional without user input.\n"
        "- Prefer ready_to_confirm only when no critical agenda item remains. Non-critical items (e.g. labeling) may be skipped or asked briefly.\n"
        "- Menus are for unresolved blockers, not a fixed questionnaire. After a compound multi-commit answer, do not re-ask resolved fields; do ask the next critical gap. option_mode=expanded only if they ask for alternatives/comparison.\n"
        "- When you emit next_decision, it must be schema-complete (question, recommendation.reason, 2-8 options each with strategy_patch); incomplete menus are dropped by the server. Critical gaps may be followed up with natural language alone—do not invent a menu just to satisfy schema.\n"
        "- Keep next_decision.question to one concise question when present. Do not embed option lists in assistant_message; the UI renders options separately. Use assistant_message to confirm what was written to the card and what (if anything) is still open.\n"
        "- Give option ids stable semantic ids when you use menus. After an option is selected, move on; never rephrase the same menu.\n"
        "- ready_to_confirm only presents the current strategy for approval; it is not user approval.\n"
        "- For natural-language approval, return action=confirm_strategy only when confirmation_context.eligible=true and the user unambiguously approves that exact current strategy. Words such as yes/ok/好的/可以 during grilling or while answering a recommendation are not confirmation. Phrases like 直接确认可搜 / 可以搜了 / 开始搜 after a complete card still require awaiting_confirm eligibility.\n"
        "- confirm_strategy never starts PRIDE. A separate caller may set grill_confirmed=true only after consuming this decision; the backend discovery-start gate remains authoritative.\n"
        "- If asked to search before confirmation, use refuse_search and explain the one remaining confirmation dependency.\n"
        "- During grilling PRIDE has not been queried. Do not invent availability, project counts, repository composition, or metadata coverage.\n"
        "- There is no repository evidence in this turn. The assistant_message must not claim that a project count, availability statement, or repository composition is known from PRIDE. If asked about one, label it as an unverified expectation and say discovery has not checked it yet.\n"
        "- Immunopeptidomics is not a PTM task. Downstream browse-only does not imply a small search scale. Preserve explicit all/exhaustive/no-limit language as exhaustive + open_ended, and never inject curated/20 from the downstream-task choice.\n"
        "- Keep prose consistent with the action and tool patch.\n"
    )

    try:
        raw = _complete_discovery_dialogue_json(
            client,
            system_prompt=_discovery_grill_turn_system_prompt(),
            dialogue_history=dialogue_history,
            user_prompt=user_prompt,
            user_message=user_message,
            session_id=session_id,
            advisor_context={
                "user_message": user_message,
                "intent_snapshot": intent_snapshot,
                "critical_decision_agenda": critical_decision_agenda,
                "gap_report": input_gap_report,
                "resolved_decision_memory": decision_memory,
            },
        )
    except Exception as exc:
        raise ValueError(f"Grill dialogue LLM failed: {_redact_secrets(str(exc))}") from exc
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Grill dialogue LLM returned an invalid object response.")
    agent_runtime = _clean_text(raw.pop("_agent_runtime", "")) or "direct_json_adapter"
    provider_compatibility_recovery = (
        _json_object(raw.pop("_provider_compatibility_recovery"))
        if isinstance(raw.get("_provider_compatibility_recovery"), Mapping)
        else {}
    )
    advisor_calls = raw.pop("_advisor_calls", [])
    if not isinstance(advisor_calls, list):
        advisor_calls = []
    # Backward-compatible marker only; the server always owns final session
    # persistence so raw SDK tool output can never become dialogue memory.
    raw.pop("_sdk_session_managed", False)

    raw_intent = _clean_text(raw.get("intent")).lower()
    allowed_legacy_intents = {
        "answer_question",
        "explain",
        "clarify",
        "multi_fill",
        "revise",
        "request_defaults",
        "request_confirm",
        "refuse_search",
        "chitchat",
    }
    if raw_intent not in allowed_legacy_intents:
        raw_intent = ""

    requested_action = _clean_text(raw.get("action") or raw.get("mode")).lower()
    raw_next_decision = raw.get("next_decision")
    next_decision = _normalise_discovery_next_decision(raw_next_decision)
    resolved_scope_removed_next_decision = False
    decision_contract_error = raw_next_decision is not None and next_decision is None
    if requested_action == "chat":
        # A social/casual turn stays conversational. Structured choices are
        # reserved for an actual scientific decision, not appended to greetings.
        # NI-1: incomplete next_decision on chat is never contract chrome.
        next_decision = None
        decision_contract_error = False
    repeated_selected_decision = bool(
        selected_decision
        and _same_discovery_decision(pending_decision, next_decision)
    )
    selected_option_contract_patch = _discovery_selected_option_strategy_patch(
        selected_decision
    )
    option_resolution_audit: dict[str, Any] | None = None
    if selected_option_contract_patch:
        # A numbered/exact option reply accepts the immutable patch authored in
        # the previous Manager turn.  The current model is still useful for the
        # acknowledgement and next scientific decision, but it is not allowed
        # to widen that earlier option (the production failure was
        # build_training silently adding run_horizon=plan_only).
        original_action = requested_action
        original_patch = _discovery_explicit_tool_patch(
            raw,
            intent_snapshot={},
            include_unchanged=True,
        )
        discarded_fields = sorted(
            field
            for field in original_patch
            if field not in selected_option_contract_patch
            or not _json_values_equal(
                original_patch[field], selected_option_contract_patch.get(field)
            )
        )
        raw["action"] = "update_strategy"
        raw["tool_calls"] = [
            {
                "name": "update_strategy",
                "arguments": {"patch": dict(selected_option_contract_patch)},
            }
        ]
        # Replace the same-model self-audit as well. Otherwise a hallucinated
        # extra field in that redundant channel could veto the safe option
        # contract before deterministic validation sees it.
        raw["turn_interpretation"] = {
            "commitments": [
                {
                    "field": field,
                    "value": _json_safe(value),
                    "source": user_message,
                }
                for field, value in selected_option_contract_patch.items()
            ],
            "consultations": [],
        }
        requested_action = "update_strategy"
        option = selected_decision.get("option") if selected_decision else None
        option_resolution_audit = {
            "contract": "predeclared_v1",
            "selected_option_id": (
                _clean_text(option.get("id")) if isinstance(option, Mapping) else ""
            ),
            "authorized_patch": _json_safe(selected_option_contract_patch),
            "model_action_overridden": original_action != "update_strategy",
            "discarded_model_fields": discarded_fields,
        }
    patch, mutation_errors = _discovery_turn_patch(
        raw,
        user_message=user_message,
        intent_snapshot=intent_snapshot,
    )
    selected_enum_repair = {}
    if selected_decision is not None and mutation_errors:
        selected_enum_repair = _repair_discovery_selected_enum_patch(
            raw,
            selected_decision,
            intent_snapshot,
        )
        if selected_enum_repair:
            patch = selected_enum_repair
            mutation_errors = []
    if (
        agent_runtime == "openai_agents"
        and requested_action == "update_strategy"
        and mutation_errors
        and mutation_errors != ["update_strategy patch contains no changed fields"]
        and manager_repair_attempt < 1
        and selected_decision is None
    ):
        remaining_seconds = request_budget_seconds - (monotonic() - turn_started_at)
        if remaining_seconds >= 6.0:
            retry_body = dict(body)
            retry_body["_manager_repair_attempt"] = manager_repair_attempt + 1
            retry_body["_manager_repair_feedback"] = {
                "kind": "invalid_strategy_patch",
                "validation_errors": [
                    _redact_secrets(_clean_text(error))[:500]
                    for error in mutation_errors[:12]
                    if _clean_text(error)
                ],
                "instruction": (
                    "The previous update_strategy arguments failed the deterministic "
                    "strategy schema. Re-read every latest-message clause and invoke "
                    "update_strategy once with a complete, canonical, schema-valid "
                    "patch. Correct only the invalid envelope; do not drop the user's "
                    "other explicit commitments and do not add recommendations."
                ),
            }
            retry_body["request_timeout_seconds"] = min(
                remaining_seconds,
                request_budget_seconds,
            )
            return _run_discovery_grill_turn(retry_body)
    commitment_authority_patch, commitment_authority_errors = (
        _discovery_turn_commitment_patch(raw, user_message=user_message)
    )
    if commitment_authority_errors:
        commitment_authority_patch = None
    if agent_runtime == "openai_agents":
        patch = _drop_uncommitted_discovery_null_placeholders(
            patch,
            commitment_authority_patch,
        )
    validated_tool_patch = _discovery_explicit_tool_patch(
        raw,
        intent_snapshot=intent_snapshot,
        include_unchanged=True,
    )
    explicit_resolution_patch = _discovery_explicit_resolution_patch(
        selected_decision,
        validated_tool_patch=validated_tool_patch,
        commitment_patch=commitment_authority_patch,
        intent_snapshot=intent_snapshot,
        requested_action=requested_action,
        agent_runtime=agent_runtime,
    )
    selected_target_fields = {
        _DISCOVERY_STRATEGY_PATCH_ALIASES.get(field, field)
        for raw_field in (selected_decision or {}).get("target_fields") or []
        if (field := _clean_text(raw_field))
    }
    # Predeclared UI option acceptance (bare "1" / exact id / exact label):
    # server already holds the Manager-authored strategy_patch. Never re-run the
    # semantic verifier against the bare selection text (rejects "1" as empty).
    selected_option_patch_is_scoped = bool(
        isinstance(selected_decision, Mapping)
        and selected_decision.get("explicit_acceptance") is True
        and bool(selected_option_contract_patch)
    )
    if selected_option_patch_is_scoped and not mutation_errors:
        # Force the card write to the immutable option contract only.
        patch = dict(selected_option_contract_patch)
        if explicit_resolution_patch:
            for key, value in explicit_resolution_patch.items():
                patch.setdefault(key, value)
    # Legacy/direct dialogue mode can still acknowledge an already-equal
    # selected value without emitting a client resolution delta.  Production
    # SDK turns use ``explicit_resolution_patch`` below so resolved_fields is
    # persisted as part of the strategy event.
    selected_value_already_applied = bool(explicit_resolution_patch) or any(
        field in validated_tool_patch
        and field in intent_snapshot
        and validated_tool_patch[field] == intent_snapshot.get(field)
        for field in selected_target_fields
    )
    if (
        selected_value_already_applied
        and mutation_errors == ["update_strategy patch contains no changed fields"]
    ):
        # This is a decision-state delta even though it is not a value delta.
        # Keeping the explicit keys lets the client mark open/default values as
        # resolved and prevents the same question from reappearing.
        patch = dict(explicit_resolution_patch)
        mutation_errors = []
    elif explicit_resolution_patch and not mutation_errors:
        patch.update(explicit_resolution_patch)
    explicit_tool_patch = _discovery_explicit_tool_patch(
        raw,
        intent_snapshot=intent_snapshot,
    )
    if not mutation_errors:
        explicit_tool_patch.update(explicit_resolution_patch)
    if agent_runtime == "openai_agents":
        explicit_tool_patch = _drop_uncommitted_discovery_null_placeholders(
            explicit_tool_patch,
            commitment_authority_patch,
        )
    if (
        agent_runtime == "openai_agents"
        and explicit_tool_patch
        and mutation_errors == ["update_strategy patch contains no changed fields"]
    ):
        # A partial optional interpretation may have filtered every SDK field.
        # Keep the structurally valid tool delta eligible for the independent
        # completeness review instead of treating that compatibility audit as
        # authoritative.
        mutation_errors = []
    verification_input_patch = dict(explicit_tool_patch)
    verification_input_patch.update(patch)
    # Complete under-specified Manager dumps on packed multi-commitment turns
    # (local flake: model writes only soft keys; user already stated species/DDA/scale).
    # LC-B write path: after compound fill, verification_input_patch and patch both
    # become filled so the low-risk skip branch can apply the full card write without
    # a second verifier pass or soft-reject under-write.
    if requested_action == "update_strategy" and not mutation_errors:
        filled = _merge_discovery_compound_commitment_hints(
            verification_input_patch,
            user_message,
            intent_snapshot=intent_snapshot,
        )
        if filled and filled != verification_input_patch:
            verification_input_patch = dict(filled)
            # Always promote filled to the apply patch (not only when patch ⊆ filled).
            # Commitment-filtered soft subsets must not win over deterministic
            # compound recovery of hard whitelist fields (species/DDA/horizon/quota).
            patch = dict(filled)
            explicit_tool_patch = {
                **dict(explicit_tool_patch),
                **{
                    field: value
                    for field, value in filled.items()
                    if field not in explicit_tool_patch
                },
            }
    # The primary tool may include model-authored convenience text that its
    # independent clause audit correctly did not classify as a user
    # commitment.  Keep every reconciled commitment atomic, while allowing the
    # semantic verifier to remove such uncommitted extras.  If the primary
    # audit is absent, fail closed and require the complete typed tool delta.
    verification_required_fields = (
        set(patch)
        if commitment_authority_patch is not None
        else set(verification_input_patch)
    )
    tool_interpretation_difference = bool(
        requested_action == "update_strategy" and explicit_tool_patch != patch
    )
    semantic_verification: dict[str, Any] | None = None
    # A separate semantic Agent is read-only. It may flag a Manager omission,
    # after which the same user-facing Manager gets one bounded retry. The
    # critic's candidate values are never applied directly.
    commitment_recovery_warranted = bool(
        agent_runtime == "openai_agents"
        and manager_repair_attempt < 1
        and selected_decision is None
        and not patch
        and not mutation_errors
        and requested_action in {"chat", "advise", "clarify"}
        # Short utterances can be complete scientific commitments (for example
        # one topic, organism, acquisition mode, or quantity).  The old
        # character-count gate silently excluded exactly those natural Agent
        # turns.  Skip the omission auditor only for an explicitly classified
        # social chat turn produced through the normal tool contract; provider
        # plain-text recovery is always audited because it has no action tool.
        and (
            bool(provider_compatibility_recovery)
            or requested_action != "chat"
            or raw_intent != "chitchat"
        )
    )
    low_risk_single_field_skip = _discovery_low_risk_single_field_verifier_skip(
        verification_input_patch,
        tool_interpretation_difference=tool_interpretation_difference,
        provider_compatibility_recovery=provider_compatibility_recovery,
    )
    if (
        low_risk_single_field_skip
        and verification_input_patch
        and not mutation_errors
        and requested_action == "update_strategy"
    ):
        # Low-risk skip write:
        # - No commitment audit → apply full verification dump (tool ∪ compound fill).
        # - With commitment audit → keep reconciled ``patch`` (drops ungrounded tool
        #   keys) and only union keys that compound hints added beyond the tool dump.
        # This recovers soft under-writes without resurrecting uncommitted tool fields.
        if commitment_authority_patch is None:
            apply_patch = dict(verification_input_patch)
        else:
            compound_extras = {
                field: value
                for field, value in verification_input_patch.items()
                if field not in explicit_tool_patch
            }
            apply_patch = {**dict(patch), **compound_extras}
        patch = _drop_unchanged_discovery_patch_fields(
            apply_patch,
            intent_snapshot,
            preserve_fields=set(explicit_resolution_patch),
        )
    # Thin SV warrant (notebook NB-2 / NI-2): do NOT force critic merely because
    # Chinese punctuation produced multiple clauses or the patch has several
    # pure low-risk keys. The low-risk helper already encodes the skip matrix
    # (whitelist subset, no scientific_constraints, compound max, plain-text
    # recovery, single-field interpretation gap). When that helper is False,
    # the patch is high-risk and SV stays warranted.
    patch_verification_warranted = bool(
        agent_runtime == "openai_agents"
        and verification_input_patch
        and not mutation_errors
        and not (
            explicit_resolution_patch
            and set(verification_input_patch).issubset(explicit_resolution_patch)
        )
        # A bare option index is not context-free text: the server has already
        # resolved it against exactly one active Agent-authored decision.  When
        # the executed SDK tool is confined to that decision's target fields,
        # re-running a stateless semantic Agent can only lose context.  Schema,
        # action, and field-scope validation above remain authoritative; any
        # out-of-scope field still takes the verifier/rejection path.
        and not selected_option_patch_is_scoped
        # Low-risk whitelist tool patches (single field or small compound dump,
        # max _DISCOVERY_LOW_RISK_COMPOUND_MAX_FIELDS) skip the second verifier.
        # scientific_constraints / non-whitelist fields / tool-interpretation gaps
        # still force the critic (helper returns False → this gate stays open).
        and not low_risk_single_field_skip
    )
    verification_warranted = bool(
        patch_verification_warranted or commitment_recovery_warranted
    )
    if verification_warranted:
        previous_attempts: list[dict[str, Any]] = []
        attempts = 0
        while attempts < _DISCOVERY_SEMANTIC_VERIFIER_MAX_ATTEMPTS:
            remaining_seconds = request_budget_seconds - (
                monotonic() - turn_started_at
            )
            if remaining_seconds < 4.0:
                semantic_verification = {
                    "verified": False,
                    "verdict": "budget_exhausted",
                    "patch": {},
                    "rationale": (
                        "The primary turn and verifier attempts consumed the "
                        "bounded dialogue budget."
                    ),
                }
                break
            attempts += 1
            try:
                semantic_verification = _run_discovery_patch_verifier_agents_sdk(
                    client,
                    user_message=user_message,
                    intent_snapshot=intent_snapshot,
                    proposed_patch=verification_input_patch,
                    timeout_seconds=min(
                        remaining_seconds,
                        _DISCOVERY_SEMANTIC_VERIFIER_ATTEMPT_SECONDS,
                    ),
                    allow_commitment_recovery=False,
                    use_update_strategy_tool=False,
                    required_fields=verification_required_fields,
                    preserve_unchanged_fields=set(explicit_resolution_patch),
                    selected_decision=selected_decision,
                )
            except Exception as exc:
                # The primary typed SDK tool remains usable when both bounded
                # reviewer attempts are unavailable.  Expose the failure
                # instead of freezing the conversation or using phrase rules.
                semantic_verification = {
                    "verified": False,
                    "verdict": "unavailable",
                    "patch": {},
                    "rationale": "Independent semantic verification was unavailable.",
                    "error": _redact_secrets(str(exc))[:500],
                }
            verification_verdict = _clean_text(
                semantic_verification.get("verdict")
            ).lower()
            if verification_verdict != "unavailable":
                break
            if attempts < _DISCOVERY_SEMANTIC_VERIFIER_MAX_ATTEMPTS:
                previous_attempts.append(
                    {
                        "verdict": "unavailable",
                        "rationale": _clean_text(
                            semantic_verification.get("rationale")
                        )[:500],
                        **(
                            {
                                "error": _clean_text(
                                    semantic_verification.get("error")
                                )[:500]
                            }
                            if _clean_text(semantic_verification.get("error"))
                            else {}
                        ),
                    }
                )
        if semantic_verification is not None:
            semantic_verification["attempts"] = attempts
            if previous_attempts:
                semantic_verification["previous_attempts"] = previous_attempts
        critic_suggested_fields = (
            semantic_verification.get("critic_suggested_fields")
            if isinstance(semantic_verification, Mapping)
            else None
        )
        if (
            commitment_recovery_warranted
            and not critic_suggested_fields
            and isinstance(semantic_verification, Mapping)
            and _clean_text(semantic_verification.get("verdict")).lower()
            in {"repair", "reject"}
        ):
            # Some verifier providers return the same read-only omission finding
            # as evidence + missing_fields instead of critic_suggested_fields.
            # Derive field *names* only; values remain exclusively authored by
            # the retried Dialogue Manager.
            missing = {
                _DISCOVERY_STRATEGY_PATCH_ALIASES.get(
                    _clean_text(field),
                    _clean_text(field),
                )
                for field in semantic_verification.get("missing_fields") or []
                if _clean_text(field)
            }
            evidence_fields: list[str] = []
            for item in semantic_verification.get("evidence") or []:
                if not isinstance(item, Mapping):
                    continue
                raw_field = _clean_text(item.get("field"))
                field = _DISCOVERY_STRATEGY_PATCH_ALIASES.get(
                    raw_field,
                    raw_field,
                )
                if (
                    field in _DISCOVERY_STRATEGY_PATCH_FIELDS
                    and (not missing or field in missing)
                    and field not in evidence_fields
                ):
                    evidence_fields.append(field)
            if evidence_fields:
                critic_suggested_fields = evidence_fields
                semantic_verification["critic_suggested_fields"] = evidence_fields
                semantic_verification["suggested_fields_derived_from_evidence"] = True
        if (
            commitment_recovery_warranted
            and isinstance(critic_suggested_fields, list)
            and critic_suggested_fields
        ):
            remaining_seconds = request_budget_seconds - (
                monotonic() - turn_started_at
            )
            if remaining_seconds >= 6.0:
                retry_body = dict(body)
                retry_body["_manager_repair_attempt"] = manager_repair_attempt + 1
                retry_body["_manager_repair_feedback"] = {
                    "kind": "omitted_commitment",
                    "suggested_fields": [
                        _clean_text(field)
                        for field in critic_suggested_fields
                        if _clean_text(field)
                    ],
                    "evidence": _json_safe(
                        semantic_verification.get("evidence") or []
                    ),
                    "instruction": (
                        "Re-read every clause. If the critic finding is justified, the "
                        "Dialogue Manager must invoke update_strategy with all and only "
                        "the user's commitments. The critic has no write authority."
                    ),
                    **(
                        {
                            "provider_compatibility_recovery": (
                                provider_compatibility_recovery
                            )
                        }
                        if provider_compatibility_recovery
                        else {}
                    ),
                }
                retry_body["request_timeout_seconds"] = min(
                    remaining_seconds,
                    request_budget_seconds,
                )
                return _run_discovery_grill_turn(retry_body)
        if semantic_verification is not None and attempts:
            verified_patch = semantic_verification.get("patch")
            verification_verdict = _clean_text(
                semantic_verification.get("verdict")
            ).lower()
            if isinstance(verified_patch, Mapping):
                canonical_verified_patch, verified_patch_errors = (
                    _validate_discovery_strategy_patch(dict(verified_patch))
                )
                if verified_patch_errors:
                    canonical_verified_patch = {}
                critic_overreach_fields = sorted(
                    field
                    for field, value in canonical_verified_patch.items()
                    if field not in verification_input_patch
                    or not _json_values_equal(
                        value, verification_input_patch.get(field)
                    )
                )
                verified_patch = {
                    field: value
                    for field, value in canonical_verified_patch.items()
                    if field not in critic_overreach_fields
                }
                semantic_verification["patch"] = verified_patch
                if critic_overreach_fields:
                    semantic_verification["critic_overreach_fields"] = (
                        critic_overreach_fields
                    )
            effective_required_fields = set(verification_required_fields)
            if verification_verdict == "repair":
                effective_required_fields.difference_update(
                    _DISCOVERY_NON_ATOMIC_CONTEXT_FIELDS
                )
            missing_verified_fields = (
                sorted(effective_required_fields.difference(verified_patch))
                if isinstance(verified_patch, Mapping)
                else sorted(effective_required_fields)
            )
            if (
                semantic_verification.get("verified") is True
                and verification_verdict in {"accept", "repair"}
                and missing_verified_fields
                and isinstance(verified_patch, Mapping)
                and verified_patch
            ):
                # Multi-field free-text strategy dumps often ground most, but not
                # every, field under a strict independent critic. Prefer the
                # grounded subset over wiping the whole card (L1 UX: partial
                # strategy apply beats "no change" + opaque reject).
                normalized_partial, partial_errors = (
                    _validate_discovery_strategy_patch(dict(verified_patch))
                )
                if partial_errors or not normalized_partial:
                    semantic_verification = {
                        **semantic_verification,
                        "verified": False,
                        "verdict": "reject",
                        "patch": {},
                        "missing_fields": missing_verified_fields,
                        "rationale": (
                            "Independent semantic verification did not ground the "
                            "complete primary strategy delta."
                        ),
                    }
                    patch = {}
                    mutation_errors.append(
                        "semantic verification returned an incomplete strategy patch: "
                        + ", ".join(missing_verified_fields)
                    )
                else:
                    semantic_verification = {
                        **semantic_verification,
                        "verified": True,
                        "verdict": "repair",
                        "patch": normalized_partial,
                        "missing_fields": missing_verified_fields,
                        "partial_grounding": True,
                        "rationale": (
                            "Independent semantic verification grounded a subset of "
                            "the primary strategy delta; ungrounded fields were "
                            "omitted instead of rejecting the whole update. Missing: "
                            + ", ".join(missing_verified_fields)
                        ),
                    }
                    patch = _drop_unchanged_discovery_patch_fields(
                        normalized_partial,
                        intent_snapshot,
                        preserve_fields=set(explicit_resolution_patch),
                    )
            elif (
                semantic_verification.get("verified") is True
                and verification_verdict in {"accept", "repair"}
                and missing_verified_fields
            ):
                semantic_verification = {
                    **semantic_verification,
                    "verified": False,
                    "verdict": "reject",
                    "patch": {},
                    "missing_fields": missing_verified_fields,
                    "rationale": (
                        "Independent semantic verification did not ground the "
                        "complete primary strategy delta."
                    ),
                }
                patch = {}
                mutation_errors.append(
                    "semantic verification returned an incomplete strategy patch: "
                    + ", ".join(missing_verified_fields)
                )
            elif (
                semantic_verification.get("verified") is True
                and verification_verdict in {"accept", "repair"}
                and isinstance(verified_patch, Mapping)
                and verified_patch
                and (
                    not commitment_recovery_warranted
                    or semantic_verification.get("tool_authority") == "update_strategy"
                )
            ):
                normalized_verified_patch, verified_errors = (
                    _validate_discovery_strategy_patch(dict(verified_patch))
                )
                if verified_errors or not normalized_verified_patch:
                    patch = {}
                    mutation_errors.append(
                        "semantic verification returned an invalid strategy patch"
                    )
                else:
                    removed_uncommitted_fields = sorted(
                        set(verification_input_patch).difference(verified_patch)
                    )
                    if removed_uncommitted_fields:
                        semantic_verification["removed_uncommitted_fields"] = (
                            removed_uncommitted_fields
                        )
                    patch = _drop_unchanged_discovery_patch_fields(
                        normalized_verified_patch,
                        intent_snapshot,
                        preserve_fields=set(explicit_resolution_patch),
                    )
            elif commitment_recovery_warranted and verification_verdict == "reject":
                # For a non-mutating primary turn, reject means the independent
                # Agent confirmed that the latest message was consultation, not
                # that a user-authored strategy patch failed.
                semantic_verification = {
                    **semantic_verification,
                    "verified": True,
                    "verdict": "accept",
                    "patch": {},
                    "no_commitment_confirmed": True,
                }
            elif verification_verdict not in {"unavailable", "budget_exhausted"}:
                # Soft-reject v2 (notebook NI-2): start from explicit SDK tool
                # patch only. Field-level critic errors keep low-risk whitelist
                # keys not named by the critic; global reject keeps soft set only.
                soft_kept = (
                    _discovery_soft_reject_kept_patch(
                        explicit_tool_patch,
                        semantic_verification=semantic_verification,
                    )
                    if (
                        not commitment_recovery_warranted
                        and explicit_tool_patch
                        and requested_action == "update_strategy"
                    )
                    else {}
                )
                if soft_kept:
                    patch = _drop_unchanged_discovery_patch_fields(
                        soft_kept,
                        intent_snapshot,
                        preserve_fields=set(explicit_resolution_patch),
                    )
                    soft_dropped = _discovery_soft_reject_dropped_fields(
                        explicit_tool_patch, patch
                    )
                    semantic_verification = {
                        **semantic_verification,
                        "verified": False,
                        "verdict": "reject",
                        "patch": dict(patch),
                        "soft_reject_kept_fields": sorted(patch),
                        "soft_reject_dropped_fields": soft_dropped,
                        "rationale": (
                            _clean_text(semantic_verification.get("rationale"))[:1200]
                            or (
                                "Independent semantic verification rejected some "
                                "fields; evidence-eligible tool fields were retained."
                            )
                        ),
                    }
                else:
                    patch = {}
                    if not commitment_recovery_warranted:
                        mutation_errors.append(
                            "semantic verification rejected or could not ground the strategy patch"
                        )
            elif patch_verification_warranted and explicit_tool_patch:
                # The typed SDK function call remains the mutation authority
                # when the bounded independent reviewer is unavailable. Never
                # degrade to a subset created only by an optional audit.
                patch = dict(explicit_tool_patch)
        else:
            semantic_verification = {
                "verified": False,
                "verdict": "budget_exhausted",
                "patch": {},
                "rationale": "The primary turn consumed the bounded dialogue budget.",
            }
    # Catch hard-reject paths that wiped the delta earlier (incomplete grounding
    # forced to reject, invalid verified patch, etc.): still retain soft keys
    # from the primary tool call so the card is not blanked on short topics.
    if (
        agent_runtime == "openai_agents"
        and isinstance(semantic_verification, Mapping)
        and _clean_text(semantic_verification.get("verdict")).lower() == "reject"
        and not patch
        and not commitment_recovery_warranted
        and explicit_tool_patch
        and requested_action == "update_strategy"
        and semantic_verification.get("soft_reject_kept_fields") is None
    ):
        soft_kept = _discovery_soft_reject_kept_patch(
            explicit_tool_patch,
            semantic_verification=semantic_verification,
        )
        if soft_kept:
            patch = _drop_unchanged_discovery_patch_fields(
                soft_kept,
                intent_snapshot,
                preserve_fields=set(explicit_resolution_patch),
            )
            soft_dropped = _discovery_soft_reject_dropped_fields(
                explicit_tool_patch, patch
            )
            semantic_verification = {
                **semantic_verification,
                "verified": False,
                "verdict": "reject",
                "patch": dict(patch),
                "soft_reject_kept_fields": sorted(patch),
                "soft_reject_dropped_fields": soft_dropped,
                "rationale": (
                    _clean_text(semantic_verification.get("rationale"))[:1200]
                    or (
                        "Independent semantic verification rejected some fields; "
                        "evidence-eligible tool fields were retained."
                    )
                ),
            }
            mutation_errors = [
                error
                for error in mutation_errors
                if "semantic verification" not in _clean_text(error).casefold()
            ]
    suppressed_uncommitted_fields: list[str] = []
    # A successfully executed SDK function call plus the field-generic schema
    # contract is the normal-path mutation authority.  Do not run that semantic
    # decision through the old vocabulary-hint filter: doing so silently
    # dropped perfectly valid phrasings such as an instrument-era preference or
    # a reviewed-candidate horizon unless they matched a small phrase list.
    # When the model supplies a grounded turn_interpretation,
    # _discovery_turn_patch already reconciles the tool patch against it.  When
    # a provider omits that optional audit, keep the validated SDK tool delta
    # rather than replacing model reasoning with keyword rules.
    patch_was_reconciled = bool(patch) and patch != explicit_tool_patch
    partial_grounding_applied = bool(
        isinstance(semantic_verification, Mapping)
        and semantic_verification.get("partial_grounding") is True
        and patch
        and not mutation_errors
    )
    soft_reject_applied = bool(
        isinstance(semantic_verification, Mapping)
        and semantic_verification.get("soft_reject_kept_fields")
        and patch
        and not mutation_errors
    )
    contract_errors = list(mutation_errors)
    blocking_contract_error = bool(mutation_errors)
    action = _normalise_discovery_turn_action(
        raw,
        legacy_intent=raw_intent,
        patch=patch,
        next_decision=next_decision,
    )
    assistant_message = _clean_text(raw.get("assistant_message") or raw.get("reply"))
    if not assistant_message:
        assistant_message = "我在听。你可以继续说明需求，当前策略保持不变。"

    if option_resolution_audit is not None and patch and not mutation_errors:
        assistant_message = _format_discovery_reconciled_update_message(patch)
    elif partial_grounding_applied:
        # Prefer explicit applied/residual copy over model prose or generic
        # reconciled text when the verifier only grounded a subset.
        assistant_message = _format_discovery_partial_grounding_message(
            patch,
            semantic_verification.get("missing_fields")
            if isinstance(semantic_verification, Mapping)
            else None,
        )
    elif soft_reject_applied:
        assistant_message = _format_discovery_soft_reject_message(
            patch,
            dropped_fields=(
                semantic_verification.get("soft_reject_dropped_fields")
                if isinstance(semantic_verification, Mapping)
                else None
            ),
        )
    elif patch_was_reconciled and not mutation_errors:
        assistant_message = _format_discovery_reconciled_update_message(patch)
    if requested_action not in _DISCOVERY_TURN_ACTIONS:
        contract_errors.append("missing or unsupported D1 action")
        blocking_contract_error = True
        patch = {}
        action = "clarify" if next_decision is not None else "advise"
        assistant_message = (
            "模型返回的动作契约不完整，本轮策略保持不变。"
            "你可以直接重试，或继续用自然语言说明你的判断。"
        )

    raw_calls = raw.get("tool_calls") if isinstance(raw.get("tool_calls"), list) else []
    has_update_tool = any(
        isinstance(call, Mapping)
        and _clean_text(call.get("name") or call.get("tool")) == "update_strategy"
        for call in raw_calls
    )
    has_confirm_tool = any(
        isinstance(call, Mapping)
        and _clean_text(call.get("name") or call.get("tool")) == "confirm_strategy"
        for call in raw_calls
    )
    if requested_action != "update_strategy" and has_update_tool:
        contract_errors.append(
            "update_strategy tool call ignored because action is not update_strategy"
        )
        blocking_contract_error = True
        patch = {}
        if action == "ready_to_confirm":
            action = "clarify" if next_decision is not None else "advise"
            assistant_message = (
                "这轮同时给出了确认提示和策略修改，契约不一致；"
                "为避免确认旧策略，本轮策略保持不变。"
            )

    if has_confirm_tool and requested_action != "confirm_strategy":
        contract_errors.append(
            "confirm_strategy tool call ignored because action is not confirm_strategy"
        )
        blocking_contract_error = True
        if requested_action == "update_strategy":
            patch = {}
            action = "advise"
            assistant_message = (
                "同一轮不能既修改策略又确认策略；为避免越过确认边界，"
                "本轮两种动作都没有生效。"
            )

    if requested_action == "update_strategy" and (mutation_errors or has_confirm_tool):
        patch = {}
        action = "clarify" if next_decision is not None else "advise"
        assistant_message = (
            "这轮没有产生可验证的策略修改，当前策略保持不变。"
            "你可以继续用自然语言说明要改什么。"
        )

    confirmation_rejected_reason = ""
    if requested_action == "confirm_strategy":
        if confirmation_eligible and not blocking_contract_error:
            action = "confirm_strategy"
            patch = {}
        else:
            confirmation_rejected_reason = (
                confirmation_reason or "confirmation action failed structural validation"
            )
            action = "clarify" if next_decision is not None else "advise"
            patch = {}
            assistant_message = (
                "当前不能确认：只有在待确认阶段、且对应当前展示的完整策略时，"
                "自然语言批准才有效；本轮没有启动搜索。"
            )

    next_decision_contract_msg = (
        "next_decision requires a question, a recommendation reason, and 2-8 options"
    )
    if decision_contract_error:
        contract_errors.append(next_decision_contract_msg)
        # Repair path filled below after remaining_critical_decisions is known.
        if action == "clarify" and not patch:
            action = "advise"
            assistant_message = _format_discovery_incomplete_next_decision_message()

    selection_was_applied = _discovery_selected_option_was_applied(
        selected_decision,
        effective_patch=patch,
        validated_tool_patch=validated_tool_patch,
        intent_snapshot=intent_snapshot,
        requested_action=requested_action,
        mutation_valid=not blocking_contract_error and not confirmation_rejected_reason,
    )
    if selection_was_applied and candidate_resolved_decision is not None:
        resolved_decision = dict(candidate_resolved_decision)
        selected_values: dict[str, Any] = {}
        for field in resolved_decision.get("target_fields") or []:
            canonical_field = _clean_text(field)
            if canonical_field in patch:
                selected_values[canonical_field] = _json_safe(patch[canonical_field])
            elif canonical_field in intent_snapshot:
                selected_values[canonical_field] = _json_safe(
                    intent_snapshot.get(canonical_field)
                )
        if selected_values:
            resolved_decision["selected_values"] = selected_values
        decision_memory = _normalise_discovery_decision_memory(
            [*decision_memory, resolved_decision]
        )
        if repeated_selected_decision:
            next_decision = None

    # A consultation asking for alternatives must not be collapsed back to a
    # two-item menu merely because the provider omitted enum entries from the
    # structured response. Expand from the canonical schema only for an open
    # (not already selected) Agent decision; accepted decisions must move on.
    if selected_decision is None:
        unscoped_next_decision = next_decision
        next_decision = _scope_discovery_next_decision_to_unresolved_fields(
            next_decision,
            resolved_fields.union(patch),
        )
        resolved_scope_removed_next_decision = (
            unscoped_next_decision is not None and next_decision is None
        )
        next_decision = _expand_discovery_enum_decision_options(
            next_decision,
            assistant_message,
        )

    input_gap = _normalise_discovery_gap_report(input_gap_report)
    redundant_next_decision = resolved_scope_removed_next_decision
    if not redundant_next_decision and not (
        selected_decision is not None
        and resolved_decision is None
        and _same_discovery_decision(pending_decision, next_decision)
    ):
        redundant_next_decision = _discovery_decision_was_resolved(
            next_decision,
            decision_memory,
            intent_snapshot=intent_snapshot,
            resolved_fields=resolved_fields.union(explicit_resolution_patch),
        )
    if redundant_next_decision:
        next_decision = None
        if action == "clarify":
            action = (
                "ready_to_confirm"
                if input_gap["ready_for_confirm"] and not input_gap["required_missing"]
                else "advise"
            )
        resolved_notice = (
            "这项决定已经记录在当前策略和会话记忆里，我不会重复询问。"
            + (
                "当前策略已足够执行，请确认后再开始搜索。"
                if action == "ready_to_confirm"
                else "你可以继续讨论、主动修改任意条件，或在策略完整后确认。"
            )
        )
        if action == "update_strategy":
            # Suppressing a stale/repeated next question must never erase the
            # acknowledgement of the strategy delta that was just committed.
            assistant_message = f"{assistant_message}\n\n{resolved_notice}"
        else:
            assistant_message = resolved_notice
    if resolved_decision is not None and not patch:
        selected_option = selected_decision.get("option")
        selected_label = (
            _clean_text(selected_option.get("label"))
            if isinstance(selected_option, Mapping)
            else ""
        )
        selection_name = selected_label or _clean_text(
            selected_decision.get("selection_text")
        )
        if input_gap["ready_for_confirm"] and not input_gap["required_missing"]:
            action = "ready_to_confirm"
            next_decision = None
            assistant_message = (
                f"已记录你选择「{selection_name}」。"
                "当前策略已经足够执行，我不会重复追问同一项；请确认是否按当前策略开始搜索。"
            )
        elif repeated_selected_decision:
            action = "advise"
            next_decision = None
            assistant_message = (
                f"已记录你选择「{selection_name}」。"
                "这个决定不需要再次回答；当前策略保持不变，你可以继续说明下一项需求。"
            )

    response_gap_source = (
        raw.get("gap_report")
        if isinstance(raw.get("gap_report"), Mapping)
        else input_gap_report
    )
    response_gap = _normalise_discovery_gap_report(response_gap_source)
    projected_snapshot = dict(intent_snapshot)
    projected_snapshot.update(patch)
    projected_resolved_fields = resolved_fields.union(patch)
    remaining_decision_agenda = _discovery_critical_decision_agenda(
        projected_snapshot,
        response_gap,
        projected_resolved_fields,
    )
    remaining_critical_decisions = [
        item for item in remaining_decision_agenda if item.get("critical") is True
    ]
    # Repair path only — do not re-open grilling after an intentional
    # redundant-next clear, pure chat, or a model that simply chose not to
    # ask (except after a successful card write with zero next_decision).
    # 1) Broken model menu (decision_contract_error)
    # 2) update_strategy wrote a patch but left next_decision empty (not
    #    because we suppressed a repeated/resolved menu).
    if remaining_critical_decisions and not (
        # NI-1: pure chat never force-repairs into a questionnaire menu.
        requested_action == "chat" and not patch
    ) and (
        decision_contract_error
        or (
            action == "update_strategy"
            and bool(patch)
            and next_decision is None
            and not redundant_next_decision
        )
    ):
        synthesized = _synthesize_discovery_next_decision_from_agenda(
            remaining_critical_decisions
        )
        if synthesized is not None:
            repaired_from_decision_contract = bool(decision_contract_error)
            next_decision = synthesized
            decision_contract_error = False
            # Drop the next_decision schema error once a full agenda menu was
            # synthesized; otherwise the turn still looks like a hard failure
            # even though the user now has a valid follow-up question.
            contract_errors = [
                error
                for error in contract_errors
                if error != next_decision_contract_msg
            ]
            if action == "update_strategy" and patch:
                assistant_message = (
                    f"{assistant_message}\n\n"
                    "接下来还需要确认一个关键点（见下方选项），以免漏掉影响搜索的设定。"
                ).strip()
            elif action in {"advise", "chat", "clarify"} and not patch:
                action = "clarify"
                if _discovery_user_asks_clarification(user_message) and (
                    contract_noise_clarification
                    or _discovery_history_has_contract_noise(dialogue_history)
                    or repaired_from_decision_contract
                ):
                    assistant_message = (
                        _format_discovery_contract_noise_clarification_message()
                    )
                elif not assistant_message or len(assistant_message) < 12:
                    assistant_message = (
                        "好的。当前策略还有关键点未定，请先看下面这一问题。"
                    )
    if (
        decision_contract_error
        and action == "update_strategy"
        and patch
        and not remaining_critical_decisions
    ):
        # A fully resolved strategy does not need another question. Treat a
        # malformed optional next_decision as absent once the strategy write
        # itself succeeded; genuine unresolved decisions still take the
        # synthesis path above and retain contract visibility if repair fails.
        decision_contract_error = False
        contract_errors = [
            error
            for error in contract_errors
            if error != next_decision_contract_msg
        ]
    ready_for_confirm = bool(
        action in {"ready_to_confirm", "confirm_strategy"}
        or raw.get("ready_for_confirm") is True
        or response_gap["ready_for_confirm"]
    )
    if (
        blocking_contract_error
        or confirmation_rejected_reason
        or remaining_critical_decisions
    ):
        ready_for_confirm = False
    if remaining_critical_decisions:
        response_gap = {**response_gap, "ready_for_confirm": False}
    if action == "ready_to_confirm" and remaining_critical_decisions:
        action = "clarify" if next_decision is not None else "advise"
        if next_decision is None:
            synthesized = _synthesize_discovery_next_decision_from_agenda(
                remaining_critical_decisions
            )
            if synthesized is not None:
                next_decision = synthesized
                action = "clarify"
                assistant_message = (
                    "还不能确认开搜：下面这些关键决定仍会影响结果，请先选一项或文字说明。"
                )
            else:
                assistant_message = (
                    "当前策略还缺少会实质影响搜索或科学可用性的关键决定："
                    + "、".join(
                        _clean_text(item.get("id"))
                        for item in remaining_critical_decisions
                    )
                    + "。我不会跳过这些问题直接让你确认。"
                )

    # User asked what a prior contract/repair notice meant: ensure friendly
    # explanation even when no critical agenda item was synthesized above.
    if (
        contract_noise_clarification
        and action in {"advise", "chat", "clarify"}
        and not patch
        and not (
            assistant_message
            and "刚才的提示是说" in assistant_message
        )
    ):
        action = "clarify" if next_decision is not None else action
        explanation = _format_discovery_contract_noise_clarification_message()
        if next_decision is not None:
            assistant_message = explanation
        elif not assistant_message or len(assistant_message) < 24:
            assistant_message = (
                "刚才的提示是说：系统生成的下一问选项不完整，所以没有展示菜单；"
                "这不是否定你的科学目标。请直接用自然语言说明你的选择或数据目标。"
            )

    if action == "update_strategy" and patch:
        tool_calls = [
            {"name": "update_strategy", "arguments": {"patch": patch}}
        ]
    elif action == "confirm_strategy":
        tool_calls = [
            {
                "name": "confirm_strategy",
                "arguments": {"strategy_fingerprint": strategy_fingerprint},
            }
        ]
    else:
        tool_calls = []

    if action != "update_strategy":
        patch = {}
    # NI-1 / notebook: pure chat/advise never surface contract chrome to the user.
    # Semantic verification may remain for server audit; FE hides SV chrome on non-write.
    # Only for turns that were already non-mutating (not demoted from failed update_strategy).
    if (
        action in {"chat", "advise"}
        and not patch
        and requested_action in {"chat", "advise"}
    ):
        contract_errors = []
        blocking_contract_error = False
        raw_msg = _clean_text(raw.get("assistant_message") or raw.get("reply"))
        chrome_markers = (
            "可验证的策略修改",
            "动作契约不完整",
            "契约不一致",
            "同一轮不能既修改策略又确认策略",
            "next_decision requires",
            "选项菜单不完整",
            "下一问选项不完整",
            "下一问结构不完整",
        )
        # Restore model prose only when server rewrote into hard contract-failure copy.
        # Do not undo friendly contract-noise explanations (e.g. 什么意思 → 刚才的提示是说…).
        hard_failure_markers = (
            "可验证的策略修改",
            "动作契约不完整",
            "契约不一致",
            "同一轮不能既修改策略又确认策略",
            "next_decision requires",
        )
        if (
            raw_msg
            and any(marker in assistant_message for marker in hard_failure_markers)
            and "刚才的提示是说" not in assistant_message
        ):
            assistant_message = raw_msg
        elif not assistant_message:
            assistant_message = "我在听。你可以继续说明需求，当前策略保持不变。"
    intent = _legacy_discovery_turn_intent(action, raw_intent)
    answer_text = _clean_text(raw.get("answer_text") or raw.get("mapped_answer"))
    understanding = _clean_text(raw.get("understanding"))
    next_focus = (
        _clean_text((next_decision or {}).get("focus"))
        or _clean_text(raw.get("next_focus"))
        or None
    )
    result: dict[str, Any] = {
        "action": action,
        "mode": action,
        "assistant_message": assistant_message,
        "tool_calls": tool_calls,
        "gap_report": response_gap,
        "intent": intent,
        "advance": action == "update_strategy",
        "answer_text": answer_text,
        "extra_fields": patch,
        "understanding": understanding,
        "next_focus": next_focus,
        "ready_for_confirm": ready_for_confirm,
        "phase": phase,
        "pending_question_id": _clean_text((pending or {}).get("id")),
        "strategy_fingerprint": strategy_fingerprint,
        "status": "completed",
        "parser": "agents_sdk_grill" if agent_runtime == "openai_agents" else "llm_grill",
        "agent_runtime": agent_runtime,
        "llm_used": True,
        "request_budget_seconds": request_budget_seconds,
        "decision_memory": decision_memory,
        "decision_agenda": remaining_decision_agenda,
        "session_id": session_id,
    }
    if advisor_calls:
        result["specialist_consultations"] = advisor_calls[:4]
    if provider_compatibility_recovery:
        result["provider_compatibility_recovery"] = (
            provider_compatibility_recovery
        )
    if manager_repair_feedback:
        result["manager_repair"] = {
            "attempt": manager_repair_attempt,
            "trigger": manager_repair_feedback,
            "writer": "dialogue_manager",
            "critic_authority": "read_only",
        }
    if resolved_decision is not None:
        result["resolved_decision"] = resolved_decision
    if next_decision is not None and action != "confirm_strategy":
        result["next_decision"] = next_decision
    if contract_errors:
        result["contract_errors"] = list(dict.fromkeys(contract_errors))
    if confirmation_rejected_reason:
        result["confirmation_rejected_reason"] = confirmation_rejected_reason
    if suppressed_uncommitted_fields:
        result["suppressed_uncommitted_fields"] = suppressed_uncommitted_fields
    if semantic_verification is not None:
        result["semantic_verification"] = semantic_verification
    if option_resolution_audit is not None:
        result["option_resolution"] = option_resolution_audit
    _store_discovery_dialogue_session_turn(
        session_id,
        user_message=user_message,
        assistant_message=assistant_message,
        action=action,
        patch=patch,
        next_decision=next_decision,
        resolved_decision=resolved_decision,
    )
    return result

def _project_delivery_quality(
    project: Mapping[str, Any],
    judgment: Mapping[str, Any],
    project_files: list[Mapping[str, Any]],
    *,
    actually_selected: bool,
    review_provenance: str = "agent_judgment_legacy_or_unaudited",
) -> dict[str, Any]:
    usable_files = [
        file
        for file in project_files
        if str(file.get("validity_status") or "") == "valid"
        and not bool(file.get("needs_review"))
    ]
    review_files = [
        file
        for file in project_files
        if bool(file.get("needs_review"))
        or str(file.get("validity_status") or "") == "needs_review"
    ]
    judgment_qualified = bool(
        judgment
        and str(judgment.get("evidence_stage") or "") == "inspection"
        and str(judgment.get("status") or "") == "evidence_backed"
        and str(judgment.get("hard_gate") or "") == "pass"
        and judgment.get("grade") in (2, 3, "2", "3")
        and str(judgment.get("decision") or "") == "include"
        and str(judgment.get("explanation") or "").strip()
    )
    project_validity_status = _clean_text(project.get("validity_status")).lower()
    project_needs_review = bool(project.get("needs_review")) or project_validity_status in {
        "needs_review",
        "exclude",
    }
    usable_for_delivery = bool(
        actually_selected
        and review_provenance == "agent_judgment_with_server_quality_audit"
        and judgment_qualified
        and not project_needs_review
        and usable_files
    )
    return {
        "actual_final_selection": actually_selected,
        "judgment_qualified": judgment_qualified,
        "project_needs_review": project_needs_review,
        "usable_file_count": len(usable_files),
        "needs_review_file_count": len(review_files),
        "usable_for_delivery": usable_for_delivery,
    }


def _discovery_review_provenance(
    summary: Mapping[str, Any],
    *,
    run_id: str,
) -> str:
    """Describe the actual review chain without inventing an audit.

    Historical and deterministic/local manifests may contain project
    judgments but no server quality audit.  Only a matching, schema-valid,
    selection-ready audit earns the passing provenance label.
    """

    audit = summary.get("latest_discovery_audit")
    if not isinstance(audit, Mapping):
        return "agent_judgment_legacy_or_unaudited"
    try:
        validated = DiscoveryQualityAudit.model_validate(dict(audit))
    except Exception:
        return "agent_judgment_legacy_or_unaudited"
    if validated.schema_version != "discovery-quality-audit/v1" or (
        validated.run_id != _clean_text(run_id)
    ):
        return "agent_judgment_legacy_or_unaudited"
    if (
        validated.status == "ready"
        and validated.ready_for_selection is True
    ):
        return "agent_judgment_with_server_quality_audit"
    return "agent_judgment_with_nonpassing_server_quality_audit"


def _merge_discovery_project_judgments(
    *sources: Any,
) -> dict[str, dict[str, Any]]:
    """Merge judgment maps by accession; later, more final sources win."""
    merged: dict[str, dict[str, Any]] = {}
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for key, value in source.items():
            accession = _clean_text(key).upper()
            if accession and isinstance(value, Mapping):
                merged[accession] = dict(value)
    return merged


def _ensure_discovery_review_artifacts(output_dir: Path) -> dict[str, Path]:
    """Always materialize project-level judgment review tables for a discovery run.

    This is a product feature: every finished discovery run must expose a
    downloadable project table with selection, 0-3 grade, reason, and evidence.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    produced: dict[str, Path] = {}
    control_summary = _read_json_if_exists(output_dir / "agents_discovery_summary.json")
    selected_manifest_path = output_dir / "final_selection" / "dataset_manifest.json"
    if not selected_manifest_path.exists():
        selected_manifest_path = output_dir / "dataset_manifest.json"
    if not selected_manifest_path.exists():
        return produced

    try:
        selected_payload = json.loads(selected_manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return produced

    selected_projects = selected_payload.get("projects") or []
    selected_run_id = _clean_text(selected_payload.get("run_id"))
    review_provenance = _discovery_review_provenance(
        control_summary,
        run_id=selected_run_id,
    )
    selected_accessions = {
        str(project.get("project_accession") or "").upper()
        for project in selected_projects
        if isinstance(project, dict) and str(project.get("project_accession") or "").strip()
    }
    candidate_path_value = control_summary.get("candidate_pool_manifest_path")
    candidate_manifest_path = (
        Path(str(candidate_path_value))
        if candidate_path_value
        else output_dir / "candidate_pool" / "dataset_manifest.json"
    )
    candidate_payload = selected_payload
    if candidate_manifest_path.exists() and candidate_manifest_path.is_file():
        candidate_payload = _read_json_if_exists(candidate_manifest_path) or selected_payload

    projects = candidate_payload.get("projects") or []
    if not isinstance(projects, list):
        projects = []
    candidate_files = candidate_payload.get("files") or []
    if not isinstance(candidate_files, list):
        candidate_files = []
    summary = (
        selected_payload.get("summary")
        if isinstance(selected_payload.get("summary"), dict)
        else {}
    )
    judgments = _merge_discovery_project_judgments(
        control_summary.get("project_judgments"),
        summary.get("project_judgments"),
    )
    if not judgments:
        judgments = _merge_discovery_project_judgments(
            _read_json_if_exists(output_dir / "project_judgments.json")
        )
    projects_by_accession = {
        str(project.get("project_accession") or "").upper(): dict(project)
        for project in projects
        if isinstance(project, dict) and str(project.get("project_accession") or "").strip()
    }
    for accession in judgments:
        projects_by_accession.setdefault(
            accession,
            {"project_accession": accession, "project_title": ""},
        )
    files_by_accession: dict[str, list[dict[str, Any]]] = {}
    for file in candidate_files:
        if not isinstance(file, dict):
            continue
        accession = str(file.get("project_accession") or "").upper()
        if accession:
            files_by_accession.setdefault(accession, []).append(file)
    # Normalize project payloads for the writer.
    normalized_projects: list[dict[str, Any]] = []
    for accession, project in projects_by_accession.items():
        item = dict(project)
        judgment = judgments.get(accession) or {}
        item["project_accession"] = accession
        item["final_grade"] = judgment.get("grade", item.get("final_grade"))
        item["hard_gate"] = judgment.get("hard_gate", item.get("hard_gate"))
        item["judgment_status"] = judgment.get("status", item.get("judgment_status"))
        item["judgment_decision"] = judgment.get("decision", item.get("judgment_decision"))
        item["judgment_confidence"] = judgment.get("confidence", item.get("judgment_confidence"))
        item["judgment_explanation"] = judgment.get("explanation", item.get("judgment_explanation") or "")
        item["judgment_evidence_stage"] = judgment.get(
            "evidence_stage", item.get("judgment_evidence_stage")
        )
        item["missing_information"] = judgment.get(
            "missing_information", item.get("missing_information") or []
        )
        item["has_project_judgment"] = bool(judgment)
        item.update(
            _project_delivery_quality(
                item,
                judgment,
                files_by_accession.get(accession, []),
                actually_selected=accession in selected_accessions,
                review_provenance=review_provenance,
            )
        )
        item["selection_rationale"] = control_summary.get("selection_rationale") or ""
        item["review_provenance"] = review_provenance
        # File count fallback from nested files if needed.
        if not item.get("selected_file_count"):
            item["selected_file_count"] = len(files_by_accession.get(accession, []))
        normalized_projects.append(item)

    table_path = _write_discovery_project_judgment_table(
        output_dir,
        normalized_projects,
        judgments,
        selected_accessions=selected_accessions,
        files=candidate_files,
        selection_rationale=str(control_summary.get("selection_rationale") or ""),
        review_provenance=review_provenance,
    )
    produced["project_judgments_table_csv"] = table_path
    selected_csv = output_dir / "selected_projects_review.csv"
    selected_json = output_dir / "selected_projects_review.json"
    judgments_json = output_dir / "project_judgments.json"
    if selected_csv.exists():
        produced["selected_projects_review_csv"] = selected_csv
    if selected_json.exists():
        produced["selected_projects_review_json"] = selected_json
    if judgments_json.exists():
        produced["project_judgments_json"] = judgments_json
    return produced


def _validated_business_completion_payload(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    try:
        decision = BusinessCompletionDecision.model_validate(dict(value))
    except Exception:
        return None
    if decision.succeeded and not business_completion_allows_success(decision):
        return None
    return decision.model_dump(mode="json")


def _public_discovery_record(
    *,
    discovery_id: str,
    output_dir: Path,
    manifest: DatasetManifest,
    paths: dict[str, Path] | None = None,
    memory_saved: bool = False,
    status: str = "completed",
    runtime: str = "workflow",
    agent: dict[str, Any] | None = None,
) -> dict[str, Any]:
    persisted_agent = manifest.summary.get("agent_runtime")
    if agent is None and isinstance(persisted_agent, dict):
        agent = dict(persisted_agent)
    if agent is not None:
        runtime = "openai_agents"
        status = _clean_text(agent.get("status")) or status
    download_files = paths or {
        key: output_dir / filename
        for key, (filename, _media_type) in _DISCOVERY_DOWNLOAD_FILES.items()
    }
    # Prefer live judgment payload from summary / control-plane artifacts.
    # Always materialize project review tables for downloadable audit.
    try:
        _ensure_discovery_review_artifacts(output_dir)
    except Exception:
        pass
    control_summary = _read_json_if_exists(output_dir / "agents_discovery_summary.json")
    provenance_summary = dict(control_summary)
    if isinstance(manifest.summary, Mapping) and isinstance(
        manifest.summary.get("latest_discovery_audit"), Mapping
    ):
        provenance_summary["latest_discovery_audit"] = manifest.summary.get(
            "latest_discovery_audit"
        )
    review_provenance = _discovery_review_provenance(
        provenance_summary,
        run_id=manifest.run_id,
    )
    summary_run_id = _clean_text(provenance_summary.get("run_id"))
    summary_matches_run = not summary_run_id or summary_run_id == manifest.run_id
    latest_discovery_audit = provenance_summary.get("latest_discovery_audit")
    if not summary_matches_run or not isinstance(latest_discovery_audit, Mapping):
        latest_discovery_audit = None
    else:
        try:
            validated_audit = DiscoveryQualityAudit.model_validate(
                dict(latest_discovery_audit)
            )
        except Exception:
            latest_discovery_audit = None
        else:
            latest_discovery_audit = (
                validated_audit.model_dump(mode="json")
                if validated_audit.run_id == manifest.run_id
                else None
            )
    runtime_provenance = provenance_summary.get("runtime_provenance")
    if not summary_matches_run or not isinstance(runtime_provenance, Mapping):
        runtime_provenance = None
    else:
        try:
            runtime_provenance = RuntimeProvenance.model_validate(
                dict(runtime_provenance)
            ).model_dump(mode="json")
        except Exception:
            runtime_provenance = None
    business_completion = (
        _validated_business_completion_payload(
            provenance_summary.get("business_completion")
        )
        if summary_matches_run
        else None
    )
    if status in {"completed", "completed_with_review"}:
        try:
            completion_decision = BusinessCompletionDecision.model_validate(
                business_completion
            )
        except Exception:
            completion_decision = None
        if not business_completion_allows_success(completion_decision):
            status = (
                "running"
                if completion_decision is not None
                and completion_decision.status == "running_progress"
                else "blocked"
            )
    judgments = _merge_discovery_project_judgments(
        control_summary.get("project_judgments"),
        (
            manifest.summary.get("project_judgments")
            if isinstance(manifest.summary, dict)
            else None
        ),
    )
    if not judgments:
        judgments = _merge_discovery_project_judgments(
            _read_json_if_exists(output_dir / "project_judgments.json")
        )
    judgment_summary = (
        (manifest.summary.get("project_judgment_summary") if isinstance(manifest.summary, dict) else None)
        or control_summary.get("project_judgment_summary")
        or {}
    )
    files = [
        file.model_dump(mode="json", exclude={"raw_record"})
        for file in manifest.files
    ]
    files_by_accession: dict[str, list[dict[str, Any]]] = {}
    for file in files:
        accession = str(file.get("project_accession") or "").upper()
        if accession:
            files_by_accession.setdefault(accession, []).append(file)
    agent_selection_committed = agent is None or (
        agent.get("selected_round_index") is not None
        and status in {"completed", "completed_with_review"}
    )
    final_accessions = (
        {project.project_accession.upper() for project in manifest.projects}
        if agent_selection_committed
        else set()
    )
    projects = []
    for project in manifest.projects:
        payload = project.model_dump(mode="json", exclude={"raw_metadata"})
        accession = str(payload.get("project_accession") or "").upper()
        judgment = judgments.get(accession) or {}
        payload["final_grade"] = judgment.get("grade")
        payload["judgment_status"] = judgment.get("status")
        payload["hard_gate"] = judgment.get("hard_gate")
        payload["judgment_decision"] = judgment.get("decision")
        payload["judgment_confidence"] = judgment.get("confidence")
        payload["judgment_explanation"] = judgment.get("explanation") or ""
        payload["judgment_evidence_stage"] = judgment.get("evidence_stage")
        payload["missing_information"] = judgment.get("missing_information") or []
        payload["next_action"] = judgment.get("next_action")
        payload["has_project_judgment"] = bool(judgment)
        payload["judgment_evidence_refs"] = judgment.get("evidence_refs") or []
        payload["judgment_constraint_assessments"] = (
            judgment.get("constraint_assessments") or []
        )
        payload["judgment_limitations"] = judgment.get("limitations") or []
        payload["judgment_rubric_version"] = judgment.get("rubric_version") or ""
        payload["review_provenance"] = review_provenance
        payload.update(
            _project_delivery_quality(
                payload,
                judgment,
                files_by_accession.get(accession, []),
                actually_selected=accession in final_accessions,
                review_provenance=review_provenance,
            )
        )
        # Keep legacy retrieval scores under explicit names for debugging only.
        payload["retrieval_project_score"] = payload.get("project_score")
        payload["retrieval_confidence"] = payload.get("confidence")
        payload["retrieval_trust_score"] = payload.get("trust_score")
        projects.append(payload)
    # Ensure a durable project-level judgment table exists for every finished run.
    # Fall back to the live manifest only when no persisted review table can be
    # materialized; never overwrite a candidate-wide table with final selection.
    try:
        review_paths = _ensure_discovery_review_artifacts(output_dir)
        if "project_judgments_table_csv" not in review_paths:
            _write_discovery_project_judgment_table(
                output_dir,
                projects,
                judgments,
                selected_accessions=final_accessions,
                files=files,
                selection_rationale=str(
                    (agent or {}).get("selection_rationale")
                    or manifest.summary.get("selection_rationale")
                    or ""
                ),
                review_provenance=review_provenance,
            )
        for key, path in review_paths.items():
            download_files[key] = path
    except Exception:
        pass
    # Register generated review artifacts into downloads.
    for key, (filename, _media) in _DISCOVERY_DOWNLOAD_FILES.items():
        candidate = output_dir / filename
        if candidate.exists():
            download_files[key] = candidate
    needs_review_count = sum(
        file.needs_review or file.validity_status == "needs_review"
        for file in manifest.files
    )
    valid_count = sum(1 for file in manifest.files if file.validity_status == "valid")
    weak_keep_count = sum(1 for file in manifest.files if file.validity_status == "weak_keep")
    usable_count = sum(
        file.validity_status == "valid" and not file.needs_review
        for file in manifest.files
    )
    graded_projects = sum(1 for item in projects if item.get("has_project_judgment"))
    deliverable_projects = sum(1 for item in projects if item.get("usable_for_delivery"))
    ungraded_projects = max(0, len(projects) - graded_projects)
    selected_file_count = sum(
        1
        for file in files
        if str(file.get("project_accession") or "").upper() in final_accessions
    )
    return {
        "discovery_id": discovery_id,
        "run_id": manifest.run_id,
        "status": status,
        "runtime": runtime,
        "agent": agent,
        "latest_discovery_audit": latest_discovery_audit,
        "business_completion": business_completion,
        "runtime_provenance": runtime_provenance,
        "request": manifest.request.model_dump(mode="json"),
        "summary": {
            **manifest.summary,
            **(
                {"latest_discovery_audit": latest_discovery_audit}
                if latest_discovery_audit is not None
                else {}
            ),
            **(
                {"runtime_provenance": runtime_provenance}
                if runtime_provenance is not None
                else {}
            ),
            "valid_files": valid_count,
            "weak_keep_files": weak_keep_count,
            "usable_files": usable_count,
            "needs_review_files": needs_review_count,
            "memory_saved": memory_saved,
            "project_judgment_summary": judgment_summary,
            "graded_projects": graded_projects,
            "deliverable_projects": deliverable_projects,
            "ungraded_projects": ungraded_projects,
            "candidate_projects": len(projects),
            "candidate_files": len(files),
            "selected_projects": len(final_accessions),
            "selected_files": selected_file_count,
            "requires_project_judgments": True,
        },
        "project_count": len(projects),
        "file_count": len(files),
        "projects": projects,
        "project_judgments": judgments,
        "files": files,
        "output_dir": str(output_dir),
        "downloads": {
            key: f"/api/discovery/{discovery_id}/download?file={key}"
            for key, path in download_files.items()
            if path.exists()
        },
    }


def _write_discovery_project_judgment_table(
    output_dir: Path,
    projects: list[dict[str, Any]],
    judgments: Mapping[str, Any],
    *,
    selected_accessions: set[str] | None = None,
    files: list[dict[str, Any]] | None = None,
    selection_rationale: str = "",
    review_provenance: str = "agent_judgment_legacy_or_unaudited",
) -> Path:
    """Persist a project-level review table: selection, 0-3 grade, and evidence/reason."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "project_judgments_table.csv"
    selected_path = output_dir / "selected_projects_review.csv"
    columns = [
        "actual_final_selection",
        "selected_for_delivery",
        "project_accession",
        "project_title",
        "final_grade",
        "hard_gate",
        "judgment_status",
        "judgment_decision",
        "judgment_confidence",
        "judgment_evidence_stage",
        "judgment_explanation",
        "judgment_evidence_refs",
        "judgment_constraint_assessments",
        "judgment_limitations",
        "judgment_rubric_version",
        "missing_information",
        "next_action",
        "species",
        "canonical_species",
        "organism_taxon_id",
        "acquisition_mode",
        "labeling_strategy",
        "immunopeptide_scope",
        "immunopeptide_evidence_terms",
        "hla_class",
        "hla_alleles",
        "ptm_type",
        "selected_file_count",
        "usable_file_count",
        "needs_review_file_count",
        "project_needs_review",
        "usable_for_delivery",
        "has_project_judgment",
        "retrieval_project_score",
        "retrieval_confidence",
        "retrieval_trust_score",
        "evidence_terms",
        "sample_file_names",
        "download_urls_sample",
        "selection_rationale",
        "review_provenance",
    ]

    def _join(values: Any) -> str:
        if values is None:
            return ""
        if isinstance(values, str):
            return values
        if isinstance(values, (list, tuple, set)):
            return ";".join(str(item).strip() for item in values if str(item).strip())
        return str(values)

    # Optional file-level context for evidence preview.
    files_by_project: dict[str, list[dict[str, Any]]] = {}
    file_payloads = files
    if file_payloads is None:
        manifest_path = output_dir / "candidate_pool" / "dataset_manifest.json"
        if not manifest_path.exists():
            manifest_path = output_dir / "dataset_manifest.json"
        if not manifest_path.exists() and (output_dir / "final_selection" / "dataset_manifest.json").exists():
            manifest_path = output_dir / "final_selection" / "dataset_manifest.json"
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
            file_payloads = payload.get("files") or []
        except Exception:
            file_payloads = []
    for file in file_payloads or []:
        if not isinstance(file, dict):
            continue
        acc = str(file.get("project_accession") or "").upper()
        if not acc:
            continue
        files_by_project.setdefault(acc, []).append(file)

    normalized_selected_accessions = {
        str(accession).upper() for accession in (selected_accessions or set())
    }

    rows: list[dict[str, Any]] = []
    for project in projects:
        accession = str(project.get("project_accession") or "").upper()
        judgment = judgments.get(accession) if isinstance(judgments, Mapping) else {}
        if not isinstance(judgment, Mapping):
            judgment = {}
        project_files = files_by_project.get(accession) or []
        evidence_terms: list[str] = []
        for key in (
            "immunopeptide_evidence_terms",
            "ptm_evidence_terms",
            "hla_class",
            "hla_alleles",
        ):
            evidence_terms.extend(
                str(item)
                for item in (
                    project.get(key)
                    or judgment.get(key)
                    or []
                )
                if str(item).strip()
            )
        # Light evidence from top files.
        for file in project_files[:8]:
            for key in ("immunopeptide_evidence_terms", "ptm_evidence_terms", "matched_intent_terms"):
                evidence_terms.extend(
                    str(item) for item in (file.get(key) or []) if str(item).strip()
                )
        # unique keep order
        seen: set[str] = set()
        evidence_unique: list[str] = []
        for term in evidence_terms:
            low = term.casefold()
            if low in seen:
                continue
            seen.add(low)
            evidence_unique.append(term)
        sample_names = [str(file.get("file_name") or "") for file in project_files[:12] if file.get("file_name")]
        sample_urls = [
            str(file.get("download_url") or "")
            for file in project_files[:8]
            if file.get("download_url")
        ]
        grade = judgment.get("grade", project.get("final_grade", ""))
        decision = judgment.get("decision", project.get("judgment_decision", ""))
        hard_gate = judgment.get("hard_gate", project.get("hard_gate", ""))
        explanation = judgment.get("explanation", project.get("judgment_explanation", ""))
        actual_selected = (
            accession in normalized_selected_accessions
            if selected_accessions is not None
            else bool(project.get("actual_final_selection"))
        )
        delivery = _project_delivery_quality(
            project,
            judgment,
            project_files,
            actually_selected=actual_selected,
            review_provenance=review_provenance,
        )
        selected = bool(delivery["usable_for_delivery"])
        rows.append(
            {
                "actual_final_selection": "yes" if actual_selected else "no",
                "selected_for_delivery": "yes" if selected else "no",
                "project_accession": accession,
                "project_title": project.get("project_title") or "",
                "final_grade": grade,
                "hard_gate": hard_gate,
                "judgment_status": judgment.get("status", project.get("judgment_status", "")),
                "judgment_decision": decision,
                "judgment_confidence": judgment.get(
                    "confidence", project.get("judgment_confidence", "")
                ),
                "judgment_evidence_stage": judgment.get(
                    "evidence_stage", project.get("judgment_evidence_stage", "")
                ),
                "judgment_explanation": explanation,
                "judgment_evidence_refs": _join(judgment.get("evidence_refs") or []),
                "judgment_constraint_assessments": json.dumps(
                    judgment.get("constraint_assessments") or [],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "judgment_limitations": _join(judgment.get("limitations") or []),
                "judgment_rubric_version": judgment.get("rubric_version") or "",
                "missing_information": _join(
                    judgment.get("missing_information")
                    or project.get("missing_information")
                    or []
                ),
                "next_action": judgment.get("next_action", project.get("next_action", "")),
                "species": _join(project.get("species") or []),
                "canonical_species": _join(project.get("canonical_species") or []),
                "organism_taxon_id": _join(project.get("organism_taxon_id") or []),
                "acquisition_mode": project.get("acquisition_mode") or "",
                "labeling_strategy": project.get("labeling_strategy") or "",
                "immunopeptide_scope": project.get("immunopeptide_scope") or "",
                "immunopeptide_evidence_terms": _join(
                    project.get("immunopeptide_evidence_terms") or []
                ),
                "hla_class": _join(project.get("hla_class") or []),
                "hla_alleles": _join(project.get("hla_alleles") or []),
                "ptm_type": project.get("ptm_type") or "",
                "selected_file_count": project.get("selected_file_count")
                or project.get("file_count")
                or len(project_files)
                or 0,
                "usable_file_count": delivery["usable_file_count"],
                "needs_review_file_count": delivery["needs_review_file_count"],
                "project_needs_review": "yes" if delivery["project_needs_review"] else "no",
                "usable_for_delivery": "yes" if selected else "no",
                "has_project_judgment": "yes"
                if (
                    project.get("has_project_judgment")
                    or judgment
                )
                else "no",
                "retrieval_project_score": project.get("retrieval_project_score")
                or project.get("project_score")
                or "",
                "retrieval_confidence": project.get("retrieval_confidence")
                or project.get("confidence")
                or "",
                "retrieval_trust_score": project.get("retrieval_trust_score")
                or project.get("trust_score")
                or "",
                "evidence_terms": _join(evidence_unique[:30]),
                "sample_file_names": _join(sample_names),
                "download_urls_sample": _join(sample_urls),
                "selection_rationale": selection_rationale
                or str(project.get("selection_rationale") or ""),
                "review_provenance": review_provenance
                or str(project.get("review_provenance") or ""),
            }
        )

    # Stable order: selected first, then grade desc.
    def _sort_key(row: dict[str, Any]) -> tuple:
        selected = 1 if row.get("selected_for_delivery") == "yes" else 0
        try:
            grade = int(row.get("final_grade"))
        except Exception:
            grade = -1
        return (-selected, -grade, str(row.get("project_accession") or ""))

    rows.sort(key=_sort_key)

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Convenience: only selected projects.
    selected_rows = [row for row in rows if row.get("actual_final_selection") == "yes"]
    with selected_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in selected_rows:
            writer.writerow(row)

    # JSON companion for APIs/UI.
    write_json(
        output_dir / "project_judgments.json",
        {
            str(key).upper(): value
            for key, value in (judgments or {}).items()
            if isinstance(value, Mapping)
        },
    )
    write_json(
        output_dir / "selected_projects_review.json",
        {
            "selected_count": len(selected_rows),
            "deliverable_count": sum(
                row.get("usable_for_delivery") == "yes" for row in selected_rows
            ),
            "total_projects": len(rows),
            "projects": selected_rows,
        },
    )
    return path


def _discovery_history_record_from_run_dir(run_dir: Path) -> dict[str, Any] | None:
    if not run_dir.exists() or not run_dir.is_dir():
        return None
    discovery_id = run_dir.name
    request = _read_json_if_exists(run_dir / "dataset_request.json")
    summary = _read_json_if_exists(run_dir / "discovery_summary.json") or _read_json_if_exists(
        run_dir / "agents_discovery_summary.json"
    )
    final_manifest = run_dir / "final_selection" / "dataset_manifest.json"
    root_manifest = run_dir / "dataset_manifest.json"
    manifest_path = final_manifest if final_manifest.exists() else root_manifest
    project_count = 0
    file_count = 0
    status = "completed" if manifest_path.exists() else "partial"
    if summary:
        project_count = int(summary.get("selected_projects") or summary.get("qualified_projects") or 0)
        file_count = int(summary.get("selected_files") or 0)
        status = _clean_text(summary.get("status") or status) or status
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            project_count = len(payload.get("projects") or project_count or [])
            file_count = len(payload.get("files") or file_count or [])
        except Exception:
            pass
    prompt = ""
    if isinstance(request, Mapping):
        # request may not include prompt; leave empty
        pass
    # Prefer job linkage if present in summary.
    job_id = _clean_text((summary or {}).get("job_id") or (summary or {}).get("source_job_id"))
    goal = _clean_text((request or {}).get("goal") if isinstance(request, Mapping) else "")
    display = goal or discovery_id
    if prompt:
        display = prompt[:80]
    elif goal:
        display = f"Discovery · {goal}"
    else:
        display = f"Discovery · {discovery_id}"
    mtime = run_dir.stat().st_mtime
    finished_at = datetime.fromtimestamp(mtime, tz=_APP_TZ).isoformat()
    bundle = run_dir / "discovery_run_bundle.zip"
    size_bytes = 0
    try:
        size_bytes = sum(p.stat().st_size for p in run_dir.rglob("*") if p.is_file())
    except Exception:
        size_bytes = bundle.stat().st_size if bundle.exists() else 0
    record = {
        "kind": "discovery",
        "history_id": f"discovery-{discovery_id}",
        "discovery_id": discovery_id,
        "run_id": discovery_id,
        "result_id": discovery_id,
        "name": discovery_id,
        "display_name": display,
        "input_value": display,
        "status": status,
        "business_completion": _validated_business_completion_payload(
            (summary or {}).get("business_completion")
        ),
        "repository": _clean_text((request or {}).get("repository") if isinstance(request, Mapping) else "")
        or "pride",
        "run_mode": "discovery",
        "project_count": project_count,
        "file_count": file_count,
        "size_bytes": size_bytes,
        "output_dir": str(run_dir),
        "can_download": manifest_path.exists() or bundle.exists(),
        "primary_action": "open_discovery",
        "created_at": finished_at,
        "updated_at": finished_at,
        "finished_at": finished_at,
        "history_time": finished_at,
        "submitter": "discovery",
        "downloads": {
            "dataset_manifest_csv": f"/api/discovery/{discovery_id}/download?file=dataset_manifest_csv",
            "discovery_run_bundle_zip": f"/api/discovery/{discovery_id}/download?file=discovery_run_bundle_zip",
        }
        if (manifest_path.exists() or bundle.exists())
        else {},
        "job_id": job_id or None,
        "goal": goal or None,
    }
    return _decorate_history_item(record)


def _list_discovery_history_records(limit: int = 100) -> list[dict[str, Any]]:
    root = _discovery_root_dir()
    if not root.exists():
        return []
    records: list[dict[str, Any]] = []
    # Prefer job index for status fidelity.
    jobs_dir = _discovery_jobs_dir()
    seen_dirs: set[str] = set()
    if jobs_dir.exists():
        for path in sorted(jobs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            job = _read_json_if_exists(path)
            if not job:
                continue
            record = job.get("record") if isinstance(job.get("record"), Mapping) else {}
            run_dir = None
            for key in ("output_dir", "run_id", "discovery_id"):
                raw = _clean_text((record or {}).get(key) or job.get(key))
                if not raw:
                    continue
                candidate = Path(raw)
                if candidate.exists() and candidate.is_dir():
                    run_dir = candidate
                    break
                candidate = root / safe_output_stem(raw)
                if candidate.exists() and candidate.is_dir():
                    run_dir = candidate
                    break
            if run_dir is None:
                # Job without materialized run still shown as history shell.
                prompt = _clean_text((job.get("body") or {}).get("prompt")) if isinstance(job.get("body"), Mapping) else ""
                item = {
                    "kind": "discovery",
                    "history_id": f"discovery-job-{job.get('job_id')}",
                    "job_id": job.get("job_id"),
                    "discovery_id": None,
                    "run_id": job.get("job_id"),
                    "result_id": job.get("job_id"),
                    "name": job.get("job_id"),
                    "display_name": (prompt[:80] if prompt else f"Discovery job {job.get('job_id')}"),
                    "input_value": prompt or job.get("job_id"),
                    "status": job.get("status") or "unknown",
                    "repository": "pride",
                    "run_mode": "discovery",
                    "project_count": (record or {}).get("project_count") or 0,
                    "file_count": (record or {}).get("file_count") or 0,
                    "size_bytes": 0,
                    "output_dir": "",
                    "can_download": False,
                    "primary_action": "open_discovery_job",
                    "created_at": job.get("created_at"),
                    "updated_at": job.get("finished_at") or job.get("started_at") or job.get("created_at"),
                    "finished_at": job.get("finished_at"),
                    "history_time": job.get("finished_at") or job.get("started_at") or job.get("created_at"),
                    "submitter": "discovery",
                    "error": job.get("error"),
                }
                records.append(_decorate_history_item(item))
                continue
            seen_dirs.add(str(run_dir.resolve()))
            item = _discovery_history_record_from_run_dir(run_dir)
            if item is None:
                continue
            item["job_id"] = job.get("job_id")
            item["status"] = job.get("status") or item.get("status")
            item["error"] = job.get("error")
            if job.get("finished_at"):
                item["finished_at"] = job.get("finished_at")
                item["history_time"] = job.get("finished_at")
            item["primary_action"] = "open_discovery"
            records.append(_decorate_history_item(item))
            if len(records) >= limit:
                return records
    # Include run dirs not linked from jobs.
    for path in sorted(root.glob("agents_*"), key=lambda p: p.stat().st_mtime, reverse=True):
        if not path.is_dir():
            continue
        key = str(path.resolve())
        if key in seen_dirs:
            continue
        item = _discovery_history_record_from_run_dir(path)
        if item is None:
            continue
        records.append(item)
        if len(records) >= limit:
            break
    records.sort(key=lambda item: str(item.get("history_time") or item.get("finished_at") or ""), reverse=True)
    return records[:limit]


def _discovery_job_history_shell(job: Mapping[str, Any]) -> dict[str, Any]:
    record = job.get("record") if isinstance(job.get("record"), Mapping) else {}
    body = job.get("body") if isinstance(job.get("body"), Mapping) else {}
    request = record.get("request") if isinstance(record.get("request"), Mapping) else {}
    prompt = _clean_text(body.get("prompt"))
    goal = _clean_text(request.get("goal") or body.get("goal") or prompt)
    job_id = _clean_text(job.get("job_id"))
    discovery_id = _clean_text(record.get("discovery_id"))
    output_dir = _clean_text(record.get("output_dir"))
    history_time = job.get("finished_at") or job.get("started_at") or job.get("created_at")
    return _decorate_history_item(
        {
            "kind": "discovery",
            "history_id": (
                f"discovery-{discovery_id}"
                if discovery_id
                else f"discovery-job-{job_id}"
            ),
            "job_id": job_id,
            "discovery_id": discovery_id or None,
            "run_id": _clean_text(record.get("run_id")) or discovery_id or job_id,
            "result_id": discovery_id or job_id,
            "name": discovery_id or job_id,
            "display_name": goal[:80] if goal else f"Discovery job {job_id}",
            "input_value": goal or job_id,
            "status": job.get("status") or "unknown",
            "repository": _clean_text(request.get("repository")) or "pride",
            "run_mode": "discovery",
            "project_count": int(record.get("project_count") or 0),
            "file_count": int(record.get("file_count") or 0),
            "size_bytes": int(record.get("size_bytes") or 0),
            "output_dir": output_dir,
            "can_download": bool(job.get("status") == "completed" and output_dir),
            "primary_action": (
                "open_discovery" if discovery_id else "open_discovery_job"
            ),
            "created_at": job.get("created_at"),
            "updated_at": history_time,
            "finished_at": job.get("finished_at"),
            "history_time": history_time,
            "submitter": "discovery",
            "error": job.get("error"),
            "resumable": bool(job.get("resumable")),
            "goal": goal or None,
        }
    )


def _list_discovery_history_records_fast(limit: int = 100) -> list[dict[str, Any]]:
    """Return compact discovery shells without loading large authoritative jobs."""
    with _discovery_jobs_lock:
        jobs = [
            dict(job)
            for job in _discovery_jobs.values()
            if _discovery_job_is_in_current_storage_scope(job)
        ]
    records = [_discovery_job_history_shell(job) for job in jobs]
    seen_job_ids = {_clean_text(record.get("job_id")) for record in records}
    summaries_dir = _discovery_job_summaries_dir()
    if summaries_dir.is_dir():
        for path in summaries_dir.glob("*.json"):
            summary = _read_json_if_exists(path)
            job_id = _clean_text((summary or {}).get("job_id"))
            if not summary or not job_id or job_id in seen_job_ids:
                continue
            if str(summary.get("status") or "").lower() in _ACTIVE_STATUSES:
                summary = dict(summary)
                summary["status"] = "interrupted"
                summary["resumable"] = True
                summary["error"] = (
                    summary.get("error")
                    or "discovery_job_interrupted_by_server_reload"
                )
            records.append(_decorate_history_item(summary))
            seen_job_ids.add(job_id)
    records.sort(
        key=lambda item: str(
            item.get("history_time") or item.get("finished_at") or item.get("created_at") or ""
        ),
        reverse=True,
    )
    return records[:limit]


def _upsert_discovery_history_record(record: Mapping[str, Any] | None, output_dir: Path | None = None) -> None:
    """Ensure finished discovery runs appear in project history."""
    try:
        history_item = None
        if isinstance(record, Mapping) and record.get("discovery_id"):
            discovery_id = _clean_text(record.get("discovery_id"))
            run_dir = Path(str(record.get("output_dir") or output_dir or (_discovery_root_dir() / discovery_id)))
            history_item = {
                "kind": "discovery",
                "history_id": f"discovery-{discovery_id}",
                "discovery_id": discovery_id,
                "run_id": _clean_text(record.get("run_id") or discovery_id),
                "result_id": discovery_id,
                "name": discovery_id,
                "display_name": (
                    _clean_text((record.get("request") or {}).get("goal"))
                    and f"Discovery · {_clean_text((record.get('request') or {}).get('goal'))}"
                )
                or discovery_id,
                "input_value": _clean_text((record.get("request") or {}).get("goal")) or discovery_id,
                "status": _clean_text(record.get("status")) or "completed",
                "repository": _clean_text((record.get("request") or {}).get("repository")) or "pride",
                "run_mode": "discovery",
                "project_count": int(record.get("project_count") or 0),
                "file_count": int(record.get("file_count") or 0),
                "size_bytes": 0,
                "output_dir": str(run_dir),
                "can_download": True,
                "primary_action": "open_discovery",
                "created_at": _now_app_iso(),
                "updated_at": _now_app_iso(),
                "finished_at": _now_app_iso(),
                "history_time": _now_app_iso(),
                "submitter": "discovery",
                "downloads": record.get("downloads") or {},
                "goal": _clean_text((record.get("request") or {}).get("goal")) or None,
            }
            # Fill size if possible.
            if run_dir.exists():
                try:
                    history_item["size_bytes"] = sum(
                        p.stat().st_size for p in run_dir.rglob("*") if p.is_file()
                    )
                except Exception:
                    pass
        elif output_dir is not None:
            history_item = _discovery_history_record_from_run_dir(output_dir)
        if history_item:
            _upsert_history_index(history_item)
    except Exception:
        return


_LOCAL_SPECIES_TOKENS = {
    "human": ("human", "homo sapiens", "hela", "hek293", "hct116", "a549", "jurkat", "293t", "9606"),
    "mouse": ("mouse", "murine", "mus musculus", "10090"),
    "rat": ("rat", "rattus", "rattus norvegicus", "10116"),
    "yeast": ("yeast", "saccharomyces", "saccharomyces cerevisiae", "559292"),
    "e coli": ("e coli", "e. coli", "ecoli", "escherichia coli", "escherichia", "562"),
}
_LOCAL_PROJECT_ACCESSION_RE = re.compile(r"\b(PXD\d{6,}|PRIDE\d+)\b", re.IGNORECASE)


def _local_text_has_token(text: str, token: str) -> bool:
    folded = text.casefold()
    token_folded = token.casefold()
    if re.fullmatch(r"[a-z0-9]+", token_folded):
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(token_folded)}(?![a-z0-9])", folded))
    return token_folded in folded


def _infer_species_from_text(text: str) -> list[str]:
    inferred, _taxon_ids = species_from_text(text)
    return inferred


def _infer_local_species_from_path(path: Path, root: Path) -> list[str]:
    try:
        text = str(path.relative_to(root))
    except ValueError:
        text = path.name
    return _infer_species_from_text(text)


def _canonical_species_set(values: list[str]) -> set[str]:
    canonical, _taxon_ids = normalize_species_values(values)
    return {item.casefold() for item in canonical}


def _local_species_conflicts_request(inferred_species: list[str], requested_species: list[str], species_policy: str = "open") -> bool:
    if species_policy == "open":
        return False
    if not inferred_species or not requested_species:
        return False
    overlap = bool(_canonical_species_set(inferred_species) & _canonical_species_set(requested_species))
    if species_policy == "exclude":
        return overlap
    return not overlap


def _local_project_accession_from_path(path: Path, root: Path) -> str | None:
    try:
        text = str(path.relative_to(root))
    except ValueError:
        text = str(path)
    match = _LOCAL_PROJECT_ACCESSION_RE.search(text)
    return match.group(1).upper() if match else None


def _metadata_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_metadata_text(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(_metadata_text(item) for item in value)
    return str(value)


def _project_species_from_metadata(project_record: dict[str, Any], sdrf_rows: list[dict[str, Any]]) -> list[str]:
    text = " ".join(
        [
            _metadata_text(project_record.get("organisms")),
            _metadata_text(project_record.get("title")),
            _metadata_text(project_record.get("projectDescription")),
            _metadata_text(sdrf_rows[:200]),
        ]
    )
    return _infer_species_from_text(text)


def _local_file_record(path: Path, size_bytes: int | None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "fileName": path.name,
        "name": path.name,
        "publicFileLocations": [{"value": str(path)}],
    }
    if size_bytes is not None:
        record["fileSizeBytes"] = size_bytes
    return record


def _project_file_name(record: dict[str, Any]) -> str:
    return str(record.get("fileName") or record.get("name") or "")


def _match_project_file(local_path: Path, records: list[dict[str, Any]]) -> dict[str, Any] | None:
    local_name = local_path.name.casefold()
    local_stem = local_path.stem.casefold()
    for record in records:
        remote_name = _project_file_name(record)
        if remote_name.casefold() == local_name:
            return record
    for record in records:
        remote_name = _project_file_name(record)
        if Path(remote_name).stem.casefold() == local_stem:
            return record
    return None


def _local_fallback_project(root: Path, request: DatasetRequest) -> DiscoveredProject:
    open_species = request.species_policy == "open"
    return DiscoveredProject(
        repository="pride",
        project_accession="LOCAL",
        project_title=f"Local directory: {root}",
        species=[],
        species_policy=request.species_policy,
        canonical_species=[],
        organism_taxon_id=[],
        acquisition_mode=request.acquisition_mode,
        ptm_type=request.ptm_type,
        modification_scope=request.modification_scope or request.ptm_type,
        labeling_strategy=request.labeling_strategy,
        project_score=35.0,
        confidence=0.55,
        trust_score=0.5,
        evidence_completeness=0.35,
        validity_status="weak_keep" if open_species else "needs_review",
        validity_reasons=["local_directory_candidate", "metadata_not_verified", "missing_species_evidence"],
        needs_review=not open_species,
        evidence=[
            DiscoveryEvidence(field="source", source="local_directory", text=str(root), weight=5.0),
            DiscoveryEvidence(field="requested_species", source="user_constraint", text=", ".join(request.species), weight=1.0),
        ],
        raw_metadata={"local_dir": str(root)},
    )


def _load_local_pride_context(
    accession: str,
    request: DatasetRequest,
    pride: PrideClient,
) -> dict[str, Any]:
    project_record = pride.get_project(accession)
    file_inventory = list_project_files_paginated_with_state(
        pride,
        accession,
        mode="budgeted",
        max_files=max(request.max_files_per_project * 10, 100),
    )
    project_files = file_inventory.records
    sdrf_rows: list[dict[str, Any]] = []
    sdrf_candidates: list[dict[str, Any]] = []
    sdrf_inventory = PridePaginationState.unavailable(
        operation="list_project_files",
        mode="budgeted",
        project_accession=accession,
        keyword="sdrf",
        page_size=100,
        stop_reason="not_started",
    )
    try:
        sdrf_result = list_project_files_paginated_with_state(
            pride,
            accession,
            mode="budgeted",
            keyword="sdrf",
            max_files=5,
        )
        sdrf_candidates = sdrf_result.records
        sdrf_inventory = sdrf_result.state
    except Exception:
        sdrf_candidates = []
        sdrf_inventory = PridePaginationState.unavailable(
            operation="list_project_files",
            mode="budgeted",
            project_accession=accession,
            keyword="sdrf",
            page_size=100,
            stop_reason="error",
        )
    sdrf_file = detect_sdrf_file([*project_files, *sdrf_candidates])
    if sdrf_file:
        sdrf_url = PrideClient.first_download_url(sdrf_file)
        if sdrf_url:
            sdrf_rows = load_sdrf_rows(pride.download_text(sdrf_url))

    project_score = score_project(project_record, request)
    project_features = extract_project_features(project_record, sdrf_rows)
    project = build_discovered_project(
        project_record,
        request,
        project_score,
        features=project_features,
    )
    actual_species = _project_species_from_metadata(project_record, sdrf_rows)
    if actual_species:
        actual_canonical_species, actual_taxon_ids = normalize_species_values(actual_species)
        project = project.model_copy(
            update={
                "species": actual_species,
                "canonical_species": actual_canonical_species,
                "organism_taxon_id": actual_taxon_ids,
                "diversity_tags": [
                    *[f"species:{item}" for item in actual_species],
                    *[f"instrument:{item}" for item in project.instrument_families or ["unknown"]],
                    *[f"fragmentation:{item}" for item in project.fragmentation_methods or ["unknown"]],
                ],
            }
        )
    if _local_species_conflicts_request(project.species, request.species, request.species_policy):
        project = project.model_copy(
            update={
                "validity_status": "exclude",
                "validity_reasons": list(dict.fromkeys([*project.validity_reasons, "species_mismatch"])),
                "needs_review": True,
                "trust_score": 0.0,
            }
        )
    return {
        "project": project,
        "project_record": project_record,
        "project_files": project_files,
        "project_features": project_features,
        "sdrf_rows": sdrf_rows,
        "file_inventory": file_inventory.state.to_dict(),
        "sdrf_inventory": sdrf_inventory.to_dict(),
    }


def _local_discovery_manifest(
    request: DatasetRequest,
    local_dir: str | Path,
    report: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> DatasetManifest:
    if not str(local_dir or "").strip():
        raise ValueError("Local discovery directory is required.")
    root = Path(_container_repo_path_hint(str(local_dir))).expanduser()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Local discovery directory not found: {root}")

    def _report(message: str) -> None:
        if report is not None:
            report(message)

    def _check_cancel() -> None:
        if should_cancel is not None and should_cancel():
            raise InterruptedError("Discovery cancelled.")

    _report(f"Scanning local discovery directory: {root}")

    def fallback_project(accession: str) -> DiscoveredProject:
        project = _local_fallback_project(root, request)
        if accession == "LOCAL":
            return project
        return project.model_copy(
            update={
                "project_accession": accession,
                "project_title": f"Local project directory: {accession}",
                "raw_metadata": {"local_dir": str(root), "project_accession_hint": accession},
            }
        )

    pride: PrideClient | None = None
    context_cache: dict[str, dict[str, Any] | None] = {}
    failures: list[dict[str, str]] = []
    projects_by_accession: dict[str, DiscoveredProject] = {}
    files: list[DiscoveredFile] = []
    excluded_files = 0

    def get_context(accession: str) -> dict[str, Any] | None:
        nonlocal pride
        if accession in context_cache:
            return context_cache[accession]
        _check_cancel()
        if pride is None:
            pride = PrideClient(timeout=8.0, read_timeout=8.0)
        try:
            _report(f"Enriching local project hint {accession} with PRIDE metadata.")
            context = _load_local_pride_context(accession, request, pride)
            _report(f"{accession}: metadata enrichment completed.")
        except Exception as exc:  # pragma: no cover - network boundary
            failures.append({"stage": "local_project_metadata", "project": accession, "error": str(exc)})
            _report(f"{accession}: metadata enrichment failed: {exc}")
            context = None
        _check_cancel()
        context_cache[accession] = context
        return context

    try:
        for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: str(item).casefold()):
            _check_cancel()
            role = classify_file_role(path.name)
            if role.role not in {"raw_acquisition", "converted_peaklist"} or role.file_type is None:
                excluded_files += 1
                continue
            _report(f"Local candidate: {path.name}")
            try:
                size_bytes = path.stat().st_size
            except OSError:
                size_bytes = None

            accession = _local_project_accession_from_path(path, root) or "LOCAL"
            context = get_context(accession) if accession != "LOCAL" else None
            project = context["project"] if context else fallback_project(accession)
            projects_by_accession.setdefault(accession, project)

            local_evidence = [
                DiscoveryEvidence(field="file_name", source="local_path", text=str(path), weight=8.0),
                DiscoveryEvidence(field="file_role", source="suffix", text=role.role, weight=4.0),
            ]
            inferred_species = _infer_local_species_from_path(path, root)
            if inferred_species:
                local_evidence.append(
                    DiscoveryEvidence(
                        field="species",
                        source="local_path_name",
                        text=", ".join(inferred_species),
                        weight=3.0,
                    )
                )

            local_record = _local_file_record(path, size_bytes)
            if context:
                remote_record = _match_project_file(path, context["project_files"]) or local_record
                remote_name = _project_file_name(remote_record) or path.name
                sdrf_rows = context["sdrf_rows"]
                matched_sdrf_rows = select_sdrf_rows_for_file(sdrf_rows, remote_name) if sdrf_rows and remote_name else []
                if not sdrf_rows:
                    sdrf_match_status = "no_sdrf"
                elif matched_sdrf_rows:
                    sdrf_match_status = "matched"
                else:
                    sdrf_match_status = "no_file_match"
                file_features = extract_file_features(remote_record, context["project_features"], matched_sdrf_rows)
                scored_file = score_file(
                    remote_record,
                    project,
                    request,
                    features=file_features,
                    sdrf_match_status=sdrf_match_status,
                )
                file = scored_file or score_file(
                    local_record,
                    project,
                    request,
                    features=file_features,
                    sdrf_match_status=sdrf_match_status,
                )
            else:
                file = None

            if file is None:
                file_canonical_species, file_taxon_ids = normalize_species_values(inferred_species)
                file = DiscoveredFile(
                    repository="pride",
                    project_accession=accession,
                    project_title=project.project_title,
                    file_name=str(path),
                    download_url=str(path),
                    file_type=role.file_type,
                    file_role=role.role,
                    file_role_reasons=role.reasons,
                    sdrf_match_status="not_checked",
                    evidence_level="file",
                    file_level_evidence_count=2,
                    expected_size_bytes=size_bytes,
                    species=inferred_species,
                    species_policy=request.species_policy,
                    canonical_species=file_canonical_species or project.canonical_species,
                    organism_taxon_id=file_taxon_ids or project.organism_taxon_id,
                    acquisition_mode=request.acquisition_mode,
                    ptm_type=request.ptm_type,
                    ptm_subtype=project.ptm_subtype,
                    ptm_evidence_terms=project.ptm_evidence_terms,
                    ptm_enrichment_methods=project.ptm_enrichment_methods,
                    semantic_metadata_confidence=project.semantic_metadata_confidence,
                    semantic_interpretation_trace=project.semantic_interpretation_trace,
                    modification_scope=request.modification_scope or request.ptm_type,
                    labeling_strategy=request.labeling_strategy,
                    project_score=project.project_score,
                    file_score=52.0 if role.role == "raw_acquisition" else 45.0,
                    confidence=0.6,
                    trust_score=0.58,
                    evidence_completeness=0.45 if size_bytes is not None else 0.35,
                    validity_status="weak_keep",
                    validity_reasons=["local_file_candidate", "metadata_not_verified"],
                    needs_review=False,
                    evidence=local_evidence,
                    diversity_tags=[f"species:{item}" for item in inferred_species],
                    raw_record={"local_path": str(path), "local_dir": str(root)},
                )
                if _local_species_conflicts_request(inferred_species, request.species, request.species_policy):
                    file = file.model_copy(
                        update={
                            "validity_status": "exclude",
                            "validity_reasons": [*file.validity_reasons, "species_mismatch"],
                            "needs_review": True,
                            "trust_score": 0.0,
                        }
                    )
                else:
                    decision = assess_file_validity(file, request)
                    file = file.model_copy(
                        update={
                            "validity_status": decision.status,
                            "validity_reasons": list(dict.fromkeys([*file.validity_reasons, *decision.reasons])),
                            "needs_review": decision.needs_review,
                        }
                    )
            else:
                species_values = file.species or project.species or inferred_species
                file_canonical_species, file_taxon_ids = normalize_species_values(species_values)
                validity_reasons = list(dict.fromkeys([*file.validity_reasons, "local_file_candidate"]))
                validity_status = file.validity_status
                needs_review = file.needs_review
                trust_score = file.trust_score
                if _local_species_conflicts_request(species_values, request.species, request.species_policy):
                    validity_status = "exclude"
                    validity_reasons = list(dict.fromkeys([*validity_reasons, "species_mismatch"]))
                    needs_review = True
                    trust_score = 0.0
                file = file.model_copy(
                    update={
                        "project_accession": accession,
                        "project_title": project.project_title,
                        "file_name": str(path),
                        "download_url": str(path),
                        "file_type": role.file_type,
                        "file_role": role.role,
                        "file_role_reasons": role.reasons,
                        "expected_size_bytes": size_bytes,
                        "species": species_values,
                        "species_policy": request.species_policy,
                        "canonical_species": file_canonical_species or file.canonical_species or project.canonical_species,
                        "organism_taxon_id": file_taxon_ids or file.organism_taxon_id or project.organism_taxon_id,
                        "modification_scope": file.modification_scope or project.modification_scope or request.modification_scope,
                        "labeling_strategy": file.labeling_strategy or project.labeling_strategy or request.labeling_strategy,
                        "validity_status": validity_status,
                        "validity_reasons": validity_reasons,
                        "needs_review": needs_review,
                        "trust_score": trust_score,
                        "evidence": [*local_evidence, *file.evidence],
                        "raw_record": {
                            "local_path": str(path),
                            "local_dir": str(root),
                            "project_accession_hint": accession,
                            "remote_record": file.raw_record,
                        },
                    }
                )

            files.append(file)
            if len(files) >= request.max_files:
                break
    finally:
        if pride is not None:
            pride.close()

    final_projects: list[DiscoveredProject] = []
    for accession, project in projects_by_accession.items():
        project_files = [file for file in files if file.project_accession == accession]
        project_species = project.species or sorted({species for file in project_files for species in file.species})
        project_canonical_species, project_taxon_ids = normalize_species_values(project_species)
        updated_project = project.model_copy(
            update={
                "species": project_species,
                "canonical_species": project_canonical_species or project.canonical_species,
                "organism_taxon_id": project_taxon_ids or project.organism_taxon_id,
            }
        )
        if updated_project.validity_status != "exclude":
            project_decision = assess_project_validity(updated_project, request)
            updated_project = updated_project.model_copy(
                update={
                    "validity_status": project_decision.status,
                    "validity_reasons": list(dict.fromkeys([*updated_project.validity_reasons, *project_decision.reasons])),
                    "needs_review": updated_project.needs_review or project_decision.needs_review,
                }
            )
        updated_project = updated_project.model_copy(
            update={
                "file_count": len(project_files),
                "selected_file_count": len([file for file in project_files if file.validity_status != "exclude"]),
            }
        )
        final_projects.append(updated_project)

    diversity = {
        "species_distribution": dict(Counter(species for file in files for species in (file.species or ["unknown"]))),
        "instrument_family_distribution": dict(Counter(value for file in files for value in (file.instrument_families or ["unknown"]))),
        "fragmentation_method_distribution": dict(Counter(value for file in files for value in (file.fragmentation_methods or ["unknown"]))),
        "lc_gradient_distribution": dict(Counter(file.lc_gradient or "unknown" for file in files)),
        "unknown_counts": {
            "species": sum(1 for file in files if not file.species),
            "instrument_family": sum(1 for file in files if not file.instrument_families),
            "fragmentation_method": sum(1 for file in files if not file.fragmentation_methods),
            "lc_gradient": sum(1 for file in files if not file.lc_gradient),
        },
    }
    validity_counts = dict(Counter(file.validity_status for file in files))
    validity_reason_counts = dict(Counter(reason for file in files for reason in file.validity_reasons))
    evidence_warning_counts = dict(Counter(warning for file in files for warning in file.evidence_warnings))
    summary = {
        "repository": "local",
        "source": "local_dir",
        "local_dir": str(root),
        "metadata_enrichment": "pride_project_hint",
        "pride_metadata_inventories": [
            {
                "project_accession": accession,
                "file_inventory": context["file_inventory"],
                "sdrf_inventory": context["sdrf_inventory"],
            }
            for accession, context in sorted(context_cache.items())
            if context is not None
        ],
        "goal": request.goal,
        "ptm_type": request.ptm_type,
        "modification_scope": request.modification_scope or request.ptm_type,
        "labeling_strategy": request.labeling_strategy,
        "canonical_species": request.canonical_species,
        "organism_taxon_id": request.organism_taxon_id,
        "queries": [str(root)],
        "candidate_projects_seen": len(final_projects),
        "eligible_projects_seen": len([project for project in final_projects if project.selected_file_count > 0]),
        "excluded_projects": sum(1 for project in final_projects if project.validity_status == "exclude"),
        "excluded_files": excluded_files + sum(1 for file in files if file.validity_status == "exclude"),
        "selected_projects": len([project for project in final_projects if project.file_count > 0]),
        "selected_files": len(files),
        "max_projects": request.max_projects,
        "max_files": request.max_files,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "failures": failures,
        "memory_used": False,
        "diversity": diversity,
        "species_distribution": diversity["species_distribution"],
        "instrument_family_distribution": diversity["instrument_family_distribution"],
        "fragmentation_method_distribution": diversity["fragmentation_method_distribution"],
        "lc_gradient_distribution": diversity["lc_gradient_distribution"],
        "unknown_counts": diversity["unknown_counts"],
        "validity": {"validity_status_counts": validity_counts, "validity_reason_counts": validity_reason_counts},
        "validity_status_counts": validity_counts,
        "validity_reason_counts": validity_reason_counts,
        "file_context": {
            "evidence_level_distribution": dict(Counter(file.evidence_level for file in files)),
            "sdrf_match_status_distribution": dict(Counter(file.sdrf_match_status for file in files)),
            "evidence_warning_counts": evidence_warning_counts,
        },
        "evidence_level_distribution": dict(Counter(file.evidence_level for file in files)),
        "sdrf_match_status_distribution": dict(Counter(file.sdrf_match_status for file in files)),
        "evidence_warning_counts": evidence_warning_counts,
    }
    return DatasetManifest(request=request, projects=final_projects if files else [], files=files, summary=summary)


def _discovery_confirmation_rejection(
    body: Mapping[str, Any],
) -> dict[str, str] | None:
    if body.get("grill_confirmed") is not True:
        return {
            "status": "rejected",
            "code": "grill_confirmation_required",
            "error": "Explicit strategy confirmation is required: grill_confirmed must be true.",
        }
    if _discovery_exhaustive_payload_conflict(body):
        return {
            "status": "rejected",
            "code": "exhaustive_intent_downgraded",
            "error": (
                "检索目标明确要求全部/搜全，但执行载荷仍是旧版有限候选池。"
                "本次未启动搜索；请刷新策略并重新确认。"
            ),
        }
    supplied_fingerprint = _clean_text(body.get("strategy_fingerprint")).lower()
    if supplied_fingerprint:
        supplied_canonical = body.get("strategy_fingerprint_payload")
        expected_fingerprint = ""
        canonical_matches_payload = True
        if supplied_canonical is not None:
            canonical_text = str(supplied_canonical)
            if not canonical_text or len(canonical_text.encode("utf-8")) > 200_000:
                canonical_matches_payload = False
            else:
                try:
                    canonical_snapshot = json.loads(canonical_text)
                except (TypeError, ValueError, json.JSONDecodeError):
                    canonical_matches_payload = False
                else:
                    canonical_matches_payload = _json_values_equal(
                        canonical_snapshot,
                        _discovery_execution_snapshot(body),
                    )
                    if canonical_matches_payload:
                        expected_fingerprint = hashlib.sha256(
                            canonical_text.encode("utf-8")
                        ).hexdigest()
        else:
            expected_fingerprint = _discovery_execution_fingerprint(body)
        if (
            not canonical_matches_payload
            or re.fullmatch(r"[0-9a-f]{64}", supplied_fingerprint) is None
            or not expected_fingerprint
            or not secrets.compare_digest(supplied_fingerprint, expected_fingerprint)
        ):
            return {
                "status": "rejected",
                "code": "strategy_confirmation_mismatch",
                "error": (
                    "The confirmation fingerprint does not match the exact discovery "
                    "payload. Reconfirm the current strategy before starting search."
                ),
            }
    return None


def _require_discovery_confirmation(body: Mapping[str, Any]) -> None:
    rejection = _discovery_confirmation_rejection(body)
    if rejection is not None:
        raise ValueError(rejection["error"])


def _run_web_discovery(
    body: dict[str, Any],
    report: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    agent_event_callback: Callable[[AgentEvent], None] | None = None,
    search_event_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    # Keep this at the execution boundary as well as every HTTP entry point so
    # legacy routes and internal callers cannot accidentally bypass grilling.
    _require_discovery_confirmation(body)

    def _report(message: str) -> None:
        if report is not None:
            report(message)

    def _check_cancel() -> None:
        if should_cancel is not None and should_cancel():
            raise InterruptedError("Discovery cancelled.")

    request = _clean_dataset_request(body)
    source = _clean_text(body.get("source") or body.get("discovery_source") or "remote").lower()
    if source in {"local", "local_dir", "local_directory"} or request.repository == "local":
        _check_cancel()
        _report("Starting local directory discovery.")
        discovery_id = safe_output_stem(
            f"local_{datetime.now(_APP_TZ).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        )
        output_dir = _discovery_root_dir() / discovery_id
        manifest = _local_discovery_manifest(
            request,
            _clean_text(body.get("local_dir") or body.get("local_directory")),
            report=_report,
            should_cancel=should_cancel,
        )
        raw_task_type = _clean_text(body.get("task_type"))
        try:
            task_type = normalize_task_type(raw_task_type) if raw_task_type else None
        except ValueError:
            task_type = None
        if task_type:
            _report(f"Annotating task readiness: {task_type}")
            manifest = annotate_manifest_task_readiness(manifest, task_type)
        _check_cancel()
        manifest = manifest.model_copy(update={"run_id": discovery_id})
        paths = write_dataset_manifest(manifest, output_dir)
        _report(f"Local discovery manifest written: {output_dir}")
        return _public_discovery_record(
            discovery_id=discovery_id,
            output_dir=output_dir,
            manifest=manifest,
            paths=paths,
            memory_saved=False,
        )
    use_memory = body.get("use_memory", True) is not False
    prior_memory = DiscoveryMemory(_discovery_memory_dir()) if use_memory else None
    agentic_enabled = body.get("agentic") is True
    agentic_plan = None
    agentic_round_records = []
    agentic_fallback: dict[str, Any] | None = None
    raw_task_type = _clean_text(body.get("task_type"))
    # browse_only / empty / unknown → no task profile (data-only discovery).
    try:
        task_type = normalize_task_type(raw_task_type) if raw_task_type else None
    except ValueError:
        task_type = None
    task_profile = get_task_profile(task_type) if task_type else None

    def _discover_for_web(
        discovery_request: DatasetRequest,
        memory: DiscoveryMemory | None = None,
        queries: list[str] | None = None,
    ) -> DatasetManifest:
        pride_client = PrideClient(timeout=15.0, read_timeout=15.0)
        try:
            discovery_kwargs = {
                "memory": memory,
                "queries": queries,
                "report": _report,
                "should_cancel": should_cancel,
                "early_stop_on_limits": True,
            }
            if discovery_request.repository == "pride":
                return discover_pride_dataset(
                    discovery_request,
                    client=pride_client,
                    **discovery_kwargs,
                )
            return discover_repository_dataset(
                discovery_request,
                client=pride_client,
                **discovery_kwargs,
            )
        finally:
            pride_client.close()

    runtime = _clean_text(body.get("runtime") or body.get("discovery_runtime") or "workflow").lower()
    if runtime in {"agent", "agents", "openai-agent", "openai_agents_sdk"}:
        runtime = "openai_agents"
    if runtime not in {"workflow", "openai_agents"}:
        raise ValueError(f"Unsupported discovery runtime: {runtime}")

    if runtime == "openai_agents":
        _check_cancel()
        prompt = _clean_text(body.get("prompt"))
        if not prompt:
            raise ValueError("Discovery request is required for OpenAI Agents mode.")
        web_llm_config = body.get("llm_config") if isinstance(body.get("llm_config"), dict) else {}
        agent_llm_config, config_error = _build_llm_config(
            web_llm_config,
            allow_server_default=not bool(body.get("_require_explicit_llm_config")),
        )
        if config_error or agent_llm_config is None:
            raise ValueError(config_error or "Invalid LLM configuration.")
        fixed_execution_id = _clean_text(body.get("_execution_discovery_id"))
        discovery_id = (
            safe_output_stem(fixed_execution_id)
            if fixed_execution_id
            else safe_output_stem(
                f"agents_{datetime.now(_APP_TZ).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
            )
        )
        output_dir = _discovery_root_dir() / discovery_id
        normalized_task_type = task_type
        discovery_mode, budget, dynamic_limits = _agent_discovery_configuration(body)
        resume_existing_run = body.get("_resume_existing_discovery_run") is True
        if (
            not resume_existing_run
            and fixed_execution_id
            and (output_dir / "agent_control.sqlite").is_file()
        ):
            # The operations console resumes through the persistent queue and
            # may not share the web process's in-memory legacy body. The stable
            # execution id and existing control DB are sufficient authority to
            # resume this exact run instead of attempting to create it again.
            from agent.control_plane.store import AgentRunStore

            existing_store = AgentRunStore(output_dir / "agent_control.sqlite")
            resume_existing_run = existing_store.load_run(discovery_id) is not None

        def _agent_discovery_func(
            discovery_request: DatasetRequest,
            memory: DiscoveryMemory | None = None,
            queries: list[str] | None = None,
        ) -> DatasetManifest:
            _check_cancel()
            query_list = list(queries or [])
            preview = "; ".join(query_list[:4])
            _report(f"Act: repository search with {len(query_list)} query term(s){': ' + preview if preview else ''}")
            observed = _discover_for_web(discovery_request, memory=memory, queries=query_list)
            _report(
                "Observe: "
                f"{int(observed.summary.get('selected_projects') or len(observed.projects))} project(s), "
                f"{int(observed.summary.get('selected_files') or len(observed.files))} file(s)."
            )
            return observed

        _report(
            "Reason: OpenAI Agents SDK is planning quality-first candidate search and inspection within server safety ceilings."
        )
        quality_client: PrideClient | None = None
        search_environment: PrideDiscoverySearchEnvironment | None = None
        if request.repository == "pride":
            quality_client = PrideClient(timeout=15.0, read_timeout=15.0)
            search_environment = PrideDiscoverySearchEnvironment(
                request=request,
                prompt=prompt,
                state_path=output_dir / "candidate_search_state.json",
                client=quality_client,
                memory=prior_memory,
                report=_report,
                search_event=search_event_callback,
                should_cancel=should_cancel,
            )
        try:
            result = run_agents_discovery(
                prompt=prompt,
                request=request,
                output_dir=output_dir,
                task_type=normalized_task_type,
                state_db=output_dir / "agent_control.sqlite",
                memory=prior_memory,
                budget=budget,
                mode=discovery_mode,  # type: ignore[arg-type]
                dynamic_limits=dynamic_limits,
                run_id=discovery_id,
                discovery_func=_agent_discovery_func,
                search_environment=search_environment,
                llm_config=agent_llm_config,
                event_callback=agent_event_callback,
                # Prefer cancel-aware streaming inside the control plane when
                # should_cancel is provided; avoid full run_sync blind spots.
                stream_events=False,
                should_cancel=should_cancel,
                resume_existing=resume_existing_run,
            )
        except InterruptedError:
            # A user stop means "stop searching and keep verified work", not
            # "discard the final short file tranche".  The control-plane store
            # is authoritative even when the agent call is interrupted before
            # it can build a public terminal record.
            try:
                from agent.control_plane.discovery import DiscoveryToolService
                from agent.control_plane.store import AgentRunStore

                state_db = output_dir / "agent_control.sqlite"
                cancel_store = AgentRunStore(
                    state_db,
                    event_listener=agent_event_callback,
                )
                cancelled_run = cancel_store.load_run(discovery_id)
                manifest_path = (
                    Path(cancelled_run.candidate_pool_manifest_path)
                    if cancelled_run
                    and cancelled_run.candidate_pool_manifest_path
                    else None
                )
                if (
                    cancelled_run is not None
                    and manifest_path is not None
                    and manifest_path.exists()
                ):
                    cancelled_manifest = DatasetManifest.model_validate_json(
                        manifest_path.read_text(encoding="utf-8")
                    )
                    DiscoveryToolService(
                        run_id=discovery_id,
                        request=request,
                        output_dir=output_dir,
                        store=cancel_store,
                        task_type=normalized_task_type,
                    ).publish_verified_file_batches(
                        manifest=cancelled_manifest,
                        terminal=True,
                    )
                    _report("Stop requested: published the final verified file batch.")
            except Exception as tail_exc:
                _report(
                    "Stop requested: final verified file batch could not be "
                    f"published ({_redact_secrets(str(tail_exc))})."
                )
            raise
        finally:
            if quality_client is not None:
                quality_client.close()
        _check_cancel()
        control_summary = _read_json_if_exists(output_dir / "agents_discovery_summary.json")
        dynamic_usage = control_summary.get("dynamic_usage") or {}
        budget_audit = control_summary.get("budget_audit") or {}
        agent_summary = {
            "runtime": "openai_agents",
            "provider": "openai_compatible",
            "requested_model_id": str(agent_llm_config.get("model") or ""),
            "model_family": str(agent_llm_config.get("model") or ""),
            "endpoint_identity": str(agent_llm_config.get("base_url") or ""),
            "identity_verification": "unverified",
            "status": result.status,
            "run_id": result.run_id,
            "discovery_rounds": result.discovery_round_count,
            "candidate_searches": int(control_summary.get("candidate_search_count") or 0),
            "candidate_inspections": int(control_summary.get("candidate_inspection_count") or 0),
            "no_gain_actions": int(control_summary.get("no_gain_action_count") or 0),
            "latest_metrics": control_summary.get("latest_metrics") or {},
            "model_usage": control_summary.get("model_usage") or {},
            "sdk_turn_count": int(result.sdk_turn_count),
            "runtime_provenance": (
                result.runtime_provenance.model_dump(mode="json")
                if result.runtime_provenance is not None
                else control_summary.get("runtime_provenance")
            ),
            "latest_discovery_audit": (
                result.latest_discovery_audit.model_dump(mode="json")
                if result.latest_discovery_audit is not None
                else control_summary.get("latest_discovery_audit")
            ),
            "business_completion": control_summary.get("business_completion"),
            "quality_budget_tier": _clean_text(budget_audit.get("quality_budget_tier")),
            "tool_calls": int(control_summary.get("tool_call_count") or 0),
            "stop_reason": _clean_text(control_summary.get("stop_reason") or result.status),
            "search_stop_reason": _clean_text(control_summary.get("search_stop_reason")),
            "final_output": result.final_output,
            "warnings": list(result.warnings),
            "blockers": list(result.blockers),
            "selected_round_index": result.selected_round_index,
            "selection_rationale": result.selection_rationale,
            "mode": discovery_mode,
            "budget": budget.model_dump(mode="json"),
            "dynamic_limits": dynamic_limits.model_dump(mode="json"),
            "query_units": int(dynamic_usage.get("query_units") or 0),
            "repository_requests": int(dynamic_usage.get("repository_requests") or 0),
            "search_batches": int(dynamic_usage.get("search_batches") or 0),
            "budget_reviews": int(dynamic_usage.get("budget_reviews") or 0),
            "hard_limits_reached": bool(budget_audit.get("hard_limits_reached")),
        }
        if result.status == "failed":
            detail = "; ".join(result.blockers or result.warnings) or "OpenAI Agents discovery failed."
            agent_summary["error"] = detail
            paths = {
                key: Path(raw_path)
                for key, raw_path in result.files.items()
                if key in _DISCOVERY_DOWNLOAD_FILES and Path(raw_path).exists()
            }
            for key, (filename, _media) in _DISCOVERY_DOWNLOAD_FILES.items():
                candidate = output_dir / filename
                if key not in paths and candidate.exists():
                    paths[key] = candidate
            empty_manifest = DatasetManifest(
                run_id=result.run_id or discovery_id,
                request=request,
                projects=[],
                files=[],
                summary={
                    "run_id": result.run_id or discovery_id,
                    "selected_projects": 0,
                    "selected_files": 0,
                    "agent_runtime": agent_summary,
                    "error": detail,
                },
            )
            _report(f"Final: Agent discovery failed; audits retained under {output_dir}.")
            return _public_discovery_record(
                discovery_id=discovery_id,
                output_dir=output_dir,
                manifest=empty_manifest,
                paths=paths,
                memory_saved=False,
                status="failed",
                runtime="openai_agents",
                agent=agent_summary,
            )
        manifest_path = Path(
            result.selected_manifest_path
            or result.files.get("dataset_manifest_json")
            or control_summary.get("candidate_pool_manifest_path")
            or output_dir / "dataset_manifest.json"
        )
        if not manifest_path.exists():
            detail = _clean_text(
                agent_summary.get("stop_reason")
                or agent_summary.get("search_stop_reason")
                or "no_persisted_dataset_manifest"
            )
            blockers = agent_summary.get("blockers") if isinstance(agent_summary.get("blockers"), list) else []
            blocker_text = ", ".join(str(item) for item in blockers[:5] if str(item).strip())
            message = (
                "OpenAI Agents discovery finished without a persisted dataset manifest"
                f" (stop_reason={detail}"
                + (f"; blockers={blocker_text}" if blocker_text else "")
                + ")."
            )
            raise RuntimeError(message)
        manifest = DatasetManifest.model_validate(json.loads(manifest_path.read_text(encoding="utf-8")))
        selection_committed = (
            result.selected_round_index is not None
            and result.status in {"completed", "completed_with_review"}
        )
        # Candidate pools from blocked runs remain auditable artifacts, but they
        # must not enter successful-discovery memory as if they were delivered.
        save_memory = body.get("save_memory", True) is not False and selection_committed
        agent_summary["pooled_selected_files"] = int(
            manifest.summary.get("selected_files") or len(manifest.files)
        )
        agent_summary["candidate_projects"] = len(manifest.projects)
        agent_summary["candidate_files"] = len(manifest.files)
        summary = {
            **manifest.summary,
            "run_id": result.run_id,
            "memory_used": use_memory,
            "memory_saved": save_memory,
            **(
                {
                    "runtime_provenance": result.runtime_provenance.model_dump(
                        mode="json"
                    )
                }
                if result.runtime_provenance is not None
                else {}
            ),
            **(
                {
                    "latest_discovery_audit": result.latest_discovery_audit.model_dump(
                        mode="json"
                    )
                }
                if result.latest_discovery_audit is not None
                else {}
            ),
            "sdk_turn_count": int(result.sdk_turn_count),
            "agent_runtime": agent_summary,
        }
        manifest = manifest.model_copy(update={"run_id": result.run_id, "summary": summary})
        paths = write_dataset_manifest(manifest, output_dir)
        for key, raw_path in result.files.items():
            path = Path(raw_path)
            if key in _DISCOVERY_DOWNLOAD_FILES and path.exists():
                paths[key] = path
        if save_memory:
            memory = DiscoveryMemory(_discovery_memory_dir())
            memory.append_run(
                build_run_record(
                    run_id=result.run_id,
                    manifest=manifest,
                    output_dir=output_dir,
                    manifest_path=paths["dataset_manifest_json"],
                )
            )
        _report(
            f"Final: Agent discovery {result.status}; "
            f"{len(manifest.projects)} project(s), {len(manifest.files)} file(s)."
        )
        return _public_discovery_record(
            discovery_id=discovery_id,
            output_dir=output_dir,
            manifest=manifest,
            paths=paths,
            memory_saved=save_memory,
            status=result.status,
            runtime="openai_agents",
            agent=agent_summary,
        )

    if agentic_enabled:
        _check_cancel()
        _report("Starting LLM agentic discovery planning.")
        try:
            web_llm_config = body.get("llm_config") if isinstance(body.get("llm_config"), dict) else {}
            planner = _agentic_discovery_planner(
                web_llm_config,
                allow_server_default=not bool(body.get("_require_explicit_llm_config")),
            )
        except Exception as exc:
            planner = None
            agentic_fallback = {
                "reason": "llm_planner_initialization_failed",
                "message": str(exc),
            }
        if planner is None:
            if agentic_fallback is None:
                agentic_fallback = {
                    "reason": "llm_unavailable",
                    "message": "No discovery LLM API key found; using deterministic discovery.",
                }
            _report(f"LLM query planning unavailable; using deterministic discovery. Reason: {agentic_fallback['reason']}")
            manifest = _discover_for_web(request, memory=prior_memory)
        else:
            prompt = _clean_text(body.get("prompt")) or (
                f"Find {', '.join(request.species)} {request.ptm_type} {request.acquisition_mode} "
                f"{request.repository.upper()} projects/files "
                "for model-building datasets. Prefer trustworthy file-level evidence and useful diversity."
            )
            def _discovery_func(request: DatasetRequest, memory: DiscoveryMemory | None = None, queries: list[str] | None = None) -> DatasetManifest:
                return _discover_for_web(request, memory=memory, queries=queries)

            try:
                result = run_agentic_discovery(
                    request=request,
                    planner=planner,
                    prompt=prompt,
                    memory=prior_memory,
                    max_rounds=_bounded_int(body.get("agentic_rounds"), default=1, minimum=1, maximum=2),
                    task_profile=task_profile,
                    discovery_func=_discovery_func,
                )
                manifest = result.manifest
                agentic_plan = result.plan
                agentic_round_records = result.rounds
            except Exception as exc:
                agentic_fallback = {
                    "reason": "llm_agentic_discovery_failed",
                    "message": str(exc),
                }
                _report(f"LLM query planning failed; using deterministic discovery. Reason: {exc}")
                manifest = _discover_for_web(request, memory=prior_memory)
    else:
        _report("Starting remote repository discovery.")
        manifest = _discover_for_web(request, memory=prior_memory)
    if task_type:
        _report(f"Annotating task readiness: {task_type}")
        manifest = annotate_manifest_task_readiness(manifest, normalize_task_type(task_type))
    _check_cancel()
    run_id = generate_discovery_run_id(request)
    discovery_id = safe_output_stem(run_id)
    if (_discovery_root_dir() / discovery_id).exists():
        run_id = f"{run_id}_{uuid.uuid4().hex[:6]}"
        discovery_id = safe_output_stem(run_id)
    output_dir = _discovery_root_dir() / discovery_id
    summary = {**manifest.summary, "run_id": run_id, "memory_used": use_memory}
    if agentic_plan is not None:
        summary["agentic"] = {
            "enabled": True,
            "rounds": len(agentic_round_records),
            "queries": agentic_plan.queries,
            "warnings": agentic_plan.warnings,
            "suggested_next_queries": agentic_plan.suggested_next_queries,
            "trace_steps": len(agentic_plan.trace),
        }
    elif agentic_fallback is not None:
        summary["agentic"] = {
            "enabled": False,
            "requested": True,
            "fallback": agentic_fallback,
            "warnings": [agentic_fallback["reason"]],
        }
    manifest = manifest.model_copy(update={"run_id": run_id, "summary": summary})
    paths = write_dataset_manifest(manifest, output_dir)
    _report(f"Discovery manifest written: {output_dir}")
    if agentic_plan is not None:
        agentic_plan_path = output_dir / "agentic_plan.json"
        write_json(agentic_plan_path, agentic_plan.model_dump(mode="json"))
        paths["agentic_plan"] = agentic_plan_path
    if agentic_round_records:
        agentic_rounds_path = output_dir / "agentic_rounds.json"
        write_json(agentic_rounds_path, [item.model_dump(mode="json") for item in agentic_round_records])
        paths["agentic_rounds"] = agentic_rounds_path

    save_memory = body.get("save_memory", True) is not False
    if save_memory:
        memory = DiscoveryMemory(_discovery_memory_dir())
        memory.append_run(
            build_run_record(
                run_id=run_id,
                manifest=manifest,
                output_dir=output_dir,
                manifest_path=paths["dataset_manifest_json"],
            )
        )
    return _public_discovery_record(
        discovery_id=discovery_id,
        output_dir=output_dir,
        manifest=manifest,
        paths=paths,
        memory_saved=save_memory,
    )


def _discovery_review_decisions_from_body(
    *,
    manifest: DatasetManifest,
    body: dict[str, Any],
) -> list[DiscoveryReviewDecision]:
    run_id = manifest.run_id or str(manifest.summary.get("run_id") or "").strip()
    if not run_id:
        raise ValueError("Discovery manifest has no run_id.")
    raw_reviews = body.get("reviews")
    if not isinstance(raw_reviews, list):
        raise ValueError("Request body must include a reviews list.")

    files_by_key = {(file.project_accession, file.file_name): file for file in manifest.files}
    decisions: list[DiscoveryReviewDecision] = []
    for index, raw_review in enumerate(raw_reviews, start=1):
        if not isinstance(raw_review, dict):
            raise ValueError(f"Review item {index} must be an object.")
        project_accession = _clean_text(raw_review.get("project_accession"))
        file_name = _clean_text(raw_review.get("file_name"))
        decision = _clean_text(raw_review.get("decision"))
        reason = _clean_text(raw_review.get("reason") or ("correct" if decision == "keep" else "unclear"))
        note = _clean_text(raw_review.get("note"))
        if not decision:
            continue
        if not project_accession or not file_name:
            raise ValueError(f"Review item {index} is missing project_accession or file_name.")
        if (project_accession, file_name) not in files_by_key:
            raise ValueError(f"Review item {index} does not match a file in this discovery manifest.")
        if decision not in VALID_REVIEW_DECISIONS:
            raise ValueError(f"Review item {index} has invalid decision: {decision!r}")
        if reason not in VALID_REVIEW_REASONS:
            raise ValueError(f"Review item {index} has invalid reason: {reason!r}")
        file = files_by_key[(project_accession, file_name)]
        decisions.append(
            DiscoveryReviewDecision(
                review_id=uuid.uuid4().hex,
                run_id=run_id,
                created_at=now_utc_iso(),
                repository=file.repository,
                project_accession=project_accession,
                file_name=file_name,
                decision=decision,
                reason=reason,
                note=note,
            )
        )
    return decisions


def _manifest_with_review_decisions(
    manifest: DatasetManifest,
    decisions: list[DiscoveryReviewDecision],
) -> DatasetManifest:
    latest = {(decision.project_accession, decision.file_name): decision for decision in decisions}
    files = []
    for file in manifest.files:
        decision = latest.get((file.project_accession, file.file_name))
        if decision is None:
            files.append(file)
            continue
        files.append(
            file.model_copy(
                update={
                    "review_decision": decision.decision,
                    "review_reason": decision.reason,
                    "review_note": decision.note,
                }
            )
        )
    return manifest.model_copy(update={"files": files})


def _save_discovery_reviews(discovery_id: str, body: dict[str, Any]) -> dict[str, Any]:
    output_dir = _safe_discovery_dir(discovery_id)
    if output_dir is None:
        return {"error": "Discovery run not found."}
    manifest_path = output_dir / "dataset_manifest.json"
    if not manifest_path.exists():
        return {"error": "Discovery run not found."}
    try:
        manifest = load_dataset_manifest(manifest_path)
        decisions = _discovery_review_decisions_from_body(manifest=manifest, body=body)
    except ValueError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        return {"error": f"Discovery manifest is not readable: {_redact_secrets(str(exc))}"}

    memory = DiscoveryMemory(_discovery_memory_dir())
    if decisions:
        memory.append_review_decisions(decisions)
        manifest = _manifest_with_review_decisions(manifest, decisions)
        write_dataset_manifest(manifest, output_dir)

    return {
        "status": "completed",
        "run_id": manifest.run_id,
        "review_decisions": len(decisions),
        "memory_summary": memory.summary(),
        "record": _public_discovery_record(
            discovery_id=discovery_id,
            output_dir=output_dir,
            manifest=manifest,
            memory_saved=bool(decisions),
        ),
    }


def _batch_item_dir(batch_dir: Path, index: int, input_value: str) -> Path:
    stem = safe_output_stem(input_value) or f"item_{index:03d}"
    return batch_dir / "items" / f"{index:03d}_{stem}"


def _json_write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _tail_text_file(path: Path, max_lines: int = 80, max_chars: int = 20000) -> list[str]:
    if not path.exists() or not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    if len(text) > max_chars:
        text = text[-max_chars:]
    return [_redact_secrets(line) for line in text.splitlines()[-max_lines:]]


def _append_batch_event_unlocked(
    batch: dict[str, Any],
    level: str,
    message: Any,
    item_index: int | None = None,
) -> None:
    event = {
        "ts": _now_iso(),
        "level": str(level or "info").lower(),
        "message": _redact_secrets(message).strip(),
    }
    if item_index is not None:
        event["item_index"] = item_index
    events = list(batch.get("events") or [])
    events.append(event)
    batch["events"] = events[-500:]
    batch["updated_at"] = event["ts"]


def _append_batch_event(batch_id: str, level: str, message: Any, item_index: int | None = None) -> None:
    with _batches_lock:
        batch = _batches.get(batch_id)
        if batch is None:
            return
        _append_batch_event_unlocked(batch, level, message, item_index=item_index)
        _write_batch_manifest(batch)


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_plain(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _plain(model_dump(mode="json"))
        except TypeError:
            return _plain(model_dump())
    if hasattr(value, "__dict__"):
        return {key: _plain(item) for key, item in vars(value).items() if not key.startswith("_")}
    return str(value)


def _attribute_value(attributes: Any, name: str) -> Any:
    attr = getattr(attributes, name, None)
    if attr is None:
        return None
    return getattr(attr, "value", attr)


def _search_parameter_hints(attributes: Any) -> dict[str, Any]:
    hints = _attribute_value(attributes, "search_parameter_hints")
    return dict(hints) if isinstance(hints, dict) else {}


def _plan_output_path(plan: Any, key: str) -> str:
    outputs = getattr(plan, "output_paths", {}) or {}
    if isinstance(outputs, dict) and outputs.get(key) is not None:
        return str(outputs[key])
    return ""


def _materialize_parameter_workflow(output_dir: Path, attributes: Any, plan: Any) -> Path | None:
    workflow = getattr(plan, "fragpipe_workflow_path", None)
    if workflow is None:
        return None
    source = Path(workflow)
    if not source.exists() or not source.is_file():
        return None
    destination = output_dir / "workflows" / source.name
    try:
        from agent.execution.workflow import materialize_workflow_with_attributes

        materialize_workflow_with_attributes(source, destination, attributes)
    except Exception:
        return None
    return destination


def _rewrite_converter_config_workflow(config_path: Path, workflow_path: Path | None) -> None:
    if workflow_path is None or not config_path.exists():
        return
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(config, dict):
        return
    fragpipe = config.get("generate_fragpipe_search_result")
    if isinstance(fragpipe, dict):
        fragpipe["workflow_path"] = _workspace_container_path(config_path.parent, workflow_path)
    _rewrite_config_paths_for_workspace(config, config_path.parent)
    _json_write(config_path, config)


def _workspace_container_path(root: Path, path: Any) -> str:
    if path in (None, ""):
        return ""
    text = str(path)
    try:
        resolved = Path(text).resolve()
        relative = resolved.relative_to(root.resolve())
    except (OSError, ValueError):
        return text
    return f"/workspace/{relative.as_posix()}"


def _rewrite_config_paths_for_workspace(value: Any, root: Path, key: str = "") -> None:
    if isinstance(value, dict):
        for child_key, child in value.items():
            if isinstance(child, str) and child and _looks_like_converter_path_key(str(child_key)):
                value[child_key] = _workspace_container_path(root, child)
            else:
                _rewrite_config_paths_for_workspace(child, root, str(child_key))
    elif isinstance(value, list):
        for item in value:
            _rewrite_config_paths_for_workspace(item, root, key)


def _looks_like_converter_path_key(key: str) -> bool:
    return key in {"data_path", "fasta_path", "workflow_path", "manifest_path", "workdir", "output"} or key.endswith("_path")


def _write_task_runtime_log(task_id: str, output_dir: Path) -> Path:
    log_path = output_dir / "logs" / "runtime.log"
    with _tasks_lock:
        task = dict(_tasks.get(task_id) or {})
    lines: list[str] = []
    for entry in _public_logs_from_task(task):
        level = str(entry.get("level") or "info").upper()
        ts = str(entry.get("ts") or "")
        message = _redact_secrets(entry.get("message") or "")
        lines.append(f"{ts}\t{level}\t{message}".strip())
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return log_path


def _result_with_plan(result: Any, plan: Any | None) -> Any:
    if plan is None or getattr(result, "plan", None) is plan:
        return result
    model_copy = getattr(result, "model_copy", None)
    if callable(model_copy):
        return model_copy(update={"plan": plan})
    values = dict(vars(result)) if hasattr(result, "__dict__") else {}
    if not values:
        values = {
            "resolution": getattr(result, "resolution", None),
            "context": getattr(result, "context", None),
            "asset": getattr(result, "asset", None),
            "attributes": getattr(result, "attributes", None),
        }
    values["plan"] = plan
    return SimpleNamespace(**values)


def _write_agent_audit_package(
    output_dir: Path,
    result: Any,
    *,
    plan: Any | None = None,
    report: Callable[[str], None] | None = None,
) -> None:
    try:
        from agent.agent_core.audit import write_agent_audit_for_result

        write_agent_audit_for_result(output_dir, _result_with_plan(result, plan))
    except Exception as exc:
        if report is not None:
            try:
                report(f"Failed to write agent audit files: {exc}")
            except Exception:
                pass


def _write_recovery_audit_package(
    output_dir: Path,
    task_obj: Any,
    *,
    stage: str,
    run_mode: str,
    events: list[Any],
    result: Any | None = None,
    plan: Any | None = None,
    artifacts: dict[str, str | Path | None] | None = None,
    report: Callable[[str], None] | None = None,
) -> None:
    try:
        from agent.agent_core.recovery import build_recovery_audit, write_recovery_audit

        resolution = getattr(result, "resolution", None)
        primary = getattr(resolution, "primary_project", None)
        context = getattr(result, "context", None)
        audit = build_recovery_audit(
            task_id=str(getattr(task_obj, "task_id", "")),
            input_file=str(getattr(task_obj, "file_name", "")),
            output_dir=output_dir,
            run_mode=run_mode,
            repository=str(getattr(context, "repository", "unknown")),
            project_accession=getattr(primary, "project_accession", None),
            stage=stage,
            events=events,
            artifacts=artifacts,
            current_threads=getattr(plan, "thread_num", None) if plan is not None else None,
            detected_by="agent.web.app",
        )
        write_recovery_audit(output_dir, audit)
    except Exception as exc:
        if report is not None:
            try:
                report(f"Failed to write recovery audit file: {exc}")
            except Exception:
                pass


def _write_parameter_audit_files(output_dir: Path, batch_id: str, index: int, input_value: str, result: Any) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    resolution = getattr(result, "resolution", None)
    primary = getattr(resolution, "primary_project", None)
    context = getattr(result, "context", None)
    attributes = getattr(result, "attributes", None)
    plan = getattr(result, "plan", None)
    asset = getattr(result, "asset", None)
    hints = _search_parameter_hints(attributes)
    asset_payload: dict[str, Any] = {}
    try:
        loaded_asset = json.loads((output_dir / "asset_resolution.json").read_text(encoding="utf-8"))
        if isinstance(loaded_asset, dict):
            asset_payload = loaded_asset
    except (OSError, json.JSONDecodeError):
        asset_payload = {}

    def asset_field(name: str) -> Any:
        value = getattr(asset, name, None)
        return value if value not in (None, "") else asset_payload.get(name)

    def first_field(*values: Any) -> Any:
        for value in values:
            if value not in (None, ""):
                return value
        return None

    repository = _clean_repository(
        first_field(
            getattr(context, "repository", None),
            getattr(primary, "repository", None),
            asset_field("repository"),
            asset_payload.get("repository"),
        )
    )

    materialized_workflow = _materialize_parameter_workflow(output_dir, attributes, plan) if attributes is not None and plan is not None else None
    converter_config = output_dir / "converter_config.json"
    _rewrite_converter_config_workflow(converter_config, materialized_workflow)

    workflow_template = getattr(plan, "fragpipe_workflow_path", None) if plan is not None else None
    fasta_path = getattr(plan, "fasta_path", None) if plan is not None else None
    fasta_url = getattr(plan, "fasta_download_url", None) if plan is not None else None
    audit = {
        "batch_id": batch_id,
        "index": index,
        "repository": repository,
        "input_value": input_value,
        "generated_at": _now_iso(),
        "project": {
            "repository": repository,
            "accession": getattr(primary, "project_accession", None),
            "native_accession": first_field(getattr(primary, "native_accession", None), getattr(context, "native_accession", None)),
            "px_accession": first_field(getattr(primary, "px_accession", None), getattr(context, "px_accession", None)),
            "matched_file": getattr(primary, "matched_file", None),
            "match_type": getattr(primary, "match_type", None),
            "match_score": getattr(primary, "match_score", None),
            "needs_review": bool(getattr(resolution, "needs_review", False)) if resolution is not None else False,
        },
        "input": {
            "original_file_name": asset_field("original_file_name") or getattr(plan, "source_file_name", None),
            "matched_project_file": asset_field("matched_project_file") or getattr(primary, "matched_file", None),
            "logical_path": asset_field("logical_path"),
            "asset_type": asset_field("resolved_asset_type"),
            "download_url": asset_field("download_url"),
            "download_urls": asset_field("download_urls") or [],
            "transfer_method": asset_field("transfer_method"),
            "expected_size_bytes": asset_field("expected_size_bytes"),
            "requires_conversion": asset_field("requires_conversion"),
        },
        "plan": {
            "source_file_name": getattr(plan, "source_file_name", None),
            "source_data_path": str(getattr(plan, "source_data_path", "")) if plan is not None else "",
            "raw_data_type": getattr(plan, "raw_data_type", None),
            "thread_num": getattr(plan, "thread_num", None),
            "needs_review": bool(getattr(plan, "needs_review", False)) if plan is not None else False,
        },
        "workflow": {
            "name": Path(str(workflow_template)).name if workflow_template else "",
            "template_path": str(workflow_template) if workflow_template else "",
            "materialized_path": str(materialized_workflow) if materialized_workflow else "",
            "parameter_overrides": hints.get("workflow_parameter_overrides")
            or hints.get("fragpipe_workflow_overrides")
            or hints.get("msfragger_parameter_overrides")
            or {},
        },
        "fasta": {
            "name": Path(str(fasta_path)).name if fasta_path else "",
            "path": str(fasta_path) if fasta_path else "",
            "selection_mode": getattr(plan, "fasta_selection_mode", None) if plan is not None else None,
            "download_url": fasta_url or hints.get("recommended_fasta_url") or hints.get("fasta_url"),
        },
        "search_parameters": {
            "acquisition_mode": _attribute_value(attributes, "acquisition_mode"),
            "species": _attribute_value(attributes, "species"),
            "instrument_name": _attribute_value(attributes, "instrument_name"),
            "enzyme": _attribute_value(attributes, "enzyme"),
            "labeling_strategy": _attribute_value(attributes, "labeling_strategy"),
            "fixed_mods": _attribute_value(attributes, "fixed_mods"),
            "variable_mods": _attribute_value(attributes, "variable_mods"),
            "hints": hints,
        },
        "files": {
            "converter_config": str(converter_config),
            "fragpipe_manifest": str(getattr(plan, "manifest_path", "")) if plan is not None else "",
            "decision_trace": str(output_dir / "decision_trace.json"),
            "attributes": str(output_dir / "attributes.json"),
            "asset_resolution": str(output_dir / "asset_resolution.json"),
        },
        "expected_outputs": {
            "rawspectrum": str(getattr(plan, "rawspectrum_output_path", "")) if plan is not None else "",
            "fp_pin": str(getattr(plan, "expected_pin_path", "")) if plan is not None else "",
            "fp_msdt": _plan_output_path(plan, "fp_msdt") if plan is not None else "",
        },
        "blocking_issues": list(getattr(plan, "blocking_issues", []) or []) if plan is not None else [],
    }
    audit = _plain(audit)
    _json_write(output_dir / "parameter_audit.json", audit)
    audit_files = [
        "project_resolution.json",
        "metadata.json",
        "asset_resolution.json",
        "attributes.json",
        "decision_trace.json",
        "agent_observation.json",
        "agent_plan.json",
        "agent_decision_trace.json",
        "parameter_audit.json",
        "task_state.json",
        "logs/runtime.log",
    ]
    manifest = {
        "package_type": "parameter_only_msdt_input_preview",
        "generated_at": _now_iso(),
        "repository": audit.get("repository"),
        "input_file": audit.get("input", {}).get("original_file_name"),
        "project_accession": audit.get("project", {}).get("accession"),
        "run_without_full_execution": True,
        "note": (
            "This package contains the planned MSDT-Converter configuration and audit files. "
            "RAW/mzML data and FASTA sequences are not downloaded in parameter-only mode."
        ),
        "msdt_converter_inputs": {
            "converter_config": "converter_config.json",
            "workflow": _relative_package_path(output_dir, materialized_workflow) if materialized_workflow else "",
            "source_data_path_expected": audit.get("plan", {}).get("source_data_path", ""),
            "fasta_path_expected": audit.get("fasta", {}).get("path", ""),
            "fasta_download_url": audit.get("fasta", {}).get("download_url", ""),
            "fragpipe_manifest_expected": audit.get("files", {}).get("fragpipe_manifest", ""),
        },
        "audit_files": [path for path in audit_files if (output_dir / path).exists()],
    }
    _json_write(output_dir / "msdt_input_manifest.json", _plain(manifest))
    return audit


def _relative_package_path(root: Path, path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _batch_audit_zip_path(batch: dict[str, Any]) -> Path:
    return Path(batch.get("output_dir", "")) / _BATCH_AUDIT_ZIP_NAME


def _include_batch_audit_file(root: Path, file: Path) -> bool:
    try:
        rel = file.relative_to(root)
    except ValueError:
        return False
    parts = {part.lower() for part in rel.parts}
    if file.name == _BATCH_AUDIT_ZIP_NAME:
        return False
    if {"downloads", "prepared", "input", "fasta"} & parts:
        return False
    if file.suffix.lower() in {".raw", ".mzml", ".mzxml", ".wiff", ".scan", ".d", ".fasta", ".fa", ".fas", ".gz", ".zip"}:
        return False
    return True


def _ensure_batch_audit_zip(batch: dict[str, Any]) -> Path | None:
    root = Path(batch.get("output_dir", ""))
    if not root.exists() or not root.is_dir():
        return None
    zip_path = root / _BATCH_AUDIT_ZIP_NAME
    files = [file for file in root.rglob("*") if file.is_file() and _include_batch_audit_file(root, file)]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(files):
            archive.write(file, file.relative_to(root).as_posix())
    return zip_path


def _write_batch_manifest(batch: dict[str, Any]) -> None:
    manifest = {key: value for key, value in batch.items() if key not in {"llm_config"}}
    _write_json_atomic(Path(batch["output_dir"]) / _BATCH_MANIFEST_FILE, manifest)
    with _batch_history_cache_lock:
        _batch_history_cache["ts"] = 0.0
        _batch_history_cache["root"] = ""
        _batch_history_cache["generation"] = int(
            _batch_history_cache.get("generation") or 0
        ) + 1
        _batch_history_cache["records"] = []
    try:
        batch_id = str(batch.get("batch_id") or "").strip()
        if batch_id:
            history_record = _batch_history_record(
                batch,
                include_file_stats=False,
            )
            get_operations_repository().upsert_history_record(
                history_record,
                kind="batch",
                source_id=batch_id,
                history_id=f"batch:{batch_id}",
                size_bytes=int(history_record.get("size_bytes") or 0),
            )
    except Exception:
        # The durable batch manifest remains the recovery authority if the
        # operations index is temporarily unavailable.
        pass



def _infer_batch_stage_from_message(message: str) -> tuple[str, str]:
    text = str(message or "")
    lower = text.casefold()
    if "下载完成" in text or ("download" in lower and "complete" in lower):
        return "download", "下载完成"
    if "download_progress" in lower or "正在下载" in text or "下载" in text or "download" in lower:
        return "download", "下载数据"
    if any(token in text for token in ("[1/5]", "解析", "resolve", "matching project", "Querying pride")):
        return "resolve_project", "解析项目/文件"
    if any(token in text for token in ("[2/5]", "[3/5]", "[4/5]", "SDRF", "属性", "metadata", "workflow", "FASTA", "搜库参数")):
        return "infer_metadata", "推断元数据/参数"
    if any(token in text for token in ("[5/5]", "执行计划", "execution plan")):
        return "plan_ready", "计划已生成"
    if any(token in lower for token in ("docker", "msdt", "convert", "转换", "proteowizard", "msconvert")):
        return "convert", "格式转换"
    if any(token in lower for token in ("fragpipe", "philosopher", "搜库", "execution", "running msdt")):
        return "execute", "执行工作流"
    if any(token in text for token in ("Excel", "audit", "打包", "zip", "export")):
        return "export", "写结果/汇总"
    if any(token in lower for token in ("error", "failed", "失败", "错误")):
        return "failed", "失败"
    return "running", "处理中"


def _summarize_batch_error(error: str) -> str:
    text = _clean_text(error)
    if not text:
        return ""
    # Keep first sentence / clause for list row.
    for sep in ("。", ". ", "\n", "; ", "；"):
        if sep in text:
            text = text.split(sep, 1)[0].strip()
            break
    return text[:180]


def _enrich_batch_item_progress(item: dict[str, Any], *, ui_language: str) -> dict[str, Any]:
    """Ensure each public item has progress + error_summary for the status bar."""
    status = str(item.get("status") or "queued").strip().lower()
    progress = dict(item.get("progress") or {}) if isinstance(item.get("progress"), dict) else {}
    log_tail = [str(line) for line in (item.get("log_tail") or []) if str(line).strip()]
    error = _localize_public_message(item.get("error", ""), ui_language, level="error")
    existing_summary = item.get("error_summary")
    if isinstance(existing_summary, dict):
        error_summary = _summarize_batch_error(
            str(existing_summary.get("public_message") or existing_summary.get("message") or error)
        ) or error
    elif existing_summary:
        error_summary = _summarize_batch_error(str(existing_summary)) or str(existing_summary)
    else:
        error_summary = _summarize_batch_error(error)

    if status == "queued" and not progress:
        progress = {
            "stage": "queued",
            "stage_label": "排队",
            "percent": None,
            "message": "等待执行",
            "updated_at": item.get("started_at") or item.get("updated_at") or "",
        }
    elif status == "completed":
        progress = {
            "stage": "completed",
            "stage_label": "完成",
            "percent": 100.0,
            "message": progress.get("message") or "已完成",
            "updated_at": item.get("finished_at") or progress.get("updated_at") or "",
            **({"download": progress["download"]} if progress.get("download") else {}),
        }
    elif status in {"failed", "needs_review", "blocked"}:
        stage = str(progress.get("stage") or "failed")
        if stage in {"queued", "running", ""}:
            stage = "failed" if status == "failed" else status
        # Infer failed stage from logs when missing.
        if not progress.get("stage") or progress.get("stage") in {"running", "queued"}:
            for line in reversed(log_tail[-15:]):
                inferred, label = _infer_batch_stage_from_message(line)
                if inferred not in {"running", "queued", "failed"}:
                    progress = {
                        "stage": inferred,
                        "stage_label": label,
                        "percent": progress.get("percent"),
                        "message": error_summary or line[:200],
                        "updated_at": item.get("finished_at") or progress.get("updated_at") or "",
                        "failed_stage": inferred,
                    }
                    break
        progress.setdefault("stage", stage)
        progress.setdefault("stage_label", "失败" if status == "failed" else "需复核")
        progress["message"] = error_summary or progress.get("message") or error or "失败"
        progress["failed_stage"] = progress.get("failed_stage") or progress.get("stage")
    elif not progress and log_tail:
        stage, label = _infer_batch_stage_from_message(log_tail[-1])
        progress = {
            "stage": stage,
            "stage_label": label,
            "percent": None,
            "message": log_tail[-1][:300],
            "updated_at": item.get("updated_at") or "",
        }
    elif not progress and status == "running":
        progress = {
            "stage": "running",
            "stage_label": "处理中",
            "percent": None,
            "message": "运行中",
            "updated_at": item.get("started_at") or "",
        }

    item["progress"] = progress
    item["error_summary"] = error_summary
    if progress.get("failed_stage"):
        item["failed_stage"] = progress.get("failed_stage")
    return item


def _public_batch_record(batch: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(batch.get("output_dir", ""))
    excel_path = output_dir / _BATCH_EXCEL_FILE
    ui_language = _clean_ui_language(batch.get("ui_language"))
    items = [dict(item) for item in batch.get("items") or []]
    for item in items:
        item_dir = Path(item.get("output_dir", ""))
        item["error"] = _localize_public_message(item.get("error", ""), ui_language, level="error")
        item["log_tail"] = [
            _localize_public_message(line, ui_language, level="info")
            for line in _tail_text_file(item_dir / "logs" / "runtime.log", max_lines=40, max_chars=12000)
        ]
        audit_path = item_dir / "parameter_audit.json"
        item["audit_path"] = str(audit_path) if audit_path.exists() else ""
        _enrich_batch_item_progress(item, ui_language=ui_language)
    events = []
    for event in list(batch.get("events") or [])[-500:]:
        public_event = dict(event)
        public_event["message"] = _localize_public_message(
            public_event.get("message", ""),
            ui_language,
            level=str(public_event.get("level") or "info"),
        )
        events.append(public_event)
    completed_items = sum(1 for item in items if item.get("status") == "completed")
    failed_items = sum(1 for item in items if item.get("status") == "failed")
    needs_review_items = sum(1 for item in items if item.get("status") in {"needs_review", "blocked"})
    cancelled_items = sum(1 for item in items if item.get("status") == "cancelled")
    interrupted_items = sum(1 for item in items if item.get("status") == "interrupted")
    running_items = sum(1 for item in items if item.get("status") == "running")
    queued_items = sum(1 for item in items if item.get("status") in {"queued", "pending", ""})
    cleanup_requested = bool(batch.get("delete_source_files_after_success"))
    cleanup_released_bytes = sum(
        int((item.get("source_cleanup") or {}).get("released_bytes") or 0)
        for item in items
    )
    cleanup_completed_items = sum(
        1
        for item in items
        if str((item.get("source_cleanup") or {}).get("status") or "")
        in {"completed", "partial"}
    )
    cleanup_failed_items = sum(
        1
        for item in items
        if str((item.get("source_cleanup") or {}).get("status") or "") == "failed"
    )
    terminal_items = (
        completed_items
        + failed_items
        + needs_review_items
        + cancelled_items
        + interrupted_items
    )
    total_items = len(items) or 1
    focus = next((item for item in items if item.get("status") == "running"), None)
    if focus is None:
        focus = next((item for item in items if item.get("status") == "failed"), None)
    summary = {
        "queued": queued_items,
        "running": running_items,
        "completed": completed_items,
        "failed": failed_items,
        "needs_review": needs_review_items,
        "cancelled": cancelled_items,
        "interrupted": interrupted_items,
        "source_cleanup_requested": cleanup_requested,
        "source_cleanup_completed": cleanup_completed_items,
        "source_cleanup_failed": cleanup_failed_items,
        "source_cleanup_released_bytes": cleanup_released_bytes,
        "percent": round(100.0 * terminal_items / total_items, 1),
        "focus_item_index": focus.get("index") if isinstance(focus, dict) else None,
        "focus_message": (
            str((focus.get("progress") or {}).get("message") or focus.get("error_summary") or focus.get("input") or "")
            if isinstance(focus, dict)
            else ""
        ),
    }
    return {
        "batch_id": batch.get("batch_id", ""),
        "status": batch.get("status", "unknown"),
        "submitter": batch.get("submitter", ""),
        "created_at": batch.get("created_at"),
        "started_at": batch.get("started_at"),
        "finished_at": batch.get("finished_at"),
        "updated_at": batch.get("updated_at") or batch.get("finished_at") or batch.get("started_at") or batch.get("created_at"),
        "item_count": len(items),
        "completed_items": completed_items,
        "failed_items": failed_items,
        "needs_review_items": needs_review_items,
        "cancelled_items": cancelled_items,
        "interrupted_items": interrupted_items,
        "running_items": running_items,
        "queued_items": queued_items,
        "completed_count": completed_items,
        "failed_count": failed_items,
        "running_count": running_items,
        "queued_count": queued_items,
        "needs_review_count": needs_review_items,
        "cancelled_count": cancelled_items,
        "progress_percent": summary["percent"],
        "summary": summary,
        "jobs": batch.get("jobs", 1),
        "ui_language": ui_language,
        "repository": _clean_repository(batch.get("repository")),
        "run_mode": _clean_batch_run_mode(batch.get("run_mode")),
        "requested_run_mode": _clean_batch_run_mode(
            batch.get("requested_run_mode") or batch.get("run_mode")
        ),
        "resource_policy": _clean_resource_policy(batch.get("resource_policy")),
        "fasta_preference": "project" if batch.get("prefer_project_fasta") else "llm",
        "delete_source_files_after_success": cleanup_requested,
        "source_discovery_job_id": batch.get("source_discovery_job_id"),
        "source_discovery_id": batch.get("source_discovery_id"),
        "source_batch_index": batch.get("source_batch_index"),
        "output_dir": str(output_dir),
        "excel_path": str(excel_path),
        "can_download": batch.get("status") in _TERMINAL_STATUSES and excel_path.exists(),
        "audit_zip_path": str(output_dir / _BATCH_AUDIT_ZIP_NAME),
        "can_download_audit": batch.get("status") in _TERMINAL_STATUSES and output_dir.exists(),
        "items": items,
        "events": events,
        "errors": [_localize_public_message(error, ui_language, level="error") for error in list(batch.get("errors") or [])],
        "interrupted": bool(batch.get("interrupted")),
        "cancel_requested": bool(batch.get("cancel_requested")),
        "cancel_requested_at": batch.get("cancel_requested_at"),
    }


def _mark_interrupted_batch(batch: dict[str, Any]) -> dict[str, Any]:
    if batch.get("status") not in _ACTIVE_STATUSES:
        return batch
    repaired = dict(batch)
    repaired["status"] = "interrupted"
    repaired["interrupted"] = True
    repaired["finished_at"] = repaired.get("finished_at") or repaired.get("updated_at") or repaired.get("started_at") or repaired.get("created_at")
    repaired["updated_at"] = repaired.get("updated_at") or repaired.get("finished_at")
    errors = [str(value) for value in repaired.get("errors") or [] if str(value)]
    if _INTERRUPTED_HISTORY_MESSAGE not in errors:
        errors.append(_INTERRUPTED_HISTORY_MESSAGE)
    repaired["errors"] = errors
    return repaired


def _batch_history_record(batch: dict[str, Any], include_file_stats: bool = True) -> dict[str, Any]:
    public = _public_batch_record(batch)
    batch_id = str(public.get("batch_id") or "").strip()
    output_dir = Path(public.get("output_dir") or "")
    file_count = int(
        batch.get("file_count")
        or public.get("file_count")
        or public.get("item_count")
        or 0
    )
    size_bytes = int(batch.get("size_bytes") or public.get("size_bytes") or 0)
    if include_file_stats and output_dir.exists():
        file_count, size_bytes, _latest_mtime = _path_file_stats(output_dir)
    run_mode = _clean_batch_run_mode(batch.get("run_mode"))
    display_name = f"Batch workflow · {run_mode} · {int(public.get('item_count') or 0)} files"
    public.update(
        {
            "kind": "batch",
            "task_id": f"batch-{batch_id}" if batch_id else "",
            "project_key": f"batch-{batch_id}" if batch_id else "batch",
            "history_id": f"batch-{batch_id}" if batch_id else "",
            "run_id": output_dir.name if output_dir.name else batch_id,
            "result_id": batch_id,
            "name": display_name,
            "display_name": display_name,
            "input_value": display_name,
            "objective": display_name,
            "run_mode": run_mode,
            "file_count": file_count,
            "size_bytes": size_bytes,
        }
    )
    # History cards only need aggregate batch progress. Returning every item and
    # event makes the history response grow with all processed files.
    public.pop("items", None)
    public.pop("events", None)
    return _decorate_history_item(public)


def _load_batch_from_disk(batch_id: str) -> dict[str, Any] | None:
    manifest_path = _batch_manifest_path(batch_id)
    if not manifest_path.exists():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _list_parameter_batch_history_records(use_cache: bool = True, include_file_stats: bool = True) -> list[dict[str, Any]]:
    cache_root = str(_batch_root_dir().resolve())
    cache_generation = 0
    if use_cache and not include_file_stats:
        with _batch_history_cache_lock:
            cache_generation = int(_batch_history_cache.get("generation") or 0)
            cached_ts = float(_batch_history_cache.get("ts") or 0.0)
            if (
                _batch_history_cache.get("root") == cache_root
                and time.time() - cached_ts < 20
            ):
                return [dict(item) for item in _batch_history_cache.get("records") or []]
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with _batches_lock:
        memory_batches = [dict(batch) for batch in _batches.values()]
    for batch in memory_batches:
        batch_id = str(batch.get("batch_id") or "").strip()
        if not batch_id:
            continue
        seen.add(batch_id)
        records.append(_batch_history_record(batch, include_file_stats=include_file_stats))

    batch_root = _batch_root_dir()
    if not batch_root.exists() or not batch_root.is_dir():
        return records
    for batch_dir in batch_root.iterdir():
        if not batch_dir.is_dir():
            continue
        batch_id = batch_dir.name
        if batch_id in seen:
            continue
        batch = _load_batch_from_disk(batch_id)
        if batch is None:
            continue
        if batch.get("status") in _ACTIVE_STATUSES:
            batch = _mark_interrupted_batch(batch)
            _write_batch_manifest(batch)
        seen.add(batch_id)
        records.append(_batch_history_record(batch, include_file_stats=include_file_stats))
    if use_cache and not include_file_stats:
        with _batch_history_cache_lock:
            if int(_batch_history_cache.get("generation") or 0) == cache_generation:
                _batch_history_cache["ts"] = time.time()
                _batch_history_cache["root"] = cache_root
                _batch_history_cache["records"] = [dict(item) for item in records]
    return records


def _format_bytes(size: int | float) -> str:
    size = float(size)
    if size >= 1024 * 1024 * 1024:
        return f"{size / 1024 / 1024 / 1024:.1f} GB"
    if size >= 1024 * 1024:
        return f"{size / 1024 / 1024:.1f} MB"
    if size >= 1024:
        return f"{size / 1024:.1f} KB"
    return f"{int(size)} B"


def _pride_cache_dir() -> Path:
    configured = os.getenv("AGENT_PRIDE_CACHE_DIR")
    if configured:
        return Path(configured)
    return Path.cwd() / ".agent_cache" / "pride"


def _path_file_stats(path: Path, excluded_names: set[str] | None = None, excluded_dir_names: set[str] | None = None) -> tuple[int, int, float]:
    excluded_names = excluded_names or set()
    excluded_dir_names = excluded_dir_names or set()
    file_count = 0
    size_bytes = 0
    latest_mtime = 0.0
    try:
        latest_mtime = path.stat().st_mtime
    except OSError:
        return 0, 0, 0.0
    for file in path.rglob("*"):
        try:
            relative = file.relative_to(path)
        except ValueError:
            continue
        if any(part in excluded_dir_names for part in relative.parts[:-1]):
            continue
        if file.is_file() and file.name in excluded_names:
            continue
        try:
            stat = file.stat()
        except OSError:
            continue
        latest_mtime = max(latest_mtime, stat.st_mtime)
        if file.is_file():
            file_count += 1
            size_bytes += stat.st_size
    return file_count, size_bytes, latest_mtime


def _active_output_dirs_locked() -> set[Path]:
    active_dirs: set[Path] = set()
    for task in _tasks.values():
        if task.get("status") not in _ACTIVE_STATUSES:
            continue
        output_dir = task.get("output_dir")
        if not output_dir:
            continue
        active_dirs.add(Path(output_dir).resolve())
    return active_dirs


def _safe_run_dir(result_id: str) -> Path | None:
    if not result_id or safe_output_stem(result_id) != result_id:
        return None
    root = _runs_dir.resolve()
    candidate = (_runs_dir / result_id).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _read_public_history(run_dir: Path) -> dict[str, Any]:
    history_path = run_dir / _PUBLIC_HISTORY_FILE
    if not history_path.exists():
        return {}
    try:
        data = json.loads(history_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _project_key_for_input(input_value: str) -> str:
    return safe_output_stem(input_value)


def _history_output_names_for_project(project_key: str) -> set[str]:
    names: set[str] = set()
    if not project_key:
        return names
    for record in _read_history_index():
        item = with_history_identity(record)
        if str(item.get("project_key") or "") != project_key:
            continue
        for field in ("output_dir", "result_id", "history_id", "run_id", "name"):
            value = str(item.get(field) or "").strip()
            if value:
                names.add(Path(value).name)
    return names


def _next_output_dir_locked(project_key: str, task_id: str) -> Path:
    base = safe_output_stem(project_key) or safe_output_stem(task_id)
    used = _history_output_names_for_project(base)
    used.update(path.name for path in _active_output_dirs_locked())
    if (_runs_dir / base).exists():
        used.add(base)
    if base not in used:
        return _runs_dir / base

    timestamp = datetime.now(_APP_TZ).strftime("%Y%m%d-%H%M%S")
    task_suffix = safe_output_stem(task_id)[:8] or uuid.uuid4().hex[:8]
    stem = f"{base}__{timestamp}__{task_suffix}"
    candidate = stem
    counter = 2
    while candidate in used or (_runs_dir / candidate).exists():
        candidate = f"{stem}-{counter}"
        counter += 1
    return _runs_dir / candidate


def _public_task_record_locked(task_id: str, task: dict[str, Any], *, include_logs: bool = False) -> dict[str, Any]:
    output_dir_raw = task.get("output_dir")
    output_dir = Path(output_dir_raw) if output_dir_raw else None
    can_download = bool(
        task.get("status") == "completed"
        and output_dir
        and output_dir.exists()
        and _is_download_zip_ready(output_dir)
    )
    logs = _public_logs_from_task(task)
    updated_at = task.get("updated_at") or task.get("finished_at") or task.get("started_at") or task.get("created_at")
    record = {
        "task_id": task_id,
        "input_value": task.get("input_value", ""),
        "project_key": task.get("project_key") or _project_key_for_input(str(task.get("input_value", ""))),
        "submitter": task.get("submitter", "未填写"),
        "status": task.get("status", "unknown"),
        "created_at": task.get("created_at"),
        "started_at": task.get("started_at"),
        "finished_at": task.get("finished_at"),
        "updated_at": updated_at,
        "step": task.get("step", 0),
        "total_steps": task.get("total_steps", 5),
        "queue_position": _queue_position_locked(task_id),
        "output_dir": Path(output_dir).name if output_dir else "",
        "run_id": Path(output_dir).name if output_dir else "",
        "log_count": len(logs),
        "blocking_issues": list(task.get("blocking_issues") or []),
        "error_summary": task.get("error_summary"),
        "workflow_outcome": task.get("workflow_outcome"),
        "usable_partial_outputs": bool(task.get("usable_partial_outputs")),
        "recovery_primary_issue": task.get("recovery_primary_issue"),
        "recovery_recommended_next_step": task.get("recovery_recommended_next_step"),
        "recovery_report_json": task.get("recovery_report_json"),
        "recovery_report_md": task.get("recovery_report_md"),
        "review_summary": task.get("review_summary"),
        "fasta_preference": "project" if task.get("prefer_project_fasta") else "llm",
        "run_mode": _clean_run_mode(task.get("run_mode")),
        "resource_policy": _clean_resource_policy(task.get("resource_policy")),
        "ui_language": _clean_ui_language(task.get("ui_language")),
        "repository": _clean_repository(task.get("repository")),
        "can_download": can_download,
    }
    if include_logs:
        record["logs"] = logs
    return _decorate_history_item(record)


def _history_index_path() -> Path:
    return _runs_dir / _HISTORY_INDEX_FILE


def _history_index_backup_path() -> Path:
    return _runs_dir / f"{_HISTORY_INDEX_FILE}.bak"


def _is_legacy_batches_history_record(record: dict[str, Any]) -> bool:
    names = {
        str(record.get("task_id") or ""),
        str(record.get("input_value") or ""),
        _identity_name(record.get("output_dir")),
        _identity_name(record.get("history_id")),
        _identity_name(record.get("run_id")),
        _identity_name(record.get("result_id")),
        _identity_name(record.get("name")),
        str(record.get("project_key") or ""),
    }
    names.discard("")
    return _BATCHES_DIR_NAME in names or names == {"batches"}


def _read_history_index_file(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict) and not _is_legacy_batches_history_record(item)]


def _read_history_index() -> list[dict[str, Any]]:
    with _history_index_lock:
        records = _read_history_index_file(_history_index_path())
        if records:
            return records
        return _read_history_index_file(_history_index_backup_path())


def _upsert_history_index(record: dict[str, Any]) -> None:
    indexed_record: dict[str, Any]
    with _history_index_lock:
        indexed_record = _compact_history_index_record(record)
        if not indexed_record.get("project_key"):
            return
        records = merge_project_history_records([*_read_history_index(), indexed_record])
        _write_history_index(records)
    try:
        kind = _clean_text(indexed_record.get("kind")).lower()
        if not kind:
            kind = (
                "batch"
                if indexed_record.get("batch_id")
                else "discovery"
                if indexed_record.get("discovery_id")
                or indexed_record.get("job_id")
                else "task"
            )
        source_id = _clean_text(
            indexed_record.get("job_id")
            if kind == "discovery"
            else indexed_record.get("batch_id")
            if kind == "batch"
            else indexed_record.get("task_id")
            or indexed_record.get("result_id")
        )
        if source_id:
            if not is_material_legacy_history_record(
                indexed_record,
                source_id=source_id,
            ):
                return
            try:
                size_bytes = int(indexed_record.get("size_bytes") or 0)
            except (TypeError, ValueError):
                size_bytes = 0
            get_operations_repository().upsert_history_record(
                indexed_record,
                kind=kind,
                source_id=source_id,
                history_id=f"{kind}:{source_id}",
                size_bytes=max(0, size_bytes),
            )
    except Exception:
        # The compatibility JSON history remains available if the operations
        # index is temporarily unavailable. Startup reconciliation is
        # idempotent and will repair the row later.
        pass


def _compact_history_index_record(record: Mapping[str, Any]) -> dict[str, Any]:
    item = with_history_identity(record)
    # Archived task details still use their concise task log after result cleanup.
    # Batch item/event arrays are the unbounded payloads and belong in manifests.
    for key in ("items", "events"):
        item.pop(key, None)
    return item


def _write_history_index(records: list[dict[str, Any]]) -> None:
    with _history_index_lock:
        try:
            _runs_dir.mkdir(parents=True, exist_ok=True)
            cleaned = [
                _compact_history_index_record(record)
                for record in records
                if not _is_legacy_batches_history_record(record)
            ]
            payload = json.dumps(cleaned, indent=2, ensure_ascii=False)
            path = _history_index_path()
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            tmp_path.write_text(payload, encoding="utf-8")
            os.replace(tmp_path, path)
            _history_index_backup_path().write_text(payload, encoding="utf-8")
        except OSError:
            return


def _write_task_history(task_id: str) -> None:
    with _tasks_lock:
        task = _tasks.get(task_id)
        if task is None:
            return
        output_dir_raw = task.get("output_dir")
        if not output_dir_raw:
            return
        record = _public_task_record_locked(task_id, task, include_logs=True)
    output_dir = Path(output_dir_raw)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / _PUBLIC_HISTORY_FILE).write_text(
            json.dumps(record, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _upsert_history_index(record)
    except OSError:
        return


def _archive_run_history(run_dir: Path, history: dict[str, Any] | None = None) -> None:
    record = dict(history or _read_public_history(run_dir))
    if not record:
        record = {
            "task_id": run_dir.name,
            "input_value": run_dir.name,
            "status": "completed",
            "output_dir": run_dir.name,
        }
    record.setdefault("output_dir", run_dir.name)
    record["can_download"] = False
    _upsert_history_index(record)


def _identity_name(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return Path(text).name


def _active_history_identity_sets_locked() -> tuple[set[str], set[str]]:
    task_ids: set[str] = set()
    history_ids: set[str] = set()
    for task_id, task in _tasks.items():
        if task.get("status") not in _ACTIVE_STATUSES:
            continue
        task_ids.add(str(task_id))
        task_ids.add(str(task.get("task_id") or ""))
        for field in ("history_id", "run_id", "result_id", "name", "output_dir"):
            name = _identity_name(task.get(field))
            if name:
                history_ids.add(name)
    task_ids.discard("")
    history_ids.discard("")
    return task_ids, history_ids


def _history_item_is_active(item: dict[str, Any], active_task_ids: set[str], active_history_ids: set[str]) -> bool:
    task_ids = {str(item.get("task_id") or ""), *(str(value or "") for value in item.get("task_ids") or [])}
    history_ids = {_identity_name(item.get(field)) for field in ("history_id", "output_dir", "run_id", "result_id", "name")}
    task_ids.discard("")
    history_ids.discard("")
    return bool(task_ids & active_task_ids or history_ids & active_history_ids)


def _mark_interrupted_history_item(item: dict[str, Any]) -> dict[str, Any]:
    if item.get("status") not in _ACTIVE_STATUSES:
        return item
    repaired = dict(item)
    repaired["status"] = "failed"
    repaired["interrupted"] = True
    repaired["finished_at"] = repaired.get("finished_at") or repaired.get("updated_at") or repaired.get("started_at") or repaired.get("created_at")
    repaired["updated_at"] = repaired.get("updated_at") or repaired.get("finished_at")
    issues = [str(value) for value in repaired.get("blocking_issues") or [] if str(value)]
    if _INTERRUPTED_HISTORY_MESSAGE not in issues:
        issues.append(_INTERRUPTED_HISTORY_MESSAGE)
    repaired["blocking_issues"] = issues
    return with_history_identity(repaired)


def _repair_interrupted_history_index() -> None:
    with _history_index_lock:
        with _tasks_lock:
            active_task_ids, active_history_ids = _active_history_identity_sets_locked()
        repaired: list[dict[str, Any]] = []
        changed = False
        for record in _read_history_index():
            item = with_history_identity(record)
            if item.get("status") in _ACTIVE_STATUSES and not _history_item_is_active(item, active_task_ids, active_history_ids):
                item = _mark_interrupted_history_item(item)
                changed = True
            repaired.append(item)
        if changed:
            _write_history_index(merge_project_history_records(repaired))


def _disk_task_history_records() -> list[dict[str, Any]]:
    if not _runs_dir.exists():
        return []
    records: list[dict[str, Any]] = []
    for run_dir in _runs_dir.iterdir():
        if not run_dir.is_dir() or run_dir.name == _BATCHES_DIR_NAME:
            continue
        history = _read_public_history(run_dir)
        if not history:
            continue
        history.setdefault("output_dir", run_dir.name)
        records.append(_decorate_history_item(history))
    return records


def _sync_history_index_from_disk() -> None:
    with _history_index_lock:
        records = [*_read_history_index(), *_disk_task_history_records()]
        for batch in _list_parameter_batch_history_records(use_cache=False):
            if batch.get("status") in _ACTIVE_STATUSES:
                continue
            records.append(batch)
        if not records:
            return
        _write_history_index(merge_project_history_records(records))


def _find_history_record(task_id: str) -> dict[str, Any] | None:
    needle = str(task_id or "")
    for record in reversed(_read_history_index()):
        item = with_history_identity(record)
        aliases = {
            str(item.get("task_id") or ""),
            str(item.get("output_dir") or ""),
            str(item.get("history_id") or ""),
            str(item.get("run_id") or ""),
            str(item.get("result_id") or ""),
            str(item.get("name") or ""),
        }
        aliases.update(str(value or "") for value in item.get("task_ids") or [])
        if needle in aliases:
            return item
    if not _runs_dir.exists():
        return None
    for run_dir in _runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        record = _read_public_history(run_dir)
        if not record:
            continue
        item = with_history_identity({**record, "output_dir": record.get("output_dir") or run_dir.name})
        aliases = {
            str(item.get("task_id") or ""),
            str(item.get("output_dir") or run_dir.name),
            str(item.get("history_id") or ""),
            str(item.get("run_id") or ""),
            str(item.get("result_id") or ""),
            str(item.get("name") or ""),
        }
        aliases.update(str(value or "") for value in item.get("task_ids") or [])
        if needle in aliases:
            return item
    return None


def _public_logs_from_history(record: dict[str, Any]) -> list[dict[str, Any]]:
    raw_logs = record.get("logs")
    if not isinstance(raw_logs, list):
        return []
    logs: list[dict[str, Any]] = []
    for entry in raw_logs[-_MAX_PERSISTED_LOGS:]:
        sanitized = _sanitize_log_entry(entry)
        if sanitized:
            logs.append(sanitized)
    return logs


def _task_detail_from_history(task_id: str, record: dict[str, Any]) -> dict[str, Any]:
    record = with_history_identity(record)
    output_dir_name = str(record.get("output_dir") or "")
    output_dir = _runs_dir / output_dir_name if output_dir_name else None
    can_download = bool(
        record.get("status") == "completed"
        and output_dir
        and output_dir.exists()
        and _is_download_zip_ready(output_dir)
    )
    logs = _public_logs_from_history(record)
    detail = {
        "task_id": str(record.get("task_id") or task_id),
        "input_value": record.get("input_value", ""),
        "submitter": record.get("submitter", "未填写"),
        "status": record.get("status", "unknown"),
        "created_at": record.get("created_at"),
        "started_at": record.get("started_at"),
        "finished_at": record.get("finished_at"),
        "updated_at": record.get("updated_at") or record.get("finished_at") or record.get("started_at") or record.get("created_at"),
        "history_time": record.get("history_time"),
        "project_key": record.get("project_key"),
        "task_ids": record.get("task_ids") or [],
        "step": record.get("step", 5 if record.get("status") == "completed" else 0),
        "total_steps": record.get("total_steps", 5),
        "log_count": len(logs),
        "logs": logs,
        "blocking_issues": list(record.get("blocking_issues") or []),
        "error_summary": record.get("error_summary"),
        "workflow_outcome": record.get("workflow_outcome"),
        "usable_partial_outputs": bool(record.get("usable_partial_outputs")),
        "recovery_primary_issue": record.get("recovery_primary_issue"),
        "recovery_recommended_next_step": record.get("recovery_recommended_next_step"),
        "recovery_report_json": record.get("recovery_report_json"),
        "recovery_report_md": record.get("recovery_report_md"),
        "review_summary": record.get("review_summary"),
        "fasta_preference": record.get("fasta_preference", "llm"),
        "run_mode": _clean_run_mode(record.get("run_mode")),
        "resource_policy": _clean_resource_policy(record.get("resource_policy")),
        "ui_language": _clean_ui_language(record.get("ui_language")),
        "repository": _clean_repository(record.get("repository")),
        "can_download": can_download,
        "archived": True,
        "queue_position": 0,
        "queue_length": 0,
        "queued_tasks": 0,
    }
    return _decorate_history_item(detail)


def _list_public_results() -> list[dict[str, Any]]:
    if not _runs_dir.exists():
        return []
    retention = _result_retention_seconds()
    now = time.time()
    results: list[dict[str, Any]] = []
    with _tasks_lock:
        active_dirs = _active_output_dirs_locked()
    protected_names = _protected_result_dir_names()
    for run_dir in _runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        if run_dir.name == _BATCHES_DIR_NAME or run_dir.name in protected_names or (run_dir / ".agent_keep").exists():
            continue
        if run_dir.resolve() in active_dirs:
            continue
        file_count, size_bytes, latest_mtime = _path_file_stats(
            run_dir,
            excluded_names={_PUBLIC_HISTORY_FILE, _HISTORY_INDEX_FILE},
            excluded_dir_names={_DOWNLOAD_CACHE_DIR},
        )
        if file_count == 0 or latest_mtime <= 0:
            continue
        history = _read_public_history(run_dir)
        status = history.get("status", "completed")
        retention_start = _history_retention_start(history, latest_mtime)
        updated_at_ts = max(latest_mtime, retention_start)
        file_updated_at = datetime.fromtimestamp(latest_mtime, _APP_TZ).isoformat()
        result_updated_at = datetime.fromtimestamp(updated_at_ts, _APP_TZ).isoformat()
        expires_at_ts = retention_start + retention
        results.append(
            {
                "result_id": run_dir.name,
                "task_id": history.get("task_id", ""),
                "name": run_dir.name,
                "input_value": history.get("input_value", run_dir.name),
                "project_key": history.get("project_key") or _project_key_for_input(str(history.get("input_value", run_dir.name))),
                "run_id": history.get("run_id") or run_dir.name,
                "history_id": history.get("history_id") or run_dir.name,
                "submitter": history.get("submitter", "未填写"),
                "status": status,
                "path": str(run_dir),
                "file_count": file_count,
                "size_bytes": size_bytes,
                "created_at": history.get("created_at"),
                "started_at": history.get("started_at"),
                "finished_at": history.get("finished_at"),
                "task_updated_at": history.get("updated_at"),
                "run_mode": _clean_run_mode(history.get("run_mode")),
                "ui_language": _clean_ui_language(history.get("ui_language")),
                "repository": _clean_repository(history.get("repository")),
                "file_updated_at": file_updated_at,
                "result_updated_at": result_updated_at,
                "updated_at": result_updated_at,
                "expires_at": datetime.fromtimestamp(expires_at_ts, _APP_TZ).isoformat(),
                "expires_in_seconds": max(0, int(expires_at_ts - now)),
                "can_download": status == "completed" and _is_download_zip_ready(run_dir),
            }
        )
    results = [_decorate_history_item(item) for item in results]
    results.sort(key=lambda item: item["updated_at"], reverse=True)
    return results


def _list_project_history_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    _repair_interrupted_history_index()
    with _tasks_lock:
        active_task_ids, active_history_ids = _active_history_identity_sets_locked()
    for record in _read_history_index():
        item = with_history_identity(record)
        if not item.get("project_key"):
            continue
        if item.get("status") in _ACTIVE_STATUSES and not _history_item_is_active(item, active_task_ids, active_history_ids):
            item = _mark_interrupted_history_item(item)
        output_dir_name = item.get("output_dir")
        output_dir = _runs_dir / str(output_dir_name) if output_dir_name else None
        if output_dir_name and not item.get("result_id"):
            item["result_id"] = str(output_dir_name)
        if output_dir_name and not item.get("run_id"):
            item["run_id"] = str(output_dir_name)
        file_count = 0
        size_bytes = 0
        if output_dir and output_dir.exists():
            file_count, size_bytes, latest_mtime = _path_file_stats(
                output_dir,
                excluded_names={_PUBLIC_HISTORY_FILE, _HISTORY_INDEX_FILE},
                excluded_dir_names={_DOWNLOAD_CACHE_DIR},
            )
            if latest_mtime:
                item["file_updated_at"] = datetime.fromtimestamp(latest_mtime, _APP_TZ).isoformat()
        item["file_count"] = file_count
        item["size_bytes"] = size_bytes
        item["can_download"] = bool(item.get("status") == "completed" and output_dir and output_dir.exists() and _is_download_zip_ready(output_dir))
        records.append(_decorate_history_item(item))

    for result in _list_public_results():
        task_updated_at = result.get("task_updated_at") or result.get("finished_at") or result.get("started_at") or result.get("created_at")
        if task_updated_at:
            result["updated_at"] = task_updated_at
        if result.get("status") in _ACTIVE_STATUSES and not _history_item_is_active(result, active_task_ids, active_history_ids):
            result = _mark_interrupted_history_item(result)
        records.append(_decorate_history_item(result))

    for batch in _list_parameter_batch_history_records(use_cache=False):
        if batch.get("status") in _ACTIVE_STATUSES:
            continue
        records.append(_decorate_history_item(batch))

    with _tasks_lock:
        for task_id, task in _tasks.items():
            if task.get("status") in _ACTIVE_STATUSES:
                continue
            records.append(_public_task_record_locked(task_id, task))

    items = [_decorate_history_item(item) for item in merge_project_history_records(records)]
    for item in items:
        item.pop("logs", None)
    items.sort(key=lambda item: (history_timestamp(item), str(item.get("history_time") or "")), reverse=True)
    return items


def _list_project_history_records_fast() -> list[dict[str, Any]]:
    _repair_interrupted_history_index()
    with _tasks_lock:
        active_task_ids, active_history_ids = _active_history_identity_sets_locked()
    records: list[dict[str, Any]] = []
    for record in _read_history_index():
        item = with_history_identity(record)
        if not item.get("project_key"):
            continue
        if item.get("status") in _ACTIVE_STATUSES and not _history_item_is_active(item, active_task_ids, active_history_ids):
            item = _mark_interrupted_history_item(item)
        output_dir_name = item.get("output_dir")
        if output_dir_name and not item.get("result_id"):
            item["result_id"] = str(output_dir_name)
        if output_dir_name and not item.get("run_id"):
            item["run_id"] = str(output_dir_name)
        item["file_count"] = int(item.get("file_count") or 0)
        item["size_bytes"] = int(item.get("size_bytes") or 0)
        item["can_download"] = bool(item.get("can_download"))
        records.append(_decorate_history_item(item))

    for batch in _list_parameter_batch_history_records(include_file_stats=False):
        if batch.get("status") in _ACTIVE_STATUSES:
            continue
        records.append(_decorate_history_item(batch))

    with _tasks_lock:
        for task_id, task in _tasks.items():
            if task.get("status") in _ACTIVE_STATUSES:
                continue
            records.append(_public_task_record_locked(task_id, task))

    items = [_decorate_history_item(item) for item in merge_project_history_records(records)]
    for item in items:
        item.pop("logs", None)
    items.sort(key=lambda item: (history_timestamp(item), str(item.get("history_time") or "")), reverse=True)
    return items


def _cleanup_expired_results() -> list[str]:
    if not _runs_dir.exists():
        return []
    with _tasks_lock:
        active_dirs = _active_output_dirs_locked()
        has_active_tasks = any(task.get("status") in _ACTIVE_STATUSES for task in _tasks.values())
    with _batches_lock:
        has_active_batches = any(batch.get("status") in _ACTIVE_STATUSES for batch in _batches.values())
    now = time.time()
    retention = _result_retention_seconds()
    candidates: list[tuple[float, Path]] = []
    removed: list[str] = []
    protected_names = _protected_result_dir_names()
    for run_dir in _runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        if run_dir.name == _BATCHES_DIR_NAME or run_dir.name in protected_names or (run_dir / ".agent_keep").exists():
            continue
        resolved = run_dir.resolve()
        if resolved in active_dirs:
            continue
        history = _read_public_history(run_dir)
        status = history.get("status", "completed")
        file_count, _size_bytes, latest_mtime = _path_file_stats(
            run_dir,
            excluded_names={_PUBLIC_HISTORY_FILE, _HISTORY_INDEX_FILE},
            excluded_dir_names={_DOWNLOAD_CACHE_DIR},
        )
        retention_start = _history_retention_start(history, latest_mtime)
        if retention_start <= 0:
            continue
        if now - retention_start >= retention:
            _archive_run_history(run_dir, history)
            shutil.rmtree(run_dir, ignore_errors=True)
            removed.append(run_dir.name)
            continue
        if status != "completed" or file_count == 0 or not _has_downloadable_result_file(run_dir):
            continue
        candidates.append((retention_start, run_dir))
    max_projects = _max_result_projects()
    candidates.sort(key=lambda item: item[0], reverse=True)
    to_remove = sorted(candidates[max_projects:], key=lambda item: item[0])
    for _mtime, run_dir in to_remove:
        _archive_run_history(run_dir)
        shutil.rmtree(run_dir, ignore_errors=True)
        removed.append(run_dir.name)
    if not has_active_tasks and not has_active_batches:
        removed.extend(_cleanup_pride_cache(now, retention))
    return removed


def _cleanup_pride_cache(now: float, retention: int) -> list[str]:
    cache_root = _pride_cache_dir()
    if not cache_root.exists() or not cache_root.is_dir():
        return []
    removed: list[str] = []
    for file in cache_root.rglob("*"):
        if not file.is_file():
            continue
        try:
            stat = file.stat()
        except OSError:
            continue
        if now - stat.st_mtime < retention:
            continue
        try:
            relative = file.relative_to(cache_root)
        except ValueError:
            relative = file.name
        try:
            file.unlink()
            removed.append(f"pride-cache/{relative}")
        except OSError:
            continue
    for directory in sorted((path for path in cache_root.rglob("*") if path.is_dir()), key=lambda path: len(path.parts), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass
    return removed


def _cleanup_loop() -> None:
    while True:
        try:
            _cleanup_expired_results()
        except Exception:
            pass
        time.sleep(120)


def _is_download_result_file(output_dir: Path, file: Path) -> bool:
    if not file.is_file() or file.name in {_PUBLIC_HISTORY_FILE, _HISTORY_INDEX_FILE}:
        return False
    try:
        relative = file.relative_to(output_dir)
    except ValueError:
        return False
    if not relative.parts:
        return False
    first = relative.parts[0]
    if first == _DOWNLOAD_CACHE_DIR:
        return False
    if first in _DOWNLOAD_RESULT_DIRS:
        return True
    if first == "fragpipe" and len(relative.parts) == 2:
        if file.name in _DOWNLOAD_FRAGPIPE_PARAMETER_FILES or file.suffix.lower() == ".workflow":
            return True
    if first == "workflows" and len(relative.parts) == 2 and file.suffix.lower() == ".workflow":
        return True
    if len(relative.parts) == 1 and file.suffix.lower() in _DOWNLOAD_ROOT_SUFFIXES:
        return True
    return False


def _has_downloadable_result_file(output_dir: Path) -> bool:
    return any(_is_download_result_file(output_dir, path) for path in output_dir.rglob("*"))


def _download_zip_path(output_dir: Path) -> Path:
    return output_dir / _DOWNLOAD_CACHE_DIR / _DOWNLOAD_ZIP_NAME


def _download_result_files(output_dir: Path) -> list[Path]:
    return sorted(path for path in output_dir.rglob("*") if _is_download_result_file(output_dir, path))


def _download_source_mtime(output_dir: Path, files: list[Path] | None = None) -> float:
    latest_mtime = 0.0
    for path in files if files is not None else _download_result_files(output_dir):
        try:
            latest_mtime = max(latest_mtime, path.stat().st_mtime)
        except OSError:
            continue
    return latest_mtime


def _zip_contains_download_files(zip_path: Path, output_dir: Path, files: list[Path]) -> bool:
    try:
        with zipfile.ZipFile(zip_path) as archive:
            names = set(archive.namelist())
    except (OSError, zipfile.BadZipFile):
        return False
    return all(file.relative_to(output_dir).as_posix() in names for file in files)


def _is_download_zip_ready(output_dir: Path) -> bool:
    zip_path = _download_zip_path(output_dir)
    try:
        files = _download_result_files(output_dir)
        source_mtime = _download_source_mtime(output_dir, files)
        return (
            source_mtime > 0
            and zip_path.exists()
            and zip_path.is_file()
            and zip_path.stat().st_size > 0
            and zip_path.stat().st_mtime >= source_mtime
            and _zip_contains_download_files(zip_path, output_dir, files)
        )
    except OSError:
        return False


def _ensure_existing_download_zip_ready(output_dir: Path) -> bool:
    if _is_download_zip_ready(output_dir):
        return True
    zip_path = _download_zip_path(output_dir)
    if not zip_path.exists():
        return False
    try:
        _zip_output_dir(output_dir)
    except OSError:
        return False
    return _is_download_zip_ready(output_dir)


def _zip_output_dir(output_dir: Path, report: Callable[[str], None] | None = None) -> Path:
    files = _download_result_files(output_dir)
    source_mtime = _download_source_mtime(output_dir, files)
    zip_path = _download_zip_path(output_dir)
    try:
        if files and zip_path.exists() and zip_path.stat().st_mtime >= source_mtime and _zip_contains_download_files(zip_path, output_dir, files):
            if report:
                report(f"结果 ZIP 已存在，复用缓存：{zip_path.name} ({_format_bytes(zip_path.stat().st_size)})")
            return zip_path
    except OSError:
        pass
    if not files:
        raise FileNotFoundError("没有可打包的结果文件。")

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = zip_path.parent / f".{uuid.uuid4().hex}.zip.tmp"
    total_size = 0
    for file in files:
        try:
            total_size += file.stat().st_size
        except OSError:
            continue
    if report:
        report(
            f"开始打包下载 ZIP：{len(files)} 个结果文件，源文件合计 {_format_bytes(total_size)}，"
            f"压缩等级 {_zip_compress_level()}"
        )
    packed_size = 0
    last_report = monotonic()
    try:
        with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=_zip_compress_level()) as zf:
            for index, file in enumerate(files, start=1):
                zf.write(file, file.relative_to(output_dir))
                try:
                    packed_size += file.stat().st_size
                except OSError:
                    pass
                now = monotonic()
                if report and (index == 1 or index == len(files) or now - last_report >= 1.0):
                    report(f"ZIP 打包进度：{index}/{len(files)}，已处理 {_format_bytes(packed_size)} / {_format_bytes(total_size)}")
                    last_report = now
        temp_path.replace(zip_path)
        if report:
            report(f"结果 ZIP 打包完成：{zip_path.name} ({_format_bytes(zip_path.stat().st_size)})")
    finally:
        temp_path.unlink(missing_ok=True)
    return zip_path


def _active_task_count_locked() -> int:
    return sum(1 for task in _tasks.values() if task.get("status") in _ACTIVE_STATUSES)


def _running_task_count_locked() -> int:
    return sum(1 for task in _tasks.values() if task.get("status") == "running")


def _queued_task_ids_locked() -> list[str]:
    return [task_id for task_id, task in _tasks.items() if task.get("status") == "queued"]


def _queue_position_locked(task_id: str) -> int:
    queued_ids = _queued_task_ids_locked()
    try:
        return queued_ids.index(task_id) + 1
    except ValueError:
        return 0


def _queue_state_locked(task_id: str | None = None) -> dict[str, int]:
    queued_tasks = len(_queued_task_ids_locked())
    state = {
        "active_tasks": _active_task_count_locked(),
        "running_tasks": _running_task_count_locked(),
        "queued_tasks": queued_tasks,
        "queue_length": queued_tasks,
        "max_concurrent_tasks": _max_concurrent_tasks(),
    }
    if task_id is not None:
        state["queue_position"] = _queue_position_locked(task_id)
    return state


def _try_start_queued_task_locked(task_id: str) -> bool:
    task = _tasks.get(task_id)
    if task is None or task.get("status") != "queued":
        return False
    running = _running_task_count_locked()
    limit = _max_concurrent_tasks()
    if running >= limit:
        return False
    available_slots = limit - running
    if task_id not in _queued_task_ids_locked()[:available_slots]:
        return False
    task["status"] = "running"
    task["started_at"] = _now_iso()
    task["logs"].append(
        {
            "type": "log",
            "ts": _now_time(),
            "level": "info",
            "message": "任务已从队列启动。",
        }
    )
    return True


def _try_start_queued_task(task_id: str) -> bool:
    with _tasks_lock:
        return _try_start_queued_task_locked(task_id)


def _start_pipeline_thread(task_id: str) -> None:
    worker = threading.Thread(
        target=_run_pipeline,
        args=(task_id,),
        name=f"agent-task-{task_id}",
        daemon=True,
    )
    worker.start()


def _start_ready_queued_tasks() -> list[str]:
    started: list[str] = []
    with _tasks_lock:
        while _running_task_count_locked() < _max_concurrent_tasks():
            queued_ids = _queued_task_ids_locked()
            if not queued_ids:
                break
            task_id = queued_ids[0]
            if not _try_start_queued_task_locked(task_id):
                break
            started.append(task_id)
    for task_id in started:
        _write_task_history(task_id)
        _start_pipeline_thread(task_id)
    return started


def _llm_config_store() -> LLMConfigStore:
    return LLMConfigStore(os.getenv("AGENT_LLM_CONFIG_PATH") or ".agent_secrets/llm_config.json")


_POOL_BUILD_LLM_PROFILE_ID = "pool-builder"
_POOL_BUILD_LLM_PROVIDERS = {
    "openai_compatible",
    "openai",
    "google",
    "xai",
    "deepseek",
}


def _pool_build_llm_config_store() -> LLMConfigStore:
    return LLMConfigStore(
        os.getenv("AGENT_POOL_BUILD_LLM_CONFIG_PATH")
        or ".agent_secrets/pool_build_llm_config.json"
    )


def _server_llm_config() -> tuple[dict[str, str] | None, str]:
    saved = _llm_config_store().load()
    if saved is not None:
        return saved, "saved"
    api_key = _clean_text(
        os.getenv("AGENT_LLM_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        return None, "default"
    return {
        "api_key": api_key,
        "base_url": _clean_text(os.getenv("AGENT_LLM_BASE_URL")) or _DEFAULT_CONFIG["base_url"],
        "model": _clean_text(os.getenv("AGENT_LLM_MODEL")) or _DEFAULT_CONFIG["model"],
        "timeout": _clean_text(os.getenv("AGENT_LLM_TIMEOUT")) or _DEFAULT_CONFIG["timeout"],
    }, "environment"


def _public_llm_config() -> dict[str, Any]:
    config, source = _server_llm_config()
    store = _llm_config_store()
    profiles = store.list_profiles(include_secrets=False)
    default_profile = next((item for item in profiles if item.get("is_default")), None)
    return {
        "api_key_set": config is not None,
        "base_url": (config or {}).get("base_url") or _DEFAULT_CONFIG["base_url"],
        "model": (config or {}).get("model") or _DEFAULT_CONFIG["model"],
        "timeout": (config or {}).get("timeout") or _DEFAULT_CONFIG["timeout"],
        "source": source,
        "default_profile_id": (default_profile or {}).get("id"),
        "profiles": profiles,
    }


def _review_developer_allowed(request: Request) -> bool:
    # Developer gates removed by product request: all local/web callers may use LLM/config/tools.
    return True


def _request_review_mode(request: Request, raw: Any) -> str:
    requested = _normalize_review_mode(raw)
    if requested != "expert" and not _review_developer_allowed(request):
        return "expert"
    return requested


def _public_pool_record(record: Mapping[str, Any], *, mode: str) -> dict[str, Any]:
    payload = dict(record)
    if mode == "expert":
        stats = dict(payload.get("stats") or {})
        payload["stats"] = {"candidate_count": int(stats.get("candidate_count") or 0)}
        payload.pop("judgment_source", None)
        payload.pop("paths", None)
    return payload


def _expert_pool_registry() -> ExpertPoolRegistry:
    return ExpertPoolRegistry(expert_review_root())


def _discovery_calibration_candidates() -> list[dict[str, Any]]:
    registry = _expert_pool_registry()
    candidates: list[dict[str, Any]] = []
    for record in registry.list_pools():
        pool_id = str(record.get("pool_id") or "")
        document = registry.load_pool_document(pool_id, prefer_reviewed=True)
        if not isinstance(document, dict):
            continue
        private_key = registry.load_private_key(pool_id) or {}
        private_by_candidate = {
            str(item.get("candidate_id") or ""): item
            for item in (private_key.get("candidates") or [])
            if isinstance(item, dict) and str(item.get("candidate_id") or "")
        }
        task_records = document.get("tasks") if isinstance(document.get("tasks"), dict) else {}
        for raw in document.get("candidates") or []:
            if not isinstance(raw, dict):
                continue
            candidate = dict(raw)
            candidate_id = str(candidate.get("candidate_id") or "")
            private = private_by_candidate.get(candidate_id) or {}
            project_accession = str(private.get("project_accession") or candidate_id)
            task_identity = str(private.get("calibration_task_id") or "")
            if not task_identity:
                task_key = f"{candidate.get('scenario_id') or ''}:{candidate.get('variant_id') or ''}"
                task_record = task_records.get(task_key) if isinstance(task_records.get(task_key), dict) else {}
                task_identity = calibration_task_identity(
                    str(task_record.get("visible_prompt") or candidate.get("visible_prompt") or ""),
                    task_record.get("visible_constraints") or {},
                    task_record.get("task_semantics") or candidate.get("task_semantics") or {},
                )
            project_identity = f"{project_accession}:{task_identity}"
            candidate["calibration_project_id"] = project_identity
            if isinstance(private.get("calibration_features"), dict):
                candidate["calibration_features"] = dict(private["calibration_features"])
            candidates.append(candidate)
    return candidates


def _normalize_review_mode(raw: Any) -> str:
    mode = str(raw or "expert").strip().lower()
    if mode in {"expert", "developer", "test"}:
        return mode
    return "expert"


_impact_sessions: dict[str, dict[str, Any]] = {}
_impact_sessions_lock = threading.Lock()


def _expert_job_manager() -> ExpertJudgeJobManager:
    store = _llm_config_store()

    def resolve_profile(profile_id: str) -> dict[str, Any]:
        profile = store.get_profile(profile_id, include_secrets=True)
        if profile is None:
            raise ValueError("profile_not_found_or_incomplete")
        return profile

    return ExpertJudgeJobManager(
        _expert_pool_registry(),
        resolve_profile=resolve_profile,
        list_profiles=lambda: store.list_profiles(include_secrets=False),
    )


def _pool_build_explicit_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _clean_text(value).casefold() in {"1", "true", "yes", "on"}


def _prompt_parser_generation_identity(llm_config: Any) -> dict[str, Any] | None:
    explicit = isinstance(llm_config, dict) and bool(llm_config)
    raw = dict(llm_config) if explicit else {}
    if not explicit:
        saved, _source = _server_llm_config()
        raw = dict(saved or {})
    complete_explicit_config = explicit and all(
        _clean_text(raw.get(field))
        for field in ("api_key", "base_url", "model", "timeout")
    )
    if complete_explicit_config:
        config = {
            "api_key": _clean_text(raw.get("api_key")),
            "base_url": _clean_text(raw.get("base_url")).rstrip("/"),
            "model": _clean_text(raw.get("model")),
            "timeout": _clean_text(raw.get("timeout")),
        }
    else:
        config, _error = _build_llm_config(raw)
    if config is None:
        return None
    requested_model = _clean_text(raw.get("requested_model_id") or config.get("model"))
    resolved_model = _clean_text(raw.get("resolved_model_id")) or None
    model_family = _clean_text(raw.get("model_family") or requested_model) or None
    verification = _clean_text(raw.get("identity_verification") or "unverified")
    if explicit or verification not in {"provider_attested", "unverified"}:
        verification = "unverified"
    return {
        "role": "prompt_parser",
        "provider": _clean_text(raw.get("provider") or "openai_compatible"),
        "requested_model_id": requested_model or None,
        "resolved_model_id": resolved_model,
        "model_family": model_family,
        "endpoint_identity": _clean_text(raw.get("endpoint_identity") or config.get("base_url")) or None,
        "identity_verification": verification,
    }


def _prompt_parser_failure_message(
    exc: Exception,
    *,
    profile_id: str,
    output_language: str,
) -> str:
    raw_message = _strip_ansi(exc)
    status_code = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
    normalized = raw_message.casefold()
    authentication_failed = status_code in {401, 403} or any(
        marker in normalized
        for marker in (
            "401",
            "403",
            "authorization required",
            "unauthorized",
            "authentication failed",
            "invalid api key",
        )
    )
    profile_label = profile_id or "default"
    if authentication_failed:
        if output_language == "zh-CN":
            if profile_id == _POOL_BUILD_LLM_PROFILE_ID:
                return (
                    "评审池构建模型认证失败。"
                    "请在“评审池构建模型配置”中更新 API Key，或更换提供商和模型后重试。"
                )
            return (
                f'Prompt 解析模型 Profile“{profile_label}”认证失败。'
                "请更新该 Profile 的 API Key，或选择另一个建池模型后重试。"
            )
        if profile_id == _POOL_BUILD_LLM_PROFILE_ID:
            return (
                "The review-pool builder model failed authentication. "
                "Update its API Key or change provider/model in Review-pool Builder Model Configuration, then retry."
            )
        return (
            f'Prompt parser Profile "{profile_label}" failed authentication. '
            "Update its API Key or select another pool-building model and retry."
        )

    safe_detail = _redact_secrets(raw_message)
    safe_detail = re.sub(r"https?://\S+", "[provider endpoint]", safe_detail, flags=re.IGNORECASE)
    safe_detail = safe_detail[:300].strip()
    if output_language == "zh-CN":
        if profile_id == _POOL_BUILD_LLM_PROFILE_ID:
            return (
                "评审池构建模型调用失败。请检查“评审池构建模型配置”中的提供商、模型、"
                f"Base URL 和网络设置后重试。{safe_detail}"
            )
        prefix = f'Prompt 解析模型 Profile“{profile_label}”调用失败。'
        return f"{prefix}请检查该 Profile 的模型、Base URL 和网络配置后重试。{safe_detail}"
    if profile_id == _POOL_BUILD_LLM_PROFILE_ID:
        return (
            "The review-pool builder model failed. Check its provider, model, Base URL, and network settings "
            f"in Review-pool Builder Model Configuration, then retry. {safe_detail}"
        )
    prefix = f'Prompt parser Profile "{profile_label}" failed. '
    return f"{prefix}Check its model, Base URL, and network settings, then retry. {safe_detail}"


def _prepare_expert_pool_discovery_request(payload: dict[str, Any]) -> dict[str, Any]:
    original = dict(payload)
    prompt = _clean_text(original.get("prompt"))
    if not prompt:
        raise ValueError("prompt_required")
    output_language = _normalise_pool_build_language(original.get("output_language"))
    explicit_scale = _normalise_pool_build_scale(
        original.get("scale_mode"),
        prompt=prompt,
        allow_auto=True,
    )
    selected_profile = _pool_build_llm_config_store().get_profile(
        _POOL_BUILD_LLM_PROFILE_ID,
        include_secrets=True,
    )
    parser_profile_id = _POOL_BUILD_LLM_PROFILE_ID if selected_profile else ""
    parser_config = dict(selected_profile) if selected_profile else {}
    untrusted_internal_fields = {
        "llm_config",
        "_generation_contributors",
        "pool_builder_profile_id",
        "_require_explicit_llm_config",
    }
    public_original = {
        key: value
        for key, value in original.items()
        if key not in untrusted_internal_fields
    }
    parser_current = {
        key: value
        for key, value in public_original.items()
    }
    try:
        parsed = _run_discovery_goal_parse(
            {
                "prompt": prompt,
                "output_language": output_language,
                "llm_config": parser_config,
                "allow_server_default": False,
                "current": parser_current,
            }
        )
    except Exception as exc:
        no_llm = "No discovery LLM API key found" in str(exc)
        if not no_llm:
            raise ValueError(
                _prompt_parser_failure_message(
                    exc,
                    profile_id=parser_profile_id,
                    output_language=output_language,
                )
            ) from exc
        broad_human = bool(
            re.search(
                r"(proteomics|proteome|peptidomics|蛋白质组|肽组|蛋白肽|shotgun|bottom[\s-]?up)",
                prompt,
                re.IGNORECASE,
            )
        ) and not re.search(
            r"(immunopeptidom|hla\b|mhc\b|ligandome|免疫肽)",
            prompt,
            re.IGNORECASE,
        )
        if is_immunopeptidomics_goal(prompt) and not broad_human:
            parsed = {
                "parser": "ontology_fallback",
                "fields": {
                    "repository": "pride",
                    "goal": "immunopeptidomics",
                    "ptm_type": "unknown_ptm",
                    "ptm_types": [],
                    "query_terms": ["immunopeptidomics", "HLA ligandome", "MHC ligandome"],
                    "scale_mode": _normalise_pool_build_scale("", prompt=prompt, allow_auto=False),
                },
                "warnings": [
                    "未配置评审池构建模型，已使用免疫肽领域降级解析。"
                    if output_language == "zh-CN"
                    else "No review-pool builder model was configured; the immunopeptidomics ontology fallback was used."
                ],
                "reasoning": (
                    "根据已知免疫肽/HLA 语义生成英文仓库检索词。"
                    if output_language == "zh-CN"
                    else "Generated English repository terms from known immunopeptidomics/HLA semantics."
                ),
            }
        elif broad_human or re.search(r"(人类|人源|human).{0,12}(蛋白|肽|proteom|peptid)", prompt, re.IGNORECASE):
            parsed = {
                "parser": "broad_human_proteomics_fallback",
                "fields": {
                    "repository": "pride",
                    "goal": "general",
                    "species": ["human"],
                    "species_policy": "include_only",
                    "acquisition_mode": "unknown",
                    "labeling_strategy": "unknown",
                    "query_terms": [
                        "human proteomics",
                        "shotgun proteomics",
                        "label free quantitation",
                        "TMT proteomics",
                        "DIA proteomics",
                        "phosphoproteomics",
                        "plasma proteomics",
                        "affinity purification mass spectrometry",
                    ],
                    "scale_mode": _normalise_pool_build_scale("", prompt=prompt, allow_auto=False),
                },
                "warnings": [
                    "未配置评审池构建模型，已使用广义人类蛋白质组降级解析。"
                    if output_language == "zh-CN"
                    else "No review-pool builder model was configured; a broad human-proteomics fallback was used."
                ],
                "reasoning": (
                    "将“人类蛋白/肽数据，越多越好”解释为广义人类蛋白质组/肽组检索，而不是免疫肽组。"
                    if output_language == "zh-CN"
                    else "Interpreted the request as broad human proteomics/peptidomics rather than immunopeptidomics."
                ),
            }
        elif _contains_cjk(prompt):
            raise ValueError(
                "prompt_parse_failed:请先填写并保存“评审池构建模型配置”，以便把中文或其他非英文请求转换为英文仓库检索词。"
                if output_language == "zh-CN"
                else "prompt_parse_failed:Configure the review-pool builder model before converting a non-English request into English repository terms."
            ) from exc
        else:
            parsed = {
                "parser": "deterministic_english_fallback",
                "fields": {
                    "repository": "pride",
                    "goal": "general",
                    "query_terms": _english_discovery_query_terms(general_query_terms_from_text(prompt)),
                    "scale_mode": _normalise_pool_build_scale("", prompt=prompt, allow_auto=False),
                },
                "warnings": [],
                "reasoning": (
                    "未配置评审池构建模型，已直接使用英文请求生成检索词。"
                    if output_language == "zh-CN"
                    else "No review-pool builder model was configured; repository terms were derived from the English request."
                ),
            }

    parsed_fields = parsed.get("fields") if isinstance(parsed.get("fields"), dict) else {}
    scale_mode = explicit_scale
    if scale_mode == "auto":
        scale_mode = _normalise_pool_build_scale(
            parsed_fields.get("scale_mode"),
            prompt=prompt,
            allow_auto=False,
        )
    preset = _POOL_BUILD_SCALE_PRESETS[scale_mode]
    explicit_query_terms = original.get("query_terms") if "query_terms" in original else None
    query_terms = _english_discovery_query_terms(
        explicit_query_terms if explicit_query_terms is not None else parsed_fields.get("query_terms")
    )
    goal = _clean_text(original.get("goal") or parsed_fields.get("goal") or "general").casefold()
    if not query_terms and goal == "immunopeptidomics":
        query_terms = ["immunopeptidomics", "HLA ligandome", "MHC ligandome"]
    if not query_terms and not _contains_cjk(prompt):
        query_terms = _english_discovery_query_terms(general_query_terms_from_text(prompt))
    if not query_terms:
        raise ValueError("prompt_parse_failed:no_english_query_terms")

    request = {**parsed_fields, **public_original}
    if selected_profile:
        request["pool_builder_profile_id"] = parser_profile_id
        request["llm_config"] = parser_config
    parser_identity = None
    if str(parsed.get("parser") or "").casefold() == "llm":
        parser_identity = _prompt_parser_generation_identity(parser_config)
    # Never let a free-text broad human proteomics request collapse into immunopeptidomics.
    goal = _clean_text(request.get("goal") or parsed_fields.get("goal") or "general").casefold()
    if goal == "immunopeptidomics" and re.search(
        r"(proteomics|proteome|peptidomics|蛋白质组|肽组|蛋白肽|shotgun|bottom[\s-]?up)",
        prompt,
        re.IGNORECASE,
    ) and not re.search(
        r"(immunopeptidom|hla\b|mhc\b|ligandome|免疫肽)",
        prompt,
        re.IGNORECASE,
    ):
        goal = "general"
        request["goal"] = "general"
        request["immunopeptide_scope"] = None
        request["immunopeptide_evidence_terms"] = []
        request["immunopeptide_enrichment_methods"] = []
        request["hla_class"] = []
        request["hla_alleles"] = []
        request["immunopeptide_metadata_confidence"] = 0.0
    if goal not in {"general", "ptm", "immunopeptidomics"}:
        goal = "general"
        request["goal"] = "general"
    # Human-only when the request is clearly about human proteomics data.
    species = request.get("species") if isinstance(request.get("species"), list) else []
    if not species and re.search(r"(human|homo sapiens|人类|人源|智人)", prompt, re.IGNORECASE):
        species = ["human"]
        request["species"] = species
    if species == ["human"] or (
        isinstance(species, list) and {str(item).casefold() for item in species} == {"human"}
    ):
        if re.search(r"(人类|人源|human)", prompt, re.IGNORECASE):
            request["species_policy"] = "include_only"
    # Soft defaults for acquisition/labeling unless user forced them.
    if _clean_text(request.get("acquisition_mode")).casefold() in {"", "dda"} and not re.search(
        r"\bdda\b|data[\s-]?dependent", prompt, re.IGNORECASE
    ):
        if re.search(r"(proteomics|蛋白质组|肽组|越多越好|尽可能多)", prompt, re.IGNORECASE):
            request["acquisition_mode"] = "unknown"
    if _clean_text(request.get("labeling_strategy")).casefold() in {"", "label_free", "label-free"} and not re.search(
        r"label[\s-]?free|lfq", prompt, re.IGNORECASE
    ):
        if re.search(r"(proteomics|蛋白质组|肽组|越多越好|尽可能多)", prompt, re.IGNORECASE):
            request["labeling_strategy"] = "unknown"
    request.update(
        {
            "prompt": prompt,
            "query_terms": query_terms,
            "scale_mode": scale_mode,
            "output_language": output_language,
            "runtime": "openai_agents",
            "source": "remote",
            "agentic": _pool_build_explicit_bool(original.get("agentic")),
            "_require_explicit_llm_config": True,
            "max_projects": _bounded_int(original.get("max_projects"), default=preset["max_projects"], minimum=1, maximum=5000),
            "max_candidate_projects": _bounded_int(original.get("max_candidate_projects"), default=preset["max_candidate_projects"], minimum=1, maximum=20000),
            "max_files": _bounded_int(original.get("max_files"), default=preset["max_files"], minimum=1, maximum=10000),
            "max_files_per_project": _bounded_int(original.get("max_files_per_project"), default=preset["max_files_per_project"], minimum=1, maximum=200),
            "hard_constraint_fields": list(
                dict.fromkeys(
                    ["repository"]
                    + (
                        ["species", "species_policy"]
                        if str(request.get("species_policy") or "") == "include_only"
                        else []
                    )
                    + (
                        ["goal"]
                        if goal in {"ptm", "immunopeptidomics"}
                        and re.search(r"(immunopeptidom|hla\b|mhc\b|ligandome|ptm|磷酸化|乙酰)", prompt, re.IGNORECASE)
                        else []
                    )
                )
            ),
            "constraint_provenance": {
                "repository": "user",
                "goal": "user" if goal in {"ptm", "immunopeptidomics"} else "inferred",
                "species": "user" if species else "inferred",
                "species_policy": "user" if str(request.get("species_policy") or "") == "include_only" else "inferred",
                "acquisition_mode": "inferred",
                "labeling_strategy": "inferred",
            },
        }
    )
    if goal == "general" and not request.get("query_terms"):
        request["query_terms"] = _english_discovery_query_terms(
            [
                "human proteomics",
                "shotgun proteomics",
                "label free quantitation",
                "TMT proteomics",
                "DIA proteomics",
                "phosphoproteomics",
                "plasma proteomics",
                "affinity purification mass spectrometry",
            ]
        )
    task_semantics = interpret_review_task(prompt, request)
    request["quantity_scope"] = task_semantics["quantity_scope"]
    request["portfolio_size_preference"] = task_semantics["portfolio_size_preference"]
    # "越多越好" = harvest every evidence-backed quality project within safety ceilings.
    # max_projects is a soft ambition / progress target, never a "stop and keep only N" knife.
    maximize = (
        request.get("quantity_scope") == "portfolio"
        or str(request.get("portfolio_size_preference") or "").startswith("maximize")
        or bool(
            re.search(
                r"(越多越好|尽可能多|尽量多|搜全|覆盖全|不设上限|无上限|全量|"
                r"(?:所有|全部).{0,24}(?:数据|项目|文件|候选)|"
                r"as many as possible|all (?:relevant )?(?:data|datasets|projects|files)|"
                r"open[-\s]?ended|exhaustive|maximize)",
                prompt,
                re.IGNORECASE,
            )
        )
    )
    if maximize:
        request["quantity_scope"] = "portfolio"
        request["portfolio_size_preference"] = (
            request.get("portfolio_size_preference") or "maximize_qualified_projects"
        )
        request["harvest_all_qualified"] = True
        request["continuous_discovery"] = True
        # Incremental delivery for "越多越好": emit every N verified usable files.
        request["partial_delivery_batch_size"] = max(
            1,
            min(5000, int(request.get("partial_delivery_batch_size") or 500)),
        )
        request["inspection_batch_size"] = max(
            1,
            min(100, int(request.get("inspection_batch_size") or 30)),
        )
        scale_mode = "exhaustive"
        request["scale_mode"] = "exhaustive"
        preset = _POOL_BUILD_SCALE_PRESETS["exhaustive"]
        request["max_projects"] = max(
            int(request.get("max_projects") or 0),
            int(preset["max_projects"]),
            2000,
        )
        request["max_candidate_projects"] = max(
            int(request.get("max_candidate_projects") or 0),
            int(preset["max_candidate_projects"]),
            5000,  # Compatibility hint only; continuous discovery does not truncate here.
        )
        request["max_files"] = max(
            int(request.get("max_files") or 0),
            int(preset["max_files"]),
            100000,
        )
        request["max_files_per_project"] = max(
            int(request.get("max_files_per_project") or 0),
            int(preset["max_files_per_project"]),
            500,
        )
    minimum = task_semantics.get("per_project_minimum")
    if isinstance(minimum, Mapping):
        if minimum.get("unit") == "files":
            request["per_project_min_files"] = int(minimum["value"])
        elif minimum.get("unit") == "samples":
            request["per_project_min_samples"] = int(minimum["value"])
    if parser_identity is not None:
        request["_generation_contributors"] = [parser_identity]
    warnings = [
        localized
        for item in (parsed.get("warnings") or [])
        if (localized := _localize_prompt_parse_warning(item, output_language))
    ] if isinstance(parsed.get("warnings"), list) else []
    scale_warning = _pool_build_scale_warning(scale_mode, output_language)
    if scale_warning:
        warnings.append(scale_warning)
    if maximize:
        warnings.append(
            "“越多越好”按合格项目尽量多收：只保留检查充分、硬门通过、2–3 分项目；不设 20/300 之类的最终数量硬砍。"
            if output_language == "zh-CN"
            else "Maximize mode harvests every evidence-backed grade 2-3 project within safety ceilings; fixed 20/300 final caps are not applied."
        )
    return {
        "request": request,
        "parser": str(parsed.get("parser") or "llm"),
        "warnings": list(dict.fromkeys(warnings)),
        "reasoning": _localize_prompt_parse_reasoning(parsed.get("reasoning"), output_language),
    }


def _expert_pool_build_manager() -> ExpertPoolBuildManager:
    def start_discovery(payload: dict[str, Any]) -> dict[str, Any]:
        return asyncio.run(start_discovery_job(payload, background_tasks=None))

    def get_discovery(job_id: str) -> dict[str, Any]:
        return asyncio.run(get_discovery_job(job_id, detail=1))

    def cancel_discovery(job_id: str) -> dict[str, Any]:
        return asyncio.run(cancel_discovery_job(job_id))

    def start_review(pool_id: str, review: Mapping[str, Any]) -> dict[str, Any]:
        if review.get("single_model") is True:
            profile_id = _clean_text(review.get("profile_id"))
            if not profile_id:
                profiles = _llm_config_store().list_profiles()
                default_profile = next((item for item in profiles if item.get("is_default")), None)
                profile_id = _clean_text((default_profile or {}).get("id"))
            if not profile_id:
                raise ValueError("review_profile_id_required")
            return _expert_job_manager().start_job(
                pool_id=pool_id,
                profile_id=profile_id,
                independent_model=bool(review.get("independent_model")),
                workers=int(review.get("workers") or 2),
            )
        generator_identity = review.get("generator_identity")
        if not isinstance(generator_identity, Mapping):
            generator_identity = {}
        return _expert_job_manager().start_consensus_job(
            pool_id=pool_id,
            generator_identity=generator_identity,
            workers=int(review.get("workers") or 1),
            idempotency_key=_clean_text(review.get("idempotency_key"))
            or f"{pool_id}:model-expert-consensus",
            output_language=_normalise_pool_build_language(review.get("output_language")),
            scale_mode=_normalise_pool_build_scale(review.get("scale_mode") or "auto"),
        )

    return ExpertPoolBuildManager(
        _expert_pool_registry(),
        start_discovery=start_discovery,
        get_discovery=get_discovery,
        cancel_discovery=cancel_discovery,
        start_review=start_review,
        prepare_discovery_request=_prepare_expert_pool_discovery_request,
    )


def _impact_path_allowed(path: str) -> bool:
    candidate = Path(path)
    if not candidate.exists() or not candidate.is_file():
        return False
    try:
        resolved = candidate.resolve()
        roots = [Path("runs").resolve(), expert_review_root().resolve()]
    except OSError:
        return False
    return any(resolved == root or root in resolved.parents for root in roots)


def _load_impact_bundle(session: Mapping[str, Any]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    key_payload = None
    runs: list[dict[str, Any]] = []
    key_path = session.get("key_path")
    if key_path:
        try:
            key_payload = load_json(key_path)
        except Exception:
            key_payload = None
    for path in session.get("run_paths") or []:
        try:
            payload = load_json(path)
        except Exception:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("runs"), list):
            runs.extend([item for item in payload["runs"] if isinstance(item, dict)])
        elif isinstance(payload, dict) and (
            isinstance(payload.get("workflow_runs"), list) or isinstance(payload.get("agent_runs"), list)
        ):
            runs.extend([item for item in payload.get("workflow_runs") or [] if isinstance(item, dict)])
            runs.extend([item for item in payload.get("agent_runs") or [] if isinstance(item, dict)])
        elif isinstance(payload, dict) and payload.get("selected_project_accessions") is not None:
            runs.append(payload)
        elif isinstance(payload, list):
            runs.extend([item for item in payload if isinstance(item, dict)])
    return key_payload, runs


def _build_llm_config(
    llm_config: dict[str, Any],
    *,
    allow_server_default: bool = True,
) -> tuple[dict[str, str] | None, str | None]:
    server_config, _ = _server_llm_config() if allow_server_default else (None, "disabled")
    fallback = server_config or {}
    api_key = _clean_text(llm_config.get("api_key")) or fallback.get("api_key", "")
    if not api_key:
        return None, "请先填写本次任务使用的 API Key"

    base_url = _clean_text(llm_config.get("base_url")) or fallback.get("base_url") or _DEFAULT_CONFIG["base_url"]
    model = _clean_text(llm_config.get("model")) or fallback.get("model") or _DEFAULT_CONFIG["model"]
    timeout = _clean_text(llm_config.get("timeout")) or fallback.get("timeout") or _DEFAULT_CONFIG["timeout"]
    try:
        if float(timeout) <= 0:
            return None, "大模型超时时间必须大于 0"
    except ValueError:
        return None, "大模型超时时间必须是数字"

    return {"api_key": api_key, "base_url": base_url.rstrip("/"), "model": model, "timeout": timeout}, None


def _discovery_llm_client(
    llm_config: dict[str, Any],
    *,
    allow_server_default: bool = True,
):
    if not llm_config and not allow_server_default:
        return None
    config, config_error = _build_llm_config(
        llm_config,
        allow_server_default=allow_server_default,
    )
    if config is not None:
        return OpenAICompatibleDiscoveryLLM(
            api_key=config["api_key"],
            base_url=config["base_url"],
            model=config["model"],
            timeout=_positive_float(config["timeout"], 120.0),
        )
    if llm_config:
        raise ValueError(config_error or "Invalid LLM configuration.")
    return default_discovery_llm_client()


def _agentic_discovery_planner(
    llm_config: dict[str, Any],
    *,
    allow_server_default: bool = True,
) -> AgenticDiscoveryPlanner | None:
    if not llm_config and not allow_server_default:
        return None
    config, config_error = _build_llm_config(
        llm_config,
        allow_server_default=allow_server_default,
    )
    if config is not None:
        return AgenticDiscoveryPlanner(
            OpenAICompatibleDiscoveryLLM(
                api_key=config["api_key"],
                base_url=config["base_url"],
                model=config["model"],
                timeout=_positive_float(config["timeout"], 120.0),
            )
        )
    if llm_config:
        raise ValueError(config_error or "Invalid LLM configuration.")
    return default_agentic_discovery_planner()


def _task_llm_reasoner(config: dict[str, str]):
    from agent.llm.reasoner import OpenAICompatibleReasoner

    return OpenAICompatibleReasoner(
        api_key=config["api_key"],
        base_url=config["base_url"],
        model=config["model"],
        timeout=_positive_float(config["timeout"], 300.0),
    )


def _display_value(value: Any) -> str:
    if value is None or value == "":
        return "无"
    if isinstance(value, dict):
        return ", ".join(f"{key}={_display_value(val)}" for key, val in value.items())
    if isinstance(value, list | tuple | set):
        return ", ".join(_display_value(item) for item in value)
    return str(value)


def _path_name(value: Any) -> str:
    try:
        return Path(value).name
    except TypeError:
        return _display_value(value)


def _append_review_item(
    items: list[dict[str, Any]],
    label: str,
    value: Any,
    *,
    source: str = "",
    confidence: float | None = None,
    conflict: bool = False,
) -> None:
    if value is None or value == "" or value == [] or value == {}:
        return
    item: dict[str, Any] = {
        "label": label,
        "value": _display_value(value),
        "source": source,
        "conflict": bool(conflict),
    }
    if confidence is not None:
        item["confidence"] = confidence
    items.append(item)


def _append_attribute_item(items: list[dict[str, Any]], label: str, attribute: Any) -> None:
    _append_review_item(
        items,
        label,
        getattr(attribute, "value", None),
        source=str(getattr(attribute, "source", "")),
        confidence=getattr(attribute, "confidence", None),
        conflict=bool(getattr(attribute, "conflict_flag", False)),
    )


def _normalized_fasta_hint_item(key: str, plan: Any) -> tuple[Any, str, float | None] | None:
    if key == "recommended_fasta_name":
        fasta_name = _path_name(getattr(plan, "fasta_path", None))
        return (fasta_name, "plan", None) if fasta_name else None
    if key == "recommended_fasta_url":
        fasta_url = getattr(plan, "fasta_download_url", None)
        return (fasta_url, "plan", None) if fasta_url else None
    if key == "recommended_fasta_source":
        fasta_url = str(getattr(plan, "fasta_download_url", "") or "")
        if "uniprot.org" in fasta_url.lower():
            return "UniProt", "plan", None
    return None


def _choice_values_from_metadata(result: Any, key: str) -> list[str]:
    context = getattr(result, "context", None)
    metadata = getattr(context, "metadata", {}) or {}
    raw = getattr(metadata.get(key), "value", None) if hasattr(metadata, "get") else None
    values = raw if isinstance(raw, list | tuple | set) else [raw] if raw else []
    unique: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in unique:
            unique.append(text)
    return unique


def _choice_values_from_attribute(attribute: Any) -> list[str]:
    raw = getattr(attribute, "value", None)
    if isinstance(raw, list | tuple | set):
        candidates = [str(item) for item in raw]
    else:
        candidates = re.split(r"\s*(?:;|\||,)\s*", str(raw or ""))
    unique: list[str] = []
    for value in candidates:
        text = value.strip()
        if text and text.lower() != "unknown" and text not in unique:
            unique.append(text)
    return unique


def _review_options(result: Any, issues: list[str]) -> list[dict[str, Any]]:
    attributes = result.attributes
    options: list[dict[str, Any]] = []
    species_attr = getattr(attributes, "species", None)
    if bool(getattr(species_attr, "conflict_flag", False)) or any("多个物种" in issue for issue in issues):
        values = _choice_values_from_metadata(result, "organisms") or _choice_values_from_attribute(species_attr)
        if len(values) > 1:
            options.append({"field": "species", "label": "选择物种", "values": values})
    instrument_attr = getattr(attributes, "instrument_name", None)
    if bool(getattr(instrument_attr, "conflict_flag", False)) or any("多个仪器" in issue for issue in issues):
        values = _choice_values_from_metadata(result, "instruments") or _choice_values_from_attribute(instrument_attr)
        if len(values) > 1:
            options.append({"field": "instrument_name", "label": "选择仪器", "values": values})
    return options


def _build_review_summary(result: Any) -> dict[str, Any]:
    attributes = result.attributes
    plan = result.plan
    items: list[dict[str, Any]] = []

    _append_review_item(items, "workflow", _path_name(getattr(plan, "fragpipe_workflow_path", None)), source="plan")
    fasta = _path_name(getattr(plan, "fasta_path", None))
    fasta_mode = getattr(plan, "fasta_selection_mode", "")
    if fasta_mode:
        fasta = f"{fasta} ({fasta_mode})"
    _append_review_item(items, "FASTA", fasta, source="plan")
    _append_review_item(items, "FASTA URL", getattr(plan, "fasta_download_url", None), source="plan")
    project_files = getattr(getattr(result, "context", None), "project_files", []) or []
    project_fastas = [
        str(file_record.get("fileName", ""))
        for file_record in project_files
        if str(file_record.get("fileName", "")).lower().endswith((".fasta", ".fa", ".faa", ".fasta.gz", ".fa.gz", ".faa.gz"))
    ]
    if project_fastas and fasta_mode != "reproduced":
        preview = ", ".join(project_fastas[:3])
        if len(project_fastas) > 3:
            preview += f", +{len(project_fastas) - 3}"
        _append_review_item(
            items,
            "项目 FASTA 可选",
            f"已默认使用大模型/物种推荐的 UniProt FASTA；PRIDE 项目中也检测到 {preview}。如需复现原项目 FASTA，勾选创建区的项目 FASTA 优先后重新提交。",
            source="pride",
        )
    _append_review_item(items, "raw_data_type", getattr(plan, "raw_data_type", None), source="plan")
    _append_review_item(items, "thread_num", getattr(plan, "thread_num", None), source="plan")

    _append_attribute_item(items, "采集模式", getattr(attributes, "acquisition_mode", None))
    _append_attribute_item(items, "物种", getattr(attributes, "species", None))
    _append_attribute_item(items, "仪器", getattr(attributes, "instrument_name", None))
    _append_attribute_item(items, "酶", getattr(attributes, "enzyme", None))
    _append_attribute_item(items, "固定修饰", getattr(attributes, "fixed_mods", None))
    _append_attribute_item(items, "可变修饰", getattr(attributes, "variable_mods", None))

    hints_attr = getattr(attributes, "search_parameter_hints", None)
    hints = getattr(hints_attr, "value", {})
    hint_source = str(getattr(hints_attr, "source", ""))
    hint_confidence = getattr(hints_attr, "confidence", None)
    if isinstance(hints, dict):
        for key in (
            "missed_cleavages",
            "precursor_tol",
            "fragment_tol",
            "min_peaks",
            "max_variable_mods",
            "data_family",
            "recommended_workflow_name",
            "workflow_parameter_overrides",
            "recommended_fasta_name",
            "recommended_fasta_url",
            "recommended_fasta_source",
        ):
            if key in hints:
                normalized = _normalized_fasta_hint_item(key, plan)
                if normalized is None:
                    value, source, confidence = hints[key], hint_source, hint_confidence
                else:
                    value, source, confidence = normalized
                _append_review_item(items, key, value, source=source, confidence=confidence)

    issues = list(getattr(plan, "blocking_issues", []) or [])
    return {
        "updated_at": _now_time(),
        "needs_review": bool(getattr(plan, "needs_review", False)),
        "issues": issues,
        "review_options": _review_options(result, issues),
        "items": items,
    }


def _set_review_summary(task_id: str, result: Any) -> None:
    task = _tasks.get(task_id)
    if task is None:
        return
    summary = _build_review_summary(result)
    task["review_summary"] = summary
    _emit(task_id, "review", summary=summary)


def _set_task_terminal_status(task_id: str, status: str) -> None:
    task = _tasks.get(task_id)
    if task is None:
        return
    task["status"] = status
    task["finished_at"] = _now_iso()
    _write_task_history(task_id)


def _attach_workflow_recovery_summary(task_id: str, output_dir: Path) -> None:
    task = _tasks.get(task_id)
    if task is None or not output_dir.exists():
        return
    try:
        from agent.agent_core.recovery_report import analyze_agent_recovery

        paths = analyze_agent_recovery(output_dir)
        payload = json.loads(paths["agent_recovery_report_json"].read_text(encoding="utf-8"))
    except Exception as exc:
        task["recovery_report_error"] = str(exc)
        return
    task["workflow_outcome"] = payload.get("workflow_outcome")
    task["usable_partial_outputs"] = bool(payload.get("usable_partial_outputs"))
    task["recovery_primary_issue"] = payload.get("primary_issue")
    task["recovery_recommended_next_step"] = payload.get("recommended_next_step")
    task["recovery_report_json"] = str(paths["agent_recovery_report_json"])
    task["recovery_report_md"] = str(paths["agent_recovery_report_md"])


def _llm_check_error(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        if status_code in {401, 403}:
            return f"API Key 无效或没有权限（HTTP {status_code}）"
        if status_code == 404:
            return "Base URL 或模型名称不可用（HTTP 404）"
        if status_code == 429:
            return "大模型 API 额度不足或触发限流（HTTP 429）"
        detail = exc.response.text[:200].strip()
        suffix = f"：{detail}" if detail else ""
        return f"大模型 API 检查失败（HTTP {status_code}）{suffix}"
    if isinstance(exc, httpx.TimeoutException):
        return "大模型 API 检查超时，请确认 Base URL、模型和网络可用"
    if isinstance(exc, httpx.RequestError):
        return f"无法连接大模型 API：{exc}"
    return f"大模型 API 检查失败：{exc}"


def _llm_trust_environment_proxy(base_url: str) -> bool:
    from urllib.parse import urlparse

    hostname = (urlparse(base_url).hostname or "").lower()
    return hostname not in {"localhost", "127.0.0.1", "::1"}


async def _check_llm_api(config: dict[str, str]) -> tuple[bool, str]:
    check_timeout = max(5.0, min(_positive_float(config["timeout"], 15.0), 15.0))
    payload = {
        "model": config["model"],
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "stream": False,
    }
    timeout = httpx.Timeout(connect=5.0, read=check_timeout, write=5.0, pool=5.0)
    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=_llm_trust_environment_proxy(config["base_url"])) as client:
            response = await client.post(
                f"{config['base_url']}/chat/completions",
                headers={"Authorization": f"Bearer {config['api_key']}"},
                json=payload,
            )
            response.raise_for_status()
    except Exception as exc:
        return False, _llm_check_error(exc)
    return True, "API Key 可用"


async def _run_llm_check(config: dict[str, str]) -> tuple[bool, str]:
    return await _check_llm_api(config)


async def _fetch_llm_models(config: dict[str, str]) -> list[str]:
    timeout = httpx.Timeout(connect=5.0, read=15.0, write=5.0, pool=5.0)
    async with httpx.AsyncClient(timeout=timeout, trust_env=_llm_trust_environment_proxy(config["base_url"])) as client:
        response = await client.get(
            f"{config['base_url']}/models",
            headers={"Authorization": f"Bearer {config['api_key']}"},
        )
        response.raise_for_status()
    payload = response.json()
    records = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        raise ValueError("Model API response has no data list")
    models = [
        _clean_text(record.get("id"))
        for record in records
        if isinstance(record, dict) and _clean_text(record.get("id"))
    ]
    return list(dict.fromkeys(models))


def _ordered_models(models: list[str], selected: str) -> list[str]:
    return [model for model in dict.fromkeys([selected, *models]) if model]


# ── 页面 ──────────────────────────────────────────────────────────
def _start_result_cleanup_worker() -> None:
    global _cleanup_thread_started
    if _cleanup_thread_started:
        return
    _cleanup_thread_started = True
    threading.Thread(target=_cleanup_loop, name="result-cleanup", daemon=True).start()


@app.get("/workbench-legacy", response_class=HTMLResponse)
async def workbench_legacy():
    return (_templates_dir / "index.html").read_text(encoding="utf-8")


@app.get("/benchmark-review-legacy", response_class=HTMLResponse)
async def benchmark_review():
    return (_templates_dir / "benchmark_review.html").read_text(encoding="utf-8")


@app.get("/benchmark-review", response_class=FileResponse)
@app.get("/benchmark-review-next", response_class=FileResponse)
@app.get("/", response_class=FileResponse)
async def carbon_workbench():
    return FileResponse(_benchmark_review_next_dir / "index.html", media_type="text/html")


@app.get("/api/expert-review/status")
async def expert_review_status(request: Request):
    return {
        "ok": True,
        "enabled": expert_review_enabled(),
        "developer_allowed": _review_developer_allowed(request),
        "max_job_workers": MAX_EXPERT_JOB_WORKERS,
    }


@app.get("/api/expert-review/pools")
async def list_expert_review_pools(request: Request, mode: str = "expert"):
    if not expert_review_enabled():
        return {"ok": False, "error": "expert_review_disabled", "pools": []}
    resolved_mode = _request_review_mode(request, mode)
    pools = [_public_pool_record(item, mode=resolved_mode) for item in _expert_pool_registry().list_pools()]
    return {"ok": True, "mode": resolved_mode, "pools": pools}


@app.post("/api/expert-review/pools/import")
async def import_expert_review_pool(body: dict[str, Any], request: Request):
    if not expert_review_enabled():
        return {"ok": False, "error": "expert_review_disabled"}
    developer = _review_developer_allowed(request)
    pool = body.get("pool")
    if not isinstance(pool, dict):
        return {"ok": False, "error": "pool object is required"}
    if not developer:
        from agent.web.expert_review.pool_registry import strip_pool_for_mode

        candidates = [item for item in (pool.get("candidates") or []) if isinstance(item, dict)]
        if any(
            item.get("grade") is not None or item.get("review_notes") or item.get("human_grades") or item.get("machine_reviews") or item.get("machine_review_runs")
            for item in candidates
        ):
            return {"ok": False, "error": "expert_import_requires_unreviewed_blinded_pool"}
        pool = strip_pool_for_mode(pool, mode="expert")
        label = "expert-blind-pool"
        pool_id = None
    else:
        label = _clean_text(body.get("label")) or None
        pool_id = _clean_text(body.get("pool_id")) or None
    try:
        record = _expert_pool_registry().import_pool(pool, label=label, pool_id=pool_id)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "pool": record}


@app.get("/api/expert-review/pools/{pool_id}")
async def get_expert_review_pool(pool_id: str, request: Request, mode: str = "expert"):
    if not expert_review_enabled():
        return {"ok": False, "error": "expert_review_disabled"}
    record = _expert_pool_registry().get_pool(pool_id)
    if record is None:
        return {"ok": False, "error": "pool_not_found"}
    resolved_mode = _request_review_mode(request, mode)
    return {"ok": True, "pool": _public_pool_record(record, mode=resolved_mode), "mode": resolved_mode}


@app.get("/api/expert-review/pools/{pool_id}/candidates")
async def list_expert_review_candidates(
    pool_id: str,
    request: Request,
    mode: str = "expert",
    reviewer_id: str = "",
    task: str | None = None,
    offset: int = 0,
    limit: int = 200,
):
    if not expert_review_enabled():
        return {"ok": False, "error": "expert_review_disabled"}
    payload = _expert_pool_registry().candidates(
        pool_id,
        mode=_request_review_mode(request, mode),
        reviewer_id=_clean_text(reviewer_id),
        task=task,
        offset=offset,
        limit=limit,
    )
    if payload is None:
        return {"ok": False, "error": "pool_not_found"}
    return {"ok": True, **payload}


@app.put("/api/expert-review/pools/{pool_id}/grades/{candidate_id}")
async def upsert_expert_review_grade(pool_id: str, candidate_id: str, body: dict[str, Any], request: Request):
    if not expert_review_enabled():
        return {"ok": False, "error": "expert_review_disabled"}
    mode = _request_review_mode(request, body.get("mode") or "expert")
    registry = _expert_pool_registry()
    clear = bool(body.get("clear"))
    notes = str(body.get("notes") or body.get("review_notes") or "")
    reviewer_id = _clean_text(body.get("reviewer_id")) or ""
    if not reviewer_id:
        return {"ok": False, "error": "reviewer_id_required"}
    change: dict[str, Any] = {}

    def mutate(document: dict[str, Any]) -> dict[str, Any]:
        candidates = [dict(item) for item in (document.get("candidates") or []) if isinstance(item, dict)]
        target = next(
            (index for index, item in enumerate(candidates) if str(item.get("candidate_id") or "") == candidate_id),
            None,
        )
        if target is None:
            raise ValueError("candidate_not_found")
        before = dict(candidates[target])
        grade_before = effective_grade(before)
        if clear:
            updated = append_human_grade(before, grade=None, notes=notes, reviewer_id=reviewer_id, clear=True)
        else:
            grade_raw = body.get("grade")
            if grade_raw is None:
                raise ValueError("grade_required")
            updated = append_human_grade(
                before,
                grade=int(grade_raw),
                notes=notes,
                reviewer_id=reviewer_id,
            )
        candidates[target] = updated
        new_document = {**document, "candidates": candidates}
        new_document.setdefault("schema_version", "discovery-judgment-pool-reviewed/v2")
        change.update(
            before=before,
            after=updated,
            document_before=document,
            document_after=new_document,
            grade_before=grade_before,
        )
        return new_document

    try:
        new_doc, record = registry.mutate_reviewed_pool(pool_id, mutate)
    except (TypeError, ValueError) as exc:
        return {"ok": False, "error": str(exc)}
    before = change["before"]
    updated = change["after"]
    document = change["document_before"]
    grade_before = change["grade_before"]

    impact = None
    if mode == "test":
        session_id = _clean_text(body.get("impact_session_id")) or pool_id
        with _impact_sessions_lock:
            session = dict(_impact_sessions.get(session_id) or {})
        key_payload, runs = _load_impact_bundle(session)
        # temporary pools for before/after
        before_doc = dict(document)
        before_doc["candidates"] = [
            before if str(item.get("candidate_id") or "") == candidate_id else item
            for item in (document.get("candidates") or [])
            if isinstance(item, dict)
        ]
        impact = compute_impact(
            pool_before=before_doc,
            pool_after=new_doc,
            key_payload=key_payload,
            runs=runs or None,
            changed_candidate_id=candidate_id,
            grade_before=grade_before,
            grade_after=effective_grade(updated),
        )

    from agent.web.expert_review.pool_registry import blind_candidate_view

    return {
        "ok": True,
        "pool": _public_pool_record(record, mode=mode),
        "candidate": blind_candidate_view(updated, mode=mode, reviewer_id=reviewer_id),
        "impact": impact,
    }


@app.post("/api/expert-review/pools/{pool_id}/export")
async def export_expert_review_pool(pool_id: str, request: Request, body: dict[str, Any] | None = None):
    if not expert_review_enabled():
        return {"ok": False, "error": "expert_review_disabled"}
    body = body or {}
    document = _expert_pool_registry().load_pool_document(pool_id, prefer_reviewed=True)
    if document is None:
        return {"ok": False, "error": "pool_not_found"}
    reviewer_id = _clean_text(body.get("reviewer_id"))
    if not reviewer_id:
        return {"ok": False, "error": "reviewer_id_required"}
    exported = apply_human_grades_for_export(document, reviewer_id=reviewer_id)
    return {"ok": True, "pool": exported}


@app.post("/api/expert-review/pools/{pool_id}/workspace.zip")
async def export_expert_review_workspace(
    pool_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    body: dict[str, Any] | None = None,
):
    if not expert_review_enabled():
        return {"ok": False, "error": "expert_review_disabled"}
    try:
        path, _manifest = export_workspace_archive(
            _expert_pool_registry(),
            pool_id,
            workspace_state=(body or {}).get("workspace"),
        )
    except WorkspaceArchiveError as exc:
        return {"ok": False, "error": str(exc)}
    background_tasks.add_task(path.unlink, missing_ok=True)
    return FileResponse(
        path=str(path),
        filename=path.name,
        media_type="application/zip",
    )


@app.post("/api/expert-review/workspaces/import")
async def import_expert_review_workspace(request: Request):
    if not expert_review_enabled():
        return {"ok": False, "error": "expert_review_disabled"}
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_WORKSPACE_ARCHIVE_BYTES:
                return {"ok": False, "error": "workspace_archive_too_large"}
        except ValueError:
            return {"ok": False, "error": "content_length_invalid"}
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > MAX_WORKSPACE_ARCHIVE_BYTES:
            return {"ok": False, "error": "workspace_archive_too_large"}
        chunks.append(chunk)
    try:
        record, workspace, restored_jobs = import_workspace_archive(
            _expert_pool_registry(),
            b"".join(chunks),
        )
    except WorkspaceArchiveError as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": True,
        "pool": record,
        "workspace": workspace,
        "restored_jobs": restored_jobs,
    }


@app.get("/api/expert-review/calibration/status")
async def discovery_calibration_status(request: Request):
    from agent.discovery.calibration import DiscoveryCalibrationStore, fit_scoring_calibration

    preview = fit_scoring_calibration(_discovery_calibration_candidates())
    return {
        "ok": True,
        "preview": preview,
        "active": DiscoveryCalibrationStore().load_active(),
    }


@app.post("/api/expert-review/calibration/preview")
async def preview_discovery_calibration(request: Request):
    from agent.discovery.calibration import DiscoveryCalibrationStore, fit_scoring_calibration

    return {
        "ok": True,
        "preview": fit_scoring_calibration(_discovery_calibration_candidates()),
        "active": DiscoveryCalibrationStore().load_active(),
    }


@app.post("/api/expert-review/calibration/activate")
async def activate_discovery_calibration(request: Request, body: dict[str, Any] | None = None):
    from agent.discovery.calibration import DiscoveryCalibrationStore, fit_scoring_calibration

    preview = fit_scoring_calibration(_discovery_calibration_candidates())
    expected_preview_id = _clean_text((body or {}).get("preview_id"))
    if not expected_preview_id or expected_preview_id != preview.get("preview_id"):
        return {"ok": False, "error": "calibration_preview_stale", "preview": preview}
    if not preview.get("eligible"):
        return {"ok": False, "error": "calibration_not_eligible", "preview": preview}
    try:
        active = DiscoveryCalibrationStore().activate(preview)
    except ValueError as exc:
        return {"ok": False, "error": str(exc), "preview": preview}
    return {"ok": True, "active": active, "preview": preview}


@app.post("/api/expert-review/impact/session")
async def bind_expert_impact_session(body: dict[str, Any], request: Request):
    mode = _request_review_mode(request, body.get("mode") or "test")
    if mode != "test":
        return {"ok": False, "error": "impact_requires_test_mode"}
    session_id = _clean_text(body.get("session_id")) or _clean_text(body.get("pool_id")) or uuid.uuid4().hex[:10]
    key_path = _clean_text(body.get("key_path"))
    run_paths = body.get("run_paths") or []
    if not isinstance(run_paths, list):
        return {"ok": False, "error": "run_paths_must_be_list"}
    cleaned_runs = [_clean_text(path) for path in run_paths if _clean_text(path)]
    if key_path and not _impact_path_allowed(key_path):
        return {"ok": False, "error": "key_path_not_allowed"}
    for path in cleaned_runs:
        if not _impact_path_allowed(path):
            return {"ok": False, "error": f"run_path_not_allowed:{path}"}
    with _impact_sessions_lock:
        _impact_sessions[session_id] = {
            "key_path": key_path or None,
            "run_paths": cleaned_runs,
            "updated_at": _now_app_iso(),
        }
    return {
        "ok": True,
        "session_id": session_id,
        "key_bound": bool(key_path),
        "run_count": len(cleaned_runs),
    }


@app.post("/api/expert-review/pools/{pool_id}/impact")
async def expert_review_impact(pool_id: str, body: dict[str, Any], request: Request):
    if not expert_review_enabled():
        return {"ok": False, "error": "expert_review_disabled"}
    mode = _request_review_mode(request, body.get("mode") or "test")
    if mode != "test":
        return {"ok": False, "error": "impact_requires_test_mode"}
    registry = _expert_pool_registry()
    document = registry.load_pool_document(pool_id, prefer_reviewed=True)
    if document is None:
        return {"ok": False, "error": "pool_not_found"}
    session_id = _clean_text(body.get("impact_session_id")) or pool_id
    with _impact_sessions_lock:
        session = dict(_impact_sessions.get(session_id) or {})
    if body.get("key_path"):
        key_path = _clean_text(body.get("key_path"))
        if not _impact_path_allowed(key_path):
            return {"ok": False, "error": "key_path_not_allowed"}
        session["key_path"] = key_path
    if body.get("run_paths"):
        run_paths = [_clean_text(p) for p in body.get("run_paths") or [] if _clean_text(p)]
        for path in run_paths:
            if not _impact_path_allowed(path):
                return {"ok": False, "error": f"run_path_not_allowed:{path}"}
        session["run_paths"] = run_paths
    key_payload, runs = _load_impact_bundle(session)
    # optional hypothetical grade change
    pool_after = dict(document)
    grade_before = None
    grade_after = None
    changed_id = _clean_text(body.get("candidate_id")) or None
    if changed_id and body.get("grade") is not None:
        candidates = []
        for item in document.get("candidates") or []:
            if not isinstance(item, dict):
                continue
            if str(item.get("candidate_id") or "") != changed_id:
                candidates.append(item)
                continue
            grade_before = effective_grade(item)
            updated = append_human_grade(
                item,
                grade=int(body.get("grade")),
                notes=str(body.get("notes") or ""),
                reviewer_id=_clean_text(body.get("reviewer_id")) or "",
            )
            grade_after = effective_grade(updated)
            candidates.append(updated)
        pool_after = {**document, "candidates": candidates}
    impact = compute_impact(
        pool_before=document,
        pool_after=pool_after,
        key_payload=key_payload,
        runs=runs or None,
        changed_candidate_id=changed_id,
        grade_before=grade_before,
        grade_after=grade_after,
    )
    return {"ok": True, "impact": impact}


@app.get("/api/expert-review/jobs")
async def list_expert_judge_jobs(request: Request, pool_id: str | None = None):
    if not expert_review_enabled():
        return {"ok": False, "error": "expert_review_disabled", "jobs": []}
    return {"ok": True, "jobs": _expert_job_manager().list_jobs(pool_id=pool_id)}


@app.post("/api/expert-review/jobs")
async def start_expert_judge_job(body: dict[str, Any], request: Request):
    if not expert_review_enabled():
        return {"ok": False, "error": "expert_review_disabled"}
    mode = _request_review_mode(request, body.get("mode") or "developer")
    if mode == "expert":
        return {"ok": False, "error": "jobs_forbidden_in_expert_mode"}
    pool_id = _clean_text(body.get("pool_id"))
    job_type = _clean_text(body.get("job_type") or "single_model")
    if job_type == "model_expert_consensus":
        if not pool_id:
            return {"ok": False, "error": "pool_id_required"}
        generator_identity = body.get("generator_identity")
        if not isinstance(generator_identity, dict):
            generator_identity = {}
        try:
            job = _expert_job_manager().start_consensus_job(
                pool_id=pool_id,
                generator_identity=generator_identity,
                workers=int(body.get("workers") or 1),
                idempotency_key=_clean_text(body.get("idempotency_key")) or None,
                output_language=_normalise_pool_build_language(body.get("output_language")),
                scale_mode=_normalise_pool_build_scale(body.get("scale_mode") or "auto"),
            )
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        except Exception as exc:
            return {"ok": False, "error": _redact_secrets(str(exc))}
        return {"ok": True, "job": job}
    profile_id = _clean_text(body.get("profile_id"))
    if not pool_id or not profile_id:
        return {"ok": False, "error": "pool_id_and_profile_id_required"}
    try:
        job = _expert_job_manager().start_job(
            pool_id=pool_id,
            profile_id=profile_id,
            independent_model=bool(body.get("independent_model")),
            workers=int(body.get("workers") or 2),
        )
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:
        return {"ok": False, "error": _redact_secrets(str(exc))}
    return {"ok": True, "job": job}


@app.get("/api/expert-review/jobs/{job_id}")
async def get_expert_judge_job(job_id: str, request: Request, detail: int = 0):
    if not expert_review_enabled():
        return {"ok": False, "error": "expert_review_disabled"}
    job = _expert_job_manager().get_job(job_id, detail=bool(detail))
    if job is None:
        return {"ok": False, "error": "job_not_found"}
    return {"ok": True, "job": job}


@app.delete("/api/expert-review/jobs/{job_id}")
async def delete_expert_judge_job(job_id: str, request: Request):
    if not expert_review_enabled():
        return {"ok": False, "error": "expert_review_disabled"}
    try:
        job = _expert_job_manager().delete_job(job_id)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    if job is None:
        return {"ok": False, "error": "job_not_found"}
    return {"ok": True, "deleted": True, "job": job}


@app.post("/api/expert-review/jobs/{job_id}/cancel")
async def cancel_expert_judge_job(job_id: str, request: Request):
    if not expert_review_enabled():
        return {"ok": False, "error": "expert_review_disabled"}
    job = _expert_job_manager().cancel_job(job_id)
    if job is None:
        return {"ok": False, "error": "job_not_found"}
    return {"ok": True, "job": job}


@app.post("/api/expert-review/jobs/{job_id}/resume")
async def resume_expert_judge_job(job_id: str, request: Request, body: dict[str, Any] | None = None):
    if not expert_review_enabled():
        return {"ok": False, "error": "expert_review_disabled"}
    try:
        raw_workers = body.get("workers") if isinstance(body, dict) else None
        workers = None if raw_workers is None or raw_workers == "" else raw_workers
        job = _expert_job_manager().resume_job(job_id, workers=workers)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    if job is None:
        return {"ok": False, "error": "job_not_found"}
    return {"ok": True, "job": job}


@app.post("/api/expert-review/jobs/{job_id}/retry-failed")
async def retry_expert_judge_job(job_id: str, request: Request, body: dict[str, Any] | None = None):
    if not expert_review_enabled():
        return {"ok": False, "error": "expert_review_disabled"}
    try:
        raw_workers = body.get("workers") if isinstance(body, dict) else None
        workers = None if raw_workers is None or raw_workers == "" else raw_workers
        job = _expert_job_manager().retry_failed(job_id, workers=workers)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    if job is None:
        return {"ok": False, "error": "job_not_found"}
    return {"ok": True, "job": job}


@app.get("/api/benchmark-review/build-llm-config")
async def get_pool_build_llm_config(request: Request):
    profile = _pool_build_llm_config_store().get_profile(
        _POOL_BUILD_LLM_PROFILE_ID,
        include_secrets=False,
    )
    return {
        "ok": True,
        "configured": profile is not None,
        "profile": profile,
    }


@app.put("/api/benchmark-review/build-llm-config")
async def save_pool_build_llm_config(body: dict[str, Any], request: Request):
    payload = body.get("profile", body)
    if not isinstance(payload, dict):
        return {"ok": False, "error": "profile payload must be an object"}
    model = _clean_text(payload.get("model"))
    base_url = _clean_text(payload.get("base_url"))
    provider = _clean_text(payload.get("provider") or "openai_compatible").casefold()
    if provider not in _POOL_BUILD_LLM_PROVIDERS:
        return {"ok": False, "error": "pool_build_provider_requires_openai_compatible_protocol"}
    existing = _pool_build_llm_config_store().get_profile_secrets(_POOL_BUILD_LLM_PROFILE_ID)
    explicit_key = _clean_text(payload.get("api_key"))
    if existing is not None and not explicit_key:
        existing_base = _clean_text(existing.get("base_url")).rstrip("/")
        if base_url.rstrip("/") != existing_base:
            return {"ok": False, "error": "api_key_required_for_new_base_url"}
    profile_payload = {
        **payload,
        "id": _POOL_BUILD_LLM_PROFILE_ID,
        "label": "评审池构建模型",
        "provider": provider,
        "requested_model_id": model,
        "model_family": _clean_text(payload.get("model_family") or model),
        "endpoint_identity": _clean_text(payload.get("endpoint_identity") or base_url),
        "routing_profile_id": _POOL_BUILD_LLM_PROFILE_ID,
        "identity_verification": "unverified",
        "enabled": False,
    }
    try:
        profile = _pool_build_llm_config_store().upsert_profile(
            profile_payload,
            make_default=True,
        )
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": True,
        "configured": True,
        "profile": profile,
    }


def _pool_build_llm_operation_config(
    body: Mapping[str, Any],
    *,
    require_model: bool,
) -> tuple[dict[str, str] | None, str | None, str]:
    payload = body.get("config", body)
    if not isinstance(payload, Mapping):
        return None, "pool_build_llm_config_must_be_object", ""
    provider = _clean_text(payload.get("provider") or "openai_compatible").casefold()
    if provider not in _POOL_BUILD_LLM_PROVIDERS:
        return None, "pool_build_provider_requires_openai_compatible_protocol", ""

    saved = _pool_build_llm_config_store().get_profile_secrets(_POOL_BUILD_LLM_PROFILE_ID)
    explicit_key = _clean_text(payload.get("api_key"))
    explicit_base = _clean_text(payload.get("base_url")).rstrip("/")
    supplied = {
        key: value
        for key, value in payload.items()
        if key in {"api_key", "base_url", "model", "timeout"}
        and _clean_text(value)
    }
    if explicit_key:
        merged = {**(saved or {}), **supplied, "api_key": explicit_key}
    elif saved is not None:
        saved_base = _clean_text(saved.get("base_url")).rstrip("/")
        if explicit_base and explicit_base != saved_base:
            return None, "api_key_required_for_new_base_url", ""
        merged = {
            **saved,
            **{
                key: value
                for key, value in supplied.items()
                if key in {"model", "timeout"}
            },
        }
    else:
        return None, "pool_build_api_key_or_saved_config_required", ""

    base_url = _clean_text(merged.get("base_url")).rstrip("/")
    selected_model = _clean_text(merged.get("model"))
    if not base_url:
        return None, "pool_build_base_url_required", selected_model
    if require_model and not selected_model:
        return None, "pool_build_model_required", selected_model
    merged["base_url"] = base_url
    merged["model"] = selected_model or "__model_discovery__"
    config, error = _build_llm_config(dict(merged), allow_server_default=False)
    return config, error, selected_model


def _safe_pool_build_llm_operation_message(message: Any, *, api_key: str) -> str:
    safe = _redact_secrets(message)
    if api_key:
        safe = safe.replace(api_key, "[redacted-api-key]")
    safe = re.sub(r"https?://\S+", "[provider endpoint]", safe, flags=re.IGNORECASE)
    return safe[:500].strip()


@app.post("/api/benchmark-review/build-llm-config/models")
async def list_pool_build_llm_models(body: dict[str, Any], request: Request):
    config, error, selected_model = _pool_build_llm_operation_config(
        body,
        require_model=False,
    )
    if error or config is None:
        return {"ok": False, "error": error, "models": []}
    try:
        models = await _fetch_llm_models(config)
    except Exception as exc:
        return {
            "ok": False,
            "error": _safe_pool_build_llm_operation_message(
                _llm_check_error(exc),
                api_key=config["api_key"],
            ),
            "models": [],
        }
    models = list(dict.fromkeys(model for model in models if model != "__model_discovery__"))
    return {
        "ok": True,
        "models": models,
        "selected": selected_model if selected_model in models else (models[0] if models else ""),
    }


@app.post("/api/benchmark-review/build-llm-config/check")
async def check_pool_build_llm_config(body: dict[str, Any], request: Request):
    config, error, _selected_model = _pool_build_llm_operation_config(
        body,
        require_model=True,
    )
    if error or config is None:
        return {"ok": False, "error": error}
    ok, message = await _run_llm_check(config)
    message = _safe_pool_build_llm_operation_message(
        message,
        api_key=config["api_key"],
    )
    if not ok:
        return {"ok": False, "error": message, "message": message}
    return {"ok": True, "message": message}


@app.get("/api/benchmark-review/builds")
@app.get("/api/expert-review/pool-builds")
async def list_expert_pool_builds(request: Request):
    if not expert_review_enabled():
        return {"ok": False, "error": "expert_review_disabled", "builds": []}
    builds = _expert_pool_build_manager().list_builds()
    jobs = _expert_job_manager().list_jobs() if any(build.get("review_job_id") for build in builds) else []
    return {"ok": True, "builds": attach_review_progress(builds, jobs)}


@app.post("/api/benchmark-review/builds")
@app.post("/api/expert-review/pool-builds")
async def start_expert_pool_build(body: dict[str, Any], request: Request):
    if not expert_review_enabled():
        return {"ok": False, "error": "expert_review_disabled"}
    prompt = _clean_text(body.get("prompt"))
    if not prompt:
        return {"ok": False, "error": "prompt_required"}
    advanced = body.get("advanced") if isinstance(body.get("advanced"), dict) else {}
    discovery = body.get("discovery") if isinstance(body.get("discovery"), dict) else {}
    output_language = _normalise_pool_build_language(
        body.get("output_language") or body.get("ui_language") or discovery.get("output_language")
    )
    scale_mode = _normalise_pool_build_scale(
        body.get("scale_mode") or discovery.get("scale_mode") or "auto",
        prompt=prompt,
        allow_auto=True,
    )
    discovery_body = {
        **advanced,
        **discovery,
        "prompt": prompt,
        "output_language": output_language,
        "scale_mode": scale_mode,
        "runtime": "openai_agents",
        "source": "remote",
    }
    request_id = _clean_text(body.get("idempotency_key") or body.get("client_request_id")) or uuid.uuid4().hex
    review = dict(body.get("review")) if isinstance(body.get("review"), dict) else {}
    review.pop("generator_identity", None)
    review["output_language"] = output_language
    review["scale_mode"] = scale_mode
    try:
        build = _expert_pool_build_manager().start_build(
            discovery_request=discovery_body,
            action=_clean_text(body.get("action") or "build_and_review"),
            label=_clean_text(body.get("label")) or None,
            preset_id=_clean_text(body.get("preset_id") or "default/v1"),
            review=review,
            idempotency_key=request_id,
        )
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:
        return {"ok": False, "error": _redact_secrets(str(exc))}
    return {"ok": True, "build": build}


@app.get("/api/benchmark-review/builds/{build_id}")
@app.get("/api/expert-review/pool-builds/{build_id}")
async def get_expert_pool_build(build_id: str, request: Request):
    if not expert_review_enabled():
        return {"ok": False, "error": "expert_review_disabled"}
    build = _expert_pool_build_manager().get_build(build_id)
    if build is None:
        return {"ok": False, "error": "build_not_found"}
    jobs = _expert_job_manager().list_jobs() if build.get("review_job_id") else []
    enriched = attach_review_progress([build], jobs)
    return {"ok": True, "build": enriched[0]}


@app.post("/api/benchmark-review/builds/{build_id}/cancel")
@app.post("/api/expert-review/pool-builds/{build_id}/cancel")
async def cancel_expert_pool_build(build_id: str, request: Request):
    if not expert_review_enabled():
        return {"ok": False, "error": "expert_review_disabled"}
    build = _expert_pool_build_manager().cancel_build(build_id)
    if build is None:
        return {"ok": False, "error": "build_not_found"}
    return {"ok": True, "build": build}


@app.post("/api/benchmark-review/builds/{build_id}/reconcile")
@app.post("/api/expert-review/pool-builds/{build_id}/reconcile")
async def reconcile_expert_pool_build(build_id: str, request: Request):
    if not expert_review_enabled():
        return {"ok": False, "error": "expert_review_disabled"}
    build = _expert_pool_build_manager().reconcile_review(build_id)
    if build is None:
        return {"ok": False, "error": "build_not_found"}
    return {"ok": True, "build": build}


# ── 健康检查 ──────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    with _tasks_lock:
        queue_state = _queue_state_locked()
    return {
        "status": "ok",
        "llm_configured": _server_llm_config()[0] is not None,
        "per_task_api_keys": True,
        "result_retention_seconds": _result_retention_seconds(),
        "max_result_projects": _max_result_projects(),
        "full_workflow_enabled": _full_workflow_enabled(),
        "system_metrics": collect_system_metrics(_runs_dir),
        **queue_state,
    }


# ── 获取当前配置（脱敏） ──────────────────────────────────────────
@app.get("/api/config")
async def get_config():
    with _tasks_lock:
        queue_state = _queue_state_locked()
    return {
        "api_key_masked": "",
        **_public_llm_config(),
        "per_task_api_keys": True,
        "result_retention_seconds": _result_retention_seconds(),
        "max_result_projects": _max_result_projects(),
        "full_workflow_enabled": _full_workflow_enabled(),
        **queue_state,
    }


@app.get("/api/llm/config")
async def get_llm_config(request: Request):
    return _public_llm_config()


@app.put("/api/llm/config")
async def save_llm_config(body: dict[str, Any], request: Request):
    payload = body.get("llm_config", body)
    if not isinstance(payload, dict):
        payload = {}
    existing = _llm_config_store().load() or {}
    api_key = _clean_text(payload.get("api_key")) or existing.get("api_key", "")
    requested_base = _clean_text(payload.get("base_url")).rstrip("/")
    existing_base = str(existing.get("base_url") or "").rstrip("/")
    if not api_key:
        return {"ok": False, "error": "API Key is required before saving configuration."}
    config, error = _build_llm_config({**existing, **payload, "api_key": api_key})
    if error or config is None:
        return {"ok": False, "error": error}
    ok, message = await _run_llm_check(config)
    if not ok:
        return {"ok": False, "error": message}
    _llm_config_store().save(config)
    return {"ok": True, **_public_llm_config(), "message": message}


@app.delete("/api/llm/config")
async def delete_llm_config(request: Request):
    return {"ok": True, "deleted": _llm_config_store().delete()}


@app.get("/api/llm/profiles")
async def list_llm_profiles(request: Request):
    return {"ok": True, "profiles": _llm_config_store().list_profiles(include_secrets=False)}


@app.post("/api/llm/profiles")
async def create_llm_profile(body: dict[str, Any], request: Request):
    payload = body.get("profile", body)
    if not isinstance(payload, dict):
        return {"ok": False, "error": "profile payload must be an object"}
    make_default = bool(body.get("make_default") or payload.get("make_default"))
    try:
        profile = _llm_config_store().upsert_profile(payload, make_default=make_default)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "profile": profile, "profiles": _llm_config_store().list_profiles(include_secrets=False)}


@app.put("/api/llm/profiles/{profile_id}")
async def update_llm_profile(profile_id: str, body: dict[str, Any], request: Request):
    payload = body.get("profile", body)
    if not isinstance(payload, dict):
        return {"ok": False, "error": "profile payload must be an object"}
    payload = {**payload, "id": profile_id}
    make_default = bool(body.get("make_default") or payload.get("make_default"))
    try:
        profile = _llm_config_store().upsert_profile(payload, make_default=make_default)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "profile": profile, "profiles": _llm_config_store().list_profiles(include_secrets=False)}


@app.delete("/api/llm/profiles/{profile_id}")
async def delete_llm_profile(profile_id: str, request: Request):
    deleted = _llm_config_store().delete_profile(profile_id)
    if not deleted:
        return {"ok": False, "error": "profile_not_found"}
    return {"ok": True, "deleted": True, "profiles": _llm_config_store().list_profiles(include_secrets=False)}


@app.post("/api/llm/models")
async def list_llm_models(body: dict[str, Any], request: Request):
    payload = body.get("llm_config", body)
    if not isinstance(payload, dict):
        payload = {}
    profile_id = _clean_text(body.get("profile_id") or payload.get("profile_id"))
    explicit_key = _clean_text(payload.get("api_key"))
    saved = _llm_config_store().get_profile_secrets(profile_id) if profile_id else None
    explicit_base = _clean_text(payload.get("base_url")).rstrip("/")
    if explicit_key:
        merged = {
            **(saved or {}),
            **{key: value for key, value in payload.items() if key in {"api_key", "base_url", "model", "timeout"}},
            "api_key": explicit_key,
        }
    elif saved is not None:
        saved_base = str(saved.get("base_url") or "").rstrip("/")
        if explicit_base and explicit_base != saved_base:
            return {"ok": False, "error": "api_key_required_for_new_base_url", "models": []}
        merged = {
            **saved,
            **{key: value for key, value in payload.items() if key in {"model", "timeout"} and str(value or "").strip()},
        }
    else:
        return {"ok": False, "error": "profile_not_found_or_api_key_required", "models": []}
    selected_model = _clean_text(merged.get("model"))
    merged["model"] = selected_model or "__model_discovery__"
    config, error = _build_llm_config(merged)
    if error or config is None:
        return {"ok": False, "error": error, "models": []}
    try:
        models = await _fetch_llm_models(config)
    except Exception as exc:
        return {"ok": False, "error": _llm_check_error(exc), "models": []}
    models = [model for model in _ordered_models(models, selected_model) if model != "__model_discovery__"]
    return {
        "ok": True,
        "models": models,
        "selected": selected_model if selected_model in models else (models[0] if models else ""),
    }


@app.post("/api/llm/check")
async def check_llm(body: dict[str, Any], request: Request):
    llm_config = body.get("llm_config", body)
    if not isinstance(llm_config, dict):
        llm_config = {}
    existing = _llm_config_store().load() or {}
    requested_base = _clean_text(llm_config.get("base_url")).rstrip("/")
    existing_base = str(existing.get("base_url") or "").rstrip("/")
    explicit_key = _clean_text(llm_config.get("api_key"))
    config, error = _build_llm_config(llm_config)
    if error or config is None:
        return {"ok": False, "error": error}
    ok, message = await _run_llm_check(config)
    if not ok:
        return {"ok": False, "error": message}
    return {"ok": True, "message": message, "base_url": config["base_url"], "model": config["model"]}


@app.get("/api/results")
async def list_public_results():
    removed = _cleanup_expired_results()
    return {
        "retention_seconds": _result_retention_seconds(),
        "max_result_projects": _max_result_projects(),
        "removed": removed,
        "results": _list_public_results(),
    }


def _find_discovery_job_for_history_id(identifier: str) -> dict[str, Any] | None:
    direct = _load_discovery_job(identifier)
    if direct:
        return direct
    jobs_dir = _discovery_jobs_dir()
    if not jobs_dir.exists():
        return None
    for path in jobs_dir.glob("*.json"):
        job = _read_json_if_exists(path)
        if not job:
            continue
        body = job.get("body") if isinstance(job.get("body"), Mapping) else {}
        record = job.get("record") if isinstance(job.get("record"), Mapping) else {}
        aliases = {
            _clean_text(job.get("job_id")),
            _clean_text(body.get("_execution_discovery_id")),
            _clean_text(record.get("discovery_id")),
            _clean_text(record.get("run_id")),
        }
        if identifier in aliases:
            return job
    return None


def _history_delete_targets(
    kind: str,
    identifier: str,
    *,
    include_linked_batches: bool,
) -> list[dict[str, Any]]:
    kind = _clean_text(kind).lower()
    identifier = _clean_text(identifier)
    if kind not in {"discovery", "batch"} or not identifier:
        raise ValueError("Unsupported history item.")
    targets: list[dict[str, Any]] = []
    if kind == "batch":
        batch = _load_batch_from_disk(identifier)
        with _batches_lock:
            batch = _batches.get(identifier) or batch
        archived = _find_history_record(identifier) or {}
        if not batch and not archived:
            raise ValueError("Batch not found.")
        batch_dir = managed_child(_batch_root_dir(), identifier)
        targets.append(
            {
                "kind": "batch",
                "id": identifier,
                "status": str((batch or {}).get("status") or archived.get("status") or "completed"),
                "path": str(batch_dir),
                "size_bytes": path_size_bytes(batch_dir),
                "result_available": batch_dir.exists(),
            }
        )
        return targets

    job = _find_discovery_job_for_history_id(identifier)
    body = job.get("body") if isinstance(job, Mapping) and isinstance(job.get("body"), Mapping) else {}
    record = job.get("record") if isinstance(job, Mapping) and isinstance(job.get("record"), Mapping) else {}
    discovery_id = _clean_text(
        record.get("discovery_id")
        or body.get("_execution_discovery_id")
        or identifier
    )
    output_dir = _safe_discovery_dir(discovery_id)
    if output_dir is None:
        raise ValueError("Discovery run not found.")
    archived = _find_history_record(identifier) or {}
    targets.append(
        {
            "kind": "discovery",
            "id": discovery_id,
            "job_id": _clean_text(job.get("job_id")) if isinstance(job, Mapping) else "",
            "status": str((job or {}).get("status") or archived.get("status") or "completed"),
            "path": str(output_dir),
            "size_bytes": path_size_bytes(output_dir),
            "result_available": output_dir.exists(),
        }
    )
    if include_linked_batches:
        job_id = _clean_text((job or {}).get("job_id"))
        for history in _list_parameter_batch_history_records(
            use_cache=False,
            include_file_stats=True,
        ):
            if not (
                _clean_text(history.get("source_discovery_job_id")) == job_id
                or _clean_text(history.get("source_discovery_id")) == discovery_id
            ):
                continue
            batch_id = _clean_text(history.get("batch_id") or history.get("result_id"))
            if not batch_id:
                continue
            targets.append(
                {
                    "kind": "batch",
                    "id": batch_id,
                    "status": str(history.get("status") or "unknown"),
                    "path": str(managed_child(_batch_root_dir(), batch_id)),
                    "size_bytes": int(history.get("size_bytes") or 0),
                }
            )
    return targets


def _history_delete_preview(
    kind: str,
    identifier: str,
    *,
    include_linked_batches: bool,
) -> dict[str, Any]:
    targets = _history_delete_targets(
        kind,
        identifier,
        include_linked_batches=include_linked_batches,
    )
    active = [target for target in targets if target["status"] in _ACTIVE_STATUSES]
    confirmation_id = secrets.token_urlsafe(24)
    payload = {
        "kind": kind,
        "id": identifier,
        "include_linked_batches": include_linked_batches,
        "targets": targets,
        "estimated_bytes": sum(int(target["size_bytes"]) for target in targets),
        "deletable": not active,
        "block_reason": "请先停止所有运行中的关联任务。" if active else "",
        "confirmation_id": confirmation_id,
    }
    with _history_delete_confirmations_lock:
        now = time.time()
        for key, value in list(_history_delete_confirmations.items()):
            if float(value.get("expires_at") or 0) <= now:
                _history_delete_confirmations.pop(key, None)
        _history_delete_confirmations[confirmation_id] = {
            **payload,
            "expires_at": now + 300,
        }
    return payload


def _remove_history_index_targets(deleted: list[dict[str, Any]]) -> None:
    aliases: set[str] = set()
    for target in deleted:
        identifier = str(target.get("id") or "")
        aliases.update(
            {
                identifier,
                f"discovery-{identifier}",
                f"batch-{identifier}",
                str(target.get("job_id") or ""),
            }
        )
    aliases.discard("")
    with _history_index_lock:
        kept = []
        for record in _read_history_index():
            item = with_history_identity(record)
            values = {
                str(item.get(field) or "")
                for field in (
                    "history_id",
                    "run_id",
                    "result_id",
                    "name",
                    "task_id",
                    "discovery_id",
                    "batch_id",
                    "job_id",
                )
            }
            values.add(Path(str(item.get("output_dir") or "")).name)
            if values & aliases:
                continue
            kept.append(item)
        _write_history_index(kept)


def _mark_operations_history_targets_deleted(
    deleted: list[dict[str, Any]],
) -> None:
    """Keep the SQL history index consistent with filesystem deletion."""
    try:
        repository = get_operations_repository()
    except Exception:
        return
    for target in deleted:
        kind = _clean_text(target.get("kind")).lower()
        source_id = _clean_text(
            target.get("job_id") if kind == "discovery" else target.get("id")
        )
        if not kind or not source_id:
            continue
        try:
            repository.mark_history_deleted(
                f"{kind}:{source_id}",
                released_bytes=int(target.get("released_bytes") or 0),
            )
        except (KeyError, ValueError):
            # Compatibility history remains authoritative for legacy rows that
            # were never imported into the SQL operations index.
            continue
        except Exception:
            # Filesystem deletion has already completed. Startup/history
            # reconciliation can retry SQL repair without resurrecting files.
            continue


def _execute_history_delete(
    kind: str,
    identifier: str,
    body: Mapping[str, Any],
) -> dict[str, Any]:
    confirmation_id = _clean_text(body.get("confirmation_id"))
    include_linked_batches = body.get("include_linked_batches") is True
    with _history_delete_confirmations_lock:
        confirmation = _history_delete_confirmations.pop(confirmation_id, None)
    if (
        not confirmation
        or float(confirmation.get("expires_at") or 0) <= time.time()
        or confirmation.get("kind") != kind
        or confirmation.get("id") != identifier
        or bool(confirmation.get("include_linked_batches")) != include_linked_batches
    ):
        raise ValueError("删除确认已失效，请重新预览删除范围。")
    targets = _history_delete_targets(
        kind,
        identifier,
        include_linked_batches=include_linked_batches,
    )
    preview_scope = {
        (str(target.get("kind") or ""), str(target.get("id") or ""))
        for target in confirmation.get("targets") or []
        if isinstance(target, Mapping)
    }
    current_scope = {
        (str(target.get("kind") or ""), str(target.get("id") or ""))
        for target in targets
    }
    if current_scope != preview_scope:
        raise ValueError(
            "The deletion scope changed after preview. Preview the deletion again."
        )
    if any(target["status"] in _ACTIVE_STATUSES for target in targets):
        raise ValueError("运行中的任务不能删除，请先停止任务。")
    deleted: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for target in targets:
        try:
            if target["kind"] == "batch":
                receipt = delete_managed_tree(_batch_root_dir(), target["id"])
                with _batches_lock:
                    _batches.pop(target["id"], None)
            else:
                receipt = delete_managed_tree(_discovery_root_dir(), target["id"])
                job_id = _clean_text(target.get("job_id"))
                if job_id:
                    with _discovery_jobs_lock:
                        _discovery_jobs.pop(job_id, None)
                    job_path = _discovery_job_path(job_id)
                    if job_path.exists():
                        job_path.unlink()
                    summary_path = _discovery_job_summary_path(job_id)
                    if summary_path.exists():
                        summary_path.unlink()
            deleted.append({**target, **receipt})
        except Exception as exc:
            failed.append({**target, "error": _redact_secrets(str(exc))})
    _remove_history_index_targets(deleted)
    _mark_operations_history_targets_deleted(deleted)
    with _batch_history_cache_lock:
        _batch_history_cache["ts"] = 0.0
        _batch_history_cache["root"] = ""
        _batch_history_cache["generation"] = int(
            _batch_history_cache.get("generation") or 0
        ) + 1
        _batch_history_cache["records"] = []
    return {
        "status": "completed" if not failed else "partial",
        "deleted": deleted,
        "failed": failed,
        "estimated_bytes": int(confirmation.get("estimated_bytes") or 0),
        "released_bytes": sum(
            int(target.get("released_bytes") or 0) for target in deleted
        ),
    }


@app.get("/api/history")
async def list_project_history(fast: bool = True, refresh: bool = False):
    if fast and not refresh:
        with _tasks_lock:
            active_tasks = [
                _public_task_record_locked(task_id, task)
                for task_id, task in _tasks.items()
                if task.get("status") in _ACTIVE_STATUSES
            ]
        active_tasks.extend(batch for batch in _list_parameter_batch_history_records(include_file_stats=False) if batch.get("status") in _ACTIVE_STATUSES)
        # Discovery jobs that are still running should also appear in history.
        discovery_history = _list_discovery_history_records_fast(limit=100)
        for item in discovery_history:
            if str(item.get("status") or "").lower() in _ACTIVE_STATUSES:
                active_tasks.append(item)
        active_tasks.sort(key=lambda item: str(item.get("created_at") or item.get("history_time") or ""))
        active_task_ids = {str(item.get("task_id") or "") for item in active_tasks}
        active_history_ids = {str(item.get("history_id") or "") for item in active_tasks}
        active_history_ids.update(str(item.get("output_dir") or "") for item in active_tasks)
        active_history_ids.update(str(item.get("run_id") or "") for item in active_tasks)
        active_task_ids.discard("")
        active_history_ids.discard("")
        results = []
        # Merge ordinary task history with discovery run history.
        combined = list(_list_project_history_records_fast())
        combined.extend(discovery_history)
        # de-dupe by history_id/run_id
        seen: set[str] = set()
        ordered: list[dict[str, Any]] = []
        for item in sorted(
            combined,
            key=lambda row: str(row.get("history_time") or row.get("finished_at") or row.get("created_at") or ""),
            reverse=True,
        ):
            key = str(item.get("history_id") or item.get("run_id") or item.get("result_id") or "")
            if not key or key in seen:
                continue
            seen.add(key)
            ordered.append(item)
        for item in ordered:
            task_ids = {str(item.get("task_id") or ""), *(str(value or "") for value in item.get("task_ids") or [])}
            history_ids = {
                str(item.get("history_id") or ""),
                str(item.get("output_dir") or ""),
                str(item.get("run_id") or ""),
                str(item.get("result_id") or ""),
                str(item.get("name") or ""),
            }
            task_ids.discard("")
            history_ids.discard("")
            if task_ids & active_task_ids or history_ids & active_history_ids:
                continue
            results.append(item)
        return {
            "retention_seconds": _result_retention_seconds(),
            "max_result_projects": _max_result_projects(),
            "removed": [],
            "summary": _history_summary(active_tasks, results),
            "active_tasks": active_tasks,
            "results": results,
            "history_mode": "fast",
        }
    _sync_history_index_from_disk()
    removed = _cleanup_expired_results()
    if removed:
        _sync_history_index_from_disk()
    with _tasks_lock:
        active_tasks = [
            _public_task_record_locked(task_id, task)
            for task_id, task in _tasks.items()
            if task.get("status") in _ACTIVE_STATUSES
        ]
    active_tasks.extend(batch for batch in _list_parameter_batch_history_records(use_cache=False) if batch.get("status") in _ACTIVE_STATUSES)
    active_tasks.sort(key=lambda item: str(item.get("created_at") or ""))
    active_task_ids = {str(item.get("task_id") or "") for item in active_tasks}
    active_history_ids = {str(item.get("history_id") or "") for item in active_tasks}
    active_history_ids.update(str(item.get("output_dir") or "") for item in active_tasks)
    active_history_ids.update(str(item.get("run_id") or "") for item in active_tasks)
    active_task_ids.discard("")
    active_history_ids.discard("")
    results = []
    for item in _list_project_history_records():
        task_ids = {str(item.get("task_id") or ""), *(str(value or "") for value in item.get("task_ids") or [])}
        history_ids = {
            str(item.get("history_id") or ""),
            str(item.get("output_dir") or ""),
            str(item.get("run_id") or ""),
            str(item.get("result_id") or ""),
            str(item.get("name") or ""),
        }
        task_ids.discard("")
        history_ids.discard("")
        if task_ids & active_task_ids or history_ids & active_history_ids:
            continue
        results.append(item)
    summary = _history_summary(active_tasks, results)
    return {
        "retention_seconds": _result_retention_seconds(),
        "max_result_projects": _max_result_projects(),
        "removed": removed,
        "summary": summary,
        "active_tasks": active_tasks,
        "results": results,
        "history_mode": "full",
    }


@app.get("/api/history/{kind}/{identifier}/delete-preview")
async def preview_history_delete(
    kind: str,
    identifier: str,
    include_linked_batches: bool = False,
):
    try:
        return _history_delete_preview(
            kind,
            identifier,
            include_linked_batches=include_linked_batches,
        )
    except Exception as exc:
        return {"error": _redact_secrets(str(exc))}


@app.delete("/api/history/{kind}/{identifier}")
async def delete_history_item(
    kind: str,
    identifier: str,
    body: dict[str, Any],
):
    try:
        return _execute_history_delete(kind, identifier, body)
    except Exception as exc:
        return {"error": _redact_secrets(str(exc))}


@app.get("/api/results/{result_id}/download")
async def download_public_result(result_id: str):
    output_dir = _safe_run_dir(result_id)
    if output_dir is None or not output_dir.exists():
        return {"error": "结果目录不存在。"}
    history = _read_public_history(output_dir)
    if history.get("status", "completed") != "completed":
        return {"error": "任务未完成，不能下载结果。"}
    if not _has_downloadable_result_file(output_dir):
        return {"error": "结果目录没有可下载文件。"}

    if not _ensure_existing_download_zip_ready(output_dir):
        return {"error": "结果 ZIP 尚未打包完成，请等待任务日志提示后再下载。"}
    zip_path = _download_zip_path(output_dir)
    return FileResponse(
        path=str(zip_path),
        filename=f"{result_id}_results.zip",
        media_type="application/zip",
    )


# ── 创建任务 ──────────────────────────────────────────────────────
def _clean_path_list(value: Any) -> list[str]:
    values: list[str] = []
    if isinstance(value, list):
        values.extend(_clean_text(item) for item in value)
    else:
        text = _clean_text(value)
        values.extend(item.strip() for item in text.splitlines())
    return [item for item in values if item]


def _container_repo_path_hint(value: str) -> str:
    text = _clean_text(value)
    if not text:
        return text
    normalized = text.replace("\\", "/")
    marker = "agent-aireadyy_project/agent-aireadyy/"
    if marker in normalized and not Path(text).exists():
        suffix = normalized.split(marker, 1)[1]
        return str(Path("/app") / suffix)
    return text


def _clean_ai_ready_task_types(body: dict[str, Any]) -> list[str]:
    raw = body.get("task_types")
    if raw is None:
        raw = body.get("task_type")
    values = _clean_path_list(raw)
    if not values:
        values = ["rt_prediction"]
    result: list[str] = []
    for value in values:
        task_type = normalize_task_type(value)
        if task_type and task_type not in result:
            result.append(task_type)
    return result


def _new_ai_ready_build_id(prefix: str = "ai_ready") -> str:
    return safe_output_stem(f"{prefix}_{datetime.now(_APP_TZ).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}")


def _model_loop_output_dir(
    output_dir: Path,
    data_scientist_loop: dict[str, Any],
    data_scientist_summary: dict[str, Any],
) -> Path | None:
    candidates: list[Any] = []
    loop_files = data_scientist_loop.get("files") if isinstance(data_scientist_loop.get("files"), dict) else {}
    summary_model_loop = data_scientist_summary.get("model_loop") if isinstance(data_scientist_summary.get("model_loop"), dict) else {}
    candidates.extend(
        [
            data_scientist_loop.get("model_loop_dir"),
            loop_files.get("model_loop_dir"),
            data_scientist_summary.get("model_loop_dir"),
            summary_model_loop.get("model_loop_dir"),
            output_dir / "model_loop",
        ]
    )
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(str(candidate))
        if path.exists() and path.is_dir():
            return path
    return None


def _public_ai_ready_record(build_id: str, output_dir: Path) -> dict[str, Any]:
    input_profile = _read_json_if_exists(output_dir / "ai_ready_input_profile.json")
    input_locations = _read_json_if_exists(output_dir / "ai_ready_input_locations.json")
    agent_run_locations = _read_json_if_exists(output_dir / "agent_run_input_locations.json")
    agent_run_summary = _read_json_if_exists(output_dir / "agent_run_build_summary.json")
    mini_e2e_summary = _read_json_if_exists(output_dir / "mini_e2e_summary.json")
    mini_e2e_batch_summary = _read_json_if_exists(output_dir / "mini_e2e_batch_summary.json")
    real_smoke_summary = _read_json_if_exists(output_dir / "real_smoke_summary.json")
    repository_smoke_summary = _read_json_if_exists(output_dir / "repository_smoke_summary.json")
    repository_audit = _read_json_if_exists(output_dir / "repository_audit.json")
    iprox_index_summary = _read_json_if_exists(output_dir / "iprox_index_summary.json")
    agent_harness_summary = _read_json_if_exists(output_dir / "agent_harness_summary.json")
    dataset_recipe = _read_json_if_exists(output_dir / "dataset_recipe.json")
    leakage_risk_report = _read_json_if_exists(output_dir / "leakage_risk_report.json")
    coverage_gap_report = _read_json_if_exists(output_dir / "coverage_gap_report.json")
    hard_benchmark = _read_json_if_exists(output_dir / "hard_benchmark_manifest.json")
    counterfactual_benchmark = _read_json_if_exists(output_dir / "counterfactual_benchmark_manifest.json")
    curation_queue = _read_json_if_exists(output_dir / "curation_queue.json")
    curation_memory_update = _read_json_if_exists(output_dir / "curation_memory_update.json")
    data_scientist_summary = _read_json_if_exists(output_dir / "real_data_scientist_agent_summary.json")
    guidance_alignment = _read_json_if_exists(output_dir / "guidance_alignment_report.json")
    data_scientist_loop = _read_json_if_exists(output_dir / "data_scientist_agent_loop_summary.json")
    model_loop_dir = _model_loop_output_dir(output_dir, data_scientist_loop, data_scientist_summary)
    model_eval_summary = _read_json_if_exists(output_dir / "model_eval_summary.json")
    model_adapter_contract = _read_json_if_exists(output_dir / "model_adapter_contract.json")
    model_adapter_input = _read_json_if_exists(output_dir / "model_adapter_input_manifest.json")
    model_failure_modes = _read_json_if_exists(output_dir / "model_failure_modes.json")
    model_gap_report = _read_json_if_exists(output_dir / "model_informed_gap_report.json")
    model_discovery_requests = _read_json_if_exists(output_dir / "model_informed_discovery_requests.json")
    model_discovery_payloads = _read_json_if_exists(output_dir / "model_informed_discovery_payloads.json")
    model_discovery_payload_queue = _read_json_if_exists(output_dir / "model_informed_discovery_payload_queue.json")
    model_informed_curation_queue = _read_json_if_exists(output_dir / "model_informed_curation_queue.json")
    if model_loop_dir is not None:
        model_eval_summary = model_eval_summary or _read_json_if_exists(model_loop_dir / "model_eval_summary.json")
        model_adapter_contract = model_adapter_contract or _read_json_if_exists(model_loop_dir / "model_adapter_contract.json")
        model_adapter_input = model_adapter_input or _read_json_if_exists(model_loop_dir / "model_adapter_input_manifest.json")
        model_failure_modes = model_failure_modes or _read_json_if_exists(model_loop_dir / "model_failure_modes.json")
        model_gap_report = model_gap_report or _read_json_if_exists(model_loop_dir / "model_informed_gap_report.json")
        model_discovery_requests = model_discovery_requests or _read_json_if_exists(model_loop_dir / "model_informed_discovery_requests.json")
        model_discovery_payloads = model_discovery_payloads or _read_json_if_exists(model_loop_dir / "model_informed_discovery_payloads.json")
        model_discovery_payload_queue = model_discovery_payload_queue or _read_json_if_exists(model_loop_dir / "model_informed_discovery_payload_queue.json")
        model_informed_curation_queue = model_informed_curation_queue or _read_json_if_exists(model_loop_dir / "model_informed_curation_queue.json")
    model_repository_plan = build_model_informed_repository_plan(model_discovery_payloads, model_discovery_payload_queue)
    validation_report = _read_json_if_exists(output_dir / "ai_ready_validation_report.json")
    build_summary = _read_json_if_exists(output_dir / "ai_ready_build_summary.json")
    builder_summary = _read_json_if_exists(output_dir / "agentic_dataset_build_summary.json")
    downloads = {
        key: f"/api/ai-ready/{build_id}/download?file={key}"
        for key, (filename, _) in _AI_READY_DOWNLOAD_FILES.items()
        if (output_dir / filename).exists()
    }
    task_cards = []
    for item in input_profile.get("task_profiles") or []:
        if isinstance(item, dict):
            task_cards.append(
                {
                    "task_type": item.get("task_type"),
                    "status": item.get("input_status"),
                    "rows_out": None,
                    "warnings": item.get("warnings") or [],
                    "blockers": item.get("blockers") or [],
                    "target_schema": _task_schema(item.get("task_type")),
                }
            )
    for row in validation_report.get("rows") or []:
        if isinstance(row, dict):
            task_cards.append(
                {
                    "task_type": row.get("task_type"),
                    "status": row.get("status"),
                    "rows_out": row.get("rows_out"),
                    "warnings": row.get("warnings") or [],
                    "blockers": _validation_row_blockers(row),
                    "target_schema": _task_schema(row.get("task_type")),
                }
            )
    for item in real_smoke_summary.get("task_results") or []:
        if isinstance(item, dict):
            task_cards.append(
                {
                    "task_type": item.get("task_type"),
                    "status": item.get("status"),
                    "rows_out": item.get("rows_out"),
                    "warnings": item.get("warnings") or [],
                    "blockers": item.get("blockers") or [],
                    "target_schema": _task_schema(item.get("task_type")),
                }
            )
    for item in agent_run_summary.get("task_results") or []:
        if isinstance(item, dict):
            task_cards.append(
                {
                    "task_type": item.get("task_type"),
                    "status": item.get("status"),
                    "rows_out": item.get("rows_out"),
                    "warnings": item.get("warnings") or [],
                    "blockers": item.get("blockers") or [],
                    "target_schema": _task_schema(item.get("task_type")),
                }
            )
    for item in mini_e2e_summary.get("task_results") or []:
        if isinstance(item, dict):
            task_cards.append(
                {
                    "task_type": item.get("task_type"),
                    "status": item.get("status"),
                    "rows_out": item.get("rows_out"),
                    "warnings": item.get("warnings") or [],
                    "blockers": item.get("blockers") or [],
                    "target_schema": _task_schema(item.get("task_type")),
                }
            )
    for run in mini_e2e_batch_summary.get("run_results") or []:
        if not isinstance(run, dict):
            continue
        run_name = Path(str(run.get("agent_run_dir") or "")).name or "batch run"
        task_statuses = run.get("task_statuses") or {}
        rows_out = run.get("rows_out") or {}
        for task_type, status in task_statuses.items():
            task_cards.append(
                {
                    "task_type": task_type,
                    "status": status,
                    "rows_out": rows_out.get(task_type),
                    "warnings": run.get("warnings") or [],
                    "blockers": [f"run: {run_name}", *list(run.get("blockers") or [])],
                    "target_schema": _task_schema(task_type),
                }
            )
    if repository_smoke_summary:
        task_cards.append(
            {
                "task_type": f"repository:{repository_smoke_summary.get('repository') or repository_smoke_summary.get('requested_repository') or 'unknown'}",
                "status": repository_smoke_summary.get("status") or "unknown",
                "rows_out": None,
                "warnings": repository_smoke_summary.get("warnings") or [],
                "blockers": repository_smoke_summary.get("blockers") or [],
                "target_schema": repository_smoke_summary.get("download_url") or repository_smoke_summary.get("next_step") or "",
            }
        )
    if repository_audit:
        audit_rows = repository_audit.get("rows") if isinstance(repository_audit.get("rows"), list) else []
        blockers = [
            f"{row.get('repository')}:{row.get('blocker')}"
            for row in audit_rows
            if isinstance(row, dict) and row.get("blocker")
        ]
        attempted = repository_audit.get("repositories_attempted") or [
            row.get("repository") for row in audit_rows if isinstance(row, dict) and row.get("repository")
        ]
        task_cards.append(
            {
                "task_type": "repository_audit",
                "status": repository_audit.get("status") or ("partial" if blockers else "available"),
                "rows_out": len(audit_rows),
                "warnings": [f"attempted:{','.join(map(str, attempted))}"] if attempted else [],
                "blockers": blockers,
                "target_schema": repository_audit.get("source") or "",
            }
        )
    if iprox_index_summary:
        task_cards.append(
            {
                "task_type": "iprox_index",
                "status": iprox_index_summary.get("status") or "available",
                "rows_out": iprox_index_summary.get("file_count"),
                "warnings": [f"projects:{iprox_index_summary.get('project_count') or 0}"],
                "blockers": [
                    str(item.get("error") or item)
                    for item in iprox_index_summary.get("failures") or []
                    if item
                ],
                "target_schema": iprox_index_summary.get("next_step") or "",
            }
        )
    for case in agent_harness_summary.get("case_results") or []:
        if isinstance(case, dict):
            inferred = case.get("inferred") if isinstance(case.get("inferred"), dict) else {}
            task_cards.append(
                {
                    "task_type": f"harness:{case.get('id') or 'case'}",
                    "status": case.get("status") or "unknown",
                    "rows_out": None,
                    "warnings": case.get("warnings") or [],
                    "blockers": case.get("blockers") or [],
                    "target_schema": inferred.get("next_action_category") or "",
                }
            )
    if dataset_recipe:
        recipe_blockers = []
        if leakage_risk_report.get("status") in {"fail", "warn"}:
            recipe_blockers.append(f"leakage:{leakage_risk_report.get('status')}")
        gap_count = len(coverage_gap_report.get("gaps") or [])
        if gap_count:
            recipe_blockers.append(f"gaps:{gap_count}")
        hard_count = int((hard_benchmark.get("row_count") if isinstance(hard_benchmark.get("row_count"), int) else None) or len(hard_benchmark.get("rows") or []))
        counterfactual_count = int((counterfactual_benchmark.get("row_count") if isinstance(counterfactual_benchmark.get("row_count"), int) else None) or len(counterfactual_benchmark.get("rows") or []))
        curation_count = int((curation_queue.get("row_count") if isinstance(curation_queue.get("row_count"), int) else None) or len(curation_queue.get("rows") or []))
        recipe_warnings = list(leakage_risk_report.get("warnings") or [])
        if hard_count:
            recipe_warnings.append(f"hard_benchmark:{hard_count}")
        if counterfactual_count:
            recipe_warnings.append(f"counterfactual_benchmark:{counterfactual_count}")
        if curation_count:
            recipe_warnings.append(f"curation_queue:{curation_count}")
        task_cards.append(
            {
                "task_type": "dataset_recipe",
                "status": dataset_recipe.get("status") or "available",
                "rows_out": len(dataset_recipe.get("selected_files") or []),
                "warnings": recipe_warnings,
                "blockers": recipe_blockers,
                "target_schema": dataset_recipe.get("split_strategy_resolved") or dataset_recipe.get("split_policy") or dataset_recipe.get("split_level") or "",
            }
        )
    if model_eval_summary:
        failure_count = len(model_failure_modes.get("failure_modes") or [])
        gap_count = len(model_gap_report.get("gaps") or [])
        model_blockers = list((model_eval_summary.get("validation") or {}).get("blockers") or [])
        model_warnings = list((model_eval_summary.get("validation") or {}).get("warnings") or [])
        for warning in model_eval_summary.get("adapter_contract_warnings") or []:
            model_warnings.append(f"adapter_contract:{warning}")
        if not model_adapter_contract:
            model_warnings.append("model_adapter_contract_missing")
        if failure_count:
            model_warnings.append(f"failure_modes:{failure_count}")
        if gap_count:
            model_warnings.append(f"model_gaps:{gap_count}")
        discovery_request_count = int(model_discovery_requests.get("request_count") or len(model_discovery_requests.get("requests") or []))
        if discovery_request_count:
            model_warnings.append(f"discovery_requests:{discovery_request_count}")
        planned_repositories = model_repository_plan.get("planned_repositories") or []
        if planned_repositories:
            model_warnings.append(f"planned_repositories:{','.join(map(str, planned_repositories))}")
        model_rows_out = (model_eval_summary.get("metrics") or {}).get("total_rows")
        if model_rows_out is None:
            model_rows_out = ((model_adapter_input.get("summary") or {}).get("total_rows_out") if model_adapter_input else None)
        task_cards.append(
            {
                "task_type": f"model_loop:{model_eval_summary.get('task_type') or 'task'}",
                "status": model_eval_summary.get("status") or "available",
                "rows_out": model_rows_out,
                "warnings": model_warnings,
                "blockers": model_blockers,
                "target_schema": model_repository_plan.get("repository_strategy")
                or f"{model_eval_summary.get('adapter') or ''}:{model_eval_summary.get('metric_status') or ''}",
            }
        )
    if data_scientist_summary:
        ds_warnings = list(data_scientist_summary.get("warnings") or [])
        leakage_status = ((data_scientist_summary.get("leakage") or {}).get("status") or "not_evaluated")
        if leakage_status in {"fail", "warn"}:
            ds_warnings.append(f"leakage:{leakage_status}")
        task_cards.append(
            {
                "task_type": "data_scientist_agent_report",
                "status": data_scientist_summary.get("status") or "available",
                "rows_out": data_scientist_summary.get("selected_count"),
                "warnings": ds_warnings,
                "blockers": [],
                "target_schema": f"model_loop:{(data_scientist_summary.get('model_loop') or {}).get('status') or 'not_available'}",
            }
        )
    if guidance_alignment:
        summary = guidance_alignment.get("summary") if isinstance(guidance_alignment.get("summary"), dict) else {}
        task_cards.append(
            {
                "task_type": "guidance_alignment",
                "status": guidance_alignment.get("status") or "available",
                "rows_out": summary.get("achieved"),
                "warnings": [f"partial:{summary.get('partial', 0)}", f"missing:{summary.get('missing', 0)}"],
                "blockers": [],
                "target_schema": "260617_to_do_alignment",
            }
        )
    if data_scientist_loop:
        task_cards.append(
            {
                "task_type": "data_scientist_agent_loop",
                "status": data_scientist_loop.get("status") or "available",
                "rows_out": data_scientist_loop.get("selected_count"),
                "warnings": data_scientist_loop.get("warnings") or [],
                "blockers": data_scientist_loop.get("blockers") or [],
                "target_schema": f"guidance:{data_scientist_loop.get('guidance_alignment_status') or 'not_available'}",
            }
        )
    if curation_memory_update:
        task_cards.append(
            {
                "task_type": "active_curation_memory",
                "status": curation_memory_update.get("status") or "available",
                "rows_out": curation_memory_update.get("imported_decision_count") or 0,
                "warnings": [f"skipped:{curation_memory_update.get('skipped_count') or 0}"],
                "blockers": [],
                "target_schema": "discovery_memory_review_decisions",
            }
        )
    recovery_cards = _ai_ready_recovery_cards(mini_e2e_summary, mini_e2e_batch_summary)
    return {
        "status": curation_memory_update.get("status") or data_scientist_loop.get("status") or guidance_alignment.get("status") or data_scientist_summary.get("status") or model_eval_summary.get("status") or dataset_recipe.get("status") or agent_harness_summary.get("status") or iprox_index_summary.get("status") or repository_smoke_summary.get("status") or mini_e2e_batch_summary.get("status") or mini_e2e_summary.get("status") or agent_run_summary.get("status") or real_smoke_summary.get("status") or build_summary.get("status") or validation_report.get("status") or input_profile.get("status") or agent_run_locations.get("status") or input_locations.get("status") or "available",
        "build_id": build_id,
        "output_dir": str(output_dir),
        "input_locations": input_locations,
        "agent_run_input_locations": agent_run_locations,
        "agent_run_build_summary": agent_run_summary,
        "mini_e2e_summary": mini_e2e_summary,
        "mini_e2e_batch_summary": mini_e2e_batch_summary,
        "input_profile": input_profile,
        "real_smoke_summary": real_smoke_summary,
        "repository_smoke_summary": repository_smoke_summary,
        "repository_audit": repository_audit,
        "iprox_index_summary": iprox_index_summary,
        "agent_harness_summary": agent_harness_summary,
        "dataset_recipe": dataset_recipe,
        "leakage_risk_report": leakage_risk_report,
        "coverage_gap_report": coverage_gap_report,
        "hard_benchmark": hard_benchmark,
        "counterfactual_benchmark": counterfactual_benchmark,
        "curation_queue": curation_queue,
        "curation_memory_update": curation_memory_update,
        "model_eval_summary": model_eval_summary,
        "model_adapter_contract": model_adapter_contract,
        "model_adapter_input_manifest": model_adapter_input,
        "model_failure_modes": model_failure_modes,
        "model_informed_gap_report": model_gap_report,
        "model_informed_discovery_requests": model_discovery_requests,
        "model_informed_discovery_payloads": model_discovery_payloads,
        "model_informed_discovery_payload_queue": model_discovery_payload_queue,
        "model_informed_repository_plan": model_repository_plan,
        "model_informed_curation_queue": model_informed_curation_queue,
        "real_data_scientist_agent_summary": data_scientist_summary,
        "guidance_alignment_report": guidance_alignment,
        "data_scientist_agent_loop_summary": data_scientist_loop,
        "validation_report": validation_report,
        "build_summary": build_summary,
        "builder_summary": builder_summary,
        "task_cards": task_cards,
        "recovery_cards": recovery_cards,
        "downloads": downloads,
    }


def _ai_ready_recovery_cards(mini_e2e_summary: dict[str, Any], mini_e2e_batch_summary: dict[str, Any]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    if mini_e2e_summary:
        upstream_issue = mini_e2e_summary.get("upstream_primary_issue")
        upstream_status = mini_e2e_summary.get("upstream_recovery_status")
        if upstream_issue or upstream_status:
            cards.append(
                {
                    "scope": "upstream_full",
                    "label": "Upstream full",
                    "status": upstream_status or "not_run",
                    "workflow_outcome": mini_e2e_summary.get("upstream_workflow_outcome") or "unknown",
                    "usable_partial_outputs": bool(mini_e2e_summary.get("upstream_usable_partial_outputs")),
                    "primary_issue": upstream_issue or "none",
                    "recommended_next_step": mini_e2e_summary.get("upstream_recommended_next_step") or "",
                }
            )
        task_issue = mini_e2e_summary.get("primary_issue")
        task_status = mini_e2e_summary.get("recovery_status")
        if task_issue or task_status:
            cards.append(
                {
                    "scope": "task_build",
                    "label": "Task build",
                    "status": task_status or "not_run",
                    "ai_ready_outcome": mini_e2e_summary.get("ai_ready_outcome") or "unknown",
                    "usable_partial_outputs": bool(mini_e2e_summary.get("usable_partial_outputs")),
                    "primary_issue": task_issue or "none",
                    "recommended_next_step": mini_e2e_summary.get("recommended_next_step") or "",
                }
            )
    for run in mini_e2e_batch_summary.get("run_results") or []:
        if not isinstance(run, dict):
            continue
        run_name = Path(str(run.get("agent_run_dir") or "")).name or "batch run"
        upstream_issue = run.get("upstream_primary_issue")
        upstream_status = run.get("upstream_recovery_status")
        if upstream_issue or upstream_status:
            cards.append(
                {
                    "scope": "upstream_full",
                    "label": f"{run_name} upstream",
                    "status": upstream_status or "not_run",
                    "workflow_outcome": run.get("upstream_workflow_outcome") or "unknown",
                    "usable_partial_outputs": bool(run.get("upstream_usable_partial_outputs")),
                    "primary_issue": upstream_issue or "none",
                    "recommended_next_step": run.get("upstream_recommended_next_step") or "",
                }
            )
    return cards


def _run_ai_ready_input_profile(body: dict[str, Any]) -> dict[str, Any]:
    build_id = safe_output_stem(_clean_text(body.get("build_id")) or _new_ai_ready_build_id("profile"))
    output_dir = _safe_ai_ready_dir(build_id)
    if output_dir is None:
        raise ValueError("Invalid build_id.")
    search_results = _clean_path_list(body.get("search_results") or body.get("search_result"))
    peaklists = _clean_path_list(body.get("peaklists") or body.get("peaklist"))
    search_dir = _clean_text(body.get("search_dir"))
    if search_dir and not search_results:
        locator = locate_ai_ready_inputs(search_dir=search_dir, output_dir=output_dir)
        located_search_results, located_peaklists = select_ai_ready_inputs(
            locator,
            task_type=(_clean_ai_ready_task_types(body) or ["rt_prediction"])[0],
        )
        search_results = [str(path) for path in located_search_results]
        if not peaklists:
            peaklists = [str(path) for path in located_peaklists]
    result = profile_ai_ready_inputs(
        search_results=search_results,
        peaklists=peaklists,
        task_types=_clean_ai_ready_task_types(body),
        output_dir=output_dir,
    )
    record = _public_ai_ready_record(build_id, output_dir)
    record["profile_rows_in"] = result.rows_in
    return record


def _run_ai_ready_input_locator(body: dict[str, Any]) -> dict[str, Any]:
    build_id = safe_output_stem(_clean_text(body.get("build_id")) or _new_ai_ready_build_id("locator"))
    output_dir = _safe_ai_ready_dir(build_id)
    if output_dir is None:
        raise ValueError("Invalid build_id.")
    search_dir = _clean_text(body.get("search_dir"))
    if not search_dir:
        raise ValueError("Search output directory is required.")
    locate_ai_ready_inputs(search_dir=search_dir, output_dir=output_dir)
    return _public_ai_ready_record(build_id, output_dir)


def _run_agent_run_input_locator(body: dict[str, Any]) -> dict[str, Any]:
    build_id = safe_output_stem(_clean_text(body.get("build_id")) or _new_ai_ready_build_id("agent_run_locator"))
    output_dir = _safe_ai_ready_dir(build_id)
    if output_dir is None:
        raise ValueError("Invalid build_id.")
    agent_run_dir = _clean_text(body.get("agent_run_dir"))
    if not agent_run_dir:
        raise ValueError("Original agent run directory is required.")
    locate_agent_run_inputs(
        agent_run_dir=agent_run_dir,
        output_dir=output_dir,
        max_input_file_mb=int(body.get("max_input_file_mb") or 2048),
        allow_large_input=bool(body.get("allow_large_input")),
    )
    return _public_ai_ready_record(build_id, output_dir)


def _run_agent_run_ai_ready_build(body: dict[str, Any]) -> dict[str, Any]:
    build_id = safe_output_stem(_clean_text(body.get("build_id")) or _new_ai_ready_build_id("agent_run_build"))
    output_dir = _safe_ai_ready_dir(build_id)
    if output_dir is None:
        raise ValueError("Invalid build_id.")
    agent_run_dir = _clean_text(body.get("agent_run_dir"))
    if not agent_run_dir:
        raise ValueError("Original agent run directory is required.")
    build_ai_ready_from_agent_run(
        agent_run_dir=agent_run_dir,
        task_types=_clean_ai_ready_task_types(body),
        output_dir=output_dir,
        max_input_file_mb=int(body.get("max_input_file_mb") or 2048),
        allow_large_input=bool(body.get("allow_large_input")),
        peaklists=_clean_path_list(body.get("peaklists") or body.get("peaklist")),
    )
    return _public_ai_ready_record(build_id, output_dir)


def _run_agent_run_mini_e2e(body: dict[str, Any]) -> dict[str, Any]:
    build_id = safe_output_stem(_clean_text(body.get("build_id")) or _new_ai_ready_build_id("mini_e2e"))
    output_dir = _safe_ai_ready_dir(build_id)
    if output_dir is None:
        raise ValueError("Invalid build_id.")
    agent_run_dir = _clean_text(body.get("agent_run_dir"))
    if not agent_run_dir:
        raise ValueError("Original agent run directory is required.")
    validate_agent_run_ai_ready_mini(
        agent_run_dir=agent_run_dir,
        task_types=_clean_ai_ready_task_types(body),
        output_dir=output_dir,
        max_input_file_mb=int(body.get("max_input_file_mb") or 2048),
        allow_large_input=bool(body.get("allow_large_input")),
        peaklists=_clean_path_list(body.get("peaklists") or body.get("peaklist")),
    )
    return _public_ai_ready_record(build_id, output_dir)


def _run_agent_runs_mini_e2e_batch(body: dict[str, Any]) -> dict[str, Any]:
    build_id = safe_output_stem(_clean_text(body.get("build_id")) or _new_ai_ready_build_id("mini_e2e_batch"))
    output_dir = _safe_ai_ready_dir(build_id)
    if output_dir is None:
        raise ValueError("Invalid build_id.")
    agent_run_dirs = _clean_path_list(body.get("agent_run_dirs") or body.get("agent_run_dir"))
    if not agent_run_dirs:
        raise ValueError("At least one original agent run directory is required.")
    validate_agent_runs_ai_ready_batch(
        agent_run_dirs=agent_run_dirs,
        task_types=_clean_ai_ready_task_types(body),
        output_dir=output_dir,
        max_input_file_mb=int(body.get("max_input_file_mb") or 2048),
        allow_large_input=bool(body.get("allow_large_input")),
        peaklists=_clean_path_list(body.get("peaklists") or body.get("peaklist")),
        auto_recover=bool(body.get("auto_recover", True)),
    )
    return _public_ai_ready_record(build_id, output_dir)


def _run_ai_ready_real_smoke(body: dict[str, Any]) -> dict[str, Any]:
    build_id = safe_output_stem(_clean_text(body.get("build_id")) or _new_ai_ready_build_id("real_smoke"))
    output_dir = _safe_ai_ready_dir(build_id)
    if output_dir is None:
        raise ValueError("Invalid build_id.")
    search_dir = _clean_text(body.get("search_dir"))
    if not search_dir:
        raise ValueError("Search output directory is required.")
    run_ai_ready_real_smoke(
        search_dir=search_dir,
        task_types=_clean_ai_ready_task_types(body),
        output_dir=output_dir,
    )
    return _public_ai_ready_record(build_id, output_dir)


def _run_repository_smoke_web(body: dict[str, Any]) -> dict[str, Any]:
    build_id = safe_output_stem(_clean_text(body.get("build_id")) or _new_ai_ready_build_id("repository_smoke"))
    output_dir = _safe_ai_ready_dir(build_id)
    if output_dir is None:
        raise ValueError("Invalid build_id.")
    repository = _clean_text(body.get("repository")) or "auto"
    input_value = _clean_text(body.get("input_value") or body.get("accession") or body.get("file"))
    if not input_value:
        raise ValueError("Repository accession or file path is required.")
    mode = _clean_text(body.get("mode")) or "parameters"
    if mode not in {"parameters", "prepare", "full"}:
        raise ValueError("Mode must be parameters, prepare, or full.")
    iprox_index_dir = _clean_text(body.get("iprox_index_dir")) or None
    registry = None
    if iprox_index_dir:
        registry = RepositoryRegistry(adapters=[PrideAdapter(), MassiveAdapter(), IproxAdapter(index_path=iprox_index_dir)])
    run_repository_smoke(
        repository=repository,
        input_value=input_value,
        mode=mode,  # type: ignore[arg-type]
        output_dir=output_dir,
        registry=registry,
    )
    return _public_ai_ready_record(build_id, output_dir)


def _split_clean_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        raw_items = value
    else:
        raw_items = re.split(r"[\r\n,;]+", str(value))
    return [text for item in raw_items if (text := _clean_text(item))]


def _run_iprox_index_refresh_web(body: dict[str, Any]) -> dict[str, Any]:
    build_id = safe_output_stem(_clean_text(body.get("build_id")) or _new_ai_ready_build_id("iprox_index"))
    output_dir = _safe_ai_ready_dir(build_id)
    if output_dir is None:
        raise ValueError("Invalid build_id.")
    years: list[int] = []
    for item in _split_clean_text(body.get("years")):
        if not str(item).isdigit():
            raise ValueError(f"Invalid iProX year: {item}")
        years.append(int(item))
    projects = _split_clean_text(body.get("projects") or body.get("project_ids") or body.get("project"))
    max_projects_raw = _clean_text(body.get("max_projects")) or None
    max_projects = int(max_projects_raw) if max_projects_raw and str(max_projects_raw).isdigit() else None
    if not years and not projects:
        raise ValueError("At least one iProX year or project accession is required.")
    refresh_public_iprox_index(
        years=years,
        project_ids=projects,
        output_dir=output_dir,
        max_projects=max_projects,
    )
    return _public_ai_ready_record(build_id, output_dir)


def _run_agent_harness_web(body: dict[str, Any]) -> dict[str, Any]:
    build_id = safe_output_stem(_clean_text(body.get("build_id")) or _new_ai_ready_build_id("agent_harness"))
    output_dir = _safe_ai_ready_dir(build_id)
    if output_dir is None:
        raise ValueError("Invalid build_id.")
    case_file = _clean_text(body.get("case_file")) or "tests/fixtures/agent_harness_cases.json"
    use_llm = bool(body.get("use_llm", True))
    run_agent_harness(
        case_file=case_file,
        output_dir=output_dir,
        use_llm=use_llm,
    )
    return _public_ai_ready_record(build_id, output_dir)


def _run_dataset_recipe_web(body: dict[str, Any]) -> dict[str, Any]:
    build_id = safe_output_stem(_clean_text(body.get("build_id")) or _new_ai_ready_build_id("dataset_recipe"))
    output_dir = _safe_ai_ready_dir(build_id)
    if output_dir is None:
        raise ValueError("Invalid build_id.")
    batch_dir = _clean_text(body.get("batch_dir"))
    if not batch_dir:
        raise ValueError("Mini E2E batch directory is required.")
    discovery_manifest = _clean_text(body.get("discovery_manifest"))
    repository_audit = _clean_text(body.get("repository_audit"))
    split_strategy = _clean_text(body.get("split_strategy")) or "auto"
    make_dataset_recipe(
        batch_dir=batch_dir,
        output_dir=output_dir,
        discovery_manifest=Path(discovery_manifest) if discovery_manifest else None,
        repository_audit=Path(repository_audit) if repository_audit else None,
        split_strategy=split_strategy,
    )
    return _public_ai_ready_record(build_id, output_dir)


def _path_list_from_body(value: Any) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        raw_items = value
    else:
        raw_items = re.split(r"[\r\n;]+", str(value))
    paths: list[Path] = []
    for item in raw_items:
        text = _clean_text(item)
        if text:
            paths.append(Path(text))
    return paths


def _run_apply_curation_decisions_web(body: dict[str, Any]) -> dict[str, Any]:
    build_id = safe_output_stem(_clean_text(body.get("build_id")) or _new_ai_ready_build_id("curation_memory"))
    output_dir = _safe_ai_ready_dir(build_id)
    if output_dir is None:
        raise ValueError("Invalid build_id.")
    curation_queue = _clean_text(body.get("curation_queue"))
    if not curation_queue:
        recipe_dir = _clean_text(body.get("recipe_dir"))
        if recipe_dir:
            curation_queue = str(Path(recipe_dir) / "curation_queue.json")
    if not curation_queue:
        raise ValueError("curation_queue or recipe_dir is required.")
    decisions_csv = _clean_text(body.get("decisions_csv")) or None
    default_decision = _clean_text(body.get("default_decision")) or None
    memory_dir = _clean_text(body.get("memory_dir")) or str(_discovery_memory_dir())
    run_id = _clean_text(body.get("run_id")) or "active_curation"
    apply_curation_decisions_to_memory(
        curation_queue=curation_queue,
        output_dir=output_dir,
        memory_dir=memory_dir,
        decisions_csv=Path(decisions_csv) if decisions_csv else None,
        default_decision=default_decision,
        run_id=run_id,
    )
    return _public_ai_ready_record(build_id, output_dir)


def _run_dataset_model_loop_web(body: dict[str, Any]) -> dict[str, Any]:
    build_id = safe_output_stem(_clean_text(body.get("build_id")) or _new_ai_ready_build_id("model_loop"))
    output_dir = _safe_ai_ready_dir(build_id)
    if output_dir is None:
        raise ValueError("Invalid build_id.")
    recipe_dir = _clean_text(body.get("recipe_dir"))
    if not recipe_dir:
        raise ValueError("Recipe directory is required.")
    task_type = normalize_task_type(_clean_text(body.get("task_type")) or "rt_prediction")
    if task_type is None:
        raise ValueError("Task type is required.")
    mode = _clean_text(body.get("mode")) or "smoke"
    adapter = _clean_text(body.get("adapter")) or "dry_run"
    adapter_command = _clean_text(body.get("adapter_command")) or None
    metrics_file = _clean_text(body.get("metrics_file")) or None
    run_dataset_model_loop(
        recipe_dir=recipe_dir,
        task_type=task_type,
        output_dir=output_dir,
        mode=mode,  # type: ignore[arg-type]
        adapter=adapter,
        adapter_command=adapter_command,
        metrics_file=Path(metrics_file) if metrics_file else None,
    )
    return _public_ai_ready_record(build_id, output_dir)


def _run_data_scientist_agent_report_web(body: dict[str, Any]) -> dict[str, Any]:
    build_id = safe_output_stem(_clean_text(body.get("build_id")) or _new_ai_ready_build_id("data_scientist_report"))
    output_dir = _safe_ai_ready_dir(build_id)
    if output_dir is None:
        raise ValueError("Invalid build_id.")
    recipe_dir = _clean_text(body.get("recipe_dir"))
    if not recipe_dir:
        raise ValueError("Recipe directory is required.")
    model_loop_dir = _clean_text(body.get("model_loop_dir")) or None
    benchmark_dir = _clean_text(body.get("benchmark_dir")) or None
    discovery_manifest = _clean_text(body.get("discovery_manifest")) or None
    guidance_alignment_dir = _clean_text(body.get("guidance_alignment_dir")) or None
    make_data_scientist_agent_report(
        recipe_dir=recipe_dir,
        output_dir=output_dir,
        model_loop_dir=Path(model_loop_dir) if model_loop_dir else None,
        benchmark_dir=Path(benchmark_dir) if benchmark_dir else None,
        discovery_manifest=Path(discovery_manifest) if discovery_manifest else None,
        guidance_alignment_dir=Path(guidance_alignment_dir) if guidance_alignment_dir else None,
    )
    return _public_ai_ready_record(build_id, output_dir)


def _run_guidance_alignment_report_web(body: dict[str, Any]) -> dict[str, Any]:
    build_id = safe_output_stem(_clean_text(body.get("build_id")) or _new_ai_ready_build_id("guidance_alignment"))
    output_dir = _safe_ai_ready_dir(build_id)
    if output_dir is None:
        raise ValueError("Invalid build_id.")
    recipe_dir = _clean_text(body.get("recipe_dir")) or None
    discovery_dir = _clean_text(body.get("discovery_dir")) or None
    discovery_manifest = _clean_text(body.get("discovery_manifest")) or None
    model_loop_dir = _clean_text(body.get("model_loop_dir")) or None
    benchmark_dir = _clean_text(body.get("benchmark_dir")) or None
    make_guidance_alignment_report(
        output_dir=output_dir,
        recipe_dir=Path(recipe_dir) if recipe_dir else None,
        discovery_dir=Path(discovery_dir) if discovery_dir else None,
        discovery_manifest=Path(discovery_manifest) if discovery_manifest else None,
        model_loop_dir=Path(model_loop_dir) if model_loop_dir else None,
        benchmark_dir=Path(benchmark_dir) if benchmark_dir else None,
    )
    return _public_ai_ready_record(build_id, output_dir)


def _run_data_scientist_agent_loop_web(body: dict[str, Any]) -> dict[str, Any]:
    build_id = safe_output_stem(_clean_text(body.get("build_id")) or _new_ai_ready_build_id("data_scientist_loop"))
    output_dir = _safe_ai_ready_dir(build_id)
    if output_dir is None:
        raise ValueError("Invalid build_id.")
    batch_dir = _clean_text(body.get("batch_dir"))
    if not batch_dir:
        raise ValueError("Batch/benchmark directory is required.")
    discovery_manifest = _clean_text(body.get("discovery_manifest")) or None
    task_type = _clean_text(body.get("task_type")) or "auto"
    split_strategy = _clean_text(body.get("split_strategy")) or "auto"
    adapter_command = _clean_text(body.get("adapter_command")) or None
    metrics_file = _clean_text(body.get("metrics_file")) or None
    strategy_case = _clean_text(body.get("strategy_comparison_case_file")) or None
    curation_decisions_csv = _clean_text(body.get("curation_decisions_csv")) or None
    curation_default_decision = _clean_text(body.get("curation_default_decision")) or None
    curation_memory_dir = _clean_text(body.get("curation_memory_dir")) or None
    repository_smoke_dirs = _path_list_from_body(body.get("repository_smoke_dirs"))
    run_data_scientist_agent_loop(
        batch_dir=Path(batch_dir),
        output_dir=output_dir,
        task_type=task_type,
        discovery_manifest=Path(discovery_manifest) if discovery_manifest else None,
        split_strategy=split_strategy,
        mode="smoke",
        adapter="external_command" if adapter_command else "dry_run",
        adapter_command=adapter_command,
        metrics_file=Path(metrics_file) if metrics_file else None,
        strategy_comparison_case_file=Path(strategy_case) if strategy_case else None,
        curation_decisions_csv=Path(curation_decisions_csv) if curation_decisions_csv else None,
        curation_default_decision=curation_default_decision,
        curation_memory_dir=Path(curation_memory_dir) if curation_memory_dir else None,
        repository_smoke_dirs=repository_smoke_dirs or None,
    )
    return _public_ai_ready_record(build_id, output_dir)


def _run_ai_ready_validation(body: dict[str, Any]) -> dict[str, Any]:
    build_id = safe_output_stem(_clean_text(body.get("build_id")) or _new_ai_ready_build_id("validate"))
    requested_dir = _clean_text(body.get("build_dir"))
    if requested_dir:
        output_dir = Path(requested_dir)
        build_id = safe_output_stem(_clean_text(body.get("build_id")) or output_dir.name or build_id)
    else:
        output_dir = _safe_ai_ready_dir(build_id)
        if output_dir is None:
            raise ValueError("Invalid build_id.")
    task_type = normalize_task_type(_clean_text(body.get("task_type")) or "rt_prediction")
    if task_type is None:
        raise ValueError("Task type is required.")
    validate_ai_ready_build(output_dir, task_type)
    return _public_ai_ready_record(build_id, output_dir)


def _task_schema(task_type: Any) -> str:
    try:
        return get_task_profile(str(task_type or "")).ai_ready_target_schema
    except Exception:
        return ""


def _validation_row_blockers(row: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if row.get("status") in {"export_missing", "export_empty", "planned_not_exported"}:
        blockers.append(str(row.get("status")))
    try:
        rows_out = int(row.get("rows_out") or 0)
    except (TypeError, ValueError):
        rows_out = 0
    if rows_out > 0:
        return blockers
    filter_counts = row.get("filter_counts") if isinstance(row.get("filter_counts"), dict) else {}
    if filter_counts.get("spectrum_not_matched"):
        blockers.append("spectrum_not_matched")
    if filter_counts.get("no_multi_peptide_assignment"):
        blockers.append("no_multi_peptide_assignment")
    missing = row.get("missing_required_column_counts")
    if isinstance(missing, dict):
        blockers.extend(f"missing_column:{key}" for key in sorted(missing))
    return blockers


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _now_app_iso() -> str:
    return datetime.now(_APP_TZ).isoformat()


def _slim_discovery_record(record: Any) -> Any:
    """Drop heavy project/file arrays from job polling payloads (ISS-04)."""
    if not isinstance(record, dict):
        return record
    slim = dict(record)
    projects = slim.pop("projects", None)
    files = slim.pop("files", None)
    if "project_count" not in slim:
        slim["project_count"] = len(projects) if isinstance(projects, list) else 0
    if "file_count" not in slim:
        slim["file_count"] = len(files) if isinstance(files, list) else 0
    slim["detail"] = "summary"
    return slim


def _bounded_discovery_log_value(
    value: Any,
    *,
    field: str = "",
    depth: int = 0,
) -> Any:
    """Keep progress semantics while bounding repeated metadata/file evidence."""
    if depth >= 7:
        return "[nested detail omitted]"
    if isinstance(value, str):
        text = _redact_secrets(value)
        if len(text) <= _DISCOVERY_LOG_STRING_LIMIT:
            return text
        return f"{text[:_DISCOVERY_LOG_STRING_LIMIT]}…"
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 60:
                result["_truncated_fields"] = len(value) - 60
                break
            name = str(key)
            result[name] = _bounded_discovery_log_value(
                item,
                field=name,
                depth=depth + 1,
            )
        return result
    if isinstance(value, (list, tuple)):
        limits = {
            "accessions": 100,
            "queries": 40,
            "query_yields": 40,
            "previews": 25,
            "project_assessments": 40,
            "inspection_outcomes": 40,
            "judgments": 40,
            "selected_file_examples": 8,
            "evidence_refs": 20,
            "available_evidence_refs": 20,
            "missing_information": 20,
            "limitations": 20,
        }
        limit = limits.get(field, 30)
        compact = [
            _bounded_discovery_log_value(item, field=field, depth=depth + 1)
            for item in value[:limit]
        ]
        if len(value) > limit:
            compact.append({"_truncated_items": len(value) - limit})
        return compact
    return _json_safe(value)


def _compact_discovery_log_entry(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: _bounded_discovery_log_value(value, field=str(key))
        for key, value in raw.items()
        if str(key).casefold() not in {"api_key", "authorization", "sdk_state_json"}
    }


def _discovery_summary_logs(logs: list[Any]) -> list[dict[str, Any]]:
    entries = [item for item in logs if isinstance(item, Mapping)]
    recent = entries[-_DISCOVERY_SUMMARY_RECENT_LOGS:]
    recent_ids = {id(item) for item in recent}

    def structural(item: Mapping[str, Any]) -> bool:
        event_type = str(item.get("type") or "")
        return (
            event_type.startswith("confirmed_theme_pipeline_")
            or event_type.startswith("repository_term_")
            or event_type in {
                "candidate_search_started",
                "candidate_search_completed",
                "candidate_search_failed",
                "verified_project_batch_published",
            }
        )

    selected = [
        item
        for item in entries
        if id(item) in recent_ids or structural(item)
    ]
    return [_compact_discovery_log_entry(item) for item in selected]


def _rebuild_discovery_result_batches(job: Mapping[str, Any]) -> list[dict[str, Any]]:
    body = job.get("body") if isinstance(job.get("body"), Mapping) else {}
    record = job.get("record") if isinstance(job.get("record"), Mapping) else {}
    discovery_id = _clean_text(
        record.get("discovery_id")
        or body.get("_execution_discovery_id")
        or record.get("run_id")
    )
    run_dir = _safe_discovery_dir(discovery_id)
    if run_dir is None:
        return []
    batches_root = run_dir / "verified_batches"
    if not batches_root.is_dir():
        return []
    rebuilt: list[dict[str, Any]] = []
    cumulative_projects: set[str] = set()
    cumulative_files = 0
    for batch_dir in sorted(batches_root.glob("batch_[0-9][0-9][0-9]")):
        manifest_path = batch_dir / "dataset_manifest.json"
        if not manifest_path.is_file() or not _path_within(
            manifest_path.resolve(),
            _discovery_root_dir().resolve(),
        ):
            continue
        try:
            manifest = load_dataset_manifest(manifest_path)
            batch_index = int(batch_dir.name.rsplit("_", 1)[-1])
        except Exception:
            continue
        project_accessions = sorted(
            {
                _clean_text(project.project_accession).upper()
                for project in manifest.projects
                if _clean_text(project.project_accession)
            }
        )
        file_identifiers = []
        for file in manifest.files:
            native = _clean_text(file.file_accession_or_path or file.file_name)
            accession = _clean_text(file.project_accession).upper()
            repository = _clean_repository(file.repository or "pride")
            if native:
                file_identifiers.append(
                    f"{repository.casefold()}:{accession}:{native}"
                )
        cumulative_projects.update(project_accessions)
        cumulative_files += len(manifest.files)
        summary = manifest.summary if isinstance(manifest.summary, Mapping) else {}
        batch_size = int(
            getattr(manifest.request, "partial_delivery_batch_size", 0)
            or summary.get("batch_size")
            or 500
        )
        rebuilt.append(
            {
                "batch_index": batch_index,
                "batch_size": batch_size,
                "project_count": len(project_accessions),
                "file_count": len(manifest.files),
                "cumulative_verified_project_count": len(cumulative_projects),
                "cumulative_verified_file_count": cumulative_files,
                "project_accessions": project_accessions,
                "file_identifiers": file_identifiers,
                "delivery_unit": "file",
                "manifest_path": str(manifest_path),
                "terminal": bool(summary.get("terminal")),
                "status": "ready",
                "published_at": datetime.fromtimestamp(
                    manifest_path.stat().st_mtime,
                    tz=_APP_TZ,
                ).isoformat(),
                "message": (
                    f"Verified file batch {batch_index} is ready: "
                    f"{len(manifest.files)} files from "
                    f"{len(project_accessions)} projects."
                ),
            }
        )
    return rebuilt


def _discovery_result_batches(job: Mapping[str, Any]) -> list[dict[str, Any]]:
    persisted = [
        dict(item)
        for item in job.get("result_batches") or []
        if isinstance(item, Mapping)
    ]
    return persisted or _rebuild_discovery_result_batches(job)


def _discovery_job_public(job: dict[str, Any], *, detail: bool = False) -> dict[str, Any]:
    record = job.get("record")
    if record is not None and not detail:
        record = _slim_discovery_record(record)
    if isinstance(record, dict):
        record = dict(record)
        record.pop("published_verified_project_batches", None)
    body = job.get("body") if isinstance(job.get("body"), dict) else {}
    output_language = _normalise_pool_build_language(job.get("output_language") or body.get("output_language"))
    raw_logs = list(job.get("logs") or [])
    selected_logs = raw_logs if detail else _discovery_summary_logs(raw_logs)
    logs = []
    for raw in selected_logs:
        if not isinstance(raw, dict):
            continue
        entry = dict(raw) if detail else _compact_discovery_log_entry(raw)
        entry["message"] = _localize_discovery_message(entry.get("message"), output_language)
        logs.append(entry)
    result_batches = []
    for raw in _discovery_result_batches(job):
        if not isinstance(raw, dict):
            continue
        batch = {
            key: value
            for key, value in raw.items()
            if key not in (
                {"manifest_path", "files"}
                if detail
                else {
                    "manifest_path",
                    "files",
                    "file_identifiers",
                    "project_accessions",
                }
            )
        }
        batch_index = int(batch.get("batch_index") or 0)
        if batch_index > 0:
            batch["download_url"] = (
                f"/api/discovery/jobs/{job.get('job_id')}/batches/"
                f"{batch_index}/download"
            )
        result_batches.append(batch)
    return {
        "job_id": job.get("job_id"),
        "idempotency_key": job.get("idempotency_key"),
        "status": job.get("status"),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "cancel_requested": bool(job.get("cancel_requested")),
        "resumable": bool(job.get("resumable")),
        "output_language": output_language,
        "logs": logs,
        "log_count": len([item for item in raw_logs if isinstance(item, Mapping)]),
        "logs_truncated": len(logs) < len(
            [item for item in raw_logs if isinstance(item, Mapping)]
        ),
        "result_batches": result_batches,
        "execution_state": job.get("execution_state"),
        "record": record,
        "error": _localize_discovery_message(job.get("error"), output_language) if job.get("error") else None,
        "detail": "full" if detail else "summary",
    }


def _localize_discovery_message(message: Any, output_language: str) -> str:
    text = _redact_secrets(message).strip()
    if output_language != "zh-CN" or not text or _contains_cjk(text):
        return _localize_public_message(text, "en" if output_language == "en" else "zh", level="info")
    exact = {
        "Discovery job queued.": "数据发现任务已排队。",
        "Discovery job started.": "数据发现任务已开始。",
        "Discovery job completed.": "数据发现任务已完成。",
        "Discovery job failed with retained audits.": "数据发现失败，已保留审计记录。",
        "Discovery job stopped at the quality gate; candidate evidence and audits were retained.": "数据发现已停止在质量闸门：候选证据与审计记录均已保留。",
        "Discovery job cancelled.": "数据发现任务已取消。",
        "Discovery cancelled.": "数据发现任务已取消。",
        "Running diversity-aware selection.": "正在执行多样性选择。",
    }
    if text in exact:
        return exact[text]
    patterns = (
        (r"^Searching PRIDE projects: (.+)$", r"正在检索 PRIDE 项目：\1"),
        (r"^Project search returned (\d+) raw records so far\.$", r"项目检索目前返回 \1 条原始记录。"),
        (r"^Deduped to (\d+) candidate project\(s\)\.$", r"去重后得到 \1 个候选项目。"),
        (r"^Inspecting project (.+)\.$", r"正在检查项目 \1。"),
        (r"^Selected (\d+) project\(s\), (\d+) file\(s\)\.$", r"已入选 \1 个项目、\2 个文件。"),
        (r"^(.+): kept (\d+) file candidate\(s\)\.$", r"\1：保留 \2 个候选文件。"),
        (r"^(.+): no usable acquisition/peaklist file candidates after filtering\.$", r"\1：过滤后没有可用的采集或峰表文件。"),
    )
    for pattern, replacement in patterns:
        if re.match(pattern, text):
            return re.sub(pattern, replacement, text)
    if text.startswith("Observe:"):
        return "观察：已记录当前数据发现状态。"
    if text.startswith("Reason:"):
        return "推理摘要：已根据当前证据更新检索决策。"
    if text.startswith("Act:"):
        return "执行：正在进行下一步受控数据发现操作。"
    # Common agent-event English → short Chinese for live trajectory
    agent_patterns = (
        (r"^Searching repository with (\d+) query plan\(s\): (.+)$", r"按 \1 组查询检索仓库：\2"),
        (r"^Search observed (\d+) candidate project\(s\), (\d+) new and (\d+) high-relevance; semantic coverage ([0-9.]+%)\.$",
         r"检索观察到 \1 个候选项目（新增 \2，高相关 \3），语义覆盖 \4。"),
        (r"^Inspecting (\d+) candidate project\(s\): (.+)$", r"正在审查 \1 个候选项目：\2"),
        (r"^Inspection produced (\d+) selected project\(s\) and (\d+) selected file\(s\); next action: (.+)\.$",
         r"审查入选 \1 个项目、\2 个文件；下一步：\3。"),
        (r"^Discovery job started\.$", "数据发现任务已开始。"),
        (r"^Discovery job completed\.$", "数据发现任务已完成。"),
    )
    for pattern, replacement in agent_patterns:
        if re.match(pattern, text):
            return re.sub(pattern, replacement, text)
    # Errors: never collapse to a generic progress fallback — keep forensic detail.
    if text.startswith("Discovery failed:"):
        detail = text[len("Discovery failed:") :].strip() or text
        return f"数据发现失败：{detail}"
    if "failed" in text.casefold() or "error" in text.casefold():
        return f"数据发现失败：{text}"
    # Unknown non-error English: keep original (better than fake progress).
    return text


def _discovery_jobs_dir() -> Path:
    return _runs_dir / "discovery_jobs"


def _discovery_job_path(job_id: str) -> Path:
    return _discovery_jobs_dir() / f"{safe_output_stem(job_id)}.json"


def _discovery_job_summaries_dir() -> Path:
    return _discovery_jobs_dir() / "_history"


def _discovery_job_summary_path(job_id: str) -> Path:
    return _discovery_job_summaries_dir() / f"{safe_output_stem(job_id)}.json"


def _discovery_storage_scope() -> str:
    """Stable, non-path identifier for the configured run storage root."""
    normalized = os.path.normcase(str(Path(_runs_dir).resolve()))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


def _discovery_job_is_in_current_storage_scope(job: Mapping[str, Any]) -> bool:
    scope = _clean_text(job.get("storage_scope"))
    # Jobs written before storage scoping remain visible for compatibility.
    return not scope or secrets.compare_digest(scope, _discovery_storage_scope())


def _discovery_job_persist_payload(job: dict[str, Any]) -> dict[str, Any]:
    """Serialize job for disk with raw English logs/error (API localizes on read).

    Credentials must never be written; mirror the in-worker body sanitization.
    """
    body = job.get("body") if isinstance(job.get("body"), dict) else {}
    safe_body = _sanitize_log_payload(dict(body))
    if isinstance(safe_body, dict):
        llm = safe_body.get("llm_config")
        if isinstance(llm, dict):
            safe_body["llm_config"] = {
                k: v for k, v in llm.items() if str(k).casefold() not in {"api_key", "authorization"}
            }
    logs: list[Any] = []
    for raw in job.get("logs") or []:
        if isinstance(raw, dict):
            logs.append(_compact_discovery_log_entry(raw))
        else:
            logs.append(_bounded_discovery_log_value(raw))
    return {
        "job_id": job.get("job_id"),
        "storage_scope": job.get("storage_scope") or _discovery_storage_scope(),
        "idempotency_key": job.get("idempotency_key"),
        "status": job.get("status"),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "cancel_requested": bool(job.get("cancel_requested")),
        "output_language": job.get("output_language"),
        "logs": logs,
        "body": safe_body,
        "record": job.get("record"),
        "execution_state": job.get("execution_state"),
        # Raw error string for forensics; _discovery_job_public localizes for clients.
        "error": _redact_secrets(job.get("error")) if job.get("error") else None,
        "detail": "full",
    }


def _write_json_atomic(path: Path, payload: Any) -> None:
    """Atomic JSON write used for discovery job durability (WP-D3)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    data = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    with temp_path.open("w", encoding="utf-8") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)
    try:
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)


def _persist_discovery_job(job: dict[str, Any], *, required: bool = False) -> None:
    """Persist discovery state without rewriting giant JSON on every event.

    The operations database is the live authority. Legacy JSON is retained as
    a compatibility/checkpoint artifact and is only rewritten for durable
    lifecycle checkpoints.
    """
    job_id = str(job.get("job_id") or "")
    job.setdefault("storage_scope", _discovery_storage_scope())
    try:
        get_operations_repository().sync_legacy_job(
            job,
            legacy_path=str(_discovery_job_path(job_id)),
        )
    except Exception as exc:
        if required:
            raise RuntimeError(
                f"operations_job_persist_failed:{job_id}:{exc}"
            ) from exc
        return
    if not required:
        return
    try:
        _write_json_atomic(
            _discovery_job_path(job_id),
            _discovery_job_persist_payload(job),
        )
        try:
            _write_json_atomic(
                _discovery_job_summary_path(job_id),
                _discovery_job_history_shell(job),
            )
        except Exception:
            # Never leave an older running shell beside a newer authoritative job.
            # The full job remains authoritative if this acceleration sidecar fails.
            try:
                _discovery_job_summary_path(job_id).unlink(missing_ok=True)
            except OSError:
                pass
        # Flush compatibility forensic files only at durable checkpoints.
        try:
            _write_discovery_job_log_file(job)
        except Exception:
            # Side-channel log file is non-authoritative.
            pass
    except Exception as exc:
        if required:
            raise RuntimeError(f"discovery_job_persist_failed:{job_id}:{exc}") from exc
        # Non-required (e.g. high-frequency log append) stays best-effort.
        return


def _discovery_run_dir_from_job(job: Mapping[str, Any] | None) -> Path | None:
    if not isinstance(job, Mapping):
        return None
    record = job.get("record") if isinstance(job.get("record"), Mapping) else {}
    for key in ("output_dir", "run_id", "discovery_id"):
        raw = _clean_text((record or {}).get(key) or job.get(key))
        if not raw:
            continue
        path = Path(raw)
        if path.exists() and path.is_dir():
            return path
        candidate = _discovery_root_dir() / safe_output_stem(raw)
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _write_discovery_job_log_file(job: Mapping[str, Any]) -> None:
    run_dir = _discovery_run_dir_from_job(job)
    if run_dir is None:
        return
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "discovery_job.jsonl"
    lines: list[str] = []
    for item in job.get("logs") or []:
        if not isinstance(item, Mapping):
            continue
        compact_item = _compact_discovery_log_entry(item)
        payload = {
            "ts": compact_item.get("ts"),
            "level": compact_item.get("level") or "info",
            "actor": compact_item.get("actor") or "Discovery Agent",
            "type": compact_item.get("type") or "job_message",
            "message": compact_item.get("message") or "",
            "metrics": compact_item.get("metrics") or {},
            "payload": compact_item.get("payload") or {},
            "job_id": job.get("job_id"),
            "status": job.get("status"),
        }
        lines.append(json.dumps(payload, ensure_ascii=False, default=str))
    log_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    # Human-readable short log as well.
    text_lines = []
    for item in job.get("logs") or []:
        if not isinstance(item, Mapping):
            continue
        text_lines.append(
            f"{item.get('ts') or ''} [{item.get('level') or 'info'}] {item.get('message') or ''}".strip()
        )
    (logs_dir / "discovery_job.log").write_text(
        "\n".join(text_lines) + ("\n" if text_lines else ""),
        encoding="utf-8",
    )


def _package_discovery_run_bundle(
    *,
    job: Mapping[str, Any] | None = None,
    run_dir: Path | None = None,
) -> Path | None:
    """Package one discovery run's state/logs/results into a single zip for forensics."""
    target = run_dir or _discovery_run_dir_from_job(job)
    if target is None or not target.exists():
        return None
    bundle_path = target / "discovery_run_bundle.zip"
    include_names = {
        "dataset_manifest.json",
        "dataset_manifest.csv",
        "dataset_manifest_valid.csv",
        "dataset_manifest_usable.csv",
        "dataset_request.json",
        "discovery_summary.json",
        "quality_report.json",
        "candidate_search_state.json",
        "agents_discovery_summary.json",
        "agents_discovery_events.json",
        "agents_discovery_report.md",
        "agents_discovery_budget.json",
        "agent_control.sqlite",
        "agents_sdk_trace.jsonl",
        "project_judgments.json",
        "project_judgments_table.csv",
        "selected_projects_review.csv",
        "selected_projects_review.json",
        "candidate_projects.json",
        "batch_inputs.txt",
        "project_accessions.txt",
        "recovery_summary.json",
    }
    include_dirs = {
        "logs",
        "final_selection",
        "candidate_pool",
        "recovered_final_selection",
    }
    # Keep rounds, but only their small summary/request files when possible.
    try:
        with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            # Always embed a compact job snapshot when available.
            if isinstance(job, Mapping):
                snapshot = {
                    "job_id": job.get("job_id"),
                    "status": job.get("status"),
                    "created_at": job.get("created_at"),
                    "started_at": job.get("started_at"),
                    "finished_at": job.get("finished_at"),
                    "error": job.get("error"),
                    "detail": job.get("detail"),
                    "output_language": job.get("output_language"),
                    "request": (job.get("body") or {}).get("prompt")
                    if isinstance(job.get("body"), Mapping)
                    else None,
                    "body_prompt": _clean_text((job.get("body") or {}).get("prompt"))
                    if isinstance(job.get("body"), Mapping)
                    else "",
                    "record_summary": {
                        k: (job.get("record") or {}).get(k)
                        for k in (
                            "discovery_id",
                            "run_id",
                            "status",
                            "project_count",
                            "file_count",
                            "output_dir",
                        )
                    }
                    if isinstance(job.get("record"), Mapping)
                    else {},
                    "log_count": len(job.get("logs") or []),
                }
                zf.writestr(
                    "job_snapshot.json",
                    json.dumps(snapshot, ensure_ascii=False, indent=2, default=str),
                )
                # Full logs separately for readability.
                zf.writestr(
                    "job_logs.json",
                    json.dumps(job.get("logs") or [], ensure_ascii=False, indent=2, default=str),
                )
            for path in target.rglob("*"):
                if not path.is_file():
                    continue
                rel = path.relative_to(target)
                rel_s = rel.as_posix()
                # Skip previous giant bundles and sqlite journals.
                if path.name.endswith((".zip", "-wal", "-shm")) and path.name != "discovery_run_bundle.zip":
                    if path.name != bundle_path.name:
                        continue
                if path.name == bundle_path.name:
                    continue
                top = rel.parts[0] if rel.parts else ""
                if top in include_dirs or path.name in include_names or rel_s.startswith("logs/"):
                    # Bound oversized evidence-heavy CSVs/json if needed later; include by default.
                    zf.write(path, arcname=rel_s)
                elif top.startswith("round_") and path.name in {
                    "dataset_request.json",
                    "discovery_summary.json",
                    "dataset_manifest.json",
                    "quality_report.json",
                }:
                    zf.write(path, arcname=rel_s)
        return bundle_path if bundle_path.exists() else None
    except Exception:
        return None


def _archive_discovery_job_artifacts(job_id: str) -> Path | None:
    with _discovery_jobs_lock:
        job = _discovery_jobs.get(job_id) or _load_discovery_job(job_id)
    if not job:
        return None
    # Ensure latest line logs are flushed before packaging.
    _write_discovery_job_log_file(job)
    run_dir = _discovery_run_dir_from_job(job)
    if run_dir is not None:
        try:
            _ensure_discovery_review_artifacts(run_dir)
        except Exception:
            pass
    bundle = _package_discovery_run_bundle(job=job)
    if bundle is not None:
        with _discovery_jobs_lock:
            live = _discovery_jobs.get(job_id)
            if live is not None:
                live["bundle_path"] = str(bundle)
                record = live.get("record") if isinstance(live.get("record"), dict) else {}
                if record is not None:
                    record = dict(record)
                    record["bundle_path"] = str(bundle)
                    downloads = dict(record.get("downloads") or {})
                    downloads["discovery_run_bundle_zip"] = str(bundle)
                    # Expose review tables if present.
                    if run_dir is not None:
                        for key, (filename, _media) in _DISCOVERY_DOWNLOAD_FILES.items():
                            candidate = Path(run_dir) / filename
                            if candidate.exists():
                                downloads[key] = f"/api/discovery/{Path(run_dir).name}/download?file={key}"
                    record["downloads"] = downloads
                    live["record"] = record
                _persist_discovery_job(live)
        return bundle
    return None


def _load_discovery_job(job_id: str) -> dict[str, Any] | None:
    payload = _read_json_if_exists(_discovery_job_path(job_id))
    if not payload:
        return None
    payload.setdefault("job_id", job_id)
    payload.setdefault("logs", [])
    payload.setdefault("cancel_requested", False)
    payload.setdefault("record", None)
    payload.setdefault("execution_state", None)
    payload.setdefault("error", None)
    payload.setdefault("output_language", "en")
    payload.setdefault("idempotency_key", None)
    payload.setdefault("storage_scope", _discovery_storage_scope())
    return payload


def _merge_operations_snapshot_into_discovery_job(
    job: Mapping[str, Any],
    snapshot: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Project the authoritative operations snapshot onto the legacy API shape.

    The compatibility endpoint remains available while the new UI reads the
    operations API directly. It must never mark an externally running Huey job
    interrupted merely because this web process has stale in-memory state.
    """

    merged = dict(job)
    if not isinstance(snapshot, Mapping):
        return merged
    operations_status = _clean_text(snapshot.get("status")).lower()
    merged["status"] = (
        "running"
        if operations_status in {"searching", "reviewing", "finalizing"}
        else operations_status or merged.get("status")
    )
    merged["cancel_requested"] = bool(snapshot.get("cancel_requested"))
    merged["resumable"] = bool(snapshot.get("resumable"))
    for field in ("started_at", "finished_at", "updated_at"):
        if snapshot.get(field):
            merged[field] = snapshot.get(field)
    error = snapshot.get("error")
    if isinstance(error, Mapping):
        merged["error"] = error.get("message") or error.get("code")
    else:
        # The operations database is authoritative.  In particular, a
        # successful terminal transition clears errors from an earlier failed
        # attempt; retaining the legacy value makes a completed job look
        # failed after resume.
        merged["error"] = None
    progress = snapshot.get("progress")
    progress = progress if isinstance(progress, Mapping) else {}
    execution = merged.get("execution_state")
    execution = dict(execution) if isinstance(execution, Mapping) else {}
    execution.update(
        {
            "phase": snapshot.get("phase") or execution.get("phase"),
            "current_term": progress.get("current_term"),
            "candidate_count": progress.get("candidate_count", 0),
            "raw_hit_count": progress.get("raw_hit_count", 0),
            "reviewed_project_count": progress.get("reviewed_count", 0),
            "pending_review_count": progress.get("pending_review_count", 0),
            "qualified_project_count": progress.get("qualified_count", 0),
            "candidate_file_count": progress.get("file_clue_count", 0),
            "usable_file_count": progress.get("usable_file_count", 0),
            "published_batch_count": progress.get("batch_count", 0),
            "review_worker_count": progress.get("worker_count", 0),
            "last_event_sequence": snapshot.get("last_event_sequence", 0),
            "heartbeat_at": snapshot.get("heartbeat_at"),
            "updated_at": snapshot.get("updated_at"),
        }
    )
    merged["execution_state"] = execution
    return merged


def _find_discovery_job_by_idempotency_key(key: str) -> dict[str, Any] | None:
    if not key:
        return None
    for job in _discovery_jobs.values():
        if (
            _discovery_job_is_in_current_storage_scope(job)
            and str(job.get("idempotency_key") or "") == key
        ):
            return job
    jobs_dir = _discovery_jobs_dir()
    if not jobs_dir.is_dir():
        return None
    for path in jobs_dir.glob("*.json"):
        payload = _read_json_if_exists(path)
        if not payload or str(payload.get("idempotency_key") or "") != key:
            continue
        job_id = _clean_text(payload.get("job_id"))
        if not job_id:
            continue
        operations_snapshot = get_operations_repository().get_job(job_id)
        payload = _merge_operations_snapshot_into_discovery_job(
            payload,
            operations_snapshot,
        )
        if operations_snapshot is None:
            payload = _mark_interrupted_discovery_job(dict(payload))
        _discovery_jobs[job_id] = payload
        _persist_discovery_job(payload)
        return payload
    return None


def _mark_interrupted_discovery_job(job: dict[str, Any]) -> dict[str, Any]:
    """On process restart, mark in-flight jobs interrupted/resumable (WP-D3).

    Disk remains the authority; we do not invent a success state in memory.
    """
    if job.get("status") not in {"queued", "running"}:
        return job
    logs = job.setdefault("logs", [])
    message = (
        "Discovery job was interrupted by a server reload and is marked interrupted/resumable. "
        "Re-submit or resume explicitly; memory is not authoritative."
    )
    if not any(item.get("message") == message for item in logs if isinstance(item, dict)):
        logs.append({"ts": _now_app_iso(), "level": "warning", "message": message})
    job["status"] = "interrupted"
    job["error"] = "discovery_job_interrupted_by_server_reload"
    job["resumable"] = True
    job["finished_at"] = job.get("finished_at") or _now_app_iso()
    return job


def _append_discovery_job_log(job_id: str, level: str, message: str) -> None:
    entry: dict[str, Any] | None = None
    with _discovery_jobs_lock:
        job = _discovery_jobs.get(job_id)
        if not job:
            return
        logs = job.setdefault("logs", [])
        sequence = max((int(item.get("sequence") or 0) for item in logs if isinstance(item, dict)), default=0) + 1
        entry = {
            "sequence": sequence,
            "ts": _now_app_iso(),
            "level": level,
            "actor": "Discovery Agent",
            "type": "job_message",
            "message": _redact_secrets(str(message)),
            "reasoning_summary": "",
            "evidence_refs": [],
            "metrics": {},
            "payload": {},
        }
        logs.append(entry)
        if len(logs) > _MAX_PERSISTED_LOGS:
            del logs[: len(logs) - _MAX_PERSISTED_LOGS]
    if entry is not None:
        try:
            get_operations_repository().append_event(
                job_id,
                event_type="job_message",
                level=level,
                actor="Discovery Agent",
                message=str(entry["message"]),
                payload={},
                created_at=str(entry["ts"]),
            )
        except Exception:
            # Scientific execution must not fail because the status stream is
            # temporarily contended; the next lifecycle checkpoint will sync.
            pass


def _event_actor(event_type: str) -> str:
    if event_type.startswith("budget_"):
        return "Budget Agent"
    if "grant" in event_type or event_type == "dynamic_search_stopped":
        return "BudgetGovernor"
    if event_type.startswith("sdk_"):
        return "OpenAI Agents SDK"
    if event_type.startswith("candidate_search_"):
        return "Repository Search"
    if event_type.startswith("repository_term_") or event_type.startswith(
        "confirmed_theme_pipeline_"
    ):
        return "Repository Scheduler"
    if event_type.startswith("candidate_review_queue_"):
        return "Candidate Review Queue"
    if event_type.startswith("candidate_inspection_"):
        return "Candidate Inspector"
    if event_type.startswith("discovery_quality_"):
        return "Quality Auditor"
    if event_type == "project_judgments_recorded":
        return "Project Judge"
    if event_type.startswith("tool_") or event_type == "repository_request_started":
        return "Repository tool"
    return "Discovery Agent"


def _event_level(event_type: str) -> str:
    if "invalid" in event_type or "rejected" in event_type or "failed" in event_type:
        return "warning"
    return "info"


def _event_message(event: AgentEvent) -> str:
    payload = event.payload
    observation = payload.get("observation") or {}
    action = payload.get("action") or {}
    if event.event_type == "candidate_search_started":
        queries = payload.get("queries") or []
        return _redact_secrets(
            f"Searching repository with {len(queries)} query plan(s): "
            + "; ".join(str(query) for query in queries[:4])
        )
    if event.event_type == "repository_term_task_started":
        return (
            f"Started confirmed repository term {int(payload.get('term_index') or 0)}/"
            f"{int(payload.get('term_count') or 0)}: {payload.get('term') or ''}."
        )
    if event.event_type == "repository_term_task_completed":
        return (
            f"Exhausted confirmed repository term {int(payload.get('term_index') or 0)}/"
            f"{int(payload.get('term_count') or 0)} after "
            f"{int(payload.get('chunks_completed') or 0)} internal pagination chunk(s); "
            f"{int(payload.get('reviewed_project_count') or 0)} project(s) reviewed."
        )
    if event.event_type == "repository_term_task_failed":
        return (
            f"Confirmed repository term {payload.get('term') or ''} did not reach "
            f"exhaustion: {payload.get('reason') or 'unknown failure'}."
        )
    if event.event_type == "candidate_search_completed":
        return (
            "Search observed "
            f"{int(observation.get('candidate_count') or 0)} candidate project(s), "
            f"{int(observation.get('new_candidate_count') or 0)} new and "
            f"{int(observation.get('high_relevance_candidate_count') or 0)} high-relevance; "
            f"semantic coverage {float(observation.get('semantic_coverage') or 0):.0%}."
        )
    if event.event_type == "candidate_inspection_started":
        accessions = action.get("accessions") or []
        return _redact_secrets(
            f"Inspecting {len(accessions)} candidate project(s): "
            + ", ".join(str(accession) for accession in accessions[:8])
        )
    if event.event_type == "candidate_inspection_completed":
        return (
            "Inspection produced "
            f"{int(observation.get('selected_projects') or 0)} selected project(s) and "
            f"{int(observation.get('selected_files') or 0)} selected file(s); "
            f"next action: {observation.get('recommended_action') or 'reassess evidence'}."
        )
    if event.event_type == "project_judgments_recorded":
        summary = payload.get("project_judgment_summary") or payload
        return (
            "Project scoring recorded for "
            f"{int(summary.get('assessed_projects') or 0)} project(s); "
            f"{int(summary.get('qualified_projects') or 0)} currently delivery-qualified."
        )
    if event.event_type == "discovery_quality_audited":
        counts = payload.get("counts") or {}
        return (
            f"Quality audit {payload.get('status') or 'completed'}: "
            f"{int(counts.get('delivery_eligible_projects') or 0)} delivery-eligible project(s), "
            f"{int(counts.get('usable_files') or 0)} usable file(s), "
            f"{len(payload.get('issues') or [])} visible issue(s)."
        )
    if event.event_type == "discovery_quality_repair_started":
        audit = payload.get("audit") or {}
        return (
            "Quality audit found repairable gaps; the Agent is continuing with "
            f"{len(audit.get('repair_actions') or [])} bounded repair action(s)."
        )
    if event.event_type == "discovery_quality_repair_completed":
        audit = payload.get("audit") or {}
        counts = audit.get("counts") or payload.get("counts") or {}
        return (
            "Autonomous quality repair attempt finished; authority audit is pending or incomplete. "
            f"{int(counts.get('delivery_eligible_projects') or 0)} project(s) currently meet intermediate review gates."
        )
    if event.event_type == "manifest_selected":
        return (
            f"Final selection retained {int(payload.get('selected_projects') or 0)} project(s) "
            f"and {int(payload.get('selected_files') or 0)} file record(s)."
        )
    return _redact_secrets(
        str(
            payload.get("reasoning_summary")
            or payload.get("reason")
            or payload.get("message")
            or event.event_type.replace("_", " ")
        )
    )


def _sanitize_log_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _sanitize_log_payload(item)
            for key, item in value.items()
            if str(key).casefold() not in {"api_key", "authorization", "sdk_state_json"}
        }
    if isinstance(value, list):
        return [_sanitize_log_payload(item) for item in value]
    if isinstance(value, str):
        return _redact_secrets(value)
    return _json_safe(value)


def _project_discovery_execution_state(
    current: Mapping[str, Any] | None,
    event: AgentEvent,
) -> dict[str, Any]:
    """Project the scheduler's durable state instead of inferring it in React."""

    state = dict(current or {})
    state.setdefault("schema_version", "discovery-execution/v1")
    terms = [
        dict(item)
        for item in state.get("terms") or []
        if isinstance(item, Mapping)
    ]
    event_type = event.event_type
    payload = event.payload
    if event_type == "confirmed_theme_pipeline_started":
        confirmed = [
            str(term).strip()
            for term in payload.get("terms") or []
            if str(term).strip()
        ]
        previous_terms = {
            (int(item.get("term_index") or 0), str(item.get("term") or "")): item
            for item in terms
        }
        terms = [
            {
                **previous_terms.get((index, term), {}),
                "term": term,
                "term_index": index,
                "term_count": len(confirmed),
                "role": "primary_theme" if index == 1 else "theme_synonym",
                "status": "pending",
                "failure_reason": "",
                "chunks_completed": int(
                    previous_terms.get((index, term), {}).get("chunks_completed") or 0
                ),
                "raw_result_count": int(
                    previous_terms.get((index, term), {}).get("raw_result_count") or 0
                ),
                "new_candidate_count": int(
                    previous_terms.get((index, term), {}).get("new_candidate_count") or 0
                ),
                "exhausted": (
                    previous_terms.get((index, term), {}).get("exhausted") is True
                ),
            }
            for index, term in enumerate(confirmed, start=1)
        ]
        state.update(
            {
                "phase": "searching",
                "active_term_index": 1 if terms else 0,
                "candidate_count": int(state.get("candidate_count") or 0),
                "reviewed_project_count": int(
                    state.get("reviewed_project_count") or 0
                ),
                "pending_review_count": int(
                    state.get("pending_review_count") or 0
                ),
                "all_terms_exhausted": False,
                "completion_ready": False,
            }
        )
    term_index = int(payload.get("term_index") or 0)
    term_item = next(
        (
            item
            for item in terms
            if int(item.get("term_index") or 0) == term_index
        ),
        None,
    )
    if event_type == "repository_term_task_started" and term_item is not None:
        term_item.update(status="running", failure_reason="")
        state.update(phase="searching", active_term_index=term_index)
    elif event_type == "repository_term_chunk_completed" and term_item is not None:
        term_item["chunks_completed"] = max(
            int(term_item.get("chunks_completed") or 0),
            int(payload.get("chunk_index") or 0),
        )
        term_item["raw_result_count"] = int(
            term_item.get("raw_result_count") or 0
        ) + int(payload.get("raw_result_count") or 0)
        term_item["new_candidate_count"] = int(
            term_item.get("new_candidate_count") or 0
        ) + int(payload.get("new_candidate_count") or 0)
        state["candidate_count"] = max(
            int(state.get("candidate_count") or 0),
            int(payload.get("candidate_count") or 0),
        )
        term_item["exhausted"] = payload.get("exhausted") is True
    elif event_type == "candidate_review_queue_batch_started":
        state.update(
            phase="reviewing",
            active_term_index=term_index,
            active_review_batch_size=int(payload.get("batch_size") or 0),
            review_workers=int(payload.get("review_workers") or 0),
        )
    elif event_type == "candidate_review_queue_batch_completed":
        state["reviewed_project_count"] = max(
            int(state.get("reviewed_project_count") or 0),
            int(payload.get("reviewed_project_count") or 0),
        )
        state["active_review_batch_size"] = 0
    elif event_type in {
        "repository_term_task_completed",
        "repository_term_task_failed",
    } and term_item is not None:
        failed = event_type.endswith("_failed")
        term_item.update(
            status="failed" if failed else "completed",
            chunks_completed=max(
                int(term_item.get("chunks_completed") or 0),
                int(payload.get("chunks_completed") or 0),
            ),
            raw_result_count=max(
                int(term_item.get("raw_result_count") or 0),
                int(payload.get("raw_result_count") or 0),
            ),
            new_candidate_count=max(
                int(term_item.get("new_candidate_count") or 0),
                int(payload.get("new_candidate_count") or 0),
            ),
            exhausted=payload.get("exhausted") is True,
            failure_reason=str(payload.get("reason") or "") if failed else "",
            reviewed_project_count=int(
                payload.get("reviewed_project_count") or 0
            ),
        )
        state["phase"] = "failed" if failed else "searching"
        state["active_term_index"] = (
            term_index if failed else min(term_index + 1, len(terms))
        )
    elif event_type == "confirmed_theme_pipeline_completed":
        complete = (
            payload.get("status") == "completed"
            and (
                payload.get("target_reached") is True
                or payload.get("all_terms_exhausted") is True
            )
            and int(payload.get("pending_review_count") or 0) == 0
        )
        state.update(
            phase="finalizing" if complete else "failed",
            active_term_index=0,
            candidate_count=max(
                int(state.get("candidate_count") or 0),
                int(payload.get("candidate_count") or 0),
            ),
            reviewed_project_count=max(
                int(state.get("reviewed_project_count") or 0),
                int(payload.get("reviewed_project_count") or 0),
            ),
            pending_review_count=int(payload.get("pending_review_count") or 0),
            all_terms_exhausted=payload.get("all_terms_exhausted") is True,
            completion_ready=complete,
        )
    state["terms"] = terms
    state["last_event_sequence"] = int(event.sequence)
    state["updated_at"] = event.created_at
    return state


def _append_discovery_job_event(job_id: str, event: AgentEvent) -> None:
    event_metrics: Any = event.payload.get("metrics") or {}
    if event.event_type == "round_value_evaluated":
        event_metrics = event.payload
    elif event.event_type == "candidate_inspection_completed":
        event_metrics = (event.payload.get("observation") or {}).get("metrics") or {}
    public_event_payload = dict(event.payload)
    if event.event_type == "verified_project_batch_published":
        public_event_payload.pop("manifest_path", None)
    entry = {
        "source_sequence": event.sequence,
        "ts": event.created_at,
        "level": _event_level(event.event_type),
        "actor": _event_actor(event.event_type),
        "type": event.event_type,
        "message": _event_message(event),
        "reasoning_summary": _redact_secrets(str(event.payload.get("reasoning_summary") or "")),
        "evidence_refs": _sanitize_log_payload(event.payload.get("evidence_refs") or []),
        "metrics": _sanitize_log_payload(event_metrics),
        "payload": _sanitize_log_payload(public_event_payload),
    }
    phase = ""
    with _discovery_jobs_lock:
        job = _discovery_jobs.get(job_id)
        if not job:
            return
        job["execution_state"] = _project_discovery_execution_state(
            job.get("execution_state")
            if isinstance(job.get("execution_state"), Mapping)
            else None,
            event,
        )
        if event.event_type == "verified_project_batch_published":
            batches = job.setdefault("result_batches", [])
            batch_index = int(event.payload.get("batch_index") or 0)
            if batch_index > 0 and not any(
                int(item.get("batch_index") or 0) == batch_index
                for item in batches
                if isinstance(item, dict)
            ):
                batches.append(
                    _json_safe(
                        {
                            **dict(event.payload),
                            "status": "ready",
                            "published_at": event.created_at,
                        }
                    )
                )
        logs = job.setdefault("logs", [])
        entry["sequence"] = max(
            (int(item.get("sequence") or 0) for item in logs if isinstance(item, dict)),
            default=0,
        ) + 1
        logs.append(entry)
        if len(logs) > _MAX_PERSISTED_LOGS:
            del logs[: len(logs) - _MAX_PERSISTED_LOGS]
        execution = job.get("execution_state")
        if isinstance(execution, Mapping):
            phase = _clean_text(execution.get("phase"))
    try:
        operations_repository = get_operations_repository()
        if event.event_type == "candidate_inspection_completed":
            observation = event.payload.get("observation") or {}
            if isinstance(observation, Mapping):
                manifest_path = _clean_text(
                    observation.get("candidate_pool_manifest_path")
                    or observation.get("manifest_path")
                )
                if manifest_path:
                    manifest = _read_json_if_exists(Path(manifest_path))
                    candidate_files = manifest.get("files")
                    if isinstance(candidate_files, list):
                        operations_repository.sync_file_review_candidates(
                            job_id,
                            candidate_files,
                        )
        operations_repository.project_file_review_event(
            job_id,
            event.event_type,
            event.payload,
        )
        operations_repository.append_event(
            job_id,
            event_type=event.event_type,
            level=str(entry["level"]),
            actor=str(entry["actor"]),
            phase=phase,
            message=str(entry["message"]),
            payload={
                **dict(entry.get("payload") or {}),
                "metrics": entry.get("metrics") or {},
            },
            created_at=event.created_at,
        )
    except Exception:
        pass


def _append_discovery_search_event(
    job_id: str,
    event_type: str,
    payload: dict[str, Any],
) -> None:
    entry = {
        "ts": _now_app_iso(),
        "level": "error" if event_type.endswith("_failed") else "info",
        "actor": "Repository Search",
        "type": event_type,
        "message": "",
        "reasoning_summary": "",
        "evidence_refs": [],
        "metrics": {},
        "payload": _sanitize_log_payload(payload),
    }
    with _discovery_jobs_lock:
        job = _discovery_jobs.get(job_id)
        if not job:
            return
        logs = job.setdefault("logs", [])
        entry["sequence"] = max(
            (int(item.get("sequence") or 0) for item in logs if isinstance(item, dict)),
            default=0,
        ) + 1
        logs.append(entry)
        if len(logs) > _MAX_PERSISTED_LOGS:
            del logs[: len(logs) - _MAX_PERSISTED_LOGS]
    try:
        get_operations_repository().append_event(
            job_id,
            event_type=event_type,
            level=str(entry["level"]),
            actor="Repository Search",
            phase="searching",
            message=_clean_text(payload.get("message") or payload.get("reason")),
            payload=entry["payload"],
            created_at=str(entry["ts"]),
        )
    except Exception:
        pass


def _discovery_cancel_requested(job_id: str) -> bool:
    try:
        if get_operations_repository().cancel_requested(job_id):
            return True
    except Exception:
        pass
    with _discovery_jobs_lock:
        job = _discovery_jobs.get(job_id)
        return bool(job and job.get("cancel_requested"))


def _run_discovery_job(job_id: str) -> None:
    with _discovery_jobs_lock:
        job = _discovery_jobs.get(job_id) or _load_discovery_job(job_id)
        if not job:
            return
        _discovery_jobs[job_id] = job
        repository = get_operations_repository()
        snapshot = repository.get_job(job_id)
        if snapshot is None:
            persisted = _discovery_job_persist_payload(job)
            safe_body = (
                persisted.get("body")
                if isinstance(persisted.get("body"), Mapping)
                else {}
            )
            snapshot = repository.create_job(
                job_id=job_id,
                payload=safe_body,
                terms=_discovery_terms_from_body(safe_body),
                idempotency_key=_clean_text(job.get("idempotency_key")) or None,
                job_type="discovery",
                created_at=_clean_text(job.get("created_at")) or None,
                legacy_path=str(_discovery_job_path(job_id)),
            )
        if (
            snapshot.get("status") == "cancelled"
            and snapshot.get("cancel_requested")
        ):
            return
        job["status"] = "running"
        job["resumable"] = False
        job["started_at"] = _now_app_iso()
        body = dict(job.get("body") or {})
        # Keep the request credential only in this worker's local copy.
        job["body"].pop("llm_config", None)
    repository.transition_job(
        job_id,
        "searching",
        phase="searching",
        reason="持久 worker 已认领任务并开始检索。",
        event_type="job_started",
        actor="Huey worker",
    )
    with _discovery_jobs_lock:
        _persist_discovery_job(_discovery_jobs[job_id], required=True)
    _append_discovery_job_log(job_id, "info", "Discovery job started.")

    def report(message: str) -> None:
        _append_discovery_job_log(job_id, "info", message)

    def should_cancel() -> bool:
        return _discovery_cancel_requested(job_id)

    try:
        if should_cancel():
            raise InterruptedError("Discovery cancelled.")
        record = _run_web_discovery(
            body,
            report=report,
            should_cancel=should_cancel,
            agent_event_callback=lambda event: _append_discovery_job_event(job_id, event),
            search_event_callback=lambda event_type, payload: _append_discovery_search_event(
                job_id,
                event_type,
                payload,
            ),
        )
        cancelled = should_cancel()
        record_status = _clean_text((record or {}).get("status")).lower() if isinstance(record, dict) else ""
        operations_progress = (
            get_operations_repository().get_job(job_id) or {}
        ).get("progress") or {}
        fixed_candidate_delivery_complete = (
            _candidate_delivery_satisfies_run_horizon(
                body,
                operations_progress,
            )
        )
        if cancelled or record_status == "cancelled":
            terminal_status = "cancelled"
        elif record_status == "failed":
            terminal_status = "failed"
        elif record_status == "blocked" and fixed_candidate_delivery_complete:
            # This product's discovery horizon is candidates_reviewed. A
            # verified terminal file batch satisfies that contract even when a
            # stricter downstream modeling/build-ready publication audit is
            # intentionally unavailable for browse-only work.
            terminal_status = "completed"
        elif record_status == "blocked":
            terminal_status = "blocked"
        else:
            terminal_status = "completed"
        get_operations_repository().transition_job(
            job_id,
            "finalizing",
            phase="finalizing",
            reason="检索与项目审查已结束，正在冻结结果和批次。",
            event_type="job_finalization_started",
            actor="Discovery Agent",
        )
        terminal_error = ""
        if terminal_status == "failed" and isinstance(record, Mapping):
            terminal_error = _redact_secrets(
                _clean_text((record.get("agent") or {}).get("error"))
                or _clean_text((record.get("summary") or {}).get("error"))
                or "Discovery failed."
            )
        get_operations_repository().transition_job(
            job_id,
            terminal_status,
            phase=terminal_status,
            reason={
                "completed": "任务完成，所有权威结果与批次已冻结。",
                "failed": "任务失败；已完成证据和断点均已保留。",
                "blocked": "任务停在质量闸门；已有证据和进度均已保留。",
                "cancelled": "任务已按用户请求安全停止。",
            }[terminal_status],
            event_type=f"job_{terminal_status}",
            level="info" if terminal_status == "completed" else "warning",
            actor="Discovery Agent",
            error_code=terminal_status if terminal_status in {"failed", "blocked"} else None,
            error_message=terminal_error or None,
            resumable=terminal_status in {"failed", "cancelled"},
        )
        with _discovery_jobs_lock:
            job = _discovery_jobs.get(job_id)
            if not job:
                return
            job["record"] = record
            job["status"] = terminal_status
            if terminal_status == "failed" and isinstance(record, dict):
                job["error"] = _redact_secrets(
                    _clean_text((record.get("agent") or {}).get("error"))
                    or _clean_text((record.get("summary") or {}).get("error"))
                    or "Discovery failed."
                )
            else:
                job["error"] = None
            job["finished_at"] = _now_app_iso()
            job["resumable"] = terminal_status in {"failed", "cancelled"}
            try:
                _persist_discovery_job(job, required=True)
            except Exception as persist_exc:
                # Durability failure: do not advertise terminal success from memory only.
                job["status"] = "durability_failed"
                job["error"] = f"discovery_job_persist_failed:{persist_exc}"
                try:
                    _persist_discovery_job(job, required=False)
                except Exception:
                    pass
                raise
        finish_message = {
            "completed": "Discovery job completed.",
            "failed": "Discovery job failed with retained audits.",
            "blocked": "Discovery job stopped at the quality gate; candidate evidence and audits were retained.",
            "cancelled": "Discovery job cancelled.",
        }[terminal_status]
        finish_level = (
            "info"
            if terminal_status == "completed"
            else "error"
            if terminal_status == "failed"
            else "warning"
        )
        _append_discovery_job_log(job_id, finish_level, finish_message)
        with _discovery_jobs_lock:
            finished_checkpoint = _discovery_jobs.get(job_id)
            if finished_checkpoint is not None:
                _persist_discovery_job(finished_checkpoint, required=True)
        # Ensure finished discovery runs appear in the main history panel.
        try:
            with _discovery_jobs_lock:
                finished_job = _discovery_jobs.get(job_id) or {}
                finished_record = finished_job.get("record") if isinstance(finished_job.get("record"), dict) else None
            if finished_record:
                _upsert_discovery_history_record(finished_record)
            else:
                # fall back to packaging path discovery
                run_dir = _discovery_run_dir_from_job(finished_job)
                if run_dir is not None:
                    _upsert_discovery_history_record(None, output_dir=run_dir)
        except Exception:
            pass
        try:
            bundle = _archive_discovery_job_artifacts(job_id)
            if bundle is not None:
                _append_discovery_job_log(
                    job_id,
                    "info",
                    f"Discovery run package saved: {bundle.name}",
                )
        except Exception as archive_exc:  # pragma: no cover - packaging must not fail the job
            _append_discovery_job_log(
                job_id,
                "warning",
                f"Discovery run packaging skipped: {_redact_secrets(str(archive_exc))}",
            )
    except InterruptedError as exc:
        try:
            get_operations_repository().transition_job(
                job_id,
                "cancelled",
                phase="cancelled",
                reason=str(exc),
                event_type="job_cancelled",
                level="warning",
                actor="Discovery Agent",
                resumable=True,
            )
        except Exception:
            pass
        with _discovery_jobs_lock:
            job = _discovery_jobs.get(job_id)
            if job:
                job["status"] = "cancelled"
                job["resumable"] = True
                job["error"] = str(exc)
                job["finished_at"] = _now_app_iso()
                _persist_discovery_job(job, required=True)
        _append_discovery_job_log(job_id, "warning", str(exc))
        try:
            _archive_discovery_job_artifacts(job_id)
        except Exception:
            pass
    except Exception as exc:  # pragma: no cover - defensive job boundary
        try:
            get_operations_repository().transition_job(
                job_id,
                "failed",
                phase="failed",
                reason="发现任务执行失败；可从持久断点恢复。",
                event_type="job_failed",
                level="error",
                actor="Discovery Agent",
                error_code="discovery_execution_failed",
                error_message=_redact_secrets(str(exc)),
                resumable=True,
            )
        except Exception:
            pass
        with _discovery_jobs_lock:
            job = _discovery_jobs.get(job_id)
            if job:
                job["status"] = "failed"
                job["resumable"] = True
                job["error"] = _redact_secrets(str(exc))
                job["finished_at"] = _now_app_iso()
                _persist_discovery_job(job, required=True)
        _append_discovery_job_log(job_id, "error", f"Discovery failed: {exc}")
        try:
            _archive_discovery_job_artifacts(job_id)
        except Exception:
            pass


def _start_discovery_job_thread(job_id: str) -> None:
    queue_task_id = enqueue_discovery_job(job_id)
    try:
        get_operations_repository().append_event(
            job_id,
            event_type="job_enqueued",
            actor="Huey queue",
            phase="queued",
            message="任务已提交到持久队列，等待 worker 认领。",
            payload={"queue_task_id": queue_task_id},
        )
    except Exception:
        pass


def _discovery_terms_from_body(body: Mapping[str, Any]) -> list[dict[str, str]]:
    for key in (
        "confirmed_theme_terms",
        "selected_search_terms",
        "search_terms",
        "query_terms",
    ):
        values = body.get(key)
        if not isinstance(values, list):
            continue
        terms = [
            _clean_text(item)
            for item in values
            if _clean_text(item)
        ]
        if terms:
            return [
                {
                    "term": term,
                    "role": "primary_theme" if index == 1 else "theme_synonym",
                }
                for index, term in enumerate(terms, start=1)
            ]
    return []


@app.post("/api/discovery/jobs")
async def start_discovery_job(body: dict[str, Any], background_tasks: BackgroundTasks = None):
    rejection = _discovery_confirmation_rejection(body)
    if rejection is not None:
        return rejection
    request_key = _clean_text(body.get("idempotency_key"))
    with _discovery_jobs_lock:
        existing = _find_discovery_job_by_idempotency_key(request_key)
        if existing is not None:
            return _discovery_job_public(existing)
        job_id = safe_output_stem(f"discovery_job_{datetime.now(_APP_TZ).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}")
        persisted_body = dict(body or {})
        persisted_body["_execution_discovery_id"] = safe_output_stem(
            f"agents_job_{job_id}"
        )
        job = {
            "job_id": job_id,
            "idempotency_key": request_key or None,
            "status": "queued",
            "created_at": _now_app_iso(),
            "started_at": None,
            "finished_at": None,
            "cancel_requested": False,
            "output_language": _normalise_pool_build_language(body.get("output_language")),
            "logs": [{"ts": _now_app_iso(), "level": "info", "message": "Discovery job queued."}],
            "result_batches": [],
            "body": persisted_body,
            "record": None,
            "error": None,
        }
        _discovery_jobs[job_id] = job
        get_operations_repository().create_job(
            job_id=job_id,
            payload=persisted_body,
            terms=_discovery_terms_from_body(persisted_body),
            idempotency_key=request_key or None,
            job_type="discovery",
            created_at=str(job["created_at"]),
            legacy_path=str(_discovery_job_path(job_id)),
        )
        _persist_discovery_job(job, required=True)
    if background_tasks is None:
        _start_discovery_job_thread(job_id)
    else:
        background_tasks.add_task(_start_discovery_job_thread, job_id)
    return _discovery_job_public(job)


@app.get("/api/discovery/jobs/{job_id}")
async def get_discovery_job(job_id: str, detail: int = 0):
    include_detail = bool(detail)
    with _discovery_jobs_lock:
        job = _discovery_jobs.get(job_id)
        if not job:
            job = _load_discovery_job(job_id)
            if not job:
                return {"error": "Discovery job not found."}
            operations_snapshot = get_operations_repository().get_job(job_id)
            job = _merge_operations_snapshot_into_discovery_job(
                job,
                operations_snapshot,
            )
            if operations_snapshot is None:
                job = _mark_interrupted_discovery_job(job)
            _discovery_jobs[job_id] = job
            _persist_discovery_job(job)
        else:
            job = _merge_operations_snapshot_into_discovery_job(
                job,
                get_operations_repository().get_job(job_id),
            )
            _discovery_jobs[job_id] = job
        return _discovery_job_public(job, detail=include_detail)


@app.post("/api/discovery/jobs/{job_id}/cancel")
async def cancel_discovery_job(job_id: str):
    with _discovery_jobs_lock:
        job = _discovery_jobs.get(job_id)
        if not job:
            job = _load_discovery_job(job_id)
            if not job:
                return {"error": "Discovery job not found."}
            job = _mark_interrupted_discovery_job(job)
            _discovery_jobs[job_id] = job
            _persist_discovery_job(job)
            return _discovery_job_public(job)
        if job.get("status") in {"completed", "failed", "blocked", "cancelled"}:
            return _discovery_job_public(job)
        operations_before_cancel = get_operations_repository().get_job(job_id)
        job["cancel_requested"] = True
        operations_snapshot = get_operations_repository().request_cancel(job_id)
        if (
            operations_before_cancel
            and operations_before_cancel.get("status") == "queued"
        ):
            revoke_discovery_job(
                str(operations_before_cancel.get("queue_task_id") or "")
            )
        if operations_snapshot.get("status") == "cancelled":
            job["status"] = "cancelled"
            job["finished_at"] = (
                operations_snapshot.get("finished_at") or _now_app_iso()
            )
            job["resumable"] = True
        _persist_discovery_job(job, required=True)
    _append_discovery_job_log(job_id, "warning", "Cancel requested. The current network call may finish before the job stops.")
    with _discovery_jobs_lock:
        job = _merge_operations_snapshot_into_discovery_job(
            _discovery_jobs[job_id],
            get_operations_repository().get_job(job_id),
        )
        _discovery_jobs[job_id] = job
        return _discovery_job_public(job)


@app.post("/api/discovery/jobs/{job_id}/resume")
async def resume_discovery_job(job_id: str):
    with _discovery_jobs_lock:
        job = _discovery_jobs.get(job_id)
        if not job:
            job = _load_discovery_job(job_id)
            if not job:
                return {"error": "Discovery job not found."}
            job = _mark_interrupted_discovery_job(job)
            _discovery_jobs[job_id] = job
        if job.get("status") not in {
            "interrupted",
            "failed",
            "durability_failed",
            "cancelled",
        }:
            return _discovery_job_public(job)
        body = job.get("body") if isinstance(job.get("body"), dict) else {}
        body.setdefault(
            "_execution_discovery_id",
            safe_output_stem(f"agents_job_{job_id}"),
        )
        body["_resume_existing_discovery_run"] = True
        job["body"] = body
        job["status"] = "queued"
        job["cancel_requested"] = False
        job["resumable"] = False
        job["error"] = None
        job["finished_at"] = None
        job["record"] = None
        get_operations_repository().transition_job(
            job_id,
            "queued",
            phase="queued",
            reason="用户请求从持久断点继续任务。",
            event_type="job_resume_requested",
            actor="user",
            resumable=False,
        )
        _persist_discovery_job(job, required=True)
    _append_discovery_job_log(
        job_id,
        "info",
        "Discovery job resume requested; persisted search cursors will be reused.",
    )
    _start_discovery_job_thread(job_id)
    with _discovery_jobs_lock:
        return _discovery_job_public(_discovery_jobs[job_id])


@app.get("/api/discovery/jobs/{job_id}/batches/{batch_index}/download")
async def download_discovery_verified_batch(job_id: str, batch_index: int):
    with _discovery_jobs_lock:
        job = _discovery_jobs.get(job_id) or _load_discovery_job(job_id)
    if job:
        batch = next(
            (
                item
                for item in _discovery_result_batches(job)
                if isinstance(item, dict)
                and int(item.get("batch_index") or 0) == batch_index
            ),
            None,
        )
        if batch is not None:
            path = Path(str(batch.get("manifest_path") or "")).resolve()
            if path.is_file() and _path_within(path, _runs_dir.resolve()):
                return FileResponse(
                    path=str(path),
                    filename=(
                        f"{job_id}_verified_batch_{batch_index:03d}.json"
                    ),
                    media_type="application/json",
                )
    try:
        handoff = _operations_discovery_batch_handoff(job_id, batch_index)
    except Exception as exc:
        return JSONResponse(
            {"error": _redact_secrets(str(exc))},
            status_code=404,
        )
    return JSONResponse(
        {
            "schema": "pride-agent-verified-file-batch/v1",
            **handoff,
        },
        headers={
            "Content-Disposition": (
                "attachment; filename="
                f'"{job_id}_verified_batch_{batch_index:03d}.json"'
            )
        },
    )


def _discovery_batch_handoff(job: Mapping[str, Any], batch_index: int) -> dict[str, Any]:
    batch = next(
        (
            item
            for item in _discovery_result_batches(job)
            if isinstance(item, dict) and int(item.get("batch_index") or 0) == batch_index
        ),
        None,
    )
    if batch is None:
        raise ValueError("Verified file batch not found.")
    path = Path(str(batch.get("manifest_path") or "")).resolve()
    if not path.is_file() or not _path_within(path, _discovery_root_dir().resolve()):
        raise ValueError("Verified file batch artifact is unavailable.")
    manifest = load_dataset_manifest(path)
    expected_count = int(batch.get("file_count") or 0)
    batch_size = int(batch.get("batch_size") or 500)
    if expected_count <= 0 or expected_count > batch_size or len(manifest.files) != expected_count:
        raise ValueError("Verified file batch count does not match its frozen manifest.")
    job_id = _clean_text(job.get("job_id"))
    body = job.get("body") if isinstance(job.get("body"), Mapping) else {}
    record = job.get("record") if isinstance(job.get("record"), Mapping) else {}
    discovery_id = _clean_text(
        record.get("discovery_id")
        or body.get("_execution_discovery_id")
        or manifest.run_id
    )
    inputs: list[str] = []
    input_records: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for file in manifest.files:
        native = _clean_text(file.file_accession_or_path or file.file_name)
        accession = _clean_text(file.project_accession).upper()
        repository = _clean_repository(file.repository or "pride")
        identifier = f"{repository.casefold()}:{accession}:{native}"
        if not native or identifier in identifiers:
            raise ValueError("Verified file batch contains a missing or duplicate file identifier.")
        identifiers.add(identifier)
        input_value = _clean_text(file.download_url) or (
            f"{accession}/{native}" if accession else native
        )
        inputs.append(input_value)
        input_records.append(
            {
                "input": input_value,
                "repository": repository,
                "project_accession": accession,
                "project_title": _clean_text(file.project_title),
                "file_name": _clean_text(file.file_name) or native,
                "download_url": _clean_text(file.download_url),
                "file_type": _clean_text(file.file_type),
                "file_role": _clean_text(file.file_role),
                "acquisition_mode": _clean_text(file.acquisition_mode),
                "source_discovery_job_id": job_id,
                "source_discovery_id": discovery_id,
                "source_batch_index": batch_index,
                "source_file_identifier": identifier,
            }
        )
    return {
        "job_id": job_id,
        "discovery_id": discovery_id,
        "batch_index": batch_index,
        "file_count": len(inputs),
        "project_count": len({record["project_accession"] for record in input_records}),
        "terminal": bool(batch.get("terminal")),
        "inputs": inputs,
        "input_records": input_records,
    }


def _operations_discovery_batch_handoff(
    job_id: str,
    batch_index: int,
) -> dict[str, Any]:
    delivery = get_operations_repository().get_batch_delivery(
        job_id,
        batch_index,
    )
    if delivery is None:
        raise ValueError("Verified file batch not found.")
    files = delivery.get("files")
    files = files if isinstance(files, list) else []
    missing = delivery.get("missing_file_identifiers")
    missing = missing if isinstance(missing, list) else []
    expected_count = int(delivery.get("file_count") or 0)
    if missing or expected_count <= 0 or len(files) != expected_count:
        raise ValueError(
            "Verified file batch is incomplete in the operations database."
        )
    inputs: list[str] = []
    input_records: list[dict[str, Any]] = []
    for file in files:
        native = _clean_text(
            file.get("native_id")
            or file.get("logical_path")
            or file.get("file_name")
        )
        accession = _clean_text(file.get("project_accession")).upper()
        repository = _clean_repository(file.get("repository") or "pride")
        download_url = _clean_text(file.get("download_url"))
        input_value = download_url or (
            f"{accession}/{native}" if accession else native
        )
        if not input_value:
            raise ValueError(
                "Verified file batch contains a file without an input value."
            )
        inputs.append(input_value)
        input_records.append(
            {
                "input": input_value,
                "repository": repository,
                "project_accession": accession,
                "project_title": "",
                "file_name": _clean_text(file.get("file_name")) or native,
                "download_url": download_url,
                "file_type": _clean_text(file.get("file_format")),
                "file_role": _clean_text(file.get("file_role")),
                "acquisition_mode": _clean_text(file.get("acquisition_mode")),
                "source_discovery_job_id": job_id,
                "source_discovery_id": safe_output_stem(
                    f"agents_job_{job_id}"
                ),
                "source_batch_index": batch_index,
                "source_file_identifier": _clean_text(
                    file.get("file_identifier")
                ),
            }
        )
    return {
        "job_id": job_id,
        "discovery_id": safe_output_stem(f"agents_job_{job_id}"),
        "batch_index": batch_index,
        "file_count": len(inputs),
        "project_count": len(
            {item["project_accession"] for item in input_records}
        ),
        "terminal": bool(delivery.get("terminal")),
        "inputs": inputs,
        "input_records": input_records,
    }


def _operations_discovery_run_bundle(discovery_id: str) -> Path | None:
    prefix = "agents_job_"
    if not discovery_id.startswith(prefix):
        return None
    job_id = discovery_id[len(prefix):]
    repository = get_operations_repository()
    snapshot = repository.get_job(job_id)
    if snapshot is None:
        return None
    batches = repository.list_batches(job_id)
    if not batches:
        return None
    bundle_root = (
        Path(repository.settings.artifact_root)
        / "recovered_discovery_bundles"
    )
    bundle_root.mkdir(parents=True, exist_ok=True)
    bundle_path = bundle_root / f"{safe_output_stem(discovery_id)}.zip"
    temporary_path = bundle_path.with_suffix(".zip.tmp")
    handoffs: list[dict[str, Any]] = []
    try:
        for batch in batches:
            handoffs.append(
                _operations_discovery_batch_handoff(
                    job_id,
                    int(batch.get("batch_index") or 0),
                )
            )
        with zipfile.ZipFile(
            temporary_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            archive.writestr(
                "job_snapshot.json",
                json.dumps(
                    snapshot,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
            )
            archive.writestr(
                "README.md",
                (
                    "# PRIDE discovery recovery bundle\n\n"
                    "This package was rebuilt from the durable operations "
                    "database after the legacy run directory was unavailable."
                    " Batch order and file identities are frozen by the SQL "
                    "batch index.\n"
                ),
            )
            all_records: list[dict[str, Any]] = []
            for handoff in handoffs:
                batch_index = int(handoff["batch_index"])
                archive.writestr(
                    f"verified_batches/batch_{batch_index:03d}.json",
                    json.dumps(
                        {
                            "schema": (
                                "pride-agent-verified-file-batch/v1"
                            ),
                            **handoff,
                        },
                        ensure_ascii=False,
                        indent=2,
                        default=str,
                    ),
                )
                archive.writestr(
                    f"batch_inputs/batch_{batch_index:03d}.txt",
                    "\n".join(handoff["inputs"]) + "\n",
                )
                all_records.extend(handoff["input_records"])
            archive.writestr(
                "all_input_records.json",
                json.dumps(
                    all_records,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
            )
        os.replace(temporary_path, bundle_path)
        return bundle_path
    except Exception:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        return None


@app.get("/api/discovery/jobs/{job_id}/batches/{batch_index}/handoff")
async def handoff_discovery_verified_batch(job_id: str, batch_index: int):
    with _discovery_jobs_lock:
        job = _discovery_jobs.get(job_id) or _load_discovery_job(job_id)
    if job:
        try:
            return _discovery_batch_handoff(job, batch_index)
        except Exception:
            # SQL is the durable authority and remains usable even when a
            # compatibility manifest path was relative to an older process.
            pass
    try:
        return _operations_discovery_batch_handoff(job_id, batch_index)
    except Exception as exc:
        return {"error": _redact_secrets(str(exc))}


@app.post("/api/discovery")
async def create_discovery(body: dict[str, Any]):
    rejection = _discovery_confirmation_rejection(body)
    if rejection is not None:
        return rejection
    try:
        return await asyncio.to_thread(_run_web_discovery, body)
    except Exception as exc:
        return {"error": _redact_secrets(str(exc))}


@app.post("/api/discovery/parse-goal")
async def parse_discovery_goal(body: dict[str, Any]):
    try:
        return await asyncio.to_thread(_run_discovery_goal_parse, body)
    except Exception as exc:
        return {"error": _redact_secrets(str(exc))}


@app.post("/api/discovery/grill-turn")
async def discovery_grill_turn(body: dict[str, Any]):
    """Conversational grill turn (LLM phrasing + mapping). Does not start discovery."""
    try:
        return await asyncio.to_thread(_run_discovery_grill_turn, body)
    except Exception as exc:
        return {
            "status": "failed",
            "parser": "llm_grill",
            "llm_used": False,
            "action": "advise",
            "mode": "advise",
            "assistant_message": (
                "模型本轮不可用或返回无效结果，当前策略保持不变。"
                "你可以直接重试；系统不会用关键词规则替你写卡。"
            ),
            "tool_calls": [],
            "extra_fields": {},
            "gap_report": {
                "required_missing": [],
                "optional_missing": [],
                "ready_for_confirm": False,
            },
            "ready_for_confirm": False,
            "failure_reason": _redact_secrets(str(exc)),
        }


@app.get("/api/discovery/{discovery_id}")
async def get_discovery(discovery_id: str):
    output_dir = _safe_discovery_dir(discovery_id)
    if output_dir is None:
        return {"error": "Discovery run not found."}
    manifest_path = output_dir / "dataset_manifest.json"
    if not manifest_path.exists():
        return {"error": "Discovery run not found."}
    try:
        manifest = load_dataset_manifest(manifest_path)
    except Exception as exc:
        return {"error": f"Discovery manifest is not readable: {_redact_secrets(str(exc))}"}
    return _public_discovery_record(discovery_id=discovery_id, output_dir=output_dir, manifest=manifest)


@app.post("/api/discovery/{discovery_id}/review")
async def review_discovery_run(discovery_id: str, body: dict[str, Any]):
    return await asyncio.to_thread(_save_discovery_reviews, discovery_id, body)


@app.get("/api/discovery/{discovery_id}/download")
async def download_discovery_file(discovery_id: str, file: str = "dataset_manifest_csv"):
    file_key = _clean_text(file)
    output_dir = _safe_discovery_dir(discovery_id)
    if output_dir is None or not output_dir.exists():
        if file_key == "discovery_run_bundle_zip":
            recovered_bundle = _operations_discovery_run_bundle(discovery_id)
            if recovered_bundle is not None:
                return FileResponse(
                    path=str(recovered_bundle),
                    filename=f"{discovery_id}_discovery_run_bundle.zip",
                    media_type="application/zip",
                )
        return JSONResponse(
            {"error": "Discovery run not found."},
            status_code=404,
        )
    entry = _DISCOVERY_DOWNLOAD_FILES.get(file_key)
    if entry is None:
        return {"error": "Discovery file not available."}
    filename, media_type = entry
    # Prefer run root; fall back to candidate_pool (manifest exports live there).
    candidates = [
        output_dir / filename,
        output_dir / "candidate_pool" / filename,
        output_dir / "final_selection" / filename,
    ]
    path = next((item for item in candidates if item.exists()), None)
    if path is None:
        return {"error": "Discovery file not available."}
    return FileResponse(path=str(path), filename=f"{discovery_id}_{filename}", media_type=media_type)


@app.post("/api/ai-ready/profile-inputs")
async def profile_ai_ready_inputs_api(body: dict[str, Any]):
    try:
        return await asyncio.to_thread(_run_ai_ready_input_profile, body)
    except Exception as exc:
        return {"error": _redact_secrets(str(exc))}


@app.post("/api/ai-ready/locate-inputs")
async def locate_ai_ready_inputs_api(body: dict[str, Any]):
    try:
        return await asyncio.to_thread(_run_ai_ready_input_locator, body)
    except Exception as exc:
        return {"error": _redact_secrets(str(exc))}


@app.post("/api/ai-ready/locate-agent-run")
async def locate_agent_run_ai_ready_inputs_api(body: dict[str, Any]):
    try:
        return await asyncio.to_thread(_run_agent_run_input_locator, body)
    except Exception as exc:
        return {"error": _redact_secrets(str(exc))}


@app.post("/api/ai-ready/build-from-agent-run")
async def build_ai_ready_from_agent_run_api(body: dict[str, Any]):
    try:
        return await asyncio.to_thread(_run_agent_run_ai_ready_build, body)
    except Exception as exc:
        return {"error": _redact_secrets(str(exc))}


@app.post("/api/ai-ready/mini-e2e")
async def validate_agent_run_mini_e2e_api(body: dict[str, Any]):
    try:
        return await asyncio.to_thread(_run_agent_run_mini_e2e, body)
    except Exception as exc:
        return {"error": _redact_secrets(str(exc))}


@app.post("/api/ai-ready/mini-e2e-batch")
async def validate_agent_runs_mini_e2e_batch_api(body: dict[str, Any]):
    try:
        return await asyncio.to_thread(_run_agent_runs_mini_e2e_batch, body)
    except Exception as exc:
        return {"error": _redact_secrets(str(exc))}


@app.post("/api/ai-ready/real-smoke")
async def run_ai_ready_real_smoke_api(body: dict[str, Any]):
    try:
        return await asyncio.to_thread(_run_ai_ready_real_smoke, body)
    except Exception as exc:
        return {"error": _redact_secrets(str(exc))}


@app.post("/api/ai-ready/repository-smoke")
async def run_repository_smoke_api(body: dict[str, Any]):
    try:
        return await asyncio.to_thread(_run_repository_smoke_web, body)
    except Exception as exc:
        return {"error": _redact_secrets(str(exc))}


@app.post("/api/ai-ready/refresh-iprox-index")
async def refresh_iprox_index_api(body: dict[str, Any]):
    try:
        return await asyncio.to_thread(_run_iprox_index_refresh_web, body)
    except Exception as exc:
        return {"error": _redact_secrets(str(exc))}


@app.post("/api/ai-ready/agent-harness")
async def run_agent_harness_api(body: dict[str, Any]):
    try:
        return await asyncio.to_thread(_run_agent_harness_web, body)
    except Exception as exc:
        return {"error": _redact_secrets(str(exc))}


@app.post("/api/ai-ready/make-dataset-recipe")
async def make_dataset_recipe_api(body: dict[str, Any]):
    try:
        return await asyncio.to_thread(_run_dataset_recipe_web, body)
    except Exception as exc:
        return {"error": _redact_secrets(str(exc))}


@app.post("/api/ai-ready/apply-curation-decisions")
async def apply_curation_decisions_api(body: dict[str, Any]):
    try:
        return await asyncio.to_thread(_run_apply_curation_decisions_web, body)
    except Exception as exc:
        return {"error": _redact_secrets(str(exc))}


@app.post("/api/ai-ready/model-loop")
async def run_dataset_model_loop_api(body: dict[str, Any]):
    try:
        return await asyncio.to_thread(_run_dataset_model_loop_web, body)
    except Exception as exc:
        return {"error": _redact_secrets(str(exc))}


@app.post("/api/ai-ready/model-informed-discovery-payload")
async def model_informed_discovery_payload_api(body: dict[str, Any]):
    try:
        return await asyncio.to_thread(_run_model_informed_discovery_payload, body)
    except Exception as exc:
        return {"error": _redact_secrets(str(exc))}


@app.post("/api/ai-ready/data-scientist-report")
async def make_data_scientist_agent_report_api(body: dict[str, Any]):
    try:
        return await asyncio.to_thread(_run_data_scientist_agent_report_web, body)
    except Exception as exc:
        return {"error": _redact_secrets(str(exc))}


@app.post("/api/ai-ready/guidance-alignment")
async def make_guidance_alignment_report_api(body: dict[str, Any]):
    try:
        return await asyncio.to_thread(_run_guidance_alignment_report_web, body)
    except Exception as exc:
        return {"error": _redact_secrets(str(exc))}


@app.post("/api/ai-ready/data-scientist-loop")
async def run_data_scientist_agent_loop_api(body: dict[str, Any]):
    try:
        return await asyncio.to_thread(_run_data_scientist_agent_loop_web, body)
    except Exception as exc:
        return {"error": _redact_secrets(str(exc))}


@app.post("/api/ai-ready/validate-build")
async def validate_ai_ready_build_api(body: dict[str, Any]):
    try:
        return await asyncio.to_thread(_run_ai_ready_validation, body)
    except Exception as exc:
        return {"error": _redact_secrets(str(exc))}


@app.get("/api/ai-ready/{build_id}")
async def get_ai_ready_build(build_id: str):
    output_dir = _safe_ai_ready_dir(build_id)
    if output_dir is None or not output_dir.exists():
        return {"error": "AI-ready build not found."}
    return _public_ai_ready_record(build_id, output_dir)


@app.get("/api/ai-ready/{build_id}/download")
async def download_ai_ready_build_file(build_id: str, file: str = "build_report_md"):
    output_dir = _safe_ai_ready_dir(build_id)
    if output_dir is None or not output_dir.exists():
        return {"error": "AI-ready build not found."}
    file_key = _clean_text(file)
    entry = _AI_READY_DOWNLOAD_FILES.get(file_key)
    if entry is None:
        return {"error": "AI-ready file not available."}
    filename, media_type = entry
    path = output_dir / filename
    if not path.exists():
        return {"error": "AI-ready file not available."}
    return FileResponse(path=str(path), filename=f"{build_id}_{Path(filename).name}", media_type=media_type)


class BatchFileReporter:
    def __init__(
        self,
        output_dir: Path,
        ui_language: str = "en",
        *,
        batch_id: str | None = None,
        item_index: int | None = None,
    ) -> None:
        self.path = output_dir / "logs" / "runtime.log"
        self.ui_language = _clean_ui_language(ui_language)
        self.batch_id = str(batch_id or "").strip() or None
        self.item_index = item_index if isinstance(item_index, int) else None
        self._lock = threading.Lock()
        self._progress_last_emit: dict[str, float] = {}

    def __call__(self, message: Any) -> None:
        if self.batch_id is not None and _batch_cancel_requested(self.batch_id):
            raise BatchCancelledError("Batch stop requested by user.")
        line_for_log: str
        if isinstance(message, dict):
            kind = str(message.get("kind") or "")
            if kind == "download_progress":
                label = _clean_text(message.get("label")) or "download"
                complete = bool(message.get("complete"))
                now = monotonic()
                last_emit = self._progress_last_emit.get(label)
                if not complete and last_emit is not None and now - last_emit < 0.5:
                    # Still update structured progress more often than log spam.
                    self._publish_download_progress(message, complete=complete)
                    return
                self._progress_last_emit[label] = now
                line_for_log = render_download_progress(message, width=16)
                if complete:
                    line_for_log = f"下载完成 {line_for_log}"
                self._publish_download_progress(message, complete=complete)
            elif kind == "activity_start":
                label = _clean_text(message.get("label")) or "处理中..."
                line_for_log = label
                self._publish_stage_message(label)
            elif kind == "activity_stop":
                line_for_log = _clean_text(message.get("message")) or "步骤完成"
                if line_for_log:
                    self._publish_stage_message(line_for_log)
            else:
                line_for_log = json.dumps(message, ensure_ascii=False, default=str)
                self._publish_stage_message(line_for_log)
        else:
            line_for_log = _redact_secrets(message)
            self._publish_stage_message(str(line_for_log))

        text = _localize_public_message(line_for_log, self.ui_language, level="info")
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(text + "\n")

    def _publish_download_progress(self, message: dict[str, Any], *, complete: bool) -> None:
        if self.batch_id is None or self.item_index is None:
            return
        try:
            downloaded = int(message.get("downloaded") or message.get("downloaded_bytes") or 0)
        except (TypeError, ValueError):
            downloaded = 0
        try:
            total_raw = message.get("total")
            if total_raw is None:
                total_raw = message.get("total_bytes")
            total = int(total_raw) if total_raw not in (None, "", 0, "0") else None
        except (TypeError, ValueError):
            total = None
        try:
            speed = message.get("speed_bps")
            speed_bps = float(speed) if speed is not None else None
        except (TypeError, ValueError):
            speed_bps = None
        try:
            eta = message.get("eta_seconds")
            eta_seconds = float(eta) if eta is not None else None
        except (TypeError, ValueError):
            eta_seconds = None
        percent = None
        if total and total > 0:
            percent = max(0.0, min(100.0, (downloaded / total) * 100.0))
            if complete:
                percent = 100.0
        label = _clean_text(message.get("label")) or "download"
        progress = {
            "stage": "download",
            "stage_label": "下载完成" if complete else "下载数据",
            "percent": percent,
            "message": f"{'下载完成' if complete else '正在下载'} {label}",
            "download": {
                "label": label,
                "downloaded_bytes": downloaded,
                "total_bytes": total,
                "speed_bps": speed_bps,
                "eta_seconds": eta_seconds,
                "complete": complete,
            },
            "updated_at": _now_iso(),
        }
        _update_batch_item(
            self.batch_id,
            self.item_index,
            progress=progress,
            write_manifest=bool(complete),
        )

    def _publish_stage_message(self, raw_message: str) -> None:
        if self.batch_id is None or self.item_index is None:
            return
        message = _clean_text(raw_message)
        if not message:
            return
        stage, stage_label = _infer_batch_stage_from_message(message)
        # Do not clobber a live download percent with a weaker stage-only update
        # unless the message clearly leaves download.
        with _batches_lock:
            batch = _batches.get(self.batch_id)
            if batch is not None:
                items = batch.get("items") or []
                if 0 <= self.item_index < len(items):
                    existing = dict(items[self.item_index].get("progress") or {})
                    if (
                        existing.get("stage") == "download"
                        and stage == "download"
                        and existing.get("download")
                        and not str(message).startswith("下载完成")
                    ):
                        return
        progress = {
            "stage": stage,
            "stage_label": stage_label,
            "percent": 100.0 if stage in {"completed", "export"} else None,
            "message": message[:500],
            "updated_at": _now_iso(),
        }
        _update_batch_item(self.batch_id, self.item_index, progress=progress)


def _update_batch_item(batch_id: str, index: int, **fields: Any) -> None:
    """Update one batch item.

    High-frequency progress ticks may pass write_manifest=False to avoid
    rewriting batch_manifest.json on every download_progress event.
    """
    write_manifest = bool(fields.pop("write_manifest", True))
    with _batches_lock:
        batch = _batches.get(batch_id)
        if batch is None:
            return
        items = batch.get("items") or []
        if index < 0 or index >= len(items):
            return
        items[index].update(fields)
        batch["updated_at"] = _now_iso()
        if write_manifest:
            _write_batch_manifest(batch)


class BatchCancelledError(RuntimeError):
    """Cooperative cancellation signal for batch items."""


def _batch_cancel_requested(batch_id: str) -> bool:
    with _batches_lock:
        batch = _batches.get(batch_id)
        return bool(batch and batch.get("cancel_requested"))


def _raise_if_batch_cancelled(batch_id: str) -> None:
    if _batch_cancel_requested(batch_id):
        raise BatchCancelledError("Batch stop requested by user.")


def _cancel_batch_item(batch_id: str, index: int, *, message: str = "Stopped by user.") -> dict[str, Any]:
    with _batches_lock:
        batch = _batches.get(batch_id)
        if batch is None:
            return {"status": "cancelled", "error": message}
        items = batch.get("items") or []
        if index < 0 or index >= len(items):
            return {"status": "cancelled", "error": message}
        item = items[index]
        if item.get("status") in {"completed", "failed", "needs_review", "blocked", "cancelled"}:
            return {"status": str(item.get("status")), "error": str(item.get("error") or "")}
        item.update(
            {
                "status": "cancelled",
                "finished_at": _now_iso(),
                "error": message,
                "progress": {
                    "stage": "cancelled",
                    "stage_label": "已停止",
                    "percent": 100.0,
                    "message": message,
                    "updated_at": _now_iso(),
                },
                "source_cleanup": _batch_item_source_cleanup(
                    requested=bool(batch.get("delete_source_files_after_success")),
                    output_dir=Path(item.get("output_dir", "")),
                    terminal_status="cancelled",
                ),
            }
        )
        batch["updated_at"] = _now_iso()
        _write_batch_manifest(batch)
    return {"status": "cancelled", "error": message}


def _write_batch_item_error(output_dir: Path, input_value: str, exc: BaseException) -> str:
    from agent.audit.review import build_task_state_snapshot, write_task_state
    from agent.errors import build_error_record, write_error_record
    from agent.execution.outputs import ExecutionFailureEvent
    from agent.input.normalizer import normalize_input

    output_dir.mkdir(parents=True, exist_ok=True)
    task_id = ""
    source_file = Path(input_value).name
    try:
        task = normalize_input(input_value)
        task_id = task.task_id
        source_file = task.file_name
    except Exception:
        task_id = f"batch-{safe_output_stem(input_value)}"
    error = build_error_record(exc, stage="planning", input_file=input_value)
    write_error_record(output_dir / "error.json", error)
    public_message = str(error.get("public_message") or error.get("message") or exc)
    write_task_state(
        output_dir / "task_state.json",
        build_task_state_snapshot(
            task_id=task_id,
            status="failed",
            stage="planning",
            source_file=source_file,
            project_accession=None,
            notes=[public_message],
        ),
    )
    if not (output_dir / "recovery_audit.json").exists():
        _write_recovery_audit_package(
            output_dir,
            SimpleNamespace(task_id=task_id, file_name=source_file),
            stage="planning",
            run_mode="batch",
            events=[
                ExecutionFailureEvent(
                    category=str(error.get("category") or "unknown"),
                    reason=str(error.get("technical_message") or public_message),
                    evidence_kind="exception",
                    marker=str(error.get("exception_type") or type(exc).__name__),
                )
            ],
            artifacts={
                "error_json": output_dir / "error.json",
                "task_state_json": output_dir / "task_state.json",
            },
        )
    return public_message


def _batch_item_recovery_fields(output_dir: Path) -> dict[str, Any]:
    try:
        from agent.agent_core.recovery_report import analyze_agent_recovery

        paths = analyze_agent_recovery(run_dir=output_dir, output_dir=output_dir)
        report_path = paths.get("agent_recovery_report_json")
        if report_path is None:
            return {}
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(report, dict):
        return {}
    workflow_outcome = _clean_text(report.get("workflow_outcome"))
    primary_issue = _clean_text(report.get("primary_issue"))
    fields: dict[str, Any] = {
        "workflow_outcome": workflow_outcome,
        "usable_partial_outputs": bool(report.get("usable_partial_outputs")),
        "recovery_primary_issue": primary_issue,
        "agent_recovery_report": str(output_dir / "agent_recovery_report.json"),
    }
    recommended = _clean_text(report.get("recommended_next_step"))
    if recommended:
        fields["recovery_recommended_next_step"] = recommended
    return {key: value for key, value in fields.items() if value not in {"", None}}


def _primary_project_error(result: Any) -> str:
    resolution = getattr(result, "resolution", None)
    primary = getattr(resolution, "primary_project", None)
    if primary is None:
        return "No exact PRIDE project match found."
    match_type = str(getattr(primary, "match_type", "") or "")
    try:
        match_score = int(getattr(primary, "match_score", 0) or 0)
    except (TypeError, ValueError):
        match_score = 0
    safe_match_types = {"exact", "stem", "known_project_local_source"}
    if match_type not in safe_match_types or match_score < 90:
        return (
            f"Non-exact PRIDE project match: {getattr(primary, 'project_accession', 'unknown')}, "
            f"match_type={match_type}, score={match_score}, matched_file={getattr(primary, 'matched_file', '')}"
        )
    if bool(getattr(resolution, "needs_review", False)):
        return f"Ambiguous PRIDE project match: {getattr(resolution, 'resolution_reason', 'manual review required')}"
    return ""


def _path_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve(strict=False).relative_to(parent.resolve(strict=False))
        return True
    except ValueError:
        return False


def _cleanup_batch_instrument_probe_files(output_dir: Path, result: Any, prepared_path: Path | None = None) -> None:
    if str(os.getenv("AGENT_BATCH_KEEP_INSTRUMENT_PROBE_FILES", "")).strip().lower() in {"1", "true", "yes", "on"}:
        return
    assets_dir = output_dir / "assets"
    allowed_roots = [assets_dir / "downloads", assets_dir / "prepared"]
    asset = getattr(result, "asset", None)
    candidates: list[Path] = []
    for raw_path in (
        prepared_path,
        getattr(asset, "prepared_path", None),
        getattr(asset, "local_path", None),
    ):
        if raw_path:
            candidates.append(Path(raw_path))
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        if not any(_path_within(candidate, root) for root in allowed_roots):
            continue
        try:
            if candidate.is_file():
                candidate.unlink()
            elif candidate.is_dir():
                shutil.rmtree(candidate)
        except OSError:
            pass


def _batch_item_source_cleanup(
    *,
    requested: bool,
    output_dir: Path,
    terminal_status: str,
) -> dict[str, Any]:
    if not requested:
        return {
            "requested": False,
            "status": "not_requested",
            "released_bytes": 0,
            "removed_paths": [],
            "errors": [],
        }
    if terminal_status != "completed":
        return {
            "requested": True,
            "status": "retained",
            "reason": f"Source files retained because item status is {terminal_status}.",
            "released_bytes": 0,
            "removed_paths": [],
            "errors": [],
            "finished_at": _now_iso(),
        }
    started_at = _now_iso()
    receipt = clean_item_source_assets(output_dir)
    return {
        "requested": True,
        "started_at": started_at,
        "finished_at": _now_iso(),
        **receipt,
    }


def _run_parameter_batch_item(batch_id: str, index: int) -> dict[str, Any]:
    with _batches_lock:
        batch = _batches[batch_id]
        item = dict(batch["items"][index])
        cancelled_before_start = bool(batch.get("cancel_requested")) or item.get("status") == "cancelled"
        llm_config = dict(batch["llm_config"])
        prefer_project_fasta = bool(batch.get("prefer_project_fasta"))
        ui_language = _clean_ui_language(batch.get("ui_language"))
        repository = _clean_repository(batch.get("repository"))
        run_mode = _clean_batch_run_mode(batch.get("run_mode"))
        delete_source_files_after_success = bool(
            batch.get("delete_source_files_after_success")
        )
    if cancelled_before_start:
        return _cancel_batch_item(batch_id, index)

    input_value = str(item["input"])
    discovery_context = dict(item.get("discovery_context") or {})
    output_dir = Path(item["output_dir"])
    _update_batch_item(batch_id, index, status="running", started_at=_now_iso(), error="")
    _append_batch_event(batch_id, "info", f"Started {input_value}", item_index=index)
    service = None
    try:
        _raise_if_batch_cancelled(batch_id)
        from agent.input.normalizer import normalize_input
        from agent.orchestrator.pipeline import AgentService

        reporter = BatchFileReporter(output_dir, ui_language=ui_language, batch_id=batch_id, item_index=index)
        service = AgentService(reporter=reporter, llm_reasoner=_task_llm_reasoner(llm_config))
        task = normalize_input(input_value)
        local_source = _known_local_source_from_input(input_value)
        if run_mode in {_RUN_MODE_PREPARE, _RUN_MODE_FULL}:
            _raise_if_batch_cancelled(batch_id)
            if local_source:
                _append_batch_event(
                    batch_id,
                    "info",
                    f"{input_value} using local cached source {local_source['project_accession']}.",
                    item_index=index,
                )
                bundle, result, prepared_path = service.prepare_known_project_local_msdt_docker_input(
                    task=task,
                    source_data_path=local_source["source_path"],
                    project_accession=local_source["project_accession"],
                    output_dir=output_dir,
                    repository="pride",
                    matched_file=local_source["matched_file"],
                    prefer_project_fasta=prefer_project_fasta,
                    context_dir=local_source.get("context_dir"),
                )
            else:
                bundle, result, prepared_path = service.prepare_repository_msdt_docker_input(
                    task=task,
                    output_dir=output_dir,
                    repository=repository,
                    prefer_project_fasta=prefer_project_fasta,
                )
            _raise_if_batch_cancelled(batch_id)
            _write_agent_audit_package(output_dir, result, plan=bundle.plan, report=reporter)
            _write_parameter_audit_files(output_dir, batch_id, index, input_value, result)
            project_error = _primary_project_error(result)
            if project_error:
                raise RuntimeError(project_error)
            if run_mode == _RUN_MODE_FULL:
                from agent.audit.review import build_task_state_snapshot, write_task_state
                from agent.execution.outputs import execution_failure_events, execution_failure_reasons
                from agent.msdt_converter.docker_runner import DockerMSDTConverterRunner

                _append_batch_event(batch_id, "info", f"{input_value} running MSDT-Converter Docker.", item_index=index)
                docker_runner = DockerMSDTConverterRunner(
                    image="guomics2017/msdt-converter:v1.3",
                    report=reporter,
                    cancel_requested=lambda: _batch_cancel_requested(batch_id),
                )
                docker_result = docker_runner.run(bundle)
                _raise_if_batch_cancelled(batch_id)
                failure_reasons = execution_failure_reasons(
                    bundle.plan,
                    docker_result.returncode,
                    docker_result.stdout,
                    docker_result.stderr,
                )
                if failure_reasons:
                    failure_events = execution_failure_events(
                        bundle.plan,
                        docker_result.returncode,
                        docker_result.stdout,
                        docker_result.stderr,
                    )
                    _write_recovery_audit_package(
                        output_dir,
                        task,
                        stage="execution",
                        run_mode=_RUN_MODE_FULL,
                        events=failure_events,
                        result=result,
                        plan=bundle.plan,
                        artifacts={
                            "task_state_json": output_dir / "task_state.json",
                            "runtime_log": output_dir / "logs" / "runtime.log",
                        },
                        report=reporter,
                    )
                    raise RuntimeError("; ".join(failure_reasons))
                project_accession = result.resolution.primary_project.project_accession if result.resolution.primary_project else None
                write_task_state(
                    output_dir / "task_state.json",
                    build_task_state_snapshot(
                        task_id=task.task_id,
                        status="completed",
                        stage="execution",
                        source_file=task.file_name,
                        project_accession=project_accession,
                        notes=[],
                    ),
                )
                _zip_output_dir(output_dir, report=reporter)
            else:
                from agent.audit.review import build_task_state_snapshot, write_task_state

                project_accession = result.resolution.primary_project.project_accession if result.resolution.primary_project else None
                write_task_state(
                    output_dir / "task_state.json",
                    build_task_state_snapshot(
                        task_id=task.task_id,
                        status="completed",
                        stage="packaging",
                        source_file=task.file_name,
                        project_accession=project_accession,
                        notes=["Prepare input package mode completed; Docker execution was not run."],
                    ),
                )
            source_cleanup = _batch_item_source_cleanup(
                requested=delete_source_files_after_success,
                output_dir=output_dir,
                terminal_status="completed",
            )
            _update_batch_item(
                batch_id,
                index,
                status="completed",
                finished_at=_now_iso(),
                error="",
                run_mode=run_mode,
                source_cleanup=source_cleanup,
            )
            _append_batch_event(batch_id, "info", f"{input_value} {run_mode} completed", item_index=index)
            return {"status": "completed", "error": ""}

        discovery_project = _clean_text(discovery_context.get("project_accession"))
        _raise_if_batch_cancelled(batch_id)
        if local_source:
            _append_batch_event(
                batch_id,
                "info",
                f"{input_value} using local cached source {local_source['project_accession']}.",
                item_index=index,
            )
            result = service.plan_dda_run_from_known_project_local_source(
                task=task,
                source_data_path=local_source["source_path"],
                project_accession=local_source["project_accession"],
                output_dir=output_dir,
                repository="pride",
                matched_file=local_source["matched_file"],
                prefer_project_fasta=prefer_project_fasta,
                context_dir=local_source.get("context_dir"),
            )
        elif discovery_project and callable(getattr(service, "plan_dda_run_from_known_project", None)):
            _append_batch_event(
                batch_id,
                "info",
                f"{input_value} using discovery handoff project {discovery_project}.",
                item_index=index,
            )
            result = service.plan_dda_run_from_known_project(
                task=task,
                project_accession=discovery_project,
                output_dir=output_dir,
                repository=repository,
                matched_file=_clean_text(discovery_context.get("file_name")) or task.file_name,
                prefer_project_fasta=prefer_project_fasta,
            )
        else:
            result = service.plan_dda_run_from_repository(
                task=task,
                output_dir=output_dir,
                repository=repository,
                prefer_project_fasta=prefer_project_fasta,
            )
        if (
            bool(getattr(result.plan, "needs_review", False))
            and callable(getattr(service, "_can_retry_with_mzml_instrument", None))
            and service._can_retry_with_mzml_instrument(result.plan)
        ):
            prepared_path: Path | None = None
            _append_batch_event(
                batch_id,
                "warning",
                f"{input_value} needs file-level instrument; downloading/converting mzML probe.",
                item_index=index,
            )
            try:
                prepared_path = service.prepare_local_asset(result.asset) if local_source else service.prepare_asset(result.asset)
                result = service.replan_with_mzml_instrument(
                    result,
                    prepared_path,
                    task,
                    output_dir,
                    prefer_project_fasta=prefer_project_fasta,
                )
                _append_batch_event(
                    batch_id,
                    "info",
                    f"{input_value} mzML instrument probe completed.",
                    item_index=index,
                )
            except Exception as probe_exc:
                reporter(f"mzML instrument probe failed; keeping needs_review. Reason: {probe_exc}")
                _append_batch_event(
                    batch_id,
                    "warning",
                    f"{input_value} mzML instrument probe failed: {probe_exc}",
                    item_index=index,
                )
            finally:
                _cleanup_batch_instrument_probe_files(output_dir, result, prepared_path)
        _raise_if_batch_cancelled(batch_id)
        service.write_task_bundle(output_dir, result.resolution, result.context, result.attributes, result.plan, asset=result.asset)
        _write_agent_audit_package(output_dir, result, report=reporter)
        _write_parameter_audit_files(output_dir, batch_id, index, input_value, result)
        project_error = _primary_project_error(result)
        if project_error:
            raise RuntimeError(project_error)
        status = "needs_review" if bool(getattr(result.plan, "needs_review", False)) else "completed"
        error = "; ".join(str(issue) for issue in getattr(result.plan, "blocking_issues", []) or [])
        source_cleanup = _batch_item_source_cleanup(
            requested=delete_source_files_after_success,
            output_dir=output_dir,
            terminal_status=status,
        )
        _update_batch_item(
            batch_id,
            index,
            status=status,
            finished_at=_now_iso(),
            error=error,
            source_cleanup=source_cleanup,
        )
        level = "warning" if status == "needs_review" else "info"
        _append_batch_event(batch_id, level, f"{input_value} {status}", item_index=index)
        return {"status": status, "error": error}
    except BatchCancelledError as exc:
        message = str(exc) or "Stopped by user."
        result = _cancel_batch_item(batch_id, index, message=message)
        _append_batch_event(batch_id, "warning", f"{input_value} stopped by user", item_index=index)
        return result
    except Exception as exc:
        from agent.errors import build_error_record, public_error_summary

        error_record = build_error_record(exc, stage="batch_item", input_file=input_value)
        classified = public_error_summary(error_record)
        message = _write_batch_item_error(output_dir, input_value, exc)
        public_message = str(classified.get("public_message") or message or exc)
        error_summary = _summarize_batch_error(public_message) or public_message
        recovery_fields = _batch_item_recovery_fields(output_dir)
        source_cleanup = _batch_item_source_cleanup(
            requested=delete_source_files_after_success,
            output_dir=output_dir,
            terminal_status="failed",
        )
        _update_batch_item(
            batch_id,
            index,
            status="failed",
            finished_at=_now_iso(),
            error=public_message,
            error_summary=error_summary,
            progress={
                "stage": str(classified.get("stage") or "failed"),
                "stage_label": "失败",
                "percent": 100.0,
                "message": error_summary,
                "failed_stage": str(classified.get("stage") or "failed"),
                "updated_at": _now_iso(),
            },
            source_cleanup=source_cleanup,
            **recovery_fields,
        )
        workflow_outcome = recovery_fields.get("workflow_outcome")
        if workflow_outcome:
            _append_batch_event(batch_id, "warning", f"{input_value} recovery outcome: {workflow_outcome}", item_index=index)
        _append_batch_event(batch_id, "error", f"{input_value} failed: {public_message}", item_index=index)
        return {
            "status": "failed",
            "error": public_message,
            "error_summary": error_summary,
            **recovery_fields,
        }
    finally:
        if service is not None:
            pride_client = getattr(service, "pride_client", None)
            close = getattr(pride_client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass


def _run_parameter_batch(batch_id: str) -> None:
    with _batches_lock:
        batch = _batches.get(batch_id)
        if batch is None:
            return
        if batch.get("cancel_requested"):
            batch["status"] = "cancelled"
            batch["finished_at"] = _now_iso()
            batch["updated_at"] = batch["finished_at"]
            _append_batch_event_unlocked(batch, "warning", "Batch stopped before execution started")
            _write_batch_manifest(batch)
            return
        batch["status"] = "running"
        batch["started_at"] = _now_iso()
        batch["updated_at"] = batch["started_at"]
        jobs = int(batch.get("jobs") or 1)
        item_count = len(batch.get("items") or [])
        _append_batch_event_unlocked(batch, "info", f"Batch started; {item_count} files; jobs={jobs}")
        _write_batch_manifest(batch)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            futures = {pool.submit(_run_parameter_batch_item, batch_id, index): index for index in range(item_count)}
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    index = futures[future]
                    with _batches_lock:
                        item = dict(_batches.get(batch_id, {}).get("items", [{}])[index])
                    from agent.errors import build_error_record, public_error_summary

                    input_value = str(item.get("input", ""))
                    error_record = build_error_record(exc, stage="batch_item", input_file=input_value)
                    classified = public_error_summary(error_record)
                    message = _write_batch_item_error(Path(item.get("output_dir", "")), input_value, exc)
                    public_message = str(classified.get("public_message") or message or exc)
                    error_summary = _summarize_batch_error(public_message) or public_message
                    recovery_fields = _batch_item_recovery_fields(Path(item.get("output_dir", "")))
                    _update_batch_item(
                        batch_id,
                        index,
                        status="failed",
                        finished_at=_now_iso(),
                        error=public_message,
                        error_summary=error_summary,
                        progress={
                            "stage": str(classified.get("stage") or "failed"),
                            "stage_label": "失败",
                            "percent": 100.0,
                            "message": error_summary,
                            "failed_stage": str(classified.get("stage") or "failed"),
                            "updated_at": _now_iso(),
                        },
                        **recovery_fields,
                    )

        from scripts.export_benchmark_excel import ResultSource, summarize_source, write_xlsx

        with _batches_lock:
            batch = _batches[batch_id]
            sources = [ResultSource(label=str(item["input"]), path=Path(item["output_dir"])) for item in batch["items"]]
            output_dir = Path(batch["output_dir"])
        rows = [summarize_source(source) for source in sources]
        write_xlsx(rows, output_dir / _BATCH_EXCEL_FILE)
        _append_batch_event(batch_id, "info", f"Excel report written: {_BATCH_EXCEL_FILE}")

        with _batches_lock:
            batch = _batches[batch_id]
            batch["status"] = "cancelled" if batch.get("cancel_requested") else "completed"
            batch["finished_at"] = _now_iso()
            batch["updated_at"] = batch["finished_at"]
            batch["excel_path"] = str(Path(batch["output_dir"]) / _BATCH_EXCEL_FILE)
            _append_batch_event_unlocked(
                batch,
                "warning" if batch["status"] == "cancelled" else "info",
                "Batch stopped; partial Excel report preserved"
                if batch["status"] == "cancelled"
                else "Batch completed",
            )
            _write_batch_manifest(batch)
    except Exception as exc:
        with _batches_lock:
            batch = _batches.get(batch_id)
            if batch is None:
                return
            batch["status"] = "failed"
            batch["finished_at"] = _now_iso()
            batch["updated_at"] = batch["finished_at"]
            batch.setdefault("errors", []).append(_redact_secrets(str(exc)))
            _append_batch_event_unlocked(batch, "error", f"Batch failed: {exc}")
            _write_batch_manifest(batch)


def _start_parameter_batch_thread(batch_id: str) -> None:
    thread = threading.Thread(target=_run_parameter_batch, args=(batch_id,), daemon=True)
    thread.start()


@app.post("/api/batches/parameters")
async def create_parameter_batch(body: dict[str, Any]):
    inputs = _clean_batch_inputs(body)
    if not inputs:
        return {"error": "Please enter at least one PRIDE file name."}
    max_items = _max_batch_items()
    if len(inputs) > max_items:
        return {"error": f"批量输入过多：{len(inputs)} 条，当前上限 {max_items}（可用环境变量 AGENT_MAX_BATCH_ITEMS 调整；0 表示接近不限制）。"}
    submitter = _clean_submitter(body.get("submitter"))
    ui_language = _clean_ui_language(body.get("ui_language"))
    repository = _clean_repository(body.get("repository"))
    run_mode = _clean_batch_run_mode(body.get("run_mode"))
    if run_mode == _RUN_MODE_FULL and not _full_workflow_enabled():
        return {
            "error": (
                "Full batch workflow is not enabled on this server. "
                "Set AGENT_WEB_FULL_WORKFLOW_ENABLED=true and restart the service; "
                "the request was not downgraded or started."
            )
        }
    resource_policy = _clean_resource_policy(body.get("resource_policy"))
    fasta_preference = _clean_text(body.get("fasta_preference")).lower()
    prefer_project_fasta = fasta_preference == "project" or body.get("prefer_project_fasta") is True
    reviewed_fasta_path, reviewed_fasta_url = _clean_reviewed_fasta(body.get("reviewed_fasta"))
    explicit_reviewed_fasta_path = _clean_text(body.get("reviewed_fasta_path"))
    explicit_reviewed_fasta_url = _clean_text(body.get("reviewed_fasta_url"))
    if explicit_reviewed_fasta_path:
        reviewed_fasta_path = explicit_reviewed_fasta_path
        reviewed_fasta_url = None
    if explicit_reviewed_fasta_url:
        reviewed_fasta_url = explicit_reviewed_fasta_url
        reviewed_fasta_path = None
    reviewed_fasta_name = _clean_text(body.get("reviewed_fasta_name")) or None
    llm_config = body.get("llm_config", {})
    if not isinstance(llm_config, dict):
        llm_config = {}
    config, config_error = _build_llm_config(llm_config)
    if config_error or config is None:
        return {"error": config_error}
    ok, message = await _run_llm_check(config)
    if not ok:
        return {"error": message}

    batch_id = uuid.uuid4().hex[:12]
    batch_dir = _batch_dir(batch_id)
    jobs = _batch_jobs(body.get("jobs"), len(inputs))
    input_records = _clean_batch_input_records(body, inputs)
    delete_source_files_after_success = body.get(
        "delete_source_files_after_success"
    ) is True
    first_context = next((record for record in input_records if record), {}) or {}
    source_discovery_job_id = _clean_text(
        body.get("source_discovery_job_id")
        or first_context.get("source_discovery_job_id")
    )
    source_discovery_id = _clean_text(
        body.get("source_discovery_id")
        or first_context.get("source_discovery_id")
    )
    try:
        source_batch_index = int(
            body.get("source_batch_index")
            or first_context.get("source_batch_index")
            or 0
        )
    except (TypeError, ValueError):
        source_batch_index = 0
    items = [
        {
            "index": index,
            "input": input_value,
            "status": "queued",
            "output_dir": str(_batch_item_dir(batch_dir, index, input_value)),
            "error": "",
            "source_cleanup": {
                "requested": delete_source_files_after_success,
                "status": "pending" if delete_source_files_after_success else "not_requested",
                "released_bytes": 0,
                "removed_paths": [],
                "errors": [],
            },
            **({"discovery_context": input_records[index - 1]} if input_records[index - 1] else {}),
        }
        for index, input_value in enumerate(inputs, start=1)
    ]
    batch = {
        "batch_id": batch_id,
        "status": "queued",
        "submitter": submitter,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "jobs": jobs,
        "ui_language": ui_language,
        "repository": repository,
        "run_mode": run_mode,
        "requested_run_mode": run_mode,
        "resource_policy": resource_policy,
        "prefer_project_fasta": prefer_project_fasta,
        "delete_source_files_after_success": delete_source_files_after_success,
        "source_discovery_job_id": source_discovery_job_id or None,
        "source_discovery_id": source_discovery_id or None,
        "source_batch_index": source_batch_index or None,
        "output_dir": str(batch_dir),
        "excel_path": str(batch_dir / _BATCH_EXCEL_FILE),
        "items": items,
        "errors": [],
        "events": [
            {
                "ts": _now_iso(),
                "level": "info",
                "message": f"Batch created with {len(items)} files; jobs={jobs}; submitter={submitter}",
            }
        ],
        "llm_config": dict(config),
    }
    with _batches_lock:
        batch_dir.mkdir(parents=True, exist_ok=True)
        _batches[batch_id] = batch
        _write_batch_manifest(batch)
    _start_parameter_batch_thread(batch_id)
    return _public_batch_record(batch)


@app.post("/api/batches/{batch_id}/cancel")
async def cancel_parameter_batch(batch_id: str):
    safe_id = safe_output_stem(batch_id)
    if safe_id != batch_id:
        return {"error": "Batch not found."}
    with _batches_lock:
        batch = _batches.get(batch_id)
        if batch is None:
            loaded = _load_batch_from_disk(batch_id)
            if loaded is None:
                return {"error": "Batch not found."}
            batch = loaded
            _batches[batch_id] = batch
        if batch.get("status") in _TERMINAL_STATUSES:
            return _public_batch_record(batch)
        now = _now_iso()
        batch["cancel_requested"] = True
        batch["cancel_requested_at"] = batch.get("cancel_requested_at") or now
        for item in batch.get("items") or []:
            if item.get("status") not in {"queued", "pending", ""}:
                continue
            item.update(
                {
                    "status": "cancelled",
                    "finished_at": now,
                    "error": "Stopped by user before processing started.",
                    "progress": {
                        "stage": "cancelled",
                        "stage_label": "已停止",
                        "percent": 100.0,
                        "message": "Stopped by user before processing started.",
                        "updated_at": now,
                    },
                    "source_cleanup": {
                        "requested": bool(batch.get("delete_source_files_after_success")),
                        "status": "retained",
                        "reason": "Source files retained because the batch was stopped.",
                        "released_bytes": 0,
                        "removed_paths": [],
                        "errors": [],
                        "finished_at": now,
                    },
                }
            )
        running_items = sum(
            1 for item in batch.get("items") or [] if item.get("status") == "running"
        )
        if running_items == 0:
            batch["status"] = "cancelled"
            batch["finished_at"] = now
        batch["updated_at"] = now
        _append_batch_event_unlocked(
            batch,
            "warning",
            (
                f"Stop requested by user; waiting for {running_items} running item(s) "
                "to reach a safe cancellation point."
                if running_items
                else "Batch stopped by user before any remaining item started."
            ),
        )
        _write_batch_manifest(batch)
        return _public_batch_record(batch)


@app.get("/api/batches/{batch_id}")
async def get_parameter_batch(batch_id: str):
    safe_id = safe_output_stem(batch_id)
    if safe_id != batch_id:
        return {"error": "Batch not found."}
    with _batches_lock:
        batch = _batches.get(batch_id)
        if batch is not None:
            return _public_batch_record(batch)
    batch = _load_batch_from_disk(batch_id)
    if batch is None:
        return {"error": "Batch not found."}
    return _public_batch_record(batch)


@app.get("/api/batches/{batch_id}/download")
async def download_parameter_batch(batch_id: str):
    safe_id = safe_output_stem(batch_id)
    if safe_id != batch_id:
        return {"error": "Batch not found."}
    batch = _load_batch_from_disk(batch_id)
    if batch is None:
        with _batches_lock:
            batch = _batches.get(batch_id)
    if batch is None:
        return {"error": "Batch not found."}
    excel_path = Path(batch.get("excel_path") or Path(batch.get("output_dir", "")) / _BATCH_EXCEL_FILE)
    if not excel_path.exists():
        return {"error": "Excel report is not ready."}
    return FileResponse(
        path=str(excel_path),
        filename=f"{batch_id}_benchmark_results.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.get("/api/batches/{batch_id}/audit.zip")
async def download_parameter_batch_audit(batch_id: str):
    safe_id = safe_output_stem(batch_id)
    if safe_id != batch_id:
        return {"error": "Batch not found."}
    batch = _load_batch_from_disk(batch_id)
    if batch is None:
        with _batches_lock:
            batch = _batches.get(batch_id)
    if batch is None:
        return {"error": "Batch not found."}
    zip_path = _ensure_batch_audit_zip(batch)
    if zip_path is None or not zip_path.exists():
        return {"error": "Audit package is not ready."}
    return FileResponse(
        path=str(zip_path),
        filename=f"{batch_id}_audit.zip",
        media_type="application/zip",
    )


@app.post("/api/preflight")
async def preflight(body: dict[str, Any]):
    inputs = _clean_batch_inputs(body)
    is_batch_request = bool(inputs)
    if not inputs:
        single = _clean_text(body.get("input_value"))
        if single:
            inputs = [single]
    if not inputs:
        return {"status": "blocked", "blocking_issues": ["No input files were provided."], "checks": []}
    run_mode = (
        _clean_batch_run_mode(body.get("run_mode"))
        if is_batch_request
        else _clean_run_mode(body.get("run_mode"))
    )
    result = run_preflight(
        inputs=inputs,
        run_mode=run_mode,
        repository=_clean_repository(body.get("repository"), default="auto"),
        output_root=_runs_dir,
        resource_policy=_clean_resource_policy(body.get("resource_policy")),
    )
    if is_batch_request and run_mode == _RUN_MODE_FULL and not _full_workflow_enabled():
        issue = "Full batch workflow is disabled on this server."
        result["status"] = "blocked"
        result.setdefault("blocking_issues", []).append(issue)
        result.setdefault("checks", []).append(
            {"name": "full_workflow", "status": "blocked", "message": issue}
        )
    return result


@app.post("/api/tasks")
async def create_task(body: dict[str, Any]):
    try:
        return await _create_task_inner(body)
    except Exception as exc:
        return {"error": f"创建任务失败：{exc}"}


async def _create_task_inner(body: dict[str, Any]):
    input_value = _clean_text(body.get("input_value"))
    if not input_value:
        return {"error": "请输入 PRIDE 文件名"}
    submitter = _clean_submitter(body.get("submitter"))
    fasta_preference = _clean_text(body.get("fasta_preference")).lower()
    prefer_project_fasta = fasta_preference == "project" or body.get("prefer_project_fasta") is True
    run_mode = _clean_run_mode(body.get("run_mode"))
    resource_policy = _clean_resource_policy(body.get("resource_policy"))
    ui_language = _clean_ui_language(body.get("ui_language"))
    repository = _clean_repository(body.get("repository"))

    # 应用用户填写的 LLM 配置
    llm_config = body.get("llm_config", {})
    if not isinstance(llm_config, dict):
        llm_config = {}
    config, config_error = _build_llm_config(llm_config)
    if config_error or config is None:
        return {"error": config_error}

    ok, message = await _run_llm_check(config)
    if not ok:
        return {"error": message}

    task_id = uuid.uuid4().hex[:12]
    project_key = _project_key_for_input(input_value)
    reviewed_fasta_path, reviewed_fasta_url = _clean_reviewed_fasta(body.get("reviewed_fasta"))
    explicit_reviewed_fasta_path = _clean_text(body.get("reviewed_fasta_path"))
    explicit_reviewed_fasta_url = _clean_text(body.get("reviewed_fasta_url"))
    if explicit_reviewed_fasta_path:
        reviewed_fasta_path = explicit_reviewed_fasta_path
        reviewed_fasta_url = None
    if explicit_reviewed_fasta_url:
        reviewed_fasta_url = explicit_reviewed_fasta_url
        reviewed_fasta_path = None
    reviewed_fasta_name = _clean_text(body.get("reviewed_fasta_name")) or None

    with _tasks_lock:
        output_dir = _next_output_dir_locked(project_key, task_id)
        _tasks[task_id] = {
            "task_id": task_id,
            "input_value": input_value,
            "project_key": project_key,
            "submitter": submitter,
            "output_dir": str(output_dir),
            "status": "queued",
            "created_at": _now_iso(),
            "logs": deque(maxlen=5000),
            "step": 0,
            "total_steps": 5,
            "blocking_issues": [],
            "prefer_project_fasta": prefer_project_fasta,
            "reviewed_fasta_path": reviewed_fasta_path,
            "reviewed_fasta_url": reviewed_fasta_url,
            "reviewed_fasta_name": reviewed_fasta_name,
            "run_mode": run_mode,
            "resource_policy": resource_policy,
            "ui_language": ui_language,
            "repository": repository,
            "llm_config": dict(config),
        }
        queue_state = _queue_state_locked(task_id)
        _tasks[task_id]["logs"].append(
            {
                "type": "log",
                "ts": _now_time(),
                "level": "info",
                "message": _localize_public_message(
                    f"任务已进入队列，当前位置 {queue_state['queue_position']}/{queue_state['queue_length']}。",
                    ui_language,
                    level="info",
                ),
            }
        )
    _write_task_history(task_id)
    _start_ready_queued_tasks()
    with _tasks_lock:
        status = _tasks[task_id]["status"]
        queue_state = _queue_state_locked(task_id)
    return {
        "task_id": task_id,
        "submitter": submitter,
        "output_dir": str(output_dir),
        "status": status,
        "run_mode": run_mode,
        "resource_policy": resource_policy,
        "ui_language": ui_language,
        "repository": repository,
        **queue_state,
    }


# ── WebSocket 实时日志 ────────────────────────────────────────────
@app.websocket("/ws/tasks/{task_id}")
async def task_ws(websocket: WebSocket, task_id: str):
    await websocket.accept()

    with _tasks_lock:
        task = _tasks.get(task_id)
    if task is None:
        await websocket.send_json({"type": "error", "message": f"任务 {task_id} 不存在"})
        await websocket.close()
        return

    for log in task["logs"]:
        await websocket.send_json(log)

    if task["status"] in _TERMINAL_STATUSES:
        await websocket.send_json({"type": "done", "status": task["status"]})
        await websocket.close()
        return

    sent = len(task["logs"])
    try:
        while task["status"] == "queued":
            with _tasks_lock:
                queue_message = {"type": "queue", "status": "queued", **_queue_state_locked(task_id)}
                new_logs = list(task["logs"])[sent:]
            await websocket.send_json(queue_message)
            for log in new_logs:
                await websocket.send_json(log)
            sent += len(new_logs)
            await asyncio.sleep(1.0)
        while task["status"] == "running":
            await asyncio.sleep(0.3)
            with _tasks_lock:
                new_logs = list(task["logs"])[sent:]
            for log in new_logs:
                await websocket.send_json(log)
            sent += len(new_logs)
        with _tasks_lock:
            new_logs = list(task["logs"])[sent:]
            final_status = task["status"]
        for log in new_logs:
            await websocket.send_json(log)
        await websocket.send_json({"type": "done", "status": final_status})
    except WebSocketDisconnect:
        pass


# ── 日志工具 ──────────────────────────────────────────────────────
def _emit(task_id: str, msg_type: str, data: Any = None, **kwargs):
    task = _tasks.get(task_id)
    if task is None:
        return
    if "message" in kwargs:
        kwargs["message"] = _localize_public_message(
            _strip_ansi(kwargs["message"]).strip(),
            _clean_ui_language(task.get("ui_language")),
            level=str(kwargs.get("level") or msg_type),
        )
    entry = {"type": msg_type, "ts": _now_time(), **kwargs}
    if data is not None:
        entry["data"] = data
    task["logs"].append(entry)


def _log(task_id: str, level: str, message: str, **kwargs):
    if not _strip_ansi(message).strip():
        return
    _emit(task_id, "log", message=message, level=level, **kwargs)


def _step(task_id: str, step: int, label: str):
    task = _tasks.get(task_id)
    if task:
        task["step"] = step
    _emit(task_id, "step", message=label, step=step)


def _notify_agent_audit_ready(task_id: str) -> None:
    """Notify the frontend that agent audit files are available for real-time visualization."""
    _emit(task_id, "agent_audit_ready")


# ── Web Reporter ──────────────────────────────────────────────────
class WebReporter:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self._progress_last_emit: dict[str, float] = {}

    def __call__(self, message):
        if isinstance(message, dict):
            kind = message.get("kind", "")
            if kind == "download_progress":
                label = _clean_text(message.get("label")) or "download"
                complete = bool(message.get("complete"))
                now = monotonic()
                last_emit = self._progress_last_emit.get(label)
                if not complete and last_emit is not None and now - last_emit < 0.5:
                    return
                self._progress_last_emit[label] = now
                msg = render_download_progress(message, width=16)
                if complete:
                    msg = f"下载完成 {msg}"
                _log(self.task_id, "info", msg, key=f"download:{label}", replace=True)
            elif kind == "activity_start":
                _log(self.task_id, "info", message.get("label", "处理中..."))
            elif kind == "activity_stop":
                if message.get("message"):
                    _log(self.task_id, "info", message["message"])
            else:
                _log(self.task_id, "info", json.dumps(message, ensure_ascii=False))
        else:
            text = str(message)
            level = "info"
            if "LLM" in text or "大模型" in text or "streaming" in text.lower():
                level = "llm"
            elif "[调试]" in text:
                level = "debug"
            elif "错误" in text or "失败" in text or "error" in text.lower():
                level = "error"
            elif any(x in text for x in ["[1/", "[2/", "[3/", "[4/", "[5/"]):
                level = "step"
            _log(self.task_id, level, text)


# ── stderr 捕获（LLM 流式输出写到 stderr） ────────────────────────
class StderrCapture:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self._original = None
        self._buffer = ""

    def __enter__(self):
        import sys
        self._original = sys.stderr
        sys.stderr = self
        return self

    def __exit__(self, *args):
        import sys
        sys.stderr = self._original
        self._flush()

    def write(self, text: str) -> int:
        if not text:
            return 0
        self._buffer += text
        if "\n" in self._buffer:
            lines = self._buffer.split("\n")
            for line in lines[:-1]:
                line = _strip_ansi(line).strip()
                if line:
                    _log(self.task_id, "llm", line)
            self._buffer = lines[-1]
        return len(text)

    def flush(self):
        return None

    def _flush(self):
        message = _strip_ansi(self._buffer).strip()
        if message:
            _log(self.task_id, "llm", message)
        self._buffer = ""

    def fileno(self):
        return self._original.fileno() if self._original else -1

    def isatty(self):
        return False


# ── 后台流水线 ────────────────────────────────────────────────────
def _run_pipeline(task_id: str):
    task = _tasks[task_id]
    input_value = task["input_value"]
    output_dir = Path(task["output_dir"])
    llm_config = task.get("llm_config")
    prefer_project_fasta = bool(task.get("prefer_project_fasta"))
    reviewed_fasta_path = _clean_text(task.get("reviewed_fasta_path")) or None
    reviewed_fasta_url = _clean_text(task.get("reviewed_fasta_url")) or None
    reviewed_fasta_name = _clean_text(task.get("reviewed_fasta_name")) or None
    run_mode = _clean_run_mode(task.get("run_mode"))
    repository = _clean_repository(task.get("repository"))
    parameter_only = run_mode == _RUN_MODE_PARAMETERS
    prepare_only = run_mode == _RUN_MODE_PREPARE
    review_overrides = dict(task.get("review_overrides") or {})
    if not isinstance(llm_config, dict):
        _set_task_terminal_status(task_id, "failed")
        _log(task_id, "error", "缺少本次任务的 API Key 配置。")
        _start_ready_queued_tasks()
        return

    reporter = WebReporter(task_id)

    try:
        from agent.input.normalizer import normalize_input
        from agent.orchestrator.pipeline import AgentService

        _log(task_id, "info", f"任务开始：{input_value}")
        _log(task_id, "info", f"输出目录：{output_dir}")
        _log(task_id, "info", f"LLM 模型：{llm_config['model']}  Base URL：{llm_config['base_url']}")
        _log(task_id, "info", f"运行模式：{_run_mode_label(run_mode)}")

        # ── 步骤 1 ──
        repository_label = repository.upper() if repository != "auto" else "Auto"
        _step(task_id, 1, f"[1/5] Resolve {repository_label} project")
        _log(task_id, "info", "正在初始化 AgentService…")
        service = AgentService(reporter=reporter, llm_reasoner=_task_llm_reasoner(llm_config))
        _log(task_id, "info", "AgentService 初始化完成")
        task_obj = normalize_input(input_value)
        local_source = _known_local_source_from_input(input_value)
        _log(task_id, "info", f"输入规范化：{task_obj.file_name}")

        if local_source:
            _log(
                task_id,
                "info",
                "Detected local cached PRIDE source; using known-project local mode "
                f"({local_source['project_accession']} / {local_source['matched_file']}).",
            )
            with StderrCapture(task_id):
                result = service.plan_dda_run_from_known_project_local_source(
                    task=task_obj,
                    source_data_path=local_source["source_path"],
                    project_accession=local_source["project_accession"],
                    output_dir=output_dir,
                    repository="pride",
                    matched_file=local_source["matched_file"],
                    reviewed_fasta_path=reviewed_fasta_path,
                    reviewed_fasta_url=reviewed_fasta_url,
                    reviewed_fasta_name=reviewed_fasta_name,
                    prefer_project_fasta=prefer_project_fasta,
                    context_dir=local_source.get("context_dir"),
                )
        else:
            _log(task_id, "info", f"Querying {repository_label} metadata and inferring parameters with the LLM...")
            with StderrCapture(task_id):
                result = service.plan_dda_run_from_repository(
                    task=task_obj,
                    output_dir=output_dir,
                    repository=repository,
                    reviewed_fasta_path=reviewed_fasta_path,
                    reviewed_fasta_url=reviewed_fasta_url,
                    reviewed_fasta_name=reviewed_fasta_name,
                    prefer_project_fasta=prefer_project_fasta,
                )
        if review_overrides:
            result = service.apply_review_overrides_to_result(
                result,
                review_overrides,
                task_obj,
                output_dir,
                prefer_project_fasta=prefer_project_fasta,
                reviewed_fasta_path=reviewed_fasta_path,
                reviewed_fasta_url=reviewed_fasta_url,
                reviewed_fasta_name=reviewed_fasta_name,
            )
            _log(task_id, "info", "已应用人工复核选择，重新生成执行计划。")
        _set_review_summary(task_id, result)
        _log(task_id, "info", f"{repository_label} metadata query and LLM inference completed")

        primary = result.resolution.primary_project
        if primary:
            _log(task_id, "info", f"项目：{primary.project_accession}  匹配文件：{primary.matched_file}  置信度：{result.resolution.resolution_confidence:.2f}")

        _log(task_id, "info", f"采集模式：{result.attributes.acquisition_mode.value}  物种：{result.attributes.species.value}")
        _log(task_id, "info", f"仪器：{result.attributes.instrument_name.value}  酶：{result.attributes.enzyme.value}")

        hints = result.attributes.search_parameter_hints.value
        if isinstance(hints, dict):
            _log(task_id, "info", f"推荐 workflow：{hints.get('recommended_workflow_name', '无')}")
            _log(task_id, "info", f"推荐 FASTA：{result.plan.fasta_path.name}")
            if result.plan.fasta_download_url:
                _log(task_id, "info", f"FASTA 下载源：{result.plan.fasta_download_url}")

        _write_agent_audit_package(output_dir, result, report=lambda message: _log(task_id, "debug", message))
        _notify_agent_audit_ready(task_id)

        prepared_path = None
        if not parameter_only and result.plan.needs_review and service._can_retry_with_mzml_instrument(result.plan):
            _log(task_id, "info", "检测到项目级多个仪器，先下载/转换 mzML，并从 mzML 读取文件级仪器信息。")
            _step(task_id, 2, "[2/5] 下载 PRIDE 数据文件")
            with StderrCapture(task_id):
                prepared_path = service.prepare_local_asset(result.asset) if local_source else service.prepare_asset(result.asset)
            _log(task_id, "info", f"数据文件已就绪：{prepared_path}")
            result = service.replan_with_mzml_instrument(
                result,
                prepared_path,
                task_obj,
                output_dir,
                reviewed_fasta_path=reviewed_fasta_path,
                reviewed_fasta_url=reviewed_fasta_url,
                reviewed_fasta_name=reviewed_fasta_name,
                prefer_project_fasta=prefer_project_fasta,
            )
            _set_review_summary(task_id, result)
            _log(task_id, "info", f"仪器复核后计划状态：{'需要人工复核' if result.plan.needs_review else '可继续运行'}")
            _write_agent_audit_package(output_dir, result, report=lambda message: _log(task_id, "debug", message))
            _notify_agent_audit_ready(task_id)

        if result.plan.needs_review:
            task["blocking_issues"] = result.plan.blocking_issues
            for issue in result.plan.blocking_issues:
                _log(task_id, "error", f"[阻断] {issue}")
            service.write_task_bundle(output_dir, result.resolution, result.context, result.attributes, result.plan, asset=result.asset)
            _write_agent_audit_package(output_dir, result, report=lambda message: _log(task_id, "debug", message))
            _notify_agent_audit_ready(task_id)
            _set_task_terminal_status(task_id, "blocked")
            return

        _log(task_id, "info", f"workflow：{result.plan.fragpipe_workflow_path.name}  FASTA：{result.plan.fasta_path.name}（{result.plan.fasta_selection_mode}）")

        if parameter_only:
            _step(task_id, 5, "[5/5] 参数推断完成")
            _log(task_id, "info", f"Parameter-only mode completed: {repository_label} project resolution, file attribute inference, workflow/FASTA/search-parameter planning, and audit package generation are complete.")
            service.write_task_bundle(output_dir, result.resolution, result.context, result.attributes, result.plan, asset=result.asset)
            _write_agent_audit_package(output_dir, result, report=lambda message: _log(task_id, "debug", message))
            try:
                from agent.audit.review import build_task_state_snapshot, write_task_state

                project_accession = result.resolution.primary_project.project_accession if result.resolution.primary_project else None
                write_task_state(
                    output_dir / "task_state.json",
                    build_task_state_snapshot(
                        task_id=task_obj.task_id,
                        status="completed",
                        stage="planning",
                        source_file=task_obj.file_name,
                        project_accession=project_accession,
                        notes=["Parameter-only mode completed; full execution was not run."],
                    ),
                )
                _write_parameter_audit_files(output_dir, task_id, 1, input_value, result)
                _write_task_runtime_log(task_id, output_dir)
                _log(task_id, "info", "Parameter package generated: converter_config, workflow, decision_trace, attributes, parameter_audit and runtime.log.")
                _log(task_id, "info", "Compressing parameter ZIP; parameter-only mode excludes RAW/mzML/FASTA payload files.")
                _zip_output_dir(output_dir, report=lambda message: _log(task_id, "info", message))
                _log(task_id, "info", "Parameter ZIP is ready to download.")
            except Exception as audit_exc:
                _log(task_id, "debug", f"Failed to write parameter-only audit files: {audit_exc}")
            _set_task_terminal_status(task_id, "completed")
            return

        # ── 步骤 2 ──
        if prepared_path is None:
            _step(task_id, 2, "[2/5] 下载 PRIDE 数据文件")
            with StderrCapture(task_id):
                prepared_path = service.prepare_local_asset(result.asset) if local_source else service.prepare_asset(result.asset)
            _log(task_id, "info", f"数据文件已就绪：{prepared_path}")
        else:
            _log(task_id, "info", f"复用已准备的数据文件：{prepared_path}")

        # ── 步骤 3 ──
        _step(task_id, 3, "[3/5] 生成 MSDT-Converter 输入包")
        result = service.validate_prepared_data_for_plan(result, prepared_path)
        if result.plan.needs_review:
            task["blocking_issues"] = result.plan.blocking_issues
            for issue in result.plan.blocking_issues:
                _log(task_id, "error", f"[阻断] {issue}")
            service.write_task_bundle(output_dir, result.resolution, result.context, result.attributes, result.plan, asset=result.asset)
            _set_review_summary(task_id, result)
            _set_task_terminal_status(task_id, "blocked")
            return

        from agent.execution.bundle import materialize_dda_task_bundle
        with StderrCapture(task_id):
            bundle = materialize_dda_task_bundle(
                task=task_obj,
                project_resolution=result.resolution,
                project_context=result.context,
                attributes=result.attributes,
                source_data_path=prepared_path,
                output_dir=output_dir,
                reviewed_fasta_path=reviewed_fasta_path,
                reviewed_fasta_url=reviewed_fasta_url,
                reviewed_fasta_name=reviewed_fasta_name,
                prefer_project_fasta=prefer_project_fasta,
                report=reporter,
            )
        service.write_task_bundle(output_dir, result.resolution, result.context, result.attributes, bundle.plan, asset=result.asset)
        _write_agent_audit_package(output_dir, result, plan=bundle.plan, report=lambda message: _log(task_id, "debug", message))
        if prepare_only:
            _step(task_id, 4, "[4/5] Package MSDT-Converter input")
            from agent.msdt_converter.docker_runner import DockerMSDTConverterRunner

            docker_runner = DockerMSDTConverterRunner(image="guomics2017/msdt-converter:v1.3", report=reporter)
            docker_runner.write_container_config(bundle)
            try:
                from agent.audit.review import build_task_state_snapshot, write_task_state

                project_accession = result.resolution.primary_project.project_accession if result.resolution.primary_project else None
                _write_parameter_audit_files(output_dir, task_id, 1, input_value, result)
                write_task_state(
                    output_dir / "task_state.json",
                    build_task_state_snapshot(
                        task_id=task_obj.task_id,
                        status="completed",
                        stage="packaging",
                        source_file=task_obj.file_name,
                        project_accession=project_accession,
                        notes=["Prepare input package mode completed; Docker execution was not run."],
                    ),
                )
                _write_task_runtime_log(task_id, output_dir)
                _log(task_id, "info", "Prepared input package generated: converter_config, workflow, FASTA reference, decision_trace, attributes, parameter_audit and runtime.log.")
                _log(task_id, "info", "Compressing input-package ZIP; large RAW/mzML payload files remain in the run directory and are not duplicated in the ZIP.")
                _zip_output_dir(output_dir, report=lambda message: _log(task_id, "info", message))
                _log(task_id, "info", "Input-package ZIP is ready to download.")
            except Exception as audit_exc:
                _log(task_id, "debug", f"Failed to write prepare-mode audit files: {audit_exc}")
            _step(task_id, 5, "[5/5] Input package ready")
            _set_task_terminal_status(task_id, "completed")
            return
        _log(task_id, "info", f"输入包已生成：{output_dir}")
        _log(task_id, "info", f"converter_config：{bundle.converter_config_path}")
        _log(task_id, "info", f"workflow：{bundle.materialized_workflow_path}")
        _log(task_id, "info", f"FASTA：{bundle.materialized_fasta_path}")

        # ── 步骤 4 ──
        _step(task_id, 4, "[4/5] 运行 MSDT-Converter Docker")
        from agent.msdt_converter.docker_runner import DockerMSDTConverterRunner
        docker_runner = DockerMSDTConverterRunner(image="guomics2017/msdt-converter:v1.3", report=reporter)
        with StderrCapture(task_id):
            docker_result = docker_runner.run(bundle)

        # ── 步骤 5 ──
        _step(task_id, 5, "[5/5] 处理结果")
        from agent.execution.outputs import execution_failure_events, execution_failure_reasons
        failure_reasons = execution_failure_reasons(
            bundle.plan,
            docker_result.returncode,
            docker_result.stdout,
            docker_result.stderr,
        )
        failure_events = execution_failure_events(
            bundle.plan,
            docker_result.returncode,
            docker_result.stdout,
            docker_result.stderr,
        )
        if not failure_reasons:
            _log(task_id, "info", "=" * 50)
            _log(task_id, "info", "全部运行完成！")
            if output_dir.exists():
                for f in sorted(output_dir.rglob("*")):
                    if f.is_file():
                        size = f.stat().st_size
                        if size > 1024 * 1024:
                            size_str = f"{size / 1024 / 1024:.1f} MB"
                        elif size > 1024:
                            size_str = f"{size / 1024:.1f} KB"
                        else:
                            size_str = f"{size} B"
                        _log(task_id, "info", f"  {f.relative_to(output_dir)}  ({size_str})")
            _log(task_id, "info", "=" * 50)
            if not _has_downloadable_result_file(output_dir):
                raise RuntimeError("No downloadable result files were produced.")
            _log(task_id, "info", "开始压缩打包结果 ZIP，打包完成后才会显示下载按钮。")
            _zip_output_dir(output_dir, report=lambda message: _log(task_id, "info", message))
            _log(task_id, "info", "结果 ZIP 已压缩打包完成，可以下载。")
            try:
                from agent.audit.review import build_task_state_snapshot, write_task_state

                project_accession = result.resolution.primary_project.project_accession if result.resolution.primary_project else None
                write_task_state(
                    output_dir / "task_state.json",
                    build_task_state_snapshot(
                        task_id=task_obj.task_id,
                        status="completed",
                        stage="execution",
                        source_file=task_obj.file_name,
                        project_accession=project_accession,
                        notes=[],
                    ),
                )
            except Exception as audit_exc:
                _log(task_id, "debug", f"Failed to write execution success audit files: {audit_exc}")
            _set_task_terminal_status(task_id, "completed")
        else:
            _log(task_id, "error", "MSDT-Converter 内部步骤失败，任务已标记为失败，不打包下载 ZIP。")
            for reason in failure_reasons:
                _log(task_id, "error", f"[failure] {reason}")
            if docker_result.stdout:
                _log(task_id, "error", f"[stdout]\n{docker_result.stdout[-2000:]}")
            if docker_result.stderr:
                _log(task_id, "error", f"[stderr]\n{docker_result.stderr[-2000:]}")
            try:
                from agent.audit.review import append_review_item, build_review_item, build_task_state_snapshot, write_task_state

                project_accession = result.resolution.primary_project.project_accession if result.resolution.primary_project else None
                write_task_state(
                    output_dir / "task_state.json",
                    build_task_state_snapshot(
                        task_id=task_obj.task_id,
                        status="failed",
                        stage="execution",
                        source_file=task_obj.file_name,
                        project_accession=project_accession,
                        notes=failure_reasons,
                    ),
                )
                append_review_item(
                    output_dir / "review_queue.json",
                    build_review_item(
                        task_id=task_obj.task_id,
                        source_file=task_obj.file_name,
                        project_accession=project_accession,
                        stage="execution",
                        reasons=failure_reasons,
                    ),
                )
                _write_recovery_audit_package(
                    output_dir,
                    task_obj,
                    stage="execution",
                    run_mode=_RUN_MODE_FULL,
                    events=failure_events,
                    result=result,
                    plan=bundle.plan,
                    artifacts={
                        "task_state_json": output_dir / "task_state.json",
                        "review_queue_json": output_dir / "review_queue.json",
                        "runtime_log": output_dir / "logs" / "runtime.log",
                    },
                    report=lambda message: _log(task_id, "debug", message),
                )
            except Exception as audit_exc:
                _log(task_id, "debug", f"Failed to write execution failure audit files: {audit_exc}")
            _attach_workflow_recovery_summary(task_id, output_dir)
            _set_task_terminal_status(task_id, "failed")

    except Exception as exc:
        from agent.errors import build_error_record, public_error_summary, write_error_record
        from agent.execution.outputs import ExecutionFailureEvent

        error_record = build_error_record(exc, stage="pipeline", input_file=input_value)
        write_error_record(output_dir / "error.json", error_record)
        summary = public_error_summary(error_record)
        task["error_summary"] = summary
        recovery_task = locals().get("task_obj") or SimpleNamespace(task_id=task_id, file_name=Path(input_value).name)
        recovery_result = locals().get("result")
        recovery_plan = getattr(locals().get("bundle"), "plan", None) or getattr(recovery_result, "plan", None)
        _write_recovery_audit_package(
            output_dir,
            recovery_task,
            stage="pipeline",
            run_mode=_clean_run_mode(task.get("run_mode")),
            events=[
                ExecutionFailureEvent(
                    category=str(error_record.get("category") or "unknown"),
                    reason=str(error_record.get("technical_message") or summary.get("public_message") or exc),
                    evidence_kind="exception",
                    marker=str(error_record.get("exception_type") or type(exc).__name__),
                )
            ],
            result=recovery_result,
            plan=recovery_plan,
            artifacts={"error_json": output_dir / "error.json"},
            report=lambda message: _log(task_id, "debug", message),
        )
        task["blocking_issues"] = [summary.get("public_message") or "任务运行失败。"]
        _log(task_id, "error", f"运行出错：{summary.get('public_message')}（{summary.get('category')}）")
        if "traceback" in error_record:
            _log(task_id, "debug", error_record["traceback"])
        _attach_workflow_recovery_summary(task_id, output_dir)
        _set_task_terminal_status(task_id, "failed")
    finally:
        _start_ready_queued_tasks()


# ── API 端点 ──────────────────────────────────────────────────────
@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    if task_id not in _tasks:
        history = _find_history_record(task_id)
        if history is not None:
            return _task_detail_from_history(task_id, history)
        return {"error": "任务不存在"}
    task = _tasks[task_id]
    with _tasks_lock:
        queue_state = _queue_state_locked(task_id)
    output_dir_raw = task.get("output_dir")
    output_dir = Path(output_dir_raw) if output_dir_raw else None
    can_download = bool(
        task.get("status") == "completed"
        and output_dir
        and output_dir.exists()
        and _is_download_zip_ready(output_dir)
    )
    return {
        "task_id": task["task_id"],
        "input_value": task["input_value"],
        "submitter": task.get("submitter", "未填写"),
        "status": task["status"],
        "step": task.get("step", 0),
        "total_steps": task.get("total_steps", 5),
        "log_count": len(task["logs"]),
        "logs": _public_logs_from_task(task),
        "blocking_issues": task.get("blocking_issues", []),
        "error_summary": task.get("error_summary"),
        "workflow_outcome": task.get("workflow_outcome"),
        "usable_partial_outputs": bool(task.get("usable_partial_outputs")),
        "recovery_primary_issue": task.get("recovery_primary_issue"),
        "recovery_recommended_next_step": task.get("recovery_recommended_next_step"),
        "recovery_report_json": task.get("recovery_report_json"),
        "recovery_report_md": task.get("recovery_report_md"),
        "review_summary": task.get("review_summary"),
        "fasta_preference": "project" if task.get("prefer_project_fasta") else "llm",
        "run_mode": _clean_run_mode(task.get("run_mode")),
        "resource_policy": _clean_resource_policy(task.get("resource_policy")),
        "ui_language": _clean_ui_language(task.get("ui_language")),
        "repository": _clean_repository(task.get("repository")),
        "can_download": can_download,
        "archived": False,
        **queue_state,
    }


def _clean_review_overrides(body: dict[str, Any]) -> dict[str, str]:
    raw = body.get("overrides") if isinstance(body.get("overrides"), dict) else body
    overrides: dict[str, str] = {}
    for field in ("species", "instrument_name"):
        value = _clean_text(raw.get(field)) if isinstance(raw, dict) else ""
        if value:
            overrides[field] = value
    return overrides


@app.post("/api/tasks/{task_id}/review")
async def submit_task_review(task_id: str, body: dict[str, Any]):
    overrides = _clean_review_overrides(body)
    if not overrides:
        return {"error": "请选择至少一个复核参数。"}
    with _tasks_lock:
        task = _tasks.get(task_id)
        if task is None:
            return {"error": "任务不存在"}
        if task.get("status") not in {"blocked", "failed"}:
            return {"error": "当前任务不在可复核状态。"}
        if not isinstance(task.get("llm_config"), dict):
            return {"error": "服务器内存中没有本次任务的 API Key，无法继续；请重新提交任务。"}
        merged = dict(task.get("review_overrides") or {})
        merged.update(overrides)
        task["review_overrides"] = merged
        task["status"] = "queued"
        task["step"] = 0
        task["blocking_issues"] = []
        task.pop("finished_at", None)
        task["logs"].append(
            {
                "type": "log",
                "ts": _now_time(),
                "level": "info",
                "message": _localize_public_message(
                    "已提交人工复核选择，任务重新进入队列。",
                    _clean_ui_language(task.get("ui_language")),
                    level="info",
                ),
            }
        )
        queue_state = _queue_state_locked(task_id)
    _write_task_history(task_id)
    _start_ready_queued_tasks()
    with _tasks_lock:
        status = _tasks.get(task_id, {}).get("status", "queued")
        queue_state = _queue_state_locked(task_id)
    return {"task_id": task_id, "status": status, "review_overrides": overrides, **queue_state}


@app.get("/api/tasks/{task_id}/download")
async def download_results(task_id: str):
    if task_id not in _tasks:
        history = _find_history_record(task_id)
        if history is not None:
            if history.get("status") != "completed":
                return {"error": "Task is not completed; results cannot be downloaded."}
            output_dir_name = str(history.get("output_dir") or "")
            output_dir = _runs_dir / output_dir_name if output_dir_name else None
            if output_dir is None or not output_dir.exists():
                return {"error": "Result directory does not exist."}
            if not _has_downloadable_result_file(output_dir):
                return {"error": "Result directory has no downloadable files."}
            if not _ensure_existing_download_zip_ready(output_dir):
                return {"error": "Result ZIP is not ready yet."}
            zip_path = _download_zip_path(output_dir)
            stem = safe_output_stem(str(history.get("input_value") or output_dir_name or task_id))
            suffix = "parameters" if _clean_run_mode(history.get("run_mode")) == _RUN_MODE_PARAMETERS else "results"
            return FileResponse(
                path=str(zip_path),
                filename=f"{stem}_{suffix}.zip",
                media_type="application/zip",
            )
        return {"error": "任务不存在"}
    task = _tasks[task_id]
    if task.get("status") != "completed":
        return {"error": "任务未完成，不能下载结果"}
    output_dir = Path(task["output_dir"])
    if not output_dir.exists():
        return {"error": "结果目录不存在"}
    if not _has_downloadable_result_file(output_dir):
        return {"error": "结果目录没有可下载文件"}

    if not _ensure_existing_download_zip_ready(output_dir):
        return {"error": "结果 ZIP 尚未打包完成，请等待任务日志提示后再下载。"}
    zip_path = _download_zip_path(output_dir)
    stem = safe_output_stem(task["input_value"])
    suffix = "parameters" if _clean_run_mode(task.get("run_mode")) == _RUN_MODE_PARAMETERS else "results"
    return FileResponse(
        path=str(zip_path),
        filename=f"{stem}_{suffix}.zip",
        media_type="application/zip",
    )


def _resolve_task_output_dir(task_id: str) -> Path | None:
    with _tasks_lock:
        task = _tasks.get(task_id)
    if task is not None:
        output_dir_raw = task.get("output_dir")
        if output_dir_raw:
            return Path(output_dir_raw)
    history = _find_history_record(task_id)
    if history is not None:
        output_dir_name = str(history.get("output_dir") or "")
        if output_dir_name:
            return _runs_dir / output_dir_name
    candidate = _runs_dir / safe_output_stem(task_id)
    if candidate.exists() and candidate.is_dir():
        return candidate
    return None


_AGENT_AUDIT_FILES = {
    "observation": "agent_observation.json",
    "plan": "agent_plan.json",
    "decision_trace": "agent_decision_trace.json",
    "recovery": "recovery_audit.json",
}


def _read_agent_audit_file(output_dir: Path, filename: str) -> tuple[dict[str, Any] | None, str | None]:
    path = output_dir / filename
    if not path.exists() or not path.is_file():
        return None, "missing"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "invalid"
    return (data, None) if isinstance(data, dict) else (None, "invalid")


@app.get("/api/tasks/{task_id}/agent-audit")
async def get_agent_audit(task_id: str):
    output_dir = _resolve_task_output_dir(task_id)
    if output_dir is None or not output_dir.exists():
        return {
            "error": "Task output directory not found.",
            "available": False,
            "available_files": [],
            "missing_files": list(_AGENT_AUDIT_FILES.values()),
            "invalid_files": [],
        }

    payloads: dict[str, dict[str, Any] | None] = {}
    available_files: list[str] = []
    missing_files: list[str] = []
    invalid_files: list[str] = []
    for key, filename in _AGENT_AUDIT_FILES.items():
        data, state = _read_agent_audit_file(output_dir, filename)
        payloads[key] = data
        if state == "missing":
            missing_files.append(filename)
        elif state == "invalid":
            invalid_files.append(filename)
        else:
            available_files.append(filename)

    if not any(payloads.values()):
        return {
            "error": "No agent audit files found.",
            "available": False,
            "available_files": available_files,
            "missing_files": missing_files,
            "invalid_files": invalid_files,
        }

    return {
        "available": True,
        "output_dir": str(output_dir),
        "available_files": available_files,
        "missing_files": missing_files,
        "invalid_files": invalid_files,
        "observation": payloads["observation"],
        "plan": payloads["plan"],
        "decision_trace": payloads["decision_trace"],
        "recovery": payloads["recovery"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
