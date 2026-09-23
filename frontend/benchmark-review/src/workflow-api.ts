import {
  decodeAgentTurnResponse,
  type AgentDialogueMessage,
  type AgentNextDecision,
  type AgentResolvedDecision,
  type AgentTurn,
} from "./agent-turn";

export type WorkflowRecord = Record<string, unknown> & {
  error?: string;
  ok?: boolean;
  status?: string;
};

/** Job lifecycle statuses — domain failures use status+error, not transport failures. */
export const DISCOVERY_JOB_STATUSES = ["queued", "running", "failed", "blocked", "cancelled", "completed", "interrupted", "durability_failed"] as const;

export function isDiscoveryJobStatus(status: unknown): boolean {
  return DISCOVERY_JOB_STATUSES.includes(String(status || "").toLowerCase() as (typeof DISCOVERY_JOB_STATUSES)[number]);
}

const apiErrorMessage = (value: unknown, status: number) => {
  if (typeof value === "string" && value.trim()) return value.trim();
  if (value && typeof value === "object" && !Array.isArray(value)) {
    const detail = value as Record<string, unknown>;
    const message = String(detail.message || detail.detail || detail.code || "").trim();
    if (message) return message;
  }
  return `HTTP ${status}`;
};

export async function workflowJson<T extends WorkflowRecord>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const response = await fetch(path, {
    ...options,
    headers: {
      ...(options.body != null ? { "Content-Type": "application/json" } : {}),
      ...(options.headers || {}),
    },
  });
  let payload: T;
  try {
    payload = (await response.json()) as T;
  } catch {
    throw new Error(`HTTP ${response.status}: invalid JSON response`);
  }
  // Discovery job bodies may carry status=failed with a domain error string.
  // That is a successful HTTP response of a job record — do not treat as transport failure.
  const payloadError = (payload as Record<string, unknown>).error;
  const jobOk =
    isDiscoveryJobStatus(payload.status) &&
    (path.includes("/api/discovery/jobs") ||
      path.includes("/api/discovery/job") ||
      path.includes("/api/ops/jobs"));
  if (!response.ok || payload.ok === false || (payloadError && !jobOk)) {
    throw new Error(apiErrorMessage(payloadError, response.status));
  }
  return payload;
}

export const getHistory = (refresh = false, signal?: AbortSignal) =>
  workflowJson<WorkflowRecord & { active_tasks?: WorkflowRecord[]; results?: WorkflowRecord[]; summary?: WorkflowRecord }>(
    `/api/history?fast=${refresh ? "0" : "1"}${refresh ? "&refresh=1" : ""}`,
    { signal },
  );

export type DatasetDirectories = WorkflowRecord & {
  path: string;
  parent: string | null;
  directories: { name: string; path: string }[];
};

export const listDatasetDirectories = (path: string, signal?: AbortSignal) =>
  workflowJson<DatasetDirectories>(`/api/local-datasets/directories?path=${encodeURIComponent(path)}`, { signal });

export const preflight = (payload: WorkflowRecord) =>
  workflowJson<WorkflowRecord & { blocking_issues?: string[]; warnings?: string[] }>("/api/preflight", {
    method: "POST",
    body: JSON.stringify(payload),
  });

export const startSingleTask = (payload: WorkflowRecord) =>
  workflowJson<WorkflowRecord & { task_id: string }>("/api/tasks", {
    method: "POST",
    body: JSON.stringify(payload),
  });

export const getSingleTask = (taskId: string, signal?: AbortSignal) =>
  workflowJson<WorkflowRecord>(`/api/tasks/${encodeURIComponent(taskId)}`, { signal });

export const submitTaskReview = (taskId: string, overrides: Record<string, string>) =>
  workflowJson<WorkflowRecord>(`/api/tasks/${encodeURIComponent(taskId)}/review`, {
    method: "POST",
    body: JSON.stringify({ overrides }),
  });

export const startBatch = (payload: WorkflowRecord) =>
  workflowJson<WorkflowRecord & { batch_id: string }>("/api/batches/parameters", {
    method: "POST",
    body: JSON.stringify(payload),
  });

export const getBatch = (batchId: string, signal?: AbortSignal) =>
  workflowJson<WorkflowRecord>(`/api/batches/${encodeURIComponent(batchId)}`, { signal });

export const cancelBatch = (batchId: string) =>
  workflowJson<WorkflowRecord>(`/api/batches/${encodeURIComponent(batchId)}/cancel`, {
    method: "POST",
  });

export type DiscoveryBatchHandoff = WorkflowRecord & {
  job_id: string;
  discovery_id: string;
  batch_index: number;
  file_count: number;
  inputs: string[];
  input_records: WorkflowRecord[];
};

export const getDiscoveryBatchHandoff = (jobId: string, batchIndex: number) =>
  workflowJson<DiscoveryBatchHandoff>(
    `/api/discovery/jobs/${encodeURIComponent(jobId)}/batches/${batchIndex}/handoff`,
  );

export const getDiscoveryRun = (discoveryId: string, signal?: AbortSignal) =>
  workflowJson<WorkflowRecord>(`/api/discovery/${encodeURIComponent(discoveryId)}`, { signal });

export const previewHistoryDelete = (
  kind: string,
  id: string,
  includeLinkedBatches = false,
) =>
  workflowJson<WorkflowRecord>(
    `/api/history/${encodeURIComponent(kind)}/${encodeURIComponent(id)}/delete-preview?include_linked_batches=${includeLinkedBatches ? "true" : "false"}`,
  );

export const deleteHistoryItem = (
  kind: string,
  id: string,
  confirmationId: string,
  includeLinkedBatches = false,
) =>
  workflowJson<WorkflowRecord>(
    `/api/history/${encodeURIComponent(kind)}/${encodeURIComponent(id)}`,
    {
      method: "DELETE",
      body: JSON.stringify({
        confirmation_id: confirmationId,
        include_linked_batches: includeLinkedBatches,
      }),
    },
  );

export type DiscoveryPublicationProgress = {
  candidate_projects?: number;
  candidate_files?: number;
  reviewed_projects?: number;
  judgment_qualified_projects?: number;
  build_ready_projects?: number;
  build_ready_files?: number;
  blocker_counts?: Record<string, number>;
};

export type BusinessCompletionDecision = WorkflowRecord & {
  schema_version?: string;
  authority_source?: "publication_contract_registry";
  succeeded?: boolean;
  status?: "blocked" | "blocked_with_progress" | "running_progress" | "build_ready_succeeded";
  package_kind?: "progress" | "build_ready";
  progress_visible?: boolean;
  progress?: DiscoveryPublicationProgress;
  build_ready_package?: WorkflowRecord | null;
  issuance_token?: string | null;
  limitations?: string[];
  success_ui_allowed?: boolean;
  /** Replay compatibility for early Wave 2 fixtures. */
  build_ready_projects?: number;
  build_ready_files?: number;
};

export type DiscoveryJobRecord = WorkflowRecord & {
  business_completion?: BusinessCompletionDecision;
  portfolio_state?: WorkflowRecord | null;
};

export type DiscoveryJob = WorkflowRecord & {
  job_id?: string;
  logs?: WorkflowRecord[];
  record?: DiscoveryJobRecord;
  execution_state?: WorkflowRecord;
  resumable?: boolean;
};

const isRecordValue = (value: unknown): value is Record<string, unknown> =>
  value != null && typeof value === "object" && !Array.isArray(value);

function businessCompletionCount(
  completion: Record<string, unknown>,
  key: "build_ready_projects" | "build_ready_files",
): number {
  const progress = isRecordValue(completion.progress) ? completion.progress : {};
  const parsed = Number(completion[key] ?? progress[key] ?? 0);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 0;
}

/** Mirror the Authority Plane's fail-closed UI success gate. */
export function businessCompletionAllowsSuccess(record: Record<string, unknown>): boolean {
  const completion = isRecordValue(record.business_completion)
    ? record.business_completion
    : null;
  if (!completion) return false;
  return (
    completion.schema_version === "business-completion/v2" &&
    completion.authority_source === "publication_contract_registry" &&
    completion.succeeded === true &&
    completion.status === "build_ready_succeeded" &&
    completion.package_kind === "build_ready" &&
    completion.success_ui_allowed === true &&
    isRecordValue(completion.build_ready_package) &&
    String(completion.issuance_token || "").trim().length > 0 &&
    businessCompletionCount(completion, "build_ready_projects") > 0 &&
    businessCompletionCount(completion, "build_ready_files") > 0
  );
}

export function honestDiscoveryStatus(
  serverStatus: string,
  record: Record<string, unknown>,
  _attemptFinishedWithoutAudit = false,
): string {
  if (serverStatus !== "completed") return serverStatus;
  if (!isRecordValue(record.business_completion)) {
    return "blocked";
  }
  if (businessCompletionAllowsSuccess(record)) return "completed";
  return record.business_completion.status === "running_progress" ? "running" : "blocked";
}

export function normalizeDiscoveryJobForUi(job: DiscoveryJob): DiscoveryJob {
  const record = (job.record || {}) as Record<string, unknown>;
  const attemptFinishedWithoutAudit = (job.logs || []).some((log) =>
    ["discovery_quality_repair_completed", "repair_attempt_finished"].includes(
      String(log.type || ""),
    ),
  );
  const status = honestDiscoveryStatus(
    String(job.status || record.status || "queued").toLowerCase(),
    record,
    attemptFinishedWithoutAudit,
  );
  return status === job.status ? job : { ...job, status };
}

export type DiscoveryGoalParse = WorkflowRecord & {
  parser?: string;
  prompt?: string;
  fields?: WorkflowRecord;
  warnings?: string[];
  reasoning?: string;
};

/** Parse a free-text goal into structured discovery fields (LLM). Soft-fails to null caller-side. */
export const parseDiscoveryGoal = (
  prompt: string,
  current: WorkflowRecord = {},
  signal?: AbortSignal,
) =>
  workflowJson<DiscoveryGoalParse>("/api/discovery/parse-goal", {
    method: "POST",
    signal,
    body: JSON.stringify({
      prompt,
      output_language: "zh-CN",
      current: { output_language: "zh-CN", ...current },
    }),
  });

export type GrillTurnResult = AgentTurn;

/** One conversational grill turn (LLM phrasing + answer mapping). Soft-fails caller-side. */
export const grillTurn = async (
  payload: {
    user_message: string;
    phase?: string;
    turn_kind?: string;
    pending_question?: WorkflowRecord | null;
    pending_decision?: AgentNextDecision | null;
    decision_memory?: AgentResolvedDecision[];
    resolved_fields?: string[];
    session_id?: string;
    intent_snapshot?: WorkflowRecord;
    answered?: WorkflowRecord;
    local_summary?: string;
    allow_server_default?: boolean;
    request_timeout_seconds?: number;
    gap_report?: WorkflowRecord | null;
    dialogue_history?: AgentDialogueMessage[];
  },
  signal?: AbortSignal,
) =>
  decodeAgentTurnResponse(await workflowJson<WorkflowRecord>("/api/discovery/grill-turn", {
    method: "POST",
    signal,
    body: JSON.stringify(payload),
  }));


/** Start discovery only with a full payload (Grill-confirmed). */
export const startDiscoveryJob = (payload: WorkflowRecord, signal?: AbortSignal) =>
  payload.grill_confirmed !== true
    ? Promise.reject(new Error("Explicit strategy confirmation is required before discovery."))
    : workflowJson<DiscoveryJob>("/api/discovery/jobs", {
        method: "POST",
        signal,
        body: JSON.stringify({
          runtime: "openai_agents",
          source: "remote",
          repository: "pride",
          output_language: "zh-CN",
          constraints_enabled: false,
          goal: "general",
          acquisition_mode: "unknown",
          labeling_strategy: "unknown",
          species: [],
          species_policy: "open",
          diversity_strategy: "balanced",
          use_memory: true,
          save_memory: true,
          hard_constraint_fields: ["repository"],
          constraint_provenance: { repository: "user" },
          idempotency_key: crypto.randomUUID(),
          ...payload,
        }),
      }).then(normalizeDiscoveryJobForUi);

export const getDiscoveryJob = (jobId: string, detail = false, signal?: AbortSignal) =>
  workflowJson<DiscoveryJob>(`/api/discovery/jobs/${encodeURIComponent(jobId)}${detail ? "?detail=1" : ""}`, { signal })
    .then(normalizeDiscoveryJobForUi);

export const cancelDiscoveryJob = (jobId: string) =>
  workflowJson<DiscoveryJob>(`/api/discovery/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST" });

export const resumeDiscoveryJob = (jobId: string) =>
  workflowJson<DiscoveryJob>(`/api/discovery/jobs/${encodeURIComponent(jobId)}/resume`, { method: "POST" })
    .then(normalizeDiscoveryJobForUi);

export type AiReadyAction =
  | "profile-inputs"
  | "locate-inputs"
  | "build-from-agent-run"
  | "mini-e2e"
  | "validate-build";

export const runAiReady = (action: AiReadyAction, payload: WorkflowRecord) =>
  workflowJson<WorkflowRecord>(`/api/ai-ready/${action}`, {
    method: "POST",
    body: JSON.stringify(payload),
  });

export const terminalWorkflowStatus = (status: unknown) =>
  ["completed", "failed", "blocked", "cancelled", "interrupted"].includes(String(status || "").toLowerCase());

export const delay = (milliseconds: number, signal?: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const timeout = window.setTimeout(resolve, milliseconds);
    signal?.addEventListener(
      "abort",
      () => {
        window.clearTimeout(timeout);
        reject(new DOMException("Cancelled", "AbortError"));
      },
      { once: true },
    );
  });

export function withoutBlankSecret<T extends Record<string, unknown>>(payload: T): T {
  if (String(payload.api_key || "").trim()) return payload;
  const { api_key: _apiKey, ...rest } = payload;
  return rest as T;
}
