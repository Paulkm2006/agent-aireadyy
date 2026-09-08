import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ChatCustomElement,
  BusEventType,
  MessageResponseTypes,
  MessageState,
  type ChatInstance,
  type MessageRequest,
  type MessageResponse,
  type PublicConfigMessaging,
} from "@carbon/ai-chat";

import {
  renderCodexUserDefined,
  timelineMessage,
  tl,
  type TimelineToolStatus,
  type TimelineEvent,
} from "./CodexTimeline";
import {
  buildDiscoveryRunView,
  discoveryProgressMessage,
  renderDiscoveryProgressUserDefined,
} from "./DiscoveryProgressMessage";
import {
  discoveryRecoveryMessage,
  formatRecoveryFailureDetail,
  renderDiscoveryRecoveryUserDefined,
  type DiscoveryRecoveryPayload,
  type RecoveryAction,
} from "./DiscoveryRecoveryMessage";

import {
  assessStrategyGaps,
  applyRecommendedDefaults,
  detectNextStepCommand,
  formatConfirmMessage,
  formatDoneMessage,
  intentSnapshotForLlm,
  isReadyForConfirm,
  normalizeSearchTerms,
  toDiscoveryJobPayload,
} from "./grill-tree";
import {
  appendAgentDialogue,
  decodeAgentTurnResponse,
  formatAgentNextDecision,
  isStrategyFingerprint,
  reduceAgentTurn,
  reduceAgentUnavailable,
  type AgentDialogueMessage,
  type AgentNextDecision,
  type AgentResolvedDecision,
  type AgentSemanticVerification,
} from "./agent-turn";
import { createEmptyIntent, type GrillPhase, type IntentSpec } from "./intent-spec";
import {
  cancelDiscoveryJob,
  resumeDiscoveryJob,
  delay,
  getDiscoveryJob,
  grillTurn,
  startDiscoveryJob,
  type DiscoveryJob,
  type GrillTurnResult,
  type WorkflowRecord,
} from "./workflow-api";
import {
  freshDialogueSessionId,
  startDialogueSessionId,
  storeDialogueSessionId,
} from "./dialogue-session";
import {
  canonicalDiscoveryPayloadFingerprint,
  canonicalDiscoveryPayloadJson,
} from "./strategy-fingerprint";

export type GrillControls = {
  confirm: () => void;
  applyDefaults: () => void;
};

export type GrillExternalCommand =
  | { type: "confirm"; queryTerms: string[] }
  | { type: "defaults" };

type Props = {
  onJob: (job: DiscoveryJob) => void;
  onIntentChange: (spec: IntentSpec) => void;
  onPhaseChange: (phase: GrillPhase) => void;
  onNavigate?: (tabIndex: number) => void;
  /** Open the discovery result modal for the bound terminal job (L1 / batch seed). */
  onOpenResults?: (jobId: string) => void;
  onRegisterControls?: (controls: GrillControls | null) => void;
  externalCommand?: GrillExternalCommand | null;
  onExternalCommandConsumed?: () => void;
  /** Controlled repository terms shared by desktop, mobile, and chat confirmation. */
  externalSelectedSearchTerms?: string[];
};

const text = (value: unknown) => String(value || "").trim();
const terminal = (job: DiscoveryJob | null) =>
  ["completed", "failed", "blocked", "cancelled", "interrupted", "durability_failed"].includes(
    String(job?.status || "").toLowerCase(),
  );

const EXPLICIT_BULK_TERM_SUFFIX =
  /(?:^|[。.!！；;,，])\s*(?:(?:把)?这些(?:关键词|检索词|主题词|词)(?:也)?(?:要)?(?:加(?:上|入)?|添加|作为主题词)(?:到主题词)?|(?:加(?:上|入)?|添加)这些(?:关键词|检索词|主题词|词))(?:\s*[,，、]\s*(?:搜|搜索|检索)(?:得)?(?:更)?全面(?:一些)?)?\s*[。.!！]*$/u;

export function parseExplicitBulkSearchTermAddition(value: unknown): string[] | null {
  const safe = String(value || "")
    .replace(/[\r\n\t]+/g, "、")
    .replace(
      /([\p{L}\p{N}])[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]+(?=[\p{L}\p{N}])/gu,
      "$1-",
    )
    .replace(/[\u0000-\u001F\u007F]+/g, " ");
  const command = EXPLICIT_BULK_TERM_SUFFIX.exec(safe);
  if (!command || command.index == null) return null;
  const terms = normalizeSearchTerms(
    safe
      .slice(0, command.index)
      .split(/[、,，;；]+/u)
      .map((term) =>
        term.replace(
          /^[\s:："'“”‘’]+|[\s。.!！:："'“”‘’]+$/gu,
          "",
        ),
      ),
  );
  return terms.length >= 2 ? terms : null;
}

/** Allow the server's bounded 180s turn, plus a 10s transport/UI margin. */
export const AGENT_TURN_TIMEOUT_MS = 190_000;

type AgentTurnRequestContext = {
  sessionId: string;
  generation: number;
  intentSnapshot: WorkflowRecord;
  snapshotKey: string;
};

type ConfirmedAgentExecution = {
  request: AgentTurnRequestContext;
  agentStrategyFingerprint: string;
};

function joinAssistantParts(...parts: Array<string | null | undefined>): string {
  return parts
    .map((part) => String(part || "").trim())
    .filter(Boolean)
    .join("\n\n");
}

function jobMessage(job: DiscoveryJob, responseId: string, requestId?: string): MessageResponse {
  return discoveryProgressMessage(responseId, buildDiscoveryRunView(job), requestId);
}
function captureIntentSnapshot(spec: IntentSpec): Pick<AgentTurnRequestContext, "intentSnapshot" | "snapshotKey"> {
  const snapshotKey = JSON.stringify(intentSnapshotForLlm(spec));
  return {
    intentSnapshot: JSON.parse(snapshotKey) as WorkflowRecord,
    snapshotKey,
  };
}

function intentSnapshotKey(spec: IntentSpec): string {
  return JSON.stringify(intentSnapshotForLlm(spec));
}

function semanticVerificationTimelineEvent(
  verification: AgentSemanticVerification | null,
): TimelineEvent | null {
  if (!verification) return null;
  const status: TimelineToolStatus = verification.verdict === "rejected"
    ? "error"
    : verification.verdict === "unavailable" || verification.verdict === "budget_exhausted"
      ? "fallback"
      : "ok";
  const detail = [
    verification.verdict,
    verification.rationale,
    verification.error,
    ...verification.errors,
  ].filter(Boolean).join(" · ");
  return tl.tool("semantic-verification", status, detail || verification.verdict);
}

export function CarbonAgentChat({
  onJob,
  onIntentChange,
  onPhaseChange,
  onNavigate,
  onOpenResults,
  onRegisterControls,
  externalCommand,
  onExternalCommandConsumed,
  externalSelectedSearchTerms = [],
}: Props) {
  const welcomed = useRef(false);
  const instanceRef = useRef<ChatInstance | null>(null);
  const phaseRef = useRef<GrillPhase>("idle");
  const intentRef = useRef<IntentSpec>(createEmptyIntent());
  const runningRef = useRef(false);
  const dialogueHistoryRef = useRef<AgentDialogueMessage[]>([]);
  const pendingDecisionRef = useRef<AgentNextDecision | null>(null);
  const decisionMemoryRef = useRef<AgentResolvedDecision[]>([]);
  const [initialDialogueSessionId] = useState(() => startDialogueSessionId());
  const dialogueSessionIdRef = useRef<string>(initialDialogueSessionId);
  const dialogueGenerationRef = useRef(0);
  const inFlightGrillControllersRef = useRef(new Set<AbortController>());
  const restartHandlerRegisteredRef = useRef(false);
  /** Last terminal checkpoint for recovery chips (job-bound). */
  const lastRecoveryRef = useRef<DiscoveryRecoveryPayload | null>(null);
  const recoveryHandlerRef = useRef<((action: RecoveryAction, payload: DiscoveryRecoveryPayload) => void) | null>(null);

  const setPhase = useCallback(
    (phase: GrillPhase) => {
      phaseRef.current = phase;
      onPhaseChange(phase);
    },
    [onPhaseChange],
  );

  const invalidateGrillRequests = useCallback(() => {
    dialogueGenerationRef.current += 1;
    for (const controller of inFlightGrillControllersRef.current) controller.abort();
    inFlightGrillControllersRef.current.clear();
  }, []);

  const abandonDialogueSession = useCallback(() => {
    // An aborted HTTP request may still finish in the server worker and append
    // its rejected turn to SQLite. Move the live card to a fresh canonical
    // session before accepting another message; request-carried history and
    // decision memory preserve the accepted conversation without replaying the
    // late turn.
    invalidateGrillRequests();
    const nextSessionId = freshDialogueSessionId();
    dialogueSessionIdRef.current = nextSessionId;
    storeDialogueSessionId(nextSessionId);
  }, [invalidateGrillRequests]);

  const setIntent = useCallback(
    (spec: IntentSpec) => {
      const strategyChanged = intentSnapshotKey(spec) !== intentSnapshotKey(intentRef.current);
      if (strategyChanged && inFlightGrillControllersRef.current.size > 0) {
        // Any card mutation invalidates every turn captured from the previous
        // snapshot. Rotate the server-side SDK session as well as aborting the
        // HTTP request so a worker that finishes late cannot taint later turns.
        abandonDialogueSession();
      }
      intentRef.current = spec;
      onIntentChange(spec);
    },
    [abandonDialogueSession, onIntentChange],
  );

  const externalSelectedSearchTermsKey = externalSelectedSearchTerms.join("\u0000");
  useEffect(() => {
    const currentKey = (intentRef.current.selectedSearchTerms || []).join("\u0000");
    if (currentKey === externalSelectedSearchTermsKey) return;
    if (inFlightGrillControllersRef.current.size > 0) {
      abandonDialogueSession();
    }
    // The parent already owns and renders this controlled selection. Mirror it
    // into the execution snapshot without emitting a redundant intent update.
    intentRef.current = {
      ...intentRef.current,
      selectedSearchTerms: [...externalSelectedSearchTerms],
      confirmed: false,
    };
  }, [
    abandonDialogueSession,
    externalSelectedSearchTermsKey,
    externalSelectedSearchTerms,
  ]);

  const resetDialogueState = useCallback(() => {
    invalidateGrillRequests();
    const nextSessionId = freshDialogueSessionId();
    dialogueSessionIdRef.current = nextSessionId;
    storeDialogueSessionId(nextSessionId);
    dialogueHistoryRef.current = [];
    pendingDecisionRef.current = null;
    decisionMemoryRef.current = [];
    lastRecoveryRef.current = null;
    welcomed.current = false;
    setIntent(createEmptyIntent());
    setPhase("idle");
  }, [invalidateGrillRequests, setIntent, setPhase]);

  useEffect(() => () => invalidateGrillRequests(), [invalidateGrillRequests]);

  const isRequestIdentityCurrent = useCallback((request: AgentTurnRequestContext) => (
    request.generation === dialogueGenerationRef.current
    && request.sessionId === dialogueSessionIdRef.current
  ), []);

  const isRequestSnapshotCurrent = useCallback((request: AgentTurnRequestContext) => (
    isRequestIdentityCurrent(request)
    && request.snapshotKey === intentSnapshotKey(intentRef.current)
  ), [isRequestIdentityCurrent]);

  const pushAssistant = useCallback(async (body: string, id?: string) => {
    const instance = instanceRef.current;
    if (!instance) return;
    await instance.messaging.addMessage({
      id: id || `grill-${crypto.randomUUID()}`,
      output: { generic: [{ response_type: MessageResponseTypes.TEXT, text: body }] },
    });
  }, []);

  const pushRecoveryCheckpoint = useCallback(
    async (
      outcome: "done" | "failed",
      job: DiscoveryJob | null,
      summary: string,
    ) => {
      const instance = instanceRef.current;
      if (!instance) return;
      const view = job ? buildDiscoveryRunView(job) : null;
      const jobId =
        String(job?.job_id || "").trim() || `checkpoint:${crypto.randomUUID()}`;
      const discoveryId = String(
        view?.discoveryId
        || (job as WorkflowRecord | null)?.discovery_id
        || (job?.record as WorkflowRecord | undefined)?.discovery_id
        || "",
      ).trim();
      const status = String(job?.status || "").toLowerCase();
      const hasResults = Boolean(
        job
        && discoveryId
        && (status === "completed" || status === "blocked"),
      );
      const payload: DiscoveryRecoveryPayload = {
        kind: "discovery_recovery",
        jobId,
        discoveryId,
        cardGeneration: intentSnapshotKey(intentRef.current),
        outcome,
        hasResults,
        summary: String(summary || "").replace(/\s+/g, " ").trim().slice(0, 240),
      };
      lastRecoveryRef.current = payload;
      await instance.messaging.addMessage(
        discoveryRecoveryMessage(`recovery:${jobId}:${crypto.randomUUID()}`, payload),
      );
    },
    [],
  );

  const runDiscovery = useCallback(
    async (
      confirmation: ConfirmedAgentExecution,
      requestId?: string,
      signal?: AbortSignal,
    ) => {
      if (runningRef.current) return;
      if (
        !isStrategyFingerprint(confirmation.agentStrategyFingerprint)
        || phaseRef.current !== "awaiting_confirm"
        || !isRequestSnapshotCurrent(confirmation.request)
      ) {
        await pushAssistant("确认绑定的策略快照已经失效；本次没有启动搜索，请重新确认当前策略。");
        return;
      }
      if (!intentRef.current.confirmed) {
        await pushAssistant("还差最后一步：确认右侧策略后我才会去 PRIDE 搜。");
        return;
      }
      if (intentRef.current.repository !== "local" && !(intentRef.current.selectedSearchTerms || []).map(text).filter(Boolean).length) {
        setIntent({ ...intentRef.current, confirmed: false });
        setPhase("awaiting_confirm");
        await pushAssistant("请至少选择或补充一个实际检索词；本次没有访问仓库。");
        return;
      }

      const confirmed: IntentSpec = {
        ...intentRef.current,
        runHorizon: "candidates_reviewed",
        answered: { ...intentRef.current.answered, Q2: true },
        confirmed: true,
      };
      setIntent(confirmed);
      setPhase("running");
      runningRef.current = true;

      let job: DiscoveryJob | null = null;
      const responseId = `discovery:${crypto.randomUUID()}`;
      const instance = instanceRef.current;

      try {
        const canonicalPayload = toDiscoveryJobPayload(confirmed) as unknown as WorkflowRecord;
        const strategyFingerprintPayload = canonicalDiscoveryPayloadJson(canonicalPayload);
        const strategyFingerprint = await canonicalDiscoveryPayloadFingerprint(canonicalPayload);
        const payload: WorkflowRecord = {
          ...canonicalPayload,
          strategy_fingerprint_payload: strategyFingerprintPayload,
          strategy_fingerprint: strategyFingerprint,
        };
        job = await startDiscoveryJob(payload, signal);
        onJob(job);

        const update = (state: MessageState) => {
          if (!instance) return Promise.resolve();
          return instance.messaging.upsertMessage(responseId, state, () =>
            jobMessage(job || {}, responseId, requestId),
          );
        };

        await update(MessageState.STREAMING);
        while (!terminal(job)) {
          await delay(1000, signal);
          job = await getDiscoveryJob(String(job.job_id), false, signal);
          onJob(job);
          await update(MessageState.STREAMING);
        }
        if (job.status === "completed" || job.status === "blocked") {
          job = await getDiscoveryJob(String(job.job_id), true, signal);
          onJob(job);
        }
        await update(MessageState.COMPLETE);

        if (job.status === "completed") {
          setPhase("done");
          const doneText = formatDoneMessage(job, confirmed);
          await pushAssistant(doneText);
          await pushRecoveryCheckpoint("done", job, doneText);
        } else if (job.status === "failed") {
          setPhase("failed");
          const detail = formatRecoveryFailureDetail(
            (typeof job.error === "string" && job.error.trim()) || "",
          );
          const failText = `数据发现失败：${detail}`;
          await pushAssistant(failText);
          await pushRecoveryCheckpoint("failed", job, failText);
        } else if (job.status === "blocked") {
          setPhase("done");
          const record = (job.record || {}) as WorkflowRecord;
          const summary = (record.summary || {}) as WorkflowRecord;
          const audit = (record.latest_discovery_audit || summary.latest_discovery_audit || {}) as WorkflowRecord;
          const auditCounts = (audit.counts || {}) as WorkflowRecord;
          const candidates = Number(summary.candidate_projects ?? record.project_count ?? 0);
          const usable = Number(
            auditCounts.usable_files ?? summary.usable_files ?? record.usable_files ?? record.file_count ?? 0,
          );
          const strictValid = Number(auditCounts.strict_valid_files ?? summary.strict_valid_files ?? 0);
          const inherited = Number(
            auditCounts.inherited_usable_files ?? summary.inherited_usable_files ?? 0,
          );
          const gaps = Array.isArray(audit.issues)
            ? (audit.issues as WorkflowRecord[])
                .map((issue) => String(issue.summary || issue.code || "").trim())
                .filter(Boolean)
                .slice(0, 3)
            : [];
          const blockedText = [
            `搜索与审查已结束：约 **${candidates}** 个候选项目，**${usable}** 条验证通过的可交付文件。`,
            strictValid > 0
              ? `其中 ${inherited} 个采用项目级同质证据继承；文件级明确冲突仍会被排除。`
              : `本轮尚无验证通过的可交付文件；未决文件保留在等待复核列表，不会混入下载清单。`,
            gaps.length ? `当前主要不足：${gaps.join("；")}。` : "",
            "请在右侧打开结果：下载「可用批量输入」，或点「送入批量参数规划」继续构建标准化格式。",
          ]
            .filter(Boolean)
            .join("\n");
          await pushAssistant(blockedText);
          await pushRecoveryCheckpoint("done", job, blockedText);
        } else {
          setPhase("done");
          await pushRecoveryCheckpoint(
            "done",
            job,
            `任务已结束（状态 ${String(job.status || "unknown")}）。可继续改策略或重置对话。`,
          );
        }
      } catch (reason) {
        if (reason instanceof DOMException && reason.name === "AbortError" && job?.job_id) {
          onJob(await cancelDiscoveryJob(job.job_id));
          setPhase("idle");
          // rethrow abort so Carbon stop-button path completes cleanly
          throw reason;
        }
        setPhase("failed");
        const failDetail = formatRecoveryFailureDetail(
          reason instanceof Error ? reason.message : String(reason),
        );
        const failText = `启动或运行数据发现失败：${failDetail}`;
        await pushAssistant(failText);
        await pushRecoveryCheckpoint("failed", job, failText);
        // do not rethrow non-abort errors — avoid Carbon catastrophic "Something went wrong"
      } finally {
        runningRef.current = false;
      }
    },
    [isRequestSnapshotCurrent, onJob, pushAssistant, pushRecoveryCheckpoint, setIntent, setPhase],
  );

  const applyDefaultsCore = useCallback((): IntentSpec => {
    const filled = applyRecommendedDefaults(intentRef.current);
    pendingDecisionRef.current = null;
    setIntent(filled);
    setPhase("awaiting_confirm");
    return filled;
  }, [setIntent, setPhase]);

  const handleDefaults = useCallback(async () => {
    const filled = applyDefaultsCore();
    await pushAssistant(
      "行，缺口我先用稳妥默认补齐了——你仍可改。确认后才搜。\n\n" +
        formatConfirmMessage(filled),
    );
  }, [applyDefaultsCore, pushAssistant]);

  const handleConfirm = useCallback(async (queryTerms?: string[]) => {
    if (phaseRef.current === "running") return;
    const rawSelectedTerms = queryTerms ?? intentRef.current.selectedSearchTerms ?? [];
    if (rawSelectedTerms.some((term) => String(term || "").trim().replace(/\s+/g, " ").length > 240)) {
      await pushAssistant("单个检索词不能超过 240 个字符；请缩短后重新确认。本次没有访问仓库。");
      return;
    }
    const selectedTerms = normalizeSearchTerms(rawSelectedTerms).slice(0, 100);
    if (intentRef.current.repository !== "local" && !selectedTerms.length) {
      await pushAssistant("请至少选择或补充一个实际检索词；本次没有访问仓库。");
      return;
    }
    if (intentRef.current.repository === "local" && !text(intentRef.current.localDir)) {
      await pushAssistant("请先选择或填写本地数据集目录；本次没有访问仓库。");
      return;
    }
    if (queryTerms || selectedTerms.join("\u0000") !== (intentRef.current.selectedSearchTerms || []).join("\u0000")) {
      setIntent({
        ...intentRef.current,
        selectedSearchTerms: selectedTerms,
        confirmed: false,
      });
    }
    if (phaseRef.current !== "awaiting_confirm" || !isReadyForConfirm(intentRef.current)) {
      await pushAssistant(
        "当前策略还没有进入待确认状态；本次没有启动搜索。请先继续和 Agent 对齐策略，或用右侧默认设置补齐。",
      );
      return;
    }
    const capturedSnapshot = captureIntentSnapshot(intentRef.current);
    const request: AgentTurnRequestContext = {
      sessionId: dialogueSessionIdRef.current,
      generation: dialogueGenerationRef.current,
      ...capturedSnapshot,
    };
    const strategyFingerprint = await canonicalDiscoveryPayloadFingerprint(
      capturedSnapshot.intentSnapshot,
    );
    const trustedButtonTurn = decodeAgentTurnResponse({
      status: "completed",
      action: "confirm_strategy",
      assistant_message: "",
      strategy_fingerprint: strategyFingerprint,
      tool_calls: [{
        name: "confirm_strategy",
        arguments: { strategy_fingerprint: strategyFingerprint },
      }],
    });
    const reduction = reduceAgentTurn(intentRef.current, trustedButtonTurn, {
      phase: phaseRef.current,
    });
    if (!isRequestSnapshotCurrent(request) || !reduction.confirmationAccepted) {
      await pushAssistant(
        "The strategy changed before confirmation could be bound. Nothing was started; please review and confirm the current card again.",
      );
      return;
    }
    setIntent(reduction.spec);
    setPhase("awaiting_confirm");
    await runDiscovery({
      request,
      agentStrategyFingerprint: strategyFingerprint,
    });
  }, [isRequestSnapshotCurrent, pushAssistant, runDiscovery, setIntent, setPhase]);

  useEffect(() => {
    onRegisterControls?.({
      confirm: () => {
        void handleConfirm();
      },
      applyDefaults: () => {
        void handleDefaults();
      },
    });
    return () => onRegisterControls?.(null);
  }, [handleConfirm, handleDefaults, onRegisterControls]);

  useEffect(() => {
    if (!externalCommand) return;
    if (externalCommand.type === "confirm") {
      void handleConfirm(externalCommand.queryTerms);
    }
    if (externalCommand.type === "defaults") void handleDefaults();
    onExternalCommandConsumed?.();
  }, [externalCommand, handleConfirm, handleDefaults, onExternalCommandConsumed]);

  const handleRecoveryAction = useCallback(
    async (action: RecoveryAction, payload: DiscoveryRecoveryPayload) => {
      const bound = lastRecoveryRef.current;
      if (!bound || bound.jobId !== payload.jobId) {
        await pushAssistant("这组恢复操作已过期；请以最新完成/失败气泡上的芯片为准。");
        return;
      }

      if (action === "view_results") {
        if (!bound.hasResults || !bound.jobId) {
          await pushAssistant("本轮没有可打开的结果绑定（缺少 job / L1 交付）。");
          return;
        }
        if (onOpenResults) {
          onOpenResults(bound.jobId);
        } else {
          onNavigate?.(0);
        }
        await pushAssistant(
          bound.discoveryId
            ? `已打开本轮结果（job ${bound.jobId}）。可在右侧下载 L1 或送入批量。`
            : `已定位本轮任务 ${bound.jobId}；若右侧无结果入口，请检查任务状态。`,
        );
        return;
      }

      if (action === "revise_strategy") {
        if (runningRef.current || phaseRef.current === "running") {
          await pushAssistant("当前已有数据发现任务在跑。可点停止取消，或等它完成后再改策略。");
          return;
        }
        // Stay in the same scientific conversation; do not auto-search.
        setIntent({ ...intentRef.current, confirmed: false });
        setPhase("grilling");
        await pushAssistant(
          "好，我们先改策略。直接说要改的字段（例如物种、终点、数量）；确认前不会访问 PRIDE。",
        );
        return;
      }

      if (action === "research_current_card") {
        // No dual job: never start while another discovery is running.
        if (runningRef.current || phaseRef.current === "running") {
          await pushAssistant("当前已有数据发现任务在跑，不会双开新搜索。可点停止取消，或等它完成。");
          return;
        }
        // Boss condition: re-search defaults to a fresh SDK session.
        abandonDialogueSession();
        const next = { ...intentRef.current, confirmed: false };
        setIntent(next);
        if (!isReadyForConfirm(next)) {
          setPhase("grilling");
          await pushAssistant(
            "已开新会话准备按当前卡再搜，但策略仍有缺口，还不能确认。先补齐关键字段，再确认后才会启动 PRIDE。",
          );
          return;
        }
        setPhase("awaiting_confirm");
        await pushAssistant(
          "已开新会话，并按**当前策略卡**进入待确认。\n\n"
          + "重新搜索需要再次确认（不会自动开搜，也不会假绿降级）。\n\n"
          + formatConfirmMessage(next),
        );
        return;
      }

      if (action === "reset_dialogue") {
        resetDialogueState();
        await pushAssistant(
          "对话已重置。策略卡与历史已清空；直接说新的数据需求即可。",
        );
      }
    },
    [
      abandonDialogueSession,
      onNavigate,
      onOpenResults,
      pushAssistant,
      resetDialogueState,
      setIntent,
      setPhase,
    ],
  );

  useEffect(() => {
    recoveryHandlerRef.current = (action, payload) => {
      void handleRecoveryAction(action, payload);
    };
    return () => {
      recoveryHandlerRef.current = null;
    };
  }, [handleRecoveryAction]);

  const addWelcomeMessage = useCallback(async (instance: ChatInstance) => {
    instanceRef.current = instance;
    if (!restartHandlerRegisteredRef.current) {
      instance.on({
        type: BusEventType.RESTART_CONVERSATION,
        handler: () => resetDialogueState(),
      });
      restartHandlerRegisteredRef.current = true;
    }
    if (welcomed.current) return;
    welcomed.current = true;
    await instance.messaging.addMessage({
      id: "pride-agent-welcome",
      output: {
        generic: [
          {
            response_type: MessageResponseTypes.TEXT,
            text:
              "你好，我是蛋白质组学数据 Agent——帮你在 PRIDE 里找对数据，而不是填表。\n\n" +
              "直接说目标就行，例如：人源免疫肽、先只要 20 个候选；或「DDA 做人源 RT 预测」。\n" +
              "术语不懂就问我；我会按你的任务给建议。右侧是实时策略，**你确认前我不会去搜**。\n" +
              "想省事可以说 **按推荐默认**。",
          },
        ],
      },
    });
  }, [resetDialogueState]);

  const lastGrillErrorRef = useRef("");
  const callGrillTurn = useCallback(
    async (
      prompt: string,
      opts: {
        phase: GrillPhase;
        turnKind: string;
        request: AgentTurnRequestContext;
        signal?: AbortSignal;
        timeoutMs?: number;
      },
    ): Promise<GrillTurnResult | null> => {
      const controller = new AbortController();
      inFlightGrillControllersRef.current.add(controller);
      const onOuterAbort = () => controller.abort();
      if (opts.signal?.aborted) controller.abort();
      else opts.signal?.addEventListener("abort", onOuterAbort);
      // DeepSeek 冷启动/JSON 补全偶发 >25s；给足余量，避免误报「超时」
      const timeoutMs = opts.timeoutMs ?? AGENT_TURN_TIMEOUT_MS;
      const timer = window.setTimeout(
        () => {
          lastGrillErrorRef.current =
            `模型响应超过 ${Math.round(timeoutMs / 1_000)} 秒，已停止本轮；策略没有被修改。`;
          controller.abort();
        },
        timeoutMs,
      );
      lastGrillErrorRef.current = "";
      try {
        const payload = {
          user_message: prompt,
          phase: opts.phase,
          turn_kind: opts.turnKind,
          pending_question: null,
          pending_decision: pendingDecisionRef.current,
          decision_memory: decisionMemoryRef.current,
          resolved_fields: intentRef.current.resolvedFields || [],
          session_id: opts.request.sessionId,
          intent_snapshot: opts.request.intentSnapshot,
          answered: intentRef.current.answered as unknown as WorkflowRecord,
          local_summary: "",
          allow_server_default: true,
          // Reserve a small network/UI margin while giving the backend enough
          // total time for Manager -> read-only verifier -> Manager repair.
          request_timeout_seconds: Math.max(1, Math.floor((timeoutMs - 10_000) / 1_000)),
          gap_report: assessStrategyGaps(intentRef.current) as unknown as WorkflowRecord,
          dialogue_history: dialogueHistoryRef.current,
          ...(opts.phase === "awaiting_confirm"
            ? { pending_strategy_snapshot: opts.request.intentSnapshot }
            : {}),
        } as Parameters<typeof grillTurn>[0] & {
          pending_strategy_snapshot?: WorkflowRecord;
        };
        const turn = await grillTurn(payload, controller.signal);
        if (controller.signal.aborted) {
          if (isRequestIdentityCurrent(opts.request)) abandonDialogueSession();
          return null;
        }
        return turn;
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err || "unknown");
        if (isRequestIdentityCurrent(opts.request)) {
          if (controller.signal.aborted) {
            abandonDialogueSession();
          } else {
            lastGrillErrorRef.current = msg.slice(0, 180);
          }
        }
        return null;
      } finally {
        window.clearTimeout(timer);
        opts.signal?.removeEventListener("abort", onOuterAbort);
        inFlightGrillControllersRef.current.delete(controller);
      }
    },
    [abandonDialogueSession, isRequestIdentityCurrent],
  );

  const messaging = useMemo<PublicConfigMessaging>(
    () => ({
      skipWelcome: true,
      messageTimeoutSecs: 0,
      messageLoadingIndicatorTimeoutSecs: 1,
      showStopButtonImmediately: true,
      async customSendMessage(request: MessageRequest, requestOptions, instance) {
        instanceRef.current = instance;
        const prompt = text(request.input.text);
        const responseId = `msg:${crypto.randomUUID()}`;
        const events: TimelineEvent[] = [];
        let lastBody = "";
        const reply = async (body: string, state: MessageState = MessageState.COMPLETE) => {
          const safeBody = String(body || "").trim() || lastBody || "…";
          if (String(body || "").trim()) lastBody = String(body).trim();
          await instance.messaging.upsertMessage(responseId, state, () =>
            timelineMessage(responseId, safeBody, [...events], {
              requestId: request.id,
            }),
          );
        };
        const action = (line: string) => {
          events.push(tl.action(line));
        };
        const toolCall = async (name: string, ok: boolean, detail: string, elapsedMs?: number) => {
          events.push(tl.tool(name, ok ? "ok" : "fallback", detail, elapsedMs));
          // Progressive Codex-like refresh so tool cards appear before the final answer.
          await reply(lastBody || "处理中…", MessageState.STREAMING);
        };

        if (!prompt) {
          await reply("随便说你的数据需求就行，也可以问我术语。");
          return;
        }

        let activeRequest: AgentTurnRequestContext | null = null;
        let expectedSnapshotKey = "";
        // Capture identity checker under a stable local name so minifiers cannot
        // later reuse the same short identifier for setInterval and break S().
        const identityStillCurrent = isRequestIdentityCurrent;
        const requestIsCurrent = () => activeRequest != null
          && identityStillCurrent(activeRequest)
          && expectedSnapshotKey === intentSnapshotKey(intentRef.current);

        try {
          // Next-step navigation after done
          if (phaseRef.current === "done" || phaseRef.current === "failed") {
            const step = detectNextStepCommand(prompt);
            if (step) {
              const tab = step === "single" ? 1 : step === "batch" ? 2 : 3;
              onNavigate?.(tab);
              await reply(
                step === "single"
                  ? "已切到「单文件处理」。请在该页填写 accession 后手动开始。"
                  : step === "batch"
                    ? "已切到「批量处理」。请上传/粘贴清单后手动开始。"
                    : "已切到「AI-ready 构建」。请确认输入后手动开始。",
              );
              return;
            }
          }

          if (phaseRef.current === "running") {
            await reply("当前已有数据发现任务在跑。可点停止取消，或等它完成。");
            return;
          }

          const explicitBulkTerms = parseExplicitBulkSearchTermAddition(prompt);
          if (explicitBulkTerms) {
            const previousTerms = normalizeSearchTerms(
              intentRef.current.selectedSearchTerms || [],
            );
            const previousKeys = new Set(
              previousTerms.map((term) => term.toLocaleLowerCase()),
            );
            const mergedTerms = normalizeSearchTerms([
              ...previousTerms,
              ...explicitBulkTerms,
            ]).slice(0, 100);
            const addedTerms = mergedTerms.filter(
              (term) => !previousKeys.has(term.toLocaleLowerCase()),
            );
            const nextSpec: IntentSpec = {
              ...intentRef.current,
              selectedSearchTerms: mergedTerms,
              confirmed: false,
            };
            pendingDecisionRef.current = null;
            setIntent(nextSpec);
            setPhase(isReadyForConfirm(nextSpec) ? "awaiting_confirm" : "grilling");
            const responseBody =
              `已直接解析 ${explicitBulkTerms.length} 个关键词：新增 ${addedTerms.length} 个，` +
              `忽略重复 ${explicitBulkTerms.length - addedTerms.length} 个。` +
              `当前共有 ${mergedTerms.length} 个待确认主题词；未调用模型，也未开始 PRIDE 搜索。`;
            events.push(
              tl.tool(
                "update-search-terms",
                "ok",
                `deterministic · ${addedTerms.length} added · ${mergedTerms.length} total`,
                0,
              ),
            );
            dialogueHistoryRef.current = appendAgentDialogue(
              dialogueHistoryRef.current,
              prompt,
              responseBody,
            );
            await reply(responseBody, MessageState.COMPLETE);
            return;
          }

          // D1 normal online path: one grill-turn owns the reply and action.
          // Natural-language confirmation also goes through this Agent boundary.
          const fallbackPhase = phaseRef.current;
          // Completion is a checkpoint in the same scientific conversation,
          // not a new questionnaire. Re-enter dialogue with the accepted card,
          // SDK session, history, and decision memory intact. Only Restart
          // performs a full reset.
          const turnPhase: GrillPhase = fallbackPhase === "done" || fallbackPhase === "failed"
            ? "grilling"
            : fallbackPhase;
          const capturedSnapshot = captureIntentSnapshot(intentRef.current);
          activeRequest = {
            sessionId: dialogueSessionIdRef.current,
            generation: dialogueGenerationRef.current,
            ...capturedSnapshot,
          };
          expectedSnapshotKey = capturedSnapshot.snapshotKey;
          setPhase("grilling");
          await reply(
            "正在调用模型对齐策略（通常 30～90 秒，请稍候）…",
            MessageState.STREAMING,
          );
          const onlineStarted = performance.now();
          // Heartbeat so the bubble never looks "frozen" while waiting on DeepSeek.
          const progressTimer = window.setInterval(() => {
            const sec = Math.round((performance.now() - onlineStarted) / 1000);
            void reply(
              `仍在等待模型（已 ${sec}s）。可点击停止按钮取消；确认前不会访问 PRIDE。`,
              MessageState.STREAMING,
            );
          }, 1_000);
          let agentTurn: GrillTurnResult | null = null;
          try {
            agentTurn = await callGrillTurn(prompt, {
              phase: turnPhase,
              turnKind: "agent_turn",
              request: activeRequest,
              signal: requestOptions.signal,
            });
          } finally {
            window.clearInterval(progressTimer);
          }
          // Stale identity after Restart/generation bump must not write late bubbles
          // (Carbon already owns the reset UI). Only close STREAMING when the HTTP
          // request was not explicitly aborted — otherwise a late worker finish
          // resurrects a wiped conversation.
          if (!requestIsCurrent()) {
            if (!requestOptions.signal?.aborted) {
              await reply(
                agentTurn
                  ? `本轮结果已返回，但对话上下文已变化（例如重复发送/重置），未写入策略。请再发一次需求。`
                  : lastGrillErrorRef.current
                    ? `${lastGrillErrorRef.current} 请直接重试，无需刷新页面。`
                    : `本轮被中断或会话已过期。请再发一次；若反复发生请 Ctrl+F5 强制刷新。`,
                MessageState.COMPLETE,
              );
            }
            return;
          }
          if (agentTurn) {
            await toolCall(
              "agent-turn",
              true,
              `${agentTurn.action} · ${agentTurn.tool_calls.length} tool call(s)`,
              Math.round(performance.now() - onlineStarted),
            );
            // NI-1: pure chat never shows semantic-verification chrome.
            // Write turns demoted to advise by an authoritative SV reject still surface SV.
            const verificationEvent =
              agentTurn.action === "chat"
                ? null
                : semanticVerificationTimelineEvent(agentTurn.semantic_verification);
            if (verificationEvent) {
              events.push(verificationEvent);
              await reply(lastBody || "处理中…", MessageState.STREAMING);
            }
            pendingDecisionRef.current = agentTurn.next_decision;
            decisionMemoryRef.current = agentTurn.decision_memory;
            const reduction = reduceAgentTurn(intentRef.current, agentTurn, { phase: turnPhase });
            if (reduction.strategyUpdated || reduction.confirmationAccepted) {
              setIntent(reduction.spec);
              expectedSnapshotKey = intentSnapshotKey(reduction.spec);
            }
            if (reduction.strategyUpdated && agentTurn.strategy_patch) {
              await toolCall(
                "update_strategy",
                true,
                `更新字段：${Object.keys(agentTurn.strategy_patch).join("、")}`,
              );
              action("Agent 已更新策略");
            }
            if (reduction.confirmationRequested) {
              await toolCall(
                "confirm_strategy",
                reduction.confirmationAccepted,
                reduction.confirmationAccepted
                  ? "已在待确认状态接受 Agent 的确认决策"
                  : reduction.confirmationFingerprint == null
                    ? "确认指纹缺失或不匹配；未启动搜索"
                    : "当前并非可接受确认的状态；未启动搜索",
              );
            }

            const nextDecisionText = formatAgentNextDecision(agentTurn.next_decision);
            const confirmationGuardText = reduction.confirmationRequested && !reduction.confirmationAccepted
              ? "本轮没有启动搜索。我先保留或展示策略预览，确认只能在策略进入待确认状态后生效。"
              : "";
            const responseBody = joinAssistantParts(
              agentTurn.assistant_message || "我在听，你可以继续用自然语言说明判断。",
              confirmationGuardText,
              nextDecisionText,
              reduction.showConfirmation ? formatConfirmMessage(reduction.spec) : "",
            );
            dialogueHistoryRef.current = appendAgentDialogue(
              dialogueHistoryRef.current,
              prompt,
              responseBody,
            );

            if (reduction.confirmationAccepted) {
              setPhase("awaiting_confirm");
              await reply(responseBody, MessageState.COMPLETE);
              if (reduction.confirmationFingerprint && activeRequest) {
                await runDiscovery({
                  request: activeRequest,
                  agentStrategyFingerprint: reduction.confirmationFingerprint,
                }, request.id, requestOptions.signal);
              }
              return;
            }

            setPhase(reduction.awaitingConfirmation ? "awaiting_confirm" : "grilling");
            await reply(responseBody, MessageState.COMPLETE);
            return;
          }

          const unavailable = reduceAgentUnavailable(intentRef.current, turnPhase);
          setPhase(unavailable.phase);
          await toolCall(
            "agent-turn",
            false,
            lastGrillErrorRef.current
              ? `Agent \u8bf7\u6c42\u5931\u8d25\uff1a${lastGrillErrorRef.current}`
              : "Agent \u672a\u8fd4\u56de\u53ef\u9a8c\u8bc1\u7684\u5bf9\u8bdd\u52a8\u4f5c",
            Math.round(performance.now() - onlineStarted),
          );

          const failureReply = unavailable.assistantMessage;
          dialogueHistoryRef.current = appendAgentDialogue(
            dialogueHistoryRef.current,
            prompt,
            failureReply,
          );
          await reply(failureReply, MessageState.COMPLETE);
          return;
        } catch (reason) {
          const msg = reason instanceof Error ? reason.message : String(reason);
          try {
            await reply(
              `处理时出了点问题：${msg}。本轮没有修改策略，也没有启动搜索；请稍后重试。`,
              MessageState.COMPLETE,
            );
          } catch {
            // swallow — never bubble to Carbon catastrophic panel
          }
        }
      },
    }),
    [
      callGrillTurn,
      isRequestIdentityCurrent,
      onNavigate,
      runDiscovery,
      setIntent,
      setPhase,
    ],
  );

  return (
    <ChatCustomElement
      className="carbon-chat-host"
      namespace="PRIDE Agent"
      assistantName="蛋白质组学数据 Agent"
      onAfterRender={addWelcomeMessage}
      openChatByDefault
      shouldTakeFocusIfOpensAutomatically={false}
      shouldSanitizeHTML
      aiEnabled
      launcher={{ isOn: false }}
      layout={{ showFrame: false, hasContentMaxWidth: false }}
      header={{
        title: "数据搜集 Agent",
        name: "意图澄清 → 数据发现",
        hideMinimizeButton: true,
        showRestartButton: true,
      }}
      messaging={messaging}
      renderUserDefinedResponse={(state) =>
        renderDiscoveryProgressUserDefined(state, (jobId) => {
          if (!jobId) return;
          void cancelDiscoveryJob(jobId).then(onJob);
        }, (jobId) => {
          if (!jobId) return;
          void resumeDiscoveryJob(jobId).then(onJob);
        })
        ?? renderDiscoveryRecoveryUserDefined(state, (action, payload) => {
          recoveryHandlerRef.current?.(action, payload);
        })
        ?? renderCodexUserDefined(state)
      }
      strings={{
        window_title: "蛋白质组学数据 Agent",
        input_placeholder: "用自然语言聊需求，或问术语 / 确认…",
        input_ariaLabel: "描述数据需求",
      }}
    />
  );
}
