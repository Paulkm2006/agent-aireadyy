import { useEffect, useState } from "react";
import {
  Button,
  ComposedModal,
  ModalBody,
  ModalHeader,
  ProgressBar,
  Tag,
  Tile,
} from "@carbon/react";
import { DataBase, Download } from "@carbon/icons-react";

import {
  buildDiscoveryRunView,
  DiscoveryProgressMessage,
  toDiscoveryProgressPayload,
  type DiscoveryRunStatus,
  type DiscoveryRunView,
} from "./DiscoveryProgressMessage";
import { IntentSpecPanel } from "./IntentSpecPanel";
import { PortfolioCoveragePanel } from "./PortfolioCoveragePanel";
import type { GrillPhase, IntentSpec } from "./intent-spec";
import { getDiscoveryBatchHandoff, type DiscoveryBatchHandoff, type DiscoveryJob } from "./workflow-api";

type Props = {
  spec: IntentSpec;
  phase: GrillPhase;
  job: DiscoveryJob | null;
  onConfirm: (queryTerms: string[]) => void;
  onIntentChange?: (spec: IntentSpec) => void;
  selectedSearchTerms?: string[];
  onSelectedSearchTermsChange?: (queryTerms: string[]) => void;
  onApplyDefaults: () => void;
  onNavigate: (tabIndex: number) => void;
  /** Load discovery L1 usable file list into batch planner. */
  onSeedBatchInputs?: (handoff: DiscoveryBatchHandoff) => void;
  /** Chat recovery chip: open result modal when job matches. */
  openResultRequest?: { jobId: string; token: number } | null;
};

function tagType(status: DiscoveryRunStatus): "blue" | "green" | "red" | "gray" | "magenta" | "purple" {
  if (status === "completed") return "green";
  if (status === "failed") return "red";
  if (status === "blocked") return "purple";
  if (status === "cancelled") return "gray";
  return "blue";
}

function RunStatusCard({
  view,
  portfolioState,
  onOpenResult,
}: {
  view: DiscoveryRunView | null;
  portfolioState?: unknown;
  onOpenResult: () => void;
}) {
  const active = view?.status === "queued" || view?.status === "running";
  const showProgress = Boolean(view && (active || view.progressPercent != null));
  const usable = view?.metrics.usableFiles ?? view?.metrics.selectedProjects ?? 0;

  return (
    <Tile className="discovery-run-card">
      <div className="section-title discovery-run-card__title">
        <DataBase size={20} />
        <h2>当前数据发现</h2>
      </div>
      {!view ? (
        <p className="empty-copy">确认检索策略后，将在此显示实时运行状态和关键指标。</p>
      ) : active ? (
        <>
          <DiscoveryProgressMessage payload={toDiscoveryProgressPayload(view)} />
          <PortfolioCoveragePanel state={portfolioState} />
        </>
      ) : (
        <>
          <div className="run-status">
            <Tag size="sm" type={tagType(view.status)}>{view.statusLabel}</Tag>
            <span>{view.statusDetail || view.summary}</span>
          </div>
          {showProgress ? (
            <ProgressBar
              label={
                view.progressPercent == null
                  ? "总体进度（服务端暂未提供百分比）"
                  : `总体进度 ${Math.round(view.progressPercent)}%`
              }
              value={view.progressPercent ?? undefined}
              max={100}
              size="small"
              status={
                view.status === "failed"
                  ? "error"
                  : view.status === "completed" || view.status === "blocked"
                    ? "finished"
                    : "active"
              }
            />
          ) : null}
          <dl className="context-metrics">
            <div><dt>项目</dt><dd>{view.metrics.projects}</dd></div>
            <div><dt>文件线索</dt><dd>{view.metrics.files}</dd></div>
            <div><dt>可交付文件</dt><dd>{usable}</dd></div>
            <div><dt>待复核</dt><dd>{view.metrics.reviews}</dd></div>
          </dl>

          {view.status === "completed" || view.status === "blocked" ? (
            <div className="discovery-result-entry">
              <p className="eyebrow">已验证文件清单</p>
              <h3>{view.status === "blocked" ? "可用文件清单已就绪" : "发现结果已就绪"}</h3>
              <p>
                {view.status === "blocked"
                  ? `约 ${usable} 条验证通过的可交付文件。项目证据继承会单独计数；未决文件不会混入下载清单。`
                  : view.summary}
              </p>
              <Button size="sm" onClick={onOpenResult}>查看结果与送入批量</Button>
            </div>
          ) : null}
          {view.status === "failed" ? (
            <div className="discovery-run-error" role="alert">
              <strong>这轮没有完成</strong>
              <p>{view.error || "可在左侧展开技术轨迹查看失败阶段。"}</p>
            </div>
          ) : null}
          {view.status === "cancelled" ? (
            <p className="empty-copy">已取消本轮搜索；可以继续修改策略后重新确认。</p>
          ) : null}
          <PortfolioCoveragePanel state={portfolioState} />
        </>
      )}
    </Tile>
  );
}

function ResultModal({
  view,
  open,
  onClose,
  onNavigate,
  onSeedBatchInputs,
}: {
  view: DiscoveryRunView | null;
  open: boolean;
  onClose: () => void;
  onNavigate: (tabIndex: number) => void;
  onSeedBatchInputs?: (handoff: DiscoveryBatchHandoff) => void;
}) {
  const [seedBusy, setSeedBusy] = useState<number | null>(null);
  const [seedError, setSeedError] = useState("");

  if (!open || !view || !["completed", "blocked"].includes(view.status)) return null;
  const base = view.discoveryId
    ? `/api/discovery/${encodeURIComponent(view.discoveryId)}/download?file=`
    : "";
  const usable = view.metrics.usableFiles ?? view.metrics.selectedProjects ?? 0;
  const strictValid = view.metrics.strictValidFiles ?? 0;

  const sendToBatch = async (batchIndex: number) => {
    if (!view.jobId) {
      setSeedError("缺少 discovery job id，无法拉取冻结批次。");
      return;
    }
    setSeedBusy(batchIndex);
    setSeedError("");
    try {
      const handoff = await getDiscoveryBatchHandoff(view.jobId, batchIndex);
      onSeedBatchInputs?.(handoff);
      onClose();
      onNavigate(2);
    } catch (reason) {
      setSeedError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setSeedBusy(null);
    }
  };

  return (
    <ComposedModal open onClose={onClose} size="lg">
      <ModalHeader
        label="DATA DISCOVERY"
        title={view.status === "blocked" ? "可用文件清单与说明" : "发现结果"}
        iconDescription="关闭结果"
        closeModal={onClose}
      />
      <ModalBody hasScrollingContent>
        <p className="discovery-result-summary">{view.summary}</p>
        <dl className="result-modal-metrics">
          <div><dt>候选项目</dt><dd>{view.metrics.projects}</dd></div>
          <div><dt>可交付文件</dt><dd>{usable}</dd></div>
          <div><dt>直接文件证据</dt><dd>{view.metrics.directUsableFiles ?? strictValid}</dd></div>
          <div><dt>项目证据继承</dt><dd>{view.metrics.inheritedUsableFiles ?? 0}</dd></div>
          <div><dt>文件线索</dt><dd>{view.metrics.files}</dd></div>
        </dl>

        <section className="result-modal-section">
          <h3>当前不足与待复核项</h3>
          {view.qualityIssues.length ? (
            <div className="discovery-audit-summary">
              <ul>{view.qualityIssues.map((issue) => <li key={issue}>{issue}</li>)}</ul>
              <p className="empty-copy">
                不足项不会混入可交付文件清单；已验证文件仍可下载并送入批量参数规划与标准化构建。
              </p>
            </div>
          ) : (
            <p className="empty-copy">未单独列出阻塞项；请以下载清单为准。</p>
          )}
        </section>

        <section className="result-modal-section">
          <h3>下载与批量</h3>
          {base ? (
            <div className="result-download-grid">
              <Button size="sm" renderIcon={Download} href={`${base}batch_inputs_usable`}>
                可用批量输入 L1
              </Button>
              <Button size="sm" kind="secondary" renderIcon={Download} href={`${base}dataset_manifest_usable_csv`}>
                可用数据清单 CSV
              </Button>
              <Button size="sm" kind="secondary" renderIcon={Download} href={`${base}agents_discovery_report_md`}>
                Agent 审查报告
              </Button>
              <Button size="sm" kind="tertiary" renderIcon={Download} href={`${base}discovery_run_bundle_zip`}>
                完整运行包
              </Button>
            </div>
          ) : (
            <p className="empty-copy">结果正在归档，下载入口尚未生成。</p>
          )}
          {seedError ? <p className="empty-copy" role="alert">{seedError}</p> : null}
          {view.resultBatches.length ? (
            <div className="history-list" style={{ marginTop: "0.75rem" }}>
              {view.resultBatches.map((batch) => (
                <div className="history-row" key={batch.batchIndex}>
                  <div>
                    <div className="history-title">文件批次 {batch.batchIndex}</div>
                    <div className="history-meta">
                      {batch.fileCount} 个文件 · {batch.projectCount} 个项目
                      {batch.cumulativeFileCount ? ` · 累计 ${batch.cumulativeFileCount}` : ""}
                      {` · ${batch.terminal ? "尾批已就绪" : "已就绪"}`}
                      {batch.publishedAt
                        ? ` · ${new Date(batch.publishedAt).toLocaleString("zh-CN", { hour12: false })}`
                        : ""}
                    </div>
                  </div>
                  <div className="button-row">
                    <Button size="sm" kind="ghost" renderIcon={Download} href={batch.downloadUrl}>下载清单</Button>
                    <Button
                      size="sm"
                      kind="primary"
                      disabled={seedBusy != null || !onSeedBatchInputs}
                      onClick={() => void sendToBatch(batch.batchIndex)}
                    >
                      {seedBusy === batch.batchIndex ? "正在加载…" : "送入批量处理"}
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <>
              <p className="empty-copy">尚未形成冻结文件批次；累计清单仅供下载，不会自动送入批量处理。</p>
              <Button size="sm" kind="primary" disabled>送入批量参数规划</Button>
            </>
          )}
          <div className="button-row" style={{ marginTop: "0.75rem" }}>
            <Button size="sm" kind="tertiary" onClick={() => { onClose(); onNavigate(2); }}>
              仅打开批量页
            </Button>
          </div>
        </section>
      </ModalBody>
    </ComposedModal>
  );
}

export function DiscoveryContextRail({
  spec,
  phase,
  job,
  onConfirm,
  onIntentChange,
  selectedSearchTerms,
  onSelectedSearchTermsChange,
  onApplyDefaults,
  onNavigate,
  onSeedBatchInputs,
  openResultRequest,
}: Props) {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [resultOpen, setResultOpen] = useState(false);
  const view = job ? buildDiscoveryRunView(job) : null;
  const active = view?.status === "queued" || view?.status === "running";
  const openResult = () => {
    setMobileOpen(false);
    setResultOpen(true);
  };

  useEffect(() => {
    if (!openResultRequest?.jobId || !job) return;
    const currentId = String(job.job_id || "").trim();
    if (!currentId || currentId !== openResultRequest.jobId) return;
    const status = String(job.status || "").toLowerCase();
    if (status !== "completed" && status !== "blocked") return;
    setResultOpen(true);
  }, [openResultRequest, job]);

  const intentPanel = (
    <IntentSpecPanel
      spec={spec}
      phase={phase}
      busy={phase === "running"}
      onConfirm={onConfirm}
      onIntentChange={onIntentChange}
      selectedSearchTerms={selectedSearchTerms}
      onSelectedSearchTermsChange={onSelectedSearchTermsChange}
      onApplyDefaults={onApplyDefaults}
    />
  );
  const runStatusCard = (
    <RunStatusCard
      view={view}
      portfolioState={job?.record?.portfolio_state}
      onOpenResult={openResult}
    />
  );
  const modalContent = (
    <div className="context-modal-stack">
      {intentPanel}
      {runStatusCard}
    </div>
  );

  return (
    <>
      <div className="context-rail-mobile-trigger">
        <Button
          kind="secondary"
          size="sm"
          aria-expanded={mobileOpen}
          onClick={() => setMobileOpen(true)}
        >
          查看策略与运行状态
        </Button>
      </div>
      <aside className="run-rail" aria-label="策略与运行上下文">
        {intentPanel}
        {!active ? runStatusCard : null}
      </aside>
      {active ? (
        <section className="run-progress-workspace" aria-label="运行进度主面板">
          {runStatusCard}
        </section>
      ) : null}

      {mobileOpen ? (
        <ComposedModal open onClose={() => setMobileOpen(false)} size="lg">
          <ModalHeader
            label="DISCOVERY CONTEXT"
            title="策略与运行状态"
            iconDescription="关闭上下文"
            closeModal={() => setMobileOpen(false)}
          />
          <ModalBody hasScrollingContent>
            {modalContent}
          </ModalBody>
        </ComposedModal>
      ) : null}

      <ResultModal
        view={view}
        open={resultOpen}
        onClose={() => setResultOpen(false)}
        onNavigate={onNavigate}
        onSeedBatchInputs={onSeedBatchInputs}
      />
    </>
  );
}
