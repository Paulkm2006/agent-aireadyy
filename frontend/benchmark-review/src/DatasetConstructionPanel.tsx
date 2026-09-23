import { useEffect, useMemo, useState } from "react";
import {
  Button, InlineNotification, NumberInput, ProgressBar, Select, SelectItem,
  Tag, TextInput, Tile,
} from "@carbon/react";
import { Download, Play, Renew, StopFilled } from "@carbon/icons-react";

import {
  cancelOperationsJob, datasetArtifactUrl, getOperationsEvents, getOperationsJob,
  operationsTerminal, resumeOperationsJob, submitDatasetConstructionJob,
  type OperationsEvent, type OperationsJob,
} from "./operations-api";

type Props = { batchOutputDir?: string; taskType?: string };

const protocols = [
  "row_random_control", "file_disjoint", "project_disjoint", "lab_disjoint",
  "instrument_disjoint", "organism_disjoint", "peptide_disjoint",
  "protein_family_disjoint", "modification_disjoint", "acquisition_disjoint",
];
const phases = ["queued", "ingesting", "validating", "identity_ledger", "planning", "auditing", "finalizing", "completed"];
const phaseLabel: Record<string, string> = {
  queued: "排队", ingesting: "读取 Batch", validating: "标签与数据合同",
  identity_ledger: "身份台账", planning: "十类拆分", auditing: "独立泄漏审计",
  finalizing: "冻结发布", completed: "完成", failed: "失败", cancelled: "已取消",
};

const normalizeTaskType = (value: string) => ({
  rt: "rt_prediction",
  retention_time: "rt_prediction",
  fragment_intensity: "fragment_intensity_prediction",
}[value] || value || "denovo");

function tagType(status: string): "green" | "red" | "gray" | "blue" | "magenta" {
  if (["pass", "ready", "completed"].includes(status)) return "green";
  if (["fail", "failed"].includes(status)) return "red";
  if (["inconclusive", "infeasible", "cancelled"].includes(status)) return "magenta";
  return ["queued", "pending"].includes(status) ? "gray" : "blue";
}

export function DatasetConstructionPanel({ batchOutputDir = "", taskType = "denovo" }: Props) {
  const [batchDir, setBatchDir] = useState(batchOutputDir);
  const [outputDir, setOutputDir] = useState(batchOutputDir ? `${batchOutputDir}\\dataset-releases\\release-v1` : "");
  const [releaseId, setReleaseId] = useState("dataset-release-v1");
  const [selectedTask, setSelectedTask] = useState(normalizeTaskType(taskType));
  const [benchmarkProfile, setBenchmarkProfile] = useState("standard");
  const [seed, setSeed] = useState(42);
  const [job, setJob] = useState<OperationsJob | null>(null);
  const [events, setEvents] = useState<OperationsEvent[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!batchOutputDir) return;
    setBatchDir(batchOutputDir);
    setOutputDir((current) => current || `${batchOutputDir}\\dataset-releases\\release-v1`);
  }, [batchOutputDir]);

  useEffect(() => {
    setSelectedTask(normalizeTaskType(taskType));
  }, [taskType]);

  useEffect(() => {
    if (selectedTask !== "denovo") setBenchmarkProfile("standard");
  }, [selectedTask]);

  useEffect(() => {
    if (!job?.job_id || operationsTerminal(job.status)) return;
    const controller = new AbortController();
    const refresh = async () => {
      const next = await getOperationsJob(job.job_id, controller.signal);
      setJob(next);
      const page = await getOperationsEvents(job.job_id, 0, 200, controller.signal);
      setEvents(page.items || []);
    };
    const timer = window.setInterval(
      () => void refresh().catch((reason) => setError(String(reason))), 1200,
    );
    void refresh().catch((reason) => setError(String(reason)));
    return () => { controller.abort(); window.clearInterval(timer); };
  }, [job?.job_id, job?.status]);

  const protocolStatuses = useMemo(() => {
    const event = [...events].reverse().find((item) => item.type === "dataset_split_suite_planned");
    return (event?.payload?.protocol_statuses || {}) as Record<string, string>;
  }, [events]);
  const auditStatuses = useMemo(() => {
    const event = [...events].reverse().find((item) => item.type === "dataset_leakage_audited");
    return (event?.payload?.audit_statuses || {}) as Record<string, string>;
  }, [events]);
  const coverage = useMemo(() => {
    const event = [...events].reverse().find((item) => item.type === "dataset_identity_ledger_built");
    return event?.payload || {};
  }, [events]);
  const benchmarkEvidence = useMemo(() => {
    const event = [...events].reverse().find((item) =>
      item.type === "dataset_contract_validated" || item.type === "dataset_ingestion_completed"
    );
    return (event?.payload?.benchmark_profile || {}) as Record<string, unknown>;
  }, [events]);
  const phaseIndex = Math.max(0, phases.indexOf(job?.phase || "queued"));

  const submit = async () => {
    setBusy(true); setError("");
    try {
      const next = await submitDatasetConstructionJob({
        batch_dir: batchDir, output_dir: outputDir, release_id: releaseId,
        task_spec: {
          task_type: selectedTask,
          version: 1,
          ...(benchmarkProfile === "diverse_denovo_v1" ? { benchmark_profile: benchmarkProfile } : {}),
        },
        ratios: [0.7, 0.15, 0.15], seed,
        policy: {
          peptide_identity_mode: "il_equivalent",
          modification_identity_mode: benchmarkProfile === "diverse_denovo_v1" ? "peptidoform" : "class",
        },
        idempotency_key: crypto.randomUUID(),
      });
      setJob(next); setEvents([]);
    } catch (reason) { setError(String(reason)); } finally { setBusy(false); }
  };

  return <div className="workspace-stack" data-testid="dataset-construction-panel">
    {error && <InlineNotification kind="error" title="数据集构建失败" subtitle={error} onCloseButtonClick={() => setError("")} />}
    <Tile>
      <div className="panel-heading"><div><p className="eyebrow">LEAKAGE-AWARE DATASET</p><h2>公平训练数据集构建</h2><p className="empty-copy">保留现有 Discovery 和 Batch；从已完成的 Batch 产物构建十类可审计的 train/validation/test split。</p></div></div>
      <div className="form-grid form-grid--four">
        <TextInput id="dataset-batch-dir" labelText="Batch 输出目录" value={batchDir} onChange={(event) => setBatchDir(event.target.value)} />
        <TextInput id="dataset-output-dir" labelText="Release 输出目录" value={outputDir} onChange={(event) => setOutputDir(event.target.value)} />
        <TextInput id="dataset-release-id" labelText="Release ID" value={releaseId} onChange={(event) => setReleaseId(event.target.value)} />
        <Select id="dataset-task-type" labelText="模型任务" value={selectedTask} onChange={(event) => setSelectedTask(event.target.value)}>
          <SelectItem value="denovo" text="De novo" /><SelectItem value="ptm_denovo" text="PTM de novo" />
          <SelectItem value="fragment_intensity_prediction" text="Fragment intensity" /><SelectItem value="rt_prediction" text="Retention time" />
          <SelectItem value="psm_scoring" text="PSM scoring" />
        </Select>
        <Select id="dataset-benchmark-profile" labelText="构建配置" value={benchmarkProfile} disabled={selectedTask !== "denovo"} onChange={(event) => setBenchmarkProfile(event.target.value)}>
          <SelectItem value="standard" text="标准数据集" />
          <SelectItem value="diverse_denovo_v1" text="多样性 De novo Benchmark（严格）" />
        </Select>
        <NumberInput id="dataset-seed" label="随机种子" min={0} value={seed} onChange={(_event, state) => setSeed(Number(state.value || 0))} />
      </div>
      {benchmarkProfile === "diverse_denovo_v1" && <InlineNotification kind="info" title="严格 De novo Benchmark" subtitle="要求 DDA、FragPipe∩Sage 各 1% FDR、指定仪器/碎裂/长短梯度/多酶/至少 3 类 PTM；证据不足时阻止发布。" hideCloseButton />}
      <Button renderIcon={Play} disabled={busy || !batchDir || !outputDir || !releaseId} onClick={() => void submit()}>构建十类拆分</Button>
    </Tile>
    {job && <Tile>
      <div className="panel-heading"><div><p className="eyebrow">JOB {job.job_id}</p><h2>{phaseLabel[job.phase] || job.phase}</h2></div><Tag type={tagType(job.status)}>{job.status}</Tag></div>
      <ProgressBar label="完整构建流程" helperText={`${phaseLabel[job.phase] || job.phase} · ${events.length} 条持久化事件`} value={operationsTerminal(job.status) ? 100 : (phaseIndex / (phases.length - 1)) * 100} max={100} size="big" status={job.status === "failed" ? "error" : job.status === "completed" ? "finished" : "active"} />
      {job.error?.message && <InlineNotification kind="error" title={job.error.code || "任务失败"} subtitle={job.error.message} hideCloseButton />}
      <div className="metric-strip"><span>Observation <strong>{String(job.result?.observation_count || "—")}</strong></span><span>Identity 维度 <strong>{String(coverage.dimension_count || "—")}</strong></span><span>缺失维度 <strong>{Array.isArray(coverage.incomplete_dimensions) ? coverage.incomplete_dimensions.length : "—"}</strong></span>{benchmarkProfile === "diverse_denovo_v1" && <span>Benchmark gate <strong>{String(benchmarkEvidence.status || "pending")}</strong></span>}</div>
      <div className="run-list" style={{ display: "grid", gap: "0.5rem", marginTop: "1rem" }}>{protocols.map((protocol) => <div key={protocol} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", borderBottom: "1px solid var(--cds-border-subtle-01)", padding: "0.5rem 0" }}><code>{protocol}</code><div style={{ display: "flex", gap: "0.4rem" }}><Tag size="sm" type={tagType(protocolStatuses[protocol] || "pending")}>{protocolStatuses[protocol] || "pending"}</Tag>{auditStatuses[protocol] && <Tag size="sm" type={tagType(auditStatuses[protocol])}>audit: {auditStatuses[protocol]}</Tag>}</div></div>)}</div>
      <div className="button-row" style={{ marginTop: "1rem" }}><Button kind="tertiary" renderIcon={Renew} onClick={() => void getOperationsJob(job.job_id).then(setJob).catch((reason) => setError(String(reason)))}>刷新</Button>{!operationsTerminal(job.status) && <Button kind="danger" renderIcon={StopFilled} onClick={() => void cancelOperationsJob(job.job_id).then(setJob)}>安全停止</Button>}{job.resumable && job.status !== "completed" && <Button kind="secondary" onClick={() => void resumeOperationsJob(job.job_id).then(setJob)}>从断点恢复</Button>}{job.status === "completed" && <Button renderIcon={Download} href={datasetArtifactUrl(job.job_id, "release_manifest_json")}>下载 Release Manifest</Button>}{job.status === "completed" && benchmarkProfile === "diverse_denovo_v1" && <Button kind="secondary" renderIcon={Download} href={datasetArtifactUrl(job.job_id, "benchmark_quality_report_json")}>下载 Benchmark 报告</Button>}</div>
      <details style={{ marginTop: "1rem" }}><summary>持久化阶段事件</summary><pre className="json-view">{JSON.stringify(events, null, 2)}</pre></details>
    </Tile>}
  </div>;
}
