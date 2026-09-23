import { Button, Tag, Tile } from "@carbon/react";
import { Checkmark, Restart } from "@carbon/icons-react";
import { useState } from "react";

import type { GrillPhase, IntentSpec } from "./intent-spec";
import {
  COVERAGE_LABELS,
  RUN_HORIZON_LABELS,
  TASK_TYPE_LABELS,
} from "./intent-spec";
import {
  assessStrategyGaps,
  buildStrategyCard,
  normalizeSearchTerms,
  searchTermCandidates,
} from "./grill-tree";

type Props = {
  spec: IntentSpec;
  phase: GrillPhase;
  onConfirm: (queryTerms: string[]) => void;
  onIntentChange?: (spec: IntentSpec) => void;
  selectedSearchTerms?: string[];
  onSelectedSearchTermsChange?: (queryTerms: string[]) => void;
  onApplyDefaults: () => void;
  busy?: boolean;
};

const PHASE_LABEL: Record<GrillPhase, string> = {
  idle: "待开始",
  grilling: "对话澄清中",
  awaiting_confirm: "待确认策略",
  running: "搜索中",
  done: "已完成",
  failed: "失败",
};

const INSTRUMENT_LABEL: Record<string, string> = {
  none: "不限定",
  newer: "较新仪器优先",
  classic: "经典仪器优先",
  newer_with_legacy_floor: "新仪器为主，保留经典数据",
};

const LABELING_LABEL: Record<string, string> = {
  label_free: "Label-free",
  tmt: "TMT",
  itraq: "iTRAQ",
  silac: "SILAC",
  dimethyl: "二甲基标记",
  unknown: "开放",
  any: "开放",
};

const SAFETY_LABEL: Record<string, string> = {
  ask: "触达安全上限时询问",
  auto_continue_within_safety: "安全上限内自动继续",
  stop: "触达安全上限即停止",
};

const TIME_BUDGET_LABEL: Record<string, string> = {
  fast: "快速单轮",
  multi_round: "允许多轮补搜",
};

const MIXED_ACQUISITION_LABEL: Record<string, string> = {
  reject_mixed: "混合项目整项排除",
  review_mixed: "混合项目文件级审查",
  allow: "混合项目可保留",
};

type KeyValue = { label: string; value: string; pending?: boolean };

function phaseTagType(phase: GrillPhase): "blue" | "purple" | "green" | "red" | "gray" {
  if (phase === "running") return "blue";
  if (phase === "awaiting_confirm") return "purple";
  if (phase === "done") return "green";
  if (phase === "failed") return "red";
  return "gray";
}

function compactStrategy(spec: IntentSpec): {
  objective: string;
  keyValues: KeyValue[];
  speciesExplicit: boolean;
  acquisitionExplicit: boolean;
  labelingExplicit: boolean;
} {
  const resolved = new Set(spec.resolvedFields || []);
  const speciesExplicit = spec.species.length > 0 || resolved.has("species") || resolved.has("species_policy");
  const mixedAcquisitionExplicit = resolved.has("mixed_acquisition_policy")
    || spec.mixedAcquisitionPolicy !== "review_mixed";
  const acquisitionExplicit = Boolean(spec.acquisitionMode)
    || resolved.has("acquisition_mode")
    || mixedAcquisitionExplicit;
  const labelingExplicit = Boolean(spec.labelingStrategy) || resolved.has("labeling_strategy");

  const task = spec.taskType
    ? TASK_TYPE_LABELS[spec.taskType] || spec.taskType
    : "待补齐";
  const species = spec.species.length
    ? `${spec.speciesPolicy === "include_only" ? "仅 " : spec.speciesPolicy === "prefer" ? "优先 " : ""}${spec.species.join("、")}`
    : speciesExplicit
      ? "开放"
      : "待决定";
  const scale = spec.quotaFlexibility === "open_ended"
    ? "尽可能多（安全上限内）"
    : spec.targetProjectCount != null
    ? `${COVERAGE_LABELS[spec.coverageMode] ? `${COVERAGE_LABELS[spec.coverageMode]} · ` : ""}${
        spec.quotaFlexibility === "fixed" ? `固定 ${spec.targetProjectCount} 个项目（待搜索）` : `约 ${spec.targetProjectCount} 个项目`
      }`
    : spec.quotaFlexibility === "fixed"
      ? "固定数量待补齐"
    : spec.coverageMode
      ? `${COVERAGE_LABELS[spec.coverageMode] || spec.coverageMode} · 数量待定`
      : "待补齐";

  const keyValues: KeyValue[] = [
    { label: "下游任务", value: task, pending: !spec.taskType },
    { label: "物种", value: species, pending: !speciesExplicit },
    {
      label: "规模",
      value: scale,
      pending: spec.quotaFlexibility === "fixed"
        ? spec.targetProjectCount == null
        : spec.quotaFlexibility !== "open_ended"
          && spec.targetProjectCount == null
          && !spec.coverageMode,
    },
  ];

  const extras: KeyValue[] = [];
  if (acquisitionExplicit) {
    const acquisition = spec.acquisitionMode === "dda" || spec.acquisitionMode === "dia"
      ? spec.acquisitionMode.toUpperCase()
      : "开放";
    extras.push({
      label: "采集方式",
      value: `${acquisition} · ${MIXED_ACQUISITION_LABEL[spec.mixedAcquisitionPolicy] || spec.mixedAcquisitionPolicy}`,
    });
  }
  if (spec.instrumentPreference) {
    extras.push({ label: "仪器", value: INSTRUMENT_LABEL[spec.instrumentPreference] || spec.instrumentPreference });
  }
  if (labelingExplicit) {
    extras.push({ label: "标记", value: LABELING_LABEL[spec.labelingStrategy] || spec.labelingStrategy || "开放" });
  }
  if (!extras.length) extras.push({ label: "采集方式", value: "待决定", pending: true });
  keyValues.push(...extras.slice(0, 1));

  return {
    objective: spec.objective.trim() || "还没有形成可执行搜索目标",
    keyValues,
    speciesExplicit,
    acquisitionExplicit,
    labelingExplicit,
  };
}

export function IntentSpecPanel({
  spec,
  phase,
  onConfirm,
  onIntentChange,
  selectedSearchTerms = [],
  onSelectedSearchTermsChange = () => undefined,
  onApplyDefaults,
  busy,
}: Props) {
  const card = buildStrategyCard(spec);
  const searchThemes = Array.from(new Set([
    ...searchTermCandidates(spec),
    ...selectedSearchTerms,
  ]));
  const [customSearchTerm, setCustomSearchTerm] = useState("");
  const compact = compactStrategy(spec);
  const gaps = assessStrategyGaps(spec);
  const canConfirm =
    phase === "awaiting_confirm" &&
    gaps.ready_for_confirm &&
    (spec.repository === "local" ? Boolean((spec.localDir || "").trim()) : selectedSearchTerms.length > 0) &&
    !busy;
  const canDefaults = (phase === "grilling" || phase === "idle" || phase === "awaiting_confirm") && !busy;

  const hardConstraints = card.hardConstraints.filter((item) => item !== "（无额外硬约束）");
  const softPreferences = card.softPreferences.filter((item) => {
    if (item === "（无额外软偏好）") return false;
    if (item === "物种开放" && !compact.speciesExplicit) return false;
    if (item === "采集方式不限" && !compact.acquisitionExplicit) return false;
    if (item === "标记：unknown" && !compact.labelingExplicit) return false;
    return true;
  });
  const requiredCount = gaps.required_missing.length;
  const optionalCount = gaps.optional_missing.length;
  const readiness = requiredCount
    ? `还差 ${requiredCount} 个关键决定`
    : optionalCount
      ? `可确认 · ${optionalCount} 个可选项仍可调整`
      : "策略已可确认";

  return (
    <Tile className="intent-spec-panel">
      <div className="intent-spec-panel__header">
        <div>
          <p className="eyebrow">LIVE STRATEGY</p>
          <h2>策略预览</h2>
        </div>
        <Tag size="sm" type={phaseTagType(phase)}>{PHASE_LABEL[phase] || phase}</Tag>
      </div>

      <div className="strategy-objective">
        <span>当前目标</span>
        <strong>{compact.objective}</strong>
      </div>

      <div className="strategy-source">
        <label htmlFor="dataset-source">数据来源</label>
        <select
          id="dataset-source"
          value={spec.repository === "local" ? "local" : "pride"}
          disabled={busy}
          onChange={(event) => onIntentChange?.({
            ...spec,
            repository: event.target.value === "local" ? "local" : "pride",
            localDir: event.target.value === "local" ? spec.localDir : "",
            confirmed: false,
          })}
        >
          <option value="pride">从 PRIDE 搜索并下载</option>
          <option value="local">使用本地数据集</option>
        </select>
        {spec.repository === "local" ? (
          <input
            aria-label="本地数据集目录"
            placeholder="输入服务器可访问的数据集目录，例如 /data/my-dataset"
            value={spec.localDir || ""}
            disabled={busy}
            onChange={(event) => onIntentChange?.({ ...spec, localDir: event.target.value, confirmed: false })}
          />
        ) : null}
      </div>

      <section className="strategy-search-themes" aria-label="待确认检索主题词">
        <span>开始前请确认检索主题词</span>
        {searchThemes.length ? (
          <ul className="strategy-search-themes__list" tabIndex={0}>
            {searchThemes.map((theme) => {
              const selected = selectedSearchTerms.includes(theme);
              const primary = selected && selectedSearchTerms[0] === theme;
              return (
                <li key={theme}>
                  <button
                    type="button"
                    aria-pressed={selected}
                    disabled={busy || phase !== "awaiting_confirm"}
                    onClick={() => onSelectedSearchTermsChange(
                      selectedSearchTerms.includes(theme)
                        ? selectedSearchTerms.filter((term) => term !== theme)
                        : [...selectedSearchTerms, theme],
                    )}
                  >
                    {selected ? "✓ " : ""}{theme}
                  </button>
                  {primary ? (
                    <span className="strategy-search-themes__primary">核心词</span>
                  ) : selected ? (
                    <button
                      type="button"
                      className="strategy-search-themes__make-primary"
                      disabled={busy || phase !== "awaiting_confirm"}
                      onClick={() => onSelectedSearchTermsChange([
                        theme,
                        ...selectedSearchTerms.filter((term) => term !== theme),
                      ])}
                    >
                      设为核心
                    </button>
                  ) : null}
                </li>
              );
            })}
          </ul>
        ) : (
          <p>尚未形成可执行主题词。</p>
        )}
        <div className="strategy-search-themes__custom">
          <input
            aria-label="添加自定义检索词"
            placeholder="补充仓库里可能出现的写法"
            maxLength={240}
            value={customSearchTerm}
            disabled={busy || phase !== "awaiting_confirm"}
            onChange={(event) => setCustomSearchTerm(event.target.value)}
          />
          <button
            type="button"
            disabled={busy || phase !== "awaiting_confirm" || !customSearchTerm.trim()}
            onClick={() => {
              const next = normalizeSearchTerms([...selectedSearchTerms, customSearchTerm]);
              onSelectedSearchTermsChange(next);
              setCustomSearchTerm("");
            }}
          >
            添加
          </button>
        </div>
        <p>这些是同一主题在项目元数据里的常见写法；请选择要实际执行的词，并指定一个核心词。核心词优先获得更深检索预算。确认前不会访问仓库。</p>
      </section>

      <dl className="strategy-key-values">
        {compact.keyValues.map((item) => (
          <div key={item.label} className={item.pending ? "is-pending" : ""}>
            <dt>{item.label}</dt>
            <dd>{item.value}</dd>
          </div>
        ))}
      </dl>

      <p className={`strategy-readiness${requiredCount ? " strategy-readiness--pending" : ""}`}>
        {readiness}
      </p>

      <details className="strategy-details">
        <summary>查看完整策略</summary>
        <div className="strategy-details__body">
          <dl className="strategy-details__meta">
            <div>
              <dt>本次终点</dt>
              <dd>{RUN_HORIZON_LABELS[spec.runHorizon] || "待补齐"}</dd>
            </div>
            <div>
              <dt>仓库</dt>
              <dd>{spec.repository || "PRIDE"}</dd>
            </div>
            <div>
              <dt>安全上限</dt>
              <dd>{SAFETY_LABEL[spec.onSafetyCeiling] || spec.onSafetyCeiling}</dd>
            </div>
            {spec.timeBudget ? (
              <div>
                <dt>时间策略</dt>
                <dd>{TIME_BUDGET_LABEL[spec.timeBudget] || spec.timeBudget}</dd>
              </div>
            ) : null}
            {spec.maxCandidateProjects != null && spec.quotaFlexibility !== "open_ended" ? (
              <div>
                <dt>候选池</dt>
                <dd>最多约 {spec.maxCandidateProjects} 个项目</dd>
              </div>
            ) : null}
            {spec.quotaFlexibility === "open_ended" ? (
              <>
                <div>
                  <dt>候选池</dt>
                  <dd>不设项目数量上限（仍受时间、请求量和资源安全上限约束）</dd>
                </div>
                <div>
                  <dt>分批交付</dt>
                  <dd>每 30 个经项目级证据验证的可用项目交付一批</dd>
                </div>
              </>
            ) : null}
          </dl>

          <section>
            <h3>硬条件</h3>
            {hardConstraints.length ? (
              <ul>{hardConstraints.map((item) => <li key={item}>{item}</li>)}</ul>
            ) : <p>尚未设置额外硬过滤。</p>}
          </section>
          <section>
            <h3>软偏好</h3>
            {softPreferences.length ? (
              <ul>{softPreferences.map((item) => <li key={item}>{item}</li>)}</ul>
            ) : <p>尚未设置额外排序偏好。</p>}
          </section>
          {spec.notes.trim() ? (
            <section><h3>补充要求</h3><p>{spec.notes}</p></section>
          ) : null}
          {spec.openRisks.length ? (
            <section>
              <h3>待核实约束</h3>
              <ul>{spec.openRisks.map((item) => <li key={item}>{item}</li>)}</ul>
            </section>
          ) : null}
          <p className="strategy-note">{card.safetyNote}</p>
        </div>
      </details>

      <p className="strategy-mutation-note">只有明确的策略更新会改卡；确认前不会访问数据源。</p>
      <div className="button-row intent-actions">
        <Button kind="tertiary" size="sm" renderIcon={Restart} disabled={!canDefaults} onClick={onApplyDefaults}>
          补齐稳妥默认
        </Button>
        <Button
          kind="primary"
          size="sm"
          renderIcon={Checkmark}
          disabled={!canConfirm}
          onClick={() => onConfirm(selectedSearchTerms)}
        >
          {spec.repository === "local" ? "确认目录，开始处理" : "确认主题词，开始搜"}
        </Button>
      </div>
    </Tile>
  );
}
