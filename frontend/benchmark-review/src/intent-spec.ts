/** Intent Spec + strategy card types for Grill (pre-discovery clarification). */

export type GrillPhase = "idle" | "grilling" | "awaiting_confirm" | "running" | "done" | "failed";

export type TaskType =
  | "rt_prediction"
  | "fragment_intensity_prediction"
  | "psm_scoring"
  | "denovo"
  | "ptm_denovo"
  | "chimeric_interpretation"
  | "browse_only"
  | "other"
  | "";

export type RunHorizon =
  | "plan_only"
  | "candidates_only"
  | "candidates_reviewed"
  | "ai_ready_table"
  | "pre_release"
  | "full_release"
  | "";

export type SpeciesPolicy = "open" | "include_only" | "prefer" | "exclude";
export type AcquisitionMode = "dda" | "dia" | "unknown" | "";
export type CoverageMode = "quick" | "curated" | "balanced" | "exhaustive" | "";
export type LabelingStrategy =
  | "label_free"
  | "tmt"
  | "itraq"
  | "silac"
  | "dimethyl"
  | "unknown"
  | "any"
  | "";
export type InstrumentPreference = "none" | "newer" | "classic" | "newer_with_legacy_floor" | "";
export type OnSafetyCeiling = "ask" | "auto_continue_within_safety" | "stop";
export type TimeBudget = "fast" | "multi_round" | "";
export type MixedAcquisitionPolicy = "reject_mixed" | "review_mixed" | "allow";

export interface ScientificConstraint {
  id: string;
  label: string;
  dimension: string;
  operator: string;
  value: string | number | boolean | null | string[] | Record<string, unknown>;
  strength: "hard" | "soft";
  scope: "project" | "file" | "sample" | "portfolio";
  evidence_required: boolean;
  rationale?: string;
  source: "user" | "accepted_recommendation" | "inferred";
}

export type QuestionId =
  | "Q1"
  | "Q2"
  | "Q3"
  | "Q4"
  | "Q5"
  | "Q6"
  | "Q7"
  | "Q8"
  | "Q9"
  | "Q10";

export interface IntentSpec {
  objective: string;
  originalPrompt: string;
  taskType: TaskType;
  runHorizon: RunHorizon;
  species: string[];
  speciesPolicy: SpeciesPolicy;
  speciesCoverage: "none" | "prefer_listed" | "broaden";
  acquisitionMode: AcquisitionMode;
  mixedAcquisitionPolicy: MixedAcquisitionPolicy;
  ptmTypes: string[];
  specialThemes: string[];
  /** Exact repository query variants selected by the user at confirmation time. */
  selectedSearchTerms: string[];
  labelingStrategy: LabelingStrategy;
  labelingHard: boolean;
  coverageMode: CoverageMode;
  targetProjectCount: number | null;
  maxCandidateProjects: number | null;
  quotaFlexibility: "fixed" | "recommended" | "open_ended";
  timeBudget: TimeBudget;
  onSafetyCeiling: OnSafetyCeiling;
  instrumentPreference: InstrumentPreference;
  legacyFloorRatio: number | null;
  excludeRules: string[];
  successCriteria: string[];
  /** Open-ended, structured requirements that do not need a product code change. */
  scientificConstraints: ScientificConstraint[];
  notes: string;
  openRisks: string[];
  /** Canonical strategy fields explicitly decided, including intentional open values. */
  resolvedFields: string[];
  repository: string;
  localDir?: string;
  confirmed: boolean;
  answered: Partial<Record<QuestionId, boolean>>;
  inferred: Partial<Record<QuestionId, boolean>>;
  parseWarnings: string[];
  parseReasoning: string;
}

export interface StrategyCard {
  summaryLines: string[];
  hardConstraints: string[];
  softPreferences: string[];
  targetQuota: string;
  safetyNote: string;
  confirmButtonLabel: string;
}

export interface GrillOption {
  id: string;
  label: string;
  recommended?: boolean;
  reason?: string;
}

export interface GrillQuestion {
  id: QuestionId;
  prompt: string;
  why: string;
  options: GrillOption[];
  freeTextHint?: string;
}

export interface DiscoveryJobPayload {
  prompt: string;
  runtime: string;
  source: string;
  repository: string;
  local_dir: string;
  output_language: string;
  constraints_enabled: boolean;
  goal: string;
  query_terms: string[];
  task_type: string;
  acquisition_mode: string;
  labeling_strategy: string;
  labeling_hard: boolean;
  mixed_acquisition_policy: string;
  species: string[];
  species_policy: string;
  diversity_strategy: string;
  scale_mode: string;
  ptm_types: string[];
  max_projects: number;
  max_candidate_projects: number;
  continuous_discovery: boolean;
  partial_delivery_batch_size: number;
  inspection_batch_size: number;
  use_memory: boolean;
  save_memory: boolean;
  hard_constraint_fields: string[];
  constraint_provenance: Record<string, string>;
  idempotency_key: string;
  grill_confirmed: boolean;
  run_horizon: string;
  quota_flexibility: string;
  quantity_scope: string;
  portfolio_size_preference: string | null;
  instrument_preference: string;
  exclude_rules: string[];
  success_criteria: string[];
  scientific_constraints: ScientificConstraint[];
  legacy_floor_ratio: number | null;
  on_safety_ceiling: string;
  time_budget_preference: string;
}

export const TASK_TYPE_LABELS: Record<string, string> = {
  rt_prediction: "RT 预测",
  fragment_intensity_prediction: "碎片强度预测",
  psm_scoring: "PSM 打分",
  denovo: "de novo",
  ptm_denovo: "PTM de novo",
  chimeric_interpretation: "嵌合谱解释",
  browse_only: "先只找数据、任务未定",
  other: "其它",
};

export const RUN_HORIZON_LABELS: Record<string, string> = {
  plan_only: "只对齐需求并给出搜索计划",
  candidates_only: "找到候选数据就停",
  candidates_reviewed: "找到并再审查一遍候选",
  ai_ready_table: "做到可训练的数据表",
  pre_release: "做到发布前候选版本",
  full_release: "走完正式发布",
};

export const COVERAGE_LABELS: Record<string, string> = {
  quick: "快速找够指定数量",
  curated: "快速找够指定数量",
  balanced: "均衡",
  exhaustive: "尽量搜全",
};

export function createEmptyIntent(prompt = ""): IntentSpec {
  return {
    objective: prompt.trim(),
    originalPrompt: prompt.trim(),
    taskType: "",
    runHorizon: "candidates_reviewed",
    species: [],
    speciesPolicy: "open",
    speciesCoverage: "none",
    acquisitionMode: "",
    mixedAcquisitionPolicy: "review_mixed",
    ptmTypes: [],
    specialThemes: [],
    selectedSearchTerms: [],
    labelingStrategy: "",
    labelingHard: false,
    coverageMode: "",
    targetProjectCount: null,
    maxCandidateProjects: null,
    quotaFlexibility: "recommended",
    timeBudget: "",
    onSafetyCeiling: "ask",
    instrumentPreference: "",
    legacyFloorRatio: null,
    excludeRules: [],
    successCriteria: [],
    scientificConstraints: [],
    notes: "",
    openRisks: [],
    resolvedFields: [],
    repository: "pride",
    localDir: "",
    confirmed: false,
    answered: { Q2: true },
    inferred: {},
    parseWarnings: [],
    parseReasoning: "",
  };
}
