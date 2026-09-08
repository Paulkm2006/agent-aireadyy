/**
 * Fixed decision-tree for Grill (C2).
 * LLM parse-goal only fills draft fields / phrasing; this tree decides what to ask.
 */

import {
  COVERAGE_LABELS,
  createEmptyIntent,
  RUN_HORIZON_LABELS,
  TASK_TYPE_LABELS,
  type AcquisitionMode,
  type CoverageMode,
  type DiscoveryJobPayload,
  type GrillQuestion,
  type IntentSpec,
  type LabelingStrategy,
  type QuestionId,
  type RunHorizon,
  type StrategyCard,
  type TaskType,
} from "./intent-spec";
import {
  businessCompletionAllowsSuccess,
  honestDiscoveryStatus,
} from "./workflow-api";

const text = (value: unknown) => String(value ?? "").trim();
const lower = (value: unknown) => text(value).toLowerCase();

const IMMUNO_HINT = /hla|mhc|immunopeptid|免疫肽|配体组|ligandome|neoantigen/i;

/** Display label for strategy card species bits. */
export function formatSpeciesLabel(species: string[]): string {
  const map: Record<string, string> = {
    human: "人源",
    mouse: "小鼠",
    fish: "鱼类",
    zebrafish: "斑马鱼",
  };
  return species.map((s) => map[s] || s).join("/");
}

/**
 * Detect species from free text. When multiple mentions appear (e.g. revise
 * after switching), the rightmost / latest mention wins so strategy cards
 * follow the conversation instead of sticky human defaults.
 */
export function detectSpeciesSignals(rawInput: string): {
  species: string[];
  policy: "include_only" | "prefer";
  coverage?: "broaden" | "none";
} | null {
  const raw = text(rawInput);
  if (!raw) return null;
  const blob = lower(raw);

  type Hit = { index: number; species: string[]; hardHint: boolean };
  const hits: Hit[] = [];

  const collect = (re: RegExp, species: string[]) => {
    const flags = re.flags.includes("g") ? re.flags : `${re.flags}g`;
    const r = new RegExp(re.source, flags);
    let m: RegExpExecArray | null;
    while ((m = r.exec(raw)) !== null) {
      const ctx = raw.slice(Math.max(0, m.index - 8), m.index + m[0].length + 8);
      const hardHint = /只要|仅|硬限制|only|必须|我要|换成|改成|切到/.test(ctx);
      hits.push({ index: m.index, species, hardHint });
    }
  };

  collect(/斑马鱼|zebrafish/gi, ["zebrafish"]);
  collect(/鱼类|鱼源|\bfish\b|salmon|鲑鱼|虹鳟/gi, ["fish"]);
  collect(/(?:^|[\s，,：:]|我要|只要|仅|换成|改成|有没有)鱼(?![类源苗])/g, ["fish"]);
  collect(/小鼠|\bmouse\b|mus musculus/gi, ["mouse"]);
  collect(/\bhuman\b|homo sapiens|人类|人源|智人/gi, ["human"]);
  collect(/(?:只要|优先|仅|必须)\s*人(?:类|源)?(?!工)/g, ["human"]);

  if (/^(人|人类|人源|human)$/i.test(raw.trim())) {
    hits.push({ index: 0, species: ["human"], hardHint: false });
  }
  if (/^(鱼|鱼类|fish|zebrafish|斑马鱼)$/i.test(raw.trim())) {
    hits.push({ index: 0, species: [/斑马鱼|zebrafish/i.test(raw) ? "zebrafish" : "fish"], hardHint: true });
  }

  // Broaden multi-species only when no concrete species was named
  if (!hits.length && /多物种|尽量.*物种|species divers/.test(blob)) {
    return { species: [], policy: "prefer", coverage: "broaden" };
  }
  if (!hits.length) return null;

  hits.sort((a, b) => a.index - b.index);
  const last = hits[hits.length - 1];
  const exactFishOnly = /^(鱼|鱼类|fish|zebrafish|斑马鱼)$/i.test(raw.trim());
  // bare human alone stays soft prefer; explicit 我要/只要 → hard
  const hard =
    last.hardHint ||
    exactFishOnly ||
    /只要|仅|硬限制|only|必须|我要/.test(blob);
  // "有没有X" exploratory → prefer; "我要/只要X" → include_only
  const exploratory = /有没有|有吗|是否有|能不能找/.test(raw) && !/我要|只要|仅|必须|换成|改成/.test(raw);
  const policy: "include_only" | "prefer" = hard && !exploratory ? "include_only" : "prefer";
  return { species: last.species, policy };
}

/** Short task brief used to personalize question phrasing. */
export function taskBrief(spec: IntentSpec): string {
  const bits: string[] = [];
  const blob = `${spec.originalPrompt} ${spec.objective} ${spec.ptmTypes.join(" ")} ${spec.specialThemes.join(" ")}`;
  const structuredThemes = `${spec.ptmTypes.join(" ")} ${spec.specialThemes.join(" ")}`.trim();
  if (structuredThemes) {
    if (IMMUNO_HINT.test(structuredThemes)) bits.push("免疫肽/HLA 配体");
    else if (spec.ptmTypes.includes("phospho") || /phospho|磷酸化/i.test(structuredThemes)) bits.push("磷酸化蛋白组");
    else if (spec.ptmTypes.length) bits.push(`${spec.ptmTypes.join("、")} 相关`);
    else bits.push(`${spec.specialThemes.join("、")} 主题`);
  } else if (IMMUNO_HINT.test(blob)) {
    bits.push("免疫肽/HLA 配体");
  }
  if (spec.species.length) bits.push(formatSpeciesLabel(spec.species));
  else if (/human|人源|人类|智人/.test(lower(blob)) && !/鱼|fish|小鼠|mouse/.test(lower(blob))) bits.push("人源");
  if (spec.taskType && spec.taskType !== "browse_only" && spec.taskType !== "other") {
    const pretty: Record<string, string> = {
      rt_prediction: "RT 预测",
      fragment_intensity_prediction: "碎片强度预测",
      psm_scoring: "PSM 打分",
      denovo: "de novo",
      ptm_denovo: "PTM de novo",
      chimeric_interpretation: "嵌合谱解释",
    };
    bits.push(`下游偏${pretty[spec.taskType] || spec.taskType}`);
  } else if (spec.taskType === "browse_only") {
    bits.push("先摸清有哪些数据");
  }
  if (spec.acquisitionMode === "dda") bits.push("DDA");
  if (spec.acquisitionMode === "dia") bits.push("DIA");
  if (spec.runHorizon === "candidates_only") bits.push("候选即停");
  const joined = bits.filter(Boolean).join(" · ");
  if (joined) return joined;
  const fallback = text(spec.objective || spec.originalPrompt);
  return fallback ? fallback.slice(0, 48) : "当前数据需求";
}

/** Bare option tokens like "7" / "7)" must never become the user-facing goal. */
export function isPollutedObjective(value: string): boolean {
  const v = text(value);
  if (!v) return true;
  if (/^[0-9]{1,2}$/.test(v)) return true;
  if (/^[0-9]{1,2}[).、.\s]*$/.test(v)) return true;
  // accidental option-label dumps
  if (/^browse_only$|^other$|^rt_prediction$/i.test(v)) return true;
  if (isOrientationPrompt(v)) return true;
  return false;
}

/**
 * Capability / orientation chat that is NOT a data-goal.
 * e.g. "你能干什么" "目前什么任务好" — never park as 目标.
 */
export function hasDomainKeywordsInText(prompt: string): boolean {
  const v = text(prompt);
  return /免疫肽|immunopeptid|hla|mhc|磷酸化|phospho|dda|dia|pride|pxd|人源|小鼠|鱼类|fish|zebrafish|de\s*novo|rt\b|psm|候选项目|搜数据|找数据|tmt|silac/i.test(
    v,
  );
}

/** User is asking for advice / thinking out loud — do not hard-push confirm card. */
export function isRecommendationPrompt(prompt: string): boolean {
  const v = text(prompt);
  if (!v) return false;
  return /有(没有)?推荐|你有推荐|推荐一下|帮我推荐|给个建议|理清思路|先聊聊|不知道选|该选什么|什么任务好|你觉得.{0,12}(好|合适|推荐)|有什么建议/.test(
    v,
  );
}

/**
 * Soft-filled strategy can preview on the right, but chat should keep offering options
 * while the user is still exploring recommendations / orientation.
 */
export function shouldDeferConfirmCard(spec: IntentSpec, lastUserPrompt: string): boolean {
  const prompt = text(lastUserPrompt);
  if (isRecommendationPrompt(prompt) || isOrientationPrompt(prompt)) return true;
  // Theme-only exploration (e.g. mentioned immunopeptide, no species/count yet)
  if (
    isImmunopeptideContext(spec) &&
    spec.species.length === 0 &&
    spec.quotaFlexibility !== "fixed" &&
    (!spec.taskType || spec.taskType === "browse_only")
  ) {
    return true;
  }
  return false;
}

export function isOrientationPrompt(prompt: string): boolean {
  const v = text(prompt);
  if (!v) return false;
  // Recommendation / thinking-out-loud is orientation even when a domain word is present
  if (isRecommendationPrompt(v) && !/确认|开始搜索|只要\s*\d+|目标\s*\d+/.test(v)) {
    // If they ALSO locked hard constraints in the same breath, not pure orientation
    if (!hasDomainKeywordsInText(v) || /有推荐|推荐一下|理清思路|你觉得|有没有推荐/.test(v)) {
      // hybrid still orientation-ish for confirm-defer; domain absorb handled separately
      if (!/确认并开始|开始搜索/.test(v)) {
        // fall through to more checks — mark true when primarily asking for advice
        if (isRecommendationPrompt(v) && v.length <= 80) return true;
      }
    }
  }
  // Already a real data ask → not orientation-only
  if (hasDomainKeywordsInText(v)) {
    // Still orientation if the domain word is absent and it's pure meta... handled below only when domain-ish
    if (!/你(觉得|能|会|可以)|能干什么|你会什么|有什么功能|怎么用|介绍一下|什么任务好|有(没有)?推荐|推荐一下/.test(v)) {
      return false;
    }
  }
  if (
    /你(觉得|能|会|可以).{0,16}(什么|干什么|做什么|任务|功能|能力)|能干什么|你会什么|有什么(功能|能力)|怎么用|介绍一下你|你是谁|帮助我了解|目前什么任务好|什么任务好/.test(
      v,
    )
  ) {
    return true;
  }
  // Question directed at the agent about itself / process, not a dataset brief
  if (
    v.length <= 40 &&
    /[？?]/.test(v) &&
    /你|咱们|系统/.test(v) &&
    /什么|怎么|能否|可以吗|干嘛/.test(v) &&
    !/免疫肽|dda|dia|pride|人源|项目数|pxd/i.test(v)
  ) {
    return true;
  }
  return false;
}

/** Human-readable goal line for strategy card / right panel (not raw option ids). */
export function deriveObjective(spec: IntentSpec): string {
  const brief = taskBrief(spec);
  if (brief && brief !== "当前数据需求") return brief;
  const raw = text(spec.objective);
  if (raw && !isPollutedObjective(raw)) return raw.slice(0, 64);
  const orig = text(spec.originalPrompt);
  if (orig && !isGreetingPrompt(orig) && !isPollutedObjective(orig)) return orig.slice(0, 64);
  return "蛋白质组数据发现";
}

/**
 * Pull structured signals from free-text even when the current gate is another question.
 * Local rules are the anchor; never trust bare numbers as the goal string.
 */
export function absorbFreeTextSignals(spec: IntentSpec, rawInput: string): IntentSpec {
  const raw = text(rawInput);
  if (!raw || isGreetingPrompt(raw)) return sanitizeIntentObjective(spec);
  // Pure orientation/chat with NO domain keywords: leave strategy untouched.
  // Hybrid like "有推荐吗，想做点免疫肽" must still absorb the domain bits.
  if (isOrientationPrompt(raw) && !hasDomainKeywordsInText(raw) && !isRecommendationPrompt(raw)) {
    return sanitizeIntentObjective(spec);
  }
  if (isOrientationPrompt(raw) && !hasDomainKeywordsInText(raw)) {
    return sanitizeIntentObjective(spec);
  }

  let next: IntentSpec = {
    ...spec,
    answered: { ...spec.answered },
    inferred: { ...spec.inferred },
    ptmTypes: [...spec.ptmTypes],
    specialThemes: [...spec.specialThemes],
    selectedSearchTerms: [...(spec.selectedSearchTerms || [])],
    species: [...spec.species],
  };
  const blob = raw.toLowerCase();

  // Themes / PTMs — latest explicit theme wins (agent patch, not sticky add-only)
  {
    const immuno = /hla|mhc|immunopeptid|免疫肽|配体组/.test(blob);
    const phospho = /phospho|磷酸化/.test(blob);
    const glyco = /glycosyl|糖基化/.test(blob);
    if (immuno || phospho || glyco) {
      const nextPtms: string[] = [];
      if (immuno) nextPtms.push("immunopeptide");
      if (phospho) nextPtms.push("phospho");
      if (glyco) nextPtms.push("glycosylation");
      // If user used switch language, replace entirely with the new theme set
      if (/改成|换成|只要|不要|别要|改为/.test(raw) || next.ptmTypes.length === 0) {
        next.ptmTypes = nextPtms;
      } else {
        for (const p of nextPtms) {
          if (!next.ptmTypes.includes(p)) next.ptmTypes.push(p);
        }
      }
      next = markAnswered(next, "Q5", true);
    }
  }

  // Species — latest explicit mention wins (fish overrides sticky human)
  {
    const hit = detectSpeciesSignals(raw);
    if (hit) {
      if (hit.coverage === "broaden") {
        next.species = [];
        next.speciesPolicy = "open";
        next.speciesCoverage = "broaden";
      } else {
        next.species = [...hit.species];
        next.speciesPolicy = hit.policy;
        if (hit.policy === "include_only") next.speciesCoverage = "none";
      }
      next = markAnswered(next, "Q3", true);
    }
  }

  // Acquisition — always patch when user names it
  if (/\bdda\b|数据依赖/.test(blob) && !/\bdia\b/.test(blob)) {
    next.acquisitionMode = "dda";
    if (!next.mixedAcquisitionPolicy) next.mixedAcquisitionPolicy = "review_mixed";
    next = markAnswered(next, "Q4", true);
  } else if (/\bdia\b|数据非依赖/.test(blob) && !/\bdda\b/.test(blob)) {
    next.acquisitionMode = "dia";
    next = markAnswered(next, "Q4", true);
  } else if (/两者都可|采集不限|不限采集/.test(blob)) {
    next.acquisitionMode = "unknown";
    next.mixedAcquisitionPolicy = "review_mixed";
    next = markAnswered(next, "Q4", true);
  }

  // Task type free-form patch
  if (/rt\s*预测|retention\s*time|保留时间/.test(blob)) {
    next.taskType = "rt_prediction";
    next = markAnswered(next, "Q1", true);
  } else if (/碎片强度|fragment\s*intens/.test(blob)) {
    next.taskType = "fragment_intensity_prediction";
    next = markAnswered(next, "Q1", true);
  } else if (/psm\s*打分|psm\s*scor/.test(blob)) {
    next.taskType = "psm_scoring";
    next = markAnswered(next, "Q1", true);
  } else if (/ptm\s*de\s*novo/.test(blob)) {
    next.taskType = "ptm_denovo";
    next = markAnswered(next, "Q1", true);
  } else if (/de\s*novo|从头测序/.test(blob)) {
    next.taskType = "denovo";
    next = markAnswered(next, "Q1", true);
  } else if (/嵌合谱|chimeric/.test(blob)) {
    next.taskType = "chimeric_interpretation";
    next = markAnswered(next, "Q1", true);
  }

  // Data-finding intent without modeling task → browse_only
  if (
    !next.taskType &&
    (/免疫肽|找数据|搜数据|只要数据|浏览|候选|项目/.test(blob) || next.ptmTypes.length > 0)
  ) {
    next.taskType = "browse_only";
    next = markAnswered(next, "Q1", true);
  }
  if (!next.runHorizon && (/先只找|候选|浏览|找数据/.test(blob) || next.taskType === "browse_only")) {
    next.runHorizon = "candidates_reviewed";
    next = markAnswered(next, "Q2", true);
  }

  // Explicit count
  const n = extractTargetProjectCount(raw);
  if (n != null) {
    next = applyTargetProjectCount(next, n);
  } else if (EXHAUSTIVE_HINT.test(raw)) {
    const exhaustive = coverageQuota("exhaustive");
    next.coverageMode = "exhaustive";
    next.targetProjectCount = exhaustive.projects;
    next.maxCandidateProjects = exhaustive.pool;
    next.quotaFlexibility = "open_ended";
    next.timeBudget = "multi_round";
    next.onSafetyCeiling = "ask";
    next = markAnswered(next, "Q7", true);
  }

  // Keep a useful originalPrompt trail
  if (!next.originalPrompt || isGreetingPrompt(next.originalPrompt) || isPollutedObjective(next.originalPrompt)) {
    if (!isPollutedObjective(raw) && raw.length >= 2) next.originalPrompt = raw.slice(0, 200);
  }

  return sanitizeIntentObjective(next);
}

/**
 * True-agent gate: once the user has given a real data goal (theme/species/task/count),
 * soft-fill remaining questionnaire slots with safe defaults and let the strategy card
 * become confirmable — no more forced Q1→Q7 form walk.
 */
export function hasDomainSubstance(spec: IntentSpec): boolean {
  if (spec.ptmTypes.length > 0) return true;
  if (spec.species.length > 0) return true;
  if (spec.taskType && spec.taskType !== "other") return true;
  if (spec.targetProjectCount != null && spec.targetProjectCount > 0) return true;
  if (spec.specialThemes.length > 0) return true;
  const blob = (`${spec.originalPrompt} ${spec.objective}`).toLowerCase();
  if (/免疫肽|immunopeptid|hla|mhc|磷酸化|phospho|de\s*novo|rt\s*预测|psm|pride|pxd|dda|dia/.test(blob)) return true;
  return false;
}

/** Soft-fill unanswered required gates when domain substance already exists. */
/**
 * D1: do NOT auto-complete the questionnaire every turn.
 * Only sanitize; applyRecommendedDefaults runs when user says 按推荐默认 / request_defaults.
 * Optional `forceDefaults` keeps tests / explicit default paths working.
 */
export function agenticSoftFill(spec: IntentSpec, opts?: { forceDefaults?: boolean }): IntentSpec {
  if (opts?.forceDefaults) {
    if (!hasDomainSubstance(spec)) return sanitizeIntentObjective(spec);
    return sanitizeIntentObjective(applyRecommendedDefaults(spec));
  }
  return sanitizeIntentObjective(spec);
}


export function sanitizeIntentObjective(spec: IntentSpec): IntentSpec {
  const next = { ...spec };
  const derived = deriveObjective(next);
  if (isPollutedObjective(next.objective) || !text(next.objective)) {
    next.objective = derived;
  } else {
    // Prefer derived when structured signals exist and raw is orientation/meta/weak.
    const raw = text(next.objective);
    const hasStructure =
      Boolean(next.taskType) ||
      next.ptmTypes.length > 0 ||
      next.species.length > 0 ||
      Boolean(next.acquisitionMode);
    if (
      derived &&
      derived !== raw &&
      (raw.length <= 4 ||
        /^[0-9]/.test(raw) ||
        isOrientationPrompt(raw) ||
        (hasStructure && /·|免疫肽|人源|DDA|DIA|数据/.test(derived)))
    ) {
      next.objective = derived;
    }
  }
  if (next.originalPrompt && isOrientationPrompt(next.originalPrompt)) {
    // Keep trail empty so orientation chatter never sticks as the goal source.
    if (next.ptmTypes.length || next.species.length || Boolean(next.taskType)) {
      next.originalPrompt = derived.slice(0, 200);
    } else {
      next.originalPrompt = "";
    }
  }
  return next;
}

function isImmunopeptideContext(spec: IntentSpec): boolean {
  const blob = `${spec.originalPrompt} ${spec.objective} ${spec.ptmTypes.join(" ")} ${spec.specialThemes.join(" ")}`;
  return IMMUNO_HINT.test(blob) || spec.ptmTypes.some((x) => /immuno/i.test(x));
}

/** If user clearly picked an option number/id/label, return canonical 1-based index string. */
export function matchOptionAnswer(question: GrillQuestion, raw: string): string | null {
  const answer = text(raw);
  if (!answer || !question.options.length) return null;
  const key = normalizeOptionKey(answer);
  const asNum = Number(key);
  if (Number.isInteger(asNum) && asNum >= 1 && asNum <= question.options.length) {
    return String(asNum);
  }
  const low = lower(answer);
  for (let i = 0; i < question.options.length; i += 1) {
    const opt = question.options[i];
    if (lower(opt.id) === low || lower(opt.label) === low) return String(i + 1);
    if (low && lower(opt.label).includes(low) && low.length >= 2) return String(i + 1);
  }
  return null;
}


const COVERAGE_QUOTAS: Record<
  Exclude<CoverageMode, "">,
  { projects: number; pool: number; label: string }
> = {
  quick: { projects: 20, pool: 80, label: "快速" },
  curated: { projects: 20, pool: 80, label: "快速" },
  balanced: { projects: 80, pool: 250, label: "均衡" },
  exhaustive: { projects: 2000, pool: 20000, label: "尽量搜全（无业务池顶，仅安全顶）" },
};

const PTM_TRIGGERS =
  /phospho|acetyl|glyco|ubiquit|methyl|hla\b|mhc\b|immunopeptid|免疫肽|磷酸化|乙酰|糖基|泛素|甲基化|ptm/i;
const NON_TRYPTIC =
  /non[- ]?tryptic|非胰酶|chymotrypsin|elastase|pepsin|Lys-N|Lys-C|Glu-C|Asp-N/i;
const DRUG_CELL = /drug[- ]?treat|化合物处理|药物处理|cell line|细胞系|HeLa|HEK293|K562/i;
const PATIENT_VS_LINE =
  /patient|clinical|clinical sample|病人|临床样本|组织样本|immortalized|永生化/i;
const EXHAUSTIVE_HINT =
  /尽量搜全|尽量多|搜全|覆盖全|越多越好|尽可能多|多一点|不设上限|无上限|全量|(?:所有|全部).{0,24}(?:数据|项目|文件|候选)|(?:数据|项目|文件|候选).{0,16}(?:所有|全部)|as many as possible|all (?:relevant )?(?:data|datasets|projects|files)|no (?:candidate pool )?limit|open[- ]?ended|exhaustive|comprehensive/i;
const LABEL_FREE_HINT = /label[- ]?free|无标记|非标记/i;
const SILAC_HINT = /\bsilac\b/i;
const TMT_HINT = /\btmt\b|itraq|标记定量|isobaric/i;
const DDA_HINT = /\bdda\b|data[- ]dependent|shotgun/i;
const DIA_HINT = /\bdia\b|data[- ]independent|dia-nn|spectronaut/i;
const NEWER_HINT = /新仪器|越新越好|newer instrument|orbitrap astral|timstof/i;
const CLASSIC_HINT = /老仪器|经典仪器|legacy instrument|classic instrument/i;
const IP_MS = /\bip[- ]?ms\b|免疫沉淀|pull[- ]?down|affinity purification|ap[- ]?ms/i;
const CROSSLINK = /cross[- ]?link|交联|xl[- ]?ms|dss|bs3/i;

export function coverageQuota(mode: CoverageMode) {
  if (mode === "quick" || mode === "curated" || mode === "balanced" || mode === "exhaustive") {
    return COVERAGE_QUOTAS[mode];
  }
  return COVERAGE_QUOTAS.balanced;
}

/** Parse an explicit project-count request from free text (e.g. "20个可用项目就行"). */
export function extractTargetProjectCount(text: string): number | null {
  const raw = String(text || "").trim();
  if (!raw) return null;

  const patterns: RegExp[] = [
    /(?:目标|只要|需要|希望|大约|大概|约|左右)?\s*(\d{1,5})\s*个?\s*(?:可用)?项目/,
    /(?:target|about|around|only|just)?\s*(\d{1,5})\s*(?:usable\s*)?projects?/i,
    /(?:改成|调整为|设为|变成|写成|改为)\s*(\d{1,5})/,
    /我?要\s*(\d{1,5})\s*个/,
    // free-form N个 mid-sentence
    /(?:约|大概|大约|只要|目标)?\s*(\d{1,5})\s*个(?:\s*(?:左右|可用|候选|项目))?/,
    /(\d{1,5})\s*个\s*(?:可用)?(?:项目)?\s*(?:就行|即可|够了|左右|可以)/,
    /(\d{1,5})\s*个\s*可用/,
    /项目数\s*(?:为|到|到约|约)?\s*(\d{1,5})/,
    /max[_\s-]?projects?\s*[=:]\s*(\d{1,5})/i,
  ];

  for (const re of patterns) {
    const m = raw.match(re);
    if (!m) continue;
    const n = Number(m[1]);
    if (Number.isFinite(n) && n >= 1 && n <= 5000) return Math.round(n);
  }
  return null;
}

/** Write an explicit target project count into IntentSpec and align coverage tier/pool. */
export function applyTargetProjectCount(
  spec: IntentSpec,
  n: number,
  opts?: { flexibility?: "fixed" | "recommended" },
): IntentSpec {
  const count = Math.min(5000, Math.max(1, Math.round(Number(n) || 0)));
  if (!Number.isFinite(count) || count < 1) return spec;

  const mode: CoverageMode = "quick";

  const q = coverageQuota(mode);
  const pool = Math.min(
    20000,
    Math.max(count * 4, Math.round(count * (q.pool / Math.max(1, q.projects)))),
  );

  let next: IntentSpec = {
    ...spec,
    answered: { ...spec.answered },
    inferred: { ...spec.inferred },
  };
  // User-stated counts stay fixed; inferred/LLM counts are recommendations only.
  const flex = opts?.flexibility || "fixed";
  if (next.quotaFlexibility === "fixed" && flex !== "fixed" && next.targetProjectCount) {
    // Do not overwrite an explicit user count with a softer inference.
    return markAnswered(next, "Q7");
  }
  next.targetProjectCount = count;
  next.maxCandidateProjects = pool;
  next.coverageMode = mode;
  next.quotaFlexibility = flex;
  next.timeBudget = "fast";
  next.onSafetyCeiling = next.onSafetyCeiling || "ask";
  if (next.successCriteria.some((c) => /可用项目数/.test(c))) {
    next.successCriteria = next.successCriteria.map((c) =>
      /可用项目数/.test(c) ? `可用项目数接近目标约 ${count}` : c,
    );
  }
  return markAnswered(next, "Q7");
}

/** Overlay only filled fields from a local parse so revise does not wipe existing answers. */
export function overlayFilledIntent(base: IntentSpec, overlay: IntentSpec): IntentSpec {
  let next: IntentSpec = {
    ...base,
    answered: { ...base.answered, ...overlay.answered },
    inferred: { ...base.inferred, ...overlay.inferred },
    parseWarnings: [...(base.parseWarnings || []), ...(overlay.parseWarnings || [])],
  };
  if (overlay.parseReasoning) next.parseReasoning = overlay.parseReasoning;

  if (overlay.taskType) next.taskType = overlay.taskType;
  next.runHorizon = "candidates_reviewed";
  next.answered.Q2 = true;
  if (overlay.species.length) next.species = overlay.species;
  if (overlay.speciesPolicy && overlay.speciesPolicy !== "open") next.speciesPolicy = overlay.speciesPolicy;
  else if (overlay.answered.Q3 && overlay.speciesPolicy) next.speciesPolicy = overlay.speciesPolicy;
  if (overlay.speciesCoverage && overlay.speciesCoverage !== "none") next.speciesCoverage = overlay.speciesCoverage;
  if (overlay.acquisitionMode) next.acquisitionMode = overlay.acquisitionMode;
  if (overlay.mixedAcquisitionPolicy) next.mixedAcquisitionPolicy = overlay.mixedAcquisitionPolicy;
  if (overlay.ptmTypes.length) {
    next.ptmTypes = Array.from(new Set([...next.ptmTypes, ...overlay.ptmTypes]));
  }
  if (overlay.specialThemes.length) {
    next.specialThemes = Array.from(new Set([...next.specialThemes, ...overlay.specialThemes]));
  }
  if (overlay.labelingStrategy) {
    next.labelingStrategy = overlay.labelingStrategy;
    next.labelingHard = overlay.labelingHard;
  }
  if (overlay.coverageMode) next.coverageMode = overlay.coverageMode;
  if (overlay.targetProjectCount != null) next.targetProjectCount = overlay.targetProjectCount;
  if (overlay.maxCandidateProjects != null) next.maxCandidateProjects = overlay.maxCandidateProjects;
  if (overlay.quotaFlexibility) next.quotaFlexibility = overlay.quotaFlexibility;
  if (overlay.timeBudget) next.timeBudget = overlay.timeBudget;
  if (overlay.onSafetyCeiling) next.onSafetyCeiling = overlay.onSafetyCeiling;
  if (overlay.instrumentPreference) next.instrumentPreference = overlay.instrumentPreference;
  if (overlay.legacyFloorRatio != null) next.legacyFloorRatio = overlay.legacyFloorRatio;
  if (overlay.excludeRules.length) {
    next.excludeRules = Array.from(new Set([...next.excludeRules, ...overlay.excludeRules]));
  }
  if (overlay.successCriteria.length) next.successCriteria = overlay.successCriteria;
  if (overlay.notes) next.notes = overlay.notes;
  if (overlay.repository) next.repository = overlay.repository;
  if (overlay.objective && !base.objective) next.objective = overlay.objective;
  return next;
}

export function shouldAskQ5(spec: IntentSpec): boolean {
  if (spec.taskType === "ptm_denovo") return true;
  if (spec.ptmTypes.length > 0) return true;
  const blob = `${spec.originalPrompt} ${spec.objective} ${spec.notes}`;
  return (
    PTM_TRIGGERS.test(blob) ||
    NON_TRYPTIC.test(blob) ||
    DRUG_CELL.test(blob) ||
    PATIENT_VS_LINE.test(blob) ||
    spec.specialThemes.length > 0
  );
}

export function shouldAskQ6(spec: IntentSpec): boolean {
  const blob = `${spec.originalPrompt} ${spec.objective}`;
  if (LABEL_FREE_HINT.test(blob) || SILAC_HINT.test(blob) || TMT_HINT.test(blob)) return true;
  return ["rt_prediction", "fragment_intensity_prediction", "psm_scoring"].includes(spec.taskType);
}

export function shouldAskQ8(spec: IntentSpec): boolean {
  // Only ask when user signaled instruments; otherwise quietly default newer.
  if (spec.instrumentPreference) return false;
  const blob = `${spec.originalPrompt} ${spec.objective}`;
  return NEWER_HINT.test(blob) || CLASSIC_HINT.test(blob);
}

export function shouldAskQ9(spec: IntentSpec): boolean {
  const blob = `${spec.originalPrompt} ${spec.objective}`;
  return IP_MS.test(blob) || CROSSLINK.test(blob) || /不要|排除|exclude|no top-down|非/i.test(blob);
}

export function shouldAskQ10(spec: IntentSpec): boolean {
  const blob = `${spec.originalPrompt} ${spec.objective}`;
  return (
    EXHAUSTIVE_HINT.test(blob) ||
    /diverse|多样性|多厂商|多物种|多梯度|vendor/i.test(blob) ||
    spec.coverageMode === "exhaustive" ||
    spec.speciesCoverage === "broaden"
  );
}

function markAnswered(spec: IntentSpec, id: QuestionId, inferred = false): IntentSpec {
  return {
    ...spec,
    answered: { ...spec.answered, [id]: true },
    inferred: inferred ? { ...spec.inferred, [id]: true } : { ...spec.inferred },
  };
}

export function defaultSuccessCriteria(spec: IntentSpec): string[] {
  const immuno = isImmunopeptideContext(spec);
  const browse = spec.taskType === "browse_only" || !spec.taskType;
  const items: string[] = [];

  if (spec.targetProjectCount) {
    items.push(`可用项目数接近目标约 ${spec.targetProjectCount}（安全上限内尽量接近）`);
  }

  if (immuno) {
    items.push("优先有 HLA/MHC 或 immunopeptidomics 标注、元数据可判读的项目");
    if (spec.speciesPolicy === "include_only" && spec.species.includes("human")) {
      items.push("人源免疫肽为主，避免被非 HLA 主题噪声冲淡");
    } else if (spec.speciesPolicy === "open") {
      items.push("物种开放时优先可解释物种，不强行凑跨界多样性");
    }
    if (browse) {
      items.push("候选清单可追溯到 PRIDE accession，便于后续挑项目深挖");
    }
    return items.length ? items : ["免疫肽相关、元数据可读、可下载的公开项目"];
  }

  if (spec.taskType === "rt_prediction") {
    items.push("优先 label-free / 有可靠 RT 信息的 DDA 数据");
    items.push("仪器与梯度不必硬凑多样性，先保证 RT 可用性");
    return items;
  }

  if (spec.taskType === "denovo" || spec.taskType === "ptm_denovo") {
    items.push("优先高质量 MS/MS 与可读峰列表的项目");
    return items;
  }

  // general discovery
  if (spec.coverageMode === "exhaustive" || spec.speciesCoverage === "broaden") {
    items.push("在配额内尽量覆盖更多仪器厂商与梯度长度");
    if (spec.speciesPolicy === "open" || spec.speciesCoverage === "broaden") {
      items.push("物种尽量覆盖动物 / 植物 / 微生物中的更多类别");
    }
  } else if (browse) {
    items.push("相关性优先：主题匹配 > 凑厂商多样性");
  } else {
    items.push("在目标规模内兼顾相关性与适度多样性");
  }
  return items.length ? items : ["找到与目标相关的可用公开项目"];
}

/** True for pure greetings / empty chatter that should not call LLM parse-goal. */
export function isGreetingPrompt(prompt: string): boolean {
  const raw = text(prompt);
  if (!raw) return true;
  // short non-domain pings: hi, hihi, 你好啊, 在吗?, test...
  if (raw.length > 24) return false;
  if (
    /^(hi+|hello+|hey+|你好+|您好+|嗨+|哈+|嗯+|在吗|test+|测试+)([\s!！。.?？~～]*)*$/i.test(raw)
  ) {
    return true;
  }
  // no proteomics signal words
  const domain =
    /rt|dda|dia|pride|proteom|蛋白|肽|鱼|fish|zebrafish|ms\b|tmt|silac|label|human|mouse|物种|仪器|ptm|de\s*novo|搜全|候选|项目|accession|pxd|我要|只要|改成|换成|必须|优先/i;
  // only ultra-short pings without any intent verb/domain token
  return raw.length <= 6 && !domain.test(raw) && !/[要找搜改换限]/.test(raw);
}

export function applyLocalParse(prompt: string): IntentSpec {
  let spec = createEmptyIntent(prompt);
  const raw = text(prompt);
  const blob = raw.toLowerCase();
  if (isGreetingPrompt(raw) || isOrientationPrompt(raw)) {
    spec.objective = "";
    spec.originalPrompt = isOrientationPrompt(raw) ? "" : spec.originalPrompt;
    spec.parseReasoning = isOrientationPrompt(raw)
      ? "用户在问能力/任务建议，先介绍可做的方向，再引导具体数据需求。"
      : "首条消息未包含可识别的数据需求，将从关键问题开始澄清。";
    return spec;
  }

  // Never park bare option numbers / orientation chatter as the goal.
  if (!isPollutedObjective(raw)) {
    spec.objective = raw;
  } else {
    spec.objective = "";
  }

  if (/rt\b|retention|保留时间|保留指数|irt/.test(blob)) {
    spec.taskType = "rt_prediction";
    spec = markAnswered(spec, "Q1", true);
  } else if (/fragment|强度预测|intensity/.test(blob)) {
    spec.taskType = "fragment_intensity_prediction";
    spec = markAnswered(spec, "Q1", true);
  } else if (/psm|打分|rescoring/.test(blob)) {
    spec.taskType = "psm_scoring";
    spec = markAnswered(spec, "Q1", true);
  } else if (/ptm\s*de\s*novo|ptm_denovo|修饰.*de\s*novo/.test(blob)) {
    spec.taskType = "ptm_denovo";
    spec = markAnswered(spec, "Q1", true);
  } else if (/de\s*novo|denovo/.test(blob)) {
    spec.taskType = "denovo";
    spec = markAnswered(spec, "Q1", true);
  } else if (/chimeric|嵌合/.test(blob)) {
    spec.taskType = "chimeric_interpretation";
    spec = markAnswered(spec, "Q1", true);
  } else if (/先只找|browse|浏览|先看候选/.test(blob)) {
    spec.taskType = "browse_only";
    spec = markAnswered(spec, "Q1", true);
  }

  if (/只.*计划|搜索计划|不要真正搜|plan only/.test(blob)) {
    spec.runHorizon = "candidates_reviewed";
    spec = markAnswered(spec, "Q2", true);
  } else if (/可训练|ai[- ]?ready|建表|训练表/.test(blob)) {
    spec.runHorizon = "candidates_reviewed";
    spec = markAnswered(spec, "Q2", true);
  } else if (/审查|复核|review candidate/.test(blob)) {
    spec.runHorizon = "candidates_reviewed";
    spec = markAnswered(spec, "Q2", true);
  } else if (/找.*候选|候选就停|只要候选|发现数据/.test(blob)) {
    spec.runHorizon = "candidates_reviewed";
    spec = markAnswered(spec, "Q2", true);
  }

  {
    const hit = detectSpeciesSignals(raw);
    if (hit) {
      if (hit.coverage === "broaden") {
        spec.species = [];
        spec.speciesPolicy = "open";
        spec.speciesCoverage = "broaden";
      } else {
        spec.species = [...hit.species];
        spec.speciesPolicy = hit.policy;
        if (hit.policy === "include_only") spec.speciesCoverage = "none";
      }
      spec = markAnswered(spec, "Q3", true);
    } else if (/多物种|尽量.*物种|species divers/.test(blob)) {
      spec.speciesPolicy = "open";
      spec.speciesCoverage = "broaden";
      spec = markAnswered(spec, "Q3", true);
    }
  }

  if (DDA_HINT.test(blob) && !DIA_HINT.test(blob)) {
    spec.acquisitionMode = "dda";
    spec.mixedAcquisitionPolicy = "review_mixed";
    spec = markAnswered(spec, "Q4", true);
  } else if (DIA_HINT.test(blob) && !DDA_HINT.test(blob)) {
    spec.acquisitionMode = "dia";
    spec = markAnswered(spec, "Q4", true);
  } else if (DDA_HINT.test(blob) && DIA_HINT.test(blob)) {
    spec.acquisitionMode = "unknown";
    spec.mixedAcquisitionPolicy = "allow";
    spec = markAnswered(spec, "Q4", true);
  }

  const ptms: string[] = [];
  if (/phospho|磷酸化|pSer|pThr|pTyr|phosphoprote/.test(blob)) ptms.push("phospho");
  if (/acetyl|乙酰/.test(blob)) ptms.push("acetyl");
  if (/glyco|糖基|糖肽/.test(blob)) ptms.push("glyco");
  if (/ubiquit|泛素|GlyGly|K-GG/.test(blob)) ptms.push("ubiquitin");
  if (/hla|mhc|immunopeptid|免疫肽/.test(blob)) ptms.push("immunopeptide");
  if (ptms.length) spec.ptmTypes = ptms;
  if (NON_TRYPTIC.test(blob)) spec.specialThemes.push("non_tryptic");
  if (DRUG_CELL.test(blob)) spec.specialThemes.push("drug_treated_cell_line");
  if (/patient|病人|临床/.test(blob)) spec.specialThemes.push("patient_sample");
  if (/immortalized|永生化|cell line|细胞系/.test(blob)) spec.specialThemes.push("immortalized_cell_line");
  if (ptms.length) spec = markAnswered(spec, "Q5", true);

  if (SILAC_HINT.test(blob)) {
    spec.labelingStrategy = "silac";
    spec.labelingHard = true;
    spec = markAnswered(spec, "Q6", true);
  } else if (LABEL_FREE_HINT.test(blob) && !TMT_HINT.test(blob)) {
    spec.labelingStrategy = "label_free";
    spec.labelingHard = true;
    spec = markAnswered(spec, "Q6", true);
  } else if (TMT_HINT.test(blob) && !LABEL_FREE_HINT.test(blob)) {
    spec.labelingStrategy = "tmt";
    spec.labelingHard = false;
    spec = markAnswered(spec, "Q6", true);
  }

  const explicitProjectCount = extractTargetProjectCount(raw);
  if (explicitProjectCount != null) {
    spec = applyTargetProjectCount(spec, explicitProjectCount);
  } else if (EXHAUSTIVE_HINT.test(blob)) {
    spec.coverageMode = "exhaustive";
    const q = coverageQuota("exhaustive");
    spec.targetProjectCount = q.projects;
    spec.maxCandidateProjects = q.pool;
    spec.quotaFlexibility = "open_ended";
    spec.timeBudget = "multi_round";
    spec.onSafetyCeiling = "ask";
    spec = markAnswered(spec, "Q7", true);
  } else if (/快速|精选|少量|先验证|quick|curated|small set/.test(blob)) {
    spec.coverageMode = "quick";
    const q = coverageQuota("quick");
    spec.targetProjectCount = q.projects;
    spec.maxCandidateProjects = q.pool;
    spec.quotaFlexibility = "fixed";
    spec.timeBudget = "fast";
    spec = markAnswered(spec, "Q7", true);
  }

  if (CLASSIC_HINT.test(blob) && NEWER_HINT.test(blob)) {
    spec.instrumentPreference = "newer_with_legacy_floor";
    spec.legacyFloorRatio = 0.2;
    spec = markAnswered(spec, "Q8", true);
  } else if (CLASSIC_HINT.test(blob)) {
    spec.instrumentPreference = "classic";
    spec = markAnswered(spec, "Q8", true);
  } else if (NEWER_HINT.test(blob)) {
    spec.instrumentPreference = "newer";
    spec = markAnswered(spec, "Q8", true);
  }

  const excludes: string[] = [];
  if (IP_MS.test(blob)) excludes.push("IP-MS / AP-MS");
  if (CROSSLINK.test(blob)) excludes.push("交联蛋白 / XL-MS");
  if (/top[- ]?down|完整蛋白|intact protein/.test(blob) && /不要|排除|exclude|非/.test(blob)) {
    excludes.push("Top-down / intact protein");
  }
  if (excludes.length) {
    spec.excludeRules = excludes;
    spec = markAnswered(spec, "Q9", true);
  }

  if (spec.coverageMode === "exhaustive" || /diverse|多样性|多厂商|多物种/.test(blob)) {
    spec.successCriteria = defaultSuccessCriteria(spec);
    spec = markAnswered(spec, "Q10", true);
  }

  spec.parseReasoning = "已从首条消息推断部分字段；仅就缺口继续追问。";
  return spec;
}

export type MergeLlmMode = "fill_gaps" | "patch";

export type MergeLlmOptions = {
  /**
   * When true, allow patching over a user-fixed project count via model fields.
   * Frontend should set this only when extractTargetProjectCount found a number
   * in the user message (or equivalent user-asserted count). Model-invented
   * counts must never silently replace a fixed user quota.
   */
  allowCountOverwrite?: boolean;
  /** The patch came from a decoded update_strategy tool call, not local parsing. */
  validatedToolPatch?: boolean;
  /** Preserve meaningful extension fields in notes/openRisks for forward compatibility. */
  preserveUnknownFields?: boolean;
};

/**
 * Apply LLM-extracted fields onto the live IntentSpec.
 * - fill_gaps: only fill empty slots (first-pass parse)
 * - patch: treat extra_fields as an update_strategy tool — overwrite fields the model set
 *   so the strategy card follows conversation (species, acq, task, PTM, …).
 *   Exception: target project count stays sticky when quotaFlexibility is "fixed"
 *   unless opts.allowCountOverwrite (user said a number in this turn).
 */
export function mergeLlmFields(
  spec: IntentSpec,
  fields: Record<string, unknown>,
  warnings: string[] = [],
  reasoning = "",
  mode: MergeLlmMode = "patch",
  opts?: MergeLlmOptions,
): IntentSpec {
  let next = { ...spec };
  next.parseWarnings = [...(warnings || []).map(String)];
  if (reasoning) next.parseReasoning = reasoning;

  const hasKey = (k: string) => Object.prototype.hasOwnProperty.call(fields, k);
  const canWrite = (occupied: boolean) => mode === "patch" || !occupied;

  const task = lower(fields.task_type);
  if (task && canWrite(Boolean(next.taskType))) {
    if (
      task in TASK_TYPE_LABELS ||
      [
        "rt_prediction",
        "fragment_intensity_prediction",
        "psm_scoring",
        "denovo",
        "ptm_denovo",
        "chimeric_interpretation",
        "browse_only",
        "other",
      ].includes(task)
    ) {
      next.taskType = task as TaskType;
      next = markAnswered(next, "Q1", true);
    }
  }

  if (hasKey("species") || hasKey("species_policy") || hasKey("species_coverage")) {
    const species = Array.isArray(fields.species)
      ? fields.species.map(String).map((s) => s.trim()).filter(Boolean)
      : null;
    const policy = lower(fields.species_policy);
    const coverage = lower(fields.species_coverage);

    // Drop non-organism tokens the model sometimes puts in species (e.g. "DIA").
    const NON_SPECIES = new Set([
      "dda", "dia", "tmt", "itraq", "silac", "label-free", "label_free", "label",
      "rt", "psm", "denovo", "quick", "curated", "balanced", "exhaustive",
    ]);
    const cleanSpecies = species
      ? species.filter((s) => s && !NON_SPECIES.has(s.toLowerCase()))
      : null;
    if (cleanSpecies && cleanSpecies.length && (mode === "patch" || next.species.length === 0)) {
      next.species = cleanSpecies;
      next = markAnswered(next, "Q3", true);
    } else if (
      mode === "patch" &&
      Array.isArray(fields.species) &&
      fields.species.length === 0 &&
      // Only clear when policy explicitly opens species, not empty schema echo
      (policy === "open" || lower(fields.species_policy) === "open")
    ) {
      next.species = [];
      next = markAnswered(next, "Q3", true);
    }

    if (policy === "include_only" || policy === "exclude" || policy === "open" || policy === "prefer") {
      if (mode === "patch" || !next.answered.Q3) {
        next.speciesPolicy = policy as IntentSpec["speciesPolicy"];
        next = markAnswered(next, "Q3", true);
      }
    } else if (species && species.length && (mode === "patch" || next.speciesPolicy === "open")) {
      next.speciesPolicy = "prefer";
    }

    if (coverage === "broaden" || coverage === "none") {
      next.speciesCoverage = coverage as IntentSpec["speciesCoverage"];
    }
    if (next.speciesPolicy === "include_only" && next.species.length) {
      next.speciesCoverage = "none";
    }
  }

  const acq = lower(fields.acquisition_mode);
  if ((acq === "dda" || acq === "dia" || acq === "unknown") && canWrite(Boolean(next.acquisitionMode))) {
    next.acquisitionMode = acq as AcquisitionMode;
    next = markAnswered(next, "Q4", true);
  }

  const labeling = lower(fields.labeling_strategy);
  if (labeling && canWrite(Boolean(next.labelingStrategy))) {
    if (labeling.includes("silac")) next.labelingStrategy = "silac";
    else if (labeling.includes("label") || labeling === "label_free") next.labelingStrategy = "label_free";
    else if (labeling.includes("tmt")) next.labelingStrategy = "tmt";
    else if (labeling.includes("itraq")) next.labelingStrategy = "itraq";
    else if (labeling.includes("dimethyl")) next.labelingStrategy = "dimethyl";
    else if (labeling === "unknown" || labeling === "any") next.labelingStrategy = labeling as LabelingStrategy;
    if (next.labelingStrategy) next = markAnswered(next, "Q6", true);
  }

  const scale = lower(fields.scale_mode);
  // Quota rules:
  // - fill_gaps: only fill when not user-fixed; scale bands must not invent fixed locks.
  // - patch: non-count fields may overwrite; project count overwrites fixed ONLY when
  //   opts.allowCountOverwrite (user said N this turn). Pure model invent cannot stomp fixed.
  const explicitN = Number(fields.max_projects ?? fields.target_project_count);
  const hasExplicitCount = Number.isFinite(explicitN) && explicitN > 0;
  const allowCountOverwrite = Boolean(opts?.allowCountOverwrite);
  const canMutateQuotaBand = mode === "patch" || next.quotaFlexibility !== "fixed";

  if (canMutateQuotaBand) {
    let modeCov: CoverageMode | "" = "";
    if (scale === "quick" || scale === "curated" || scale === "balanced" || scale === "exhaustive") {
      modeCov = scale === "curated" ? "quick" : scale as CoverageMode;
    } else if (isImmunopeptideContext(next) && !next.coverageMode) {
      modeCov = "quick";
    }

    if (mode === "patch" && hasExplicitCount) {
      if (next.quotaFlexibility !== "fixed" || allowCountOverwrite) {
        next = applyTargetProjectCount(next, Math.round(explicitN), { flexibility: "fixed" });
      }
      // else: keep sticky user-fixed count; ignore model-invented max_projects / target_project_count
    } else if (modeCov && next.quotaFlexibility !== "fixed") {
      next.coverageMode = modeCov;
      const q = coverageQuota(modeCov);
      next.targetProjectCount = q.projects;
      next.maxCandidateProjects =
        Number(fields.max_candidate_projects) > 0
          ? Number(fields.max_candidate_projects)
          : q.pool;
      next.quotaFlexibility =
        modeCov === "exhaustive" ? "open_ended" : modeCov === "quick" ? "fixed" : "recommended";
      next.timeBudget = modeCov === "exhaustive" ? "multi_round" : modeCov === "quick" ? "fast" : "multi_round";
      next = markAnswered(next, "Q7", true);
    } else if (mode === "patch" && modeCov && allowCountOverwrite) {
      // User asserted a new scale this turn — reband even if previously fixed.
      next.coverageMode = modeCov;
      const q = coverageQuota(modeCov);
      next.targetProjectCount = q.projects;
      next.maxCandidateProjects =
        Number(fields.max_candidate_projects) > 0
          ? Number(fields.max_candidate_projects)
          : q.pool;
      next.quotaFlexibility =
        modeCov === "exhaustive" ? "open_ended" : modeCov === "quick" ? "fixed" : "recommended";
      next.timeBudget = modeCov === "exhaustive" ? "multi_round" : modeCov === "quick" ? "fast" : "multi_round";
      next = markAnswered(next, "Q7", true);
    }
  }

  const ptms = Array.isArray(fields.ptm_types) ? fields.ptm_types.map(String).filter(Boolean) : [];
  if (ptms.length && canWrite(next.ptmTypes.length > 0)) {
    next.ptmTypes = ptms;
    next = markAnswered(next, "Q5", true);
  }

  const goal = lower(fields.goal);
  if (goal === "immunopeptidomics" && !next.ptmTypes.includes("immunopeptide")) {
    next.ptmTypes = [...next.ptmTypes, "immunopeptide"];
  }

  if (hasKey("objective") || hasKey("goal_summary")) {
    const obj = text(fields.objective ?? fields.goal_summary);
    if (obj && !isPollutedObjective(obj) && mode === "patch") {
      next.objective = obj.slice(0, 120);
    }
  }

  if (fields.repository) next.repository = String(fields.repository);

  next = sanitizeIntentObjective(next);
  return next;
}

export function applyRecommendedDefaults(spec: IntentSpec): IntentSpec {
  let next = {
    ...spec,
    resolvedFields: [...(spec.resolvedFields || [])],
    answered: { ...spec.answered },
    inferred: { ...spec.inferred },
  };

  if (!next.taskType) {
    next.taskType = "browse_only";
    next = markAnswered(next, "Q1", true);
  }
  if (!next.runHorizon) {
    next.runHorizon = "candidates_reviewed";
    next = markAnswered(next, "Q2", true);
  }
  if (!next.answered.Q3) {
    if (!next.species.length) {
      next.speciesPolicy = "open";
      next.speciesCoverage = "none";
    }
    next = markAnswered(next, "Q3", true);
  }
  if (!next.acquisitionMode) {
    next.acquisitionMode = next.taskType === "rt_prediction" ? "dda" : "unknown";
    next.mixedAcquisitionPolicy = "review_mixed";
    next = markAnswered(next, "Q4", true);
  }
  if (shouldAskQ5(next) && !next.answered.Q5) {
    next = markAnswered(next, "Q5", true);
  }
  if (shouldAskQ6(next) && !next.labelingStrategy) {
    next.labelingStrategy = next.taskType === "rt_prediction" ? "label_free" : "any";
    next.labelingHard = next.taskType === "rt_prediction";
    next = markAnswered(next, "Q6", true);
  } else if (!next.answered.Q6 && !shouldAskQ6(next)) {
    next.labelingStrategy = "any";
    next = markAnswered(next, "Q6", true);
  }
  if (!next.coverageMode) {
    // Immunopeptide: start with a concrete quick target; generic stays balanced.
    const immuno = isImmunopeptideContext(next);
    const mode: CoverageMode = immuno ? "quick" : "balanced";
    const q = coverageQuota(mode);
    next.coverageMode = mode;
    next.targetProjectCount = q.projects;
    next.maxCandidateProjects = q.pool;
    next.quotaFlexibility = mode === "quick" ? "fixed" : "recommended";
    next.timeBudget = mode === "quick" ? "fast" : "multi_round";
    next.onSafetyCeiling = "ask";
    next = markAnswered(next, "Q7", true);
  }
  if (!next.instrumentPreference) {
    next.instrumentPreference = "newer";
    next = markAnswered(next, "Q8", true);
  }
  if (!next.answered.Q9) {
    next.excludeRules = next.excludeRules.length ? next.excludeRules : [];
    next = markAnswered(next, "Q9", true);
  }
  if (!next.answered.Q10) {
    next.successCriteria = defaultSuccessCriteria(next);
    next = markAnswered(next, "Q10", true);
  }

  if (!next.objective) {
    const immuno = isImmunopeptideContext(next);
    const taskLabel = next.taskType ? TASK_TYPE_LABELS[next.taskType] || next.taskType : "";
    if (immuno && (next.taskType === "browse_only" || !next.taskType)) {
      next.objective = "找免疫肽相关公开项目（先候选清单）";
    } else if (taskLabel && next.taskType !== "browse_only") {
      next.objective = `搜集适合「${taskLabel}」的蛋白质组学公开数据`;
    } else {
      next.objective = "搜集蛋白质组学公开项目与候选文件";
    }
  }

  next.resolvedFields = Array.from(new Set([
    ...next.resolvedFields,
    "objective",
    "task_type",
    "run_horizon",
    "species",
    "species_policy",
    "species_coverage",
    "acquisition_mode",
    "mixed_acquisition_policy",
    "labeling_strategy",
    "labeling_hard",
    "coverage_mode",
    "target_project_count",
    "max_candidate_projects",
    "quota_flexibility",
    "time_budget",
    "on_safety_ceiling",
    "instrument_preference",
    "exclude_rules",
    "success_criteria",
  ]));

  next.confirmed = false;
  return next;
}

export function isReadyForConfirm(spec: IntentSpec): boolean {
  return assessStrategyGaps(spec).ready_for_confirm;
}

export type StrategySlot =
  | "task"
  | "horizon"
  | "species"
  | "acquisition"
  | "theme"
  | "coverage"
  | "labeling"
  | "objective"
  | "instrument";

export type StrategyGapReport = {
  required_missing: StrategySlot[];
  optional_missing: StrategySlot[];
  ready_for_confirm: boolean;
};

/** Semantic slots only: questionnaire completion flags never affect this report. */
export function assessStrategyGaps(spec: IntentSpec): StrategyGapReport {
  const required_missing: StrategySlot[] = [];
  const optional_missing: StrategySlot[] = [];
  const resolved = new Set(spec.resolvedFields || []);
  if (!spec.taskType) required_missing.push("task");
  if (
    spec.targetProjectCount == null
    && (spec.quotaFlexibility === "fixed" || !spec.coverageMode)
  ) required_missing.push("coverage");
  const objective = text(spec.objective);
  if (!objective || isPollutedObjective(objective) || isOrientationPrompt(objective)) {
    required_missing.push("objective");
  }
  if (!spec.species.length && !resolved.has("species") && !resolved.has("species_policy")) {
    optional_missing.push("species");
  }
  if (
    (!spec.acquisitionMode || spec.acquisitionMode === "unknown")
    && !resolved.has("acquisition_mode")
  ) optional_missing.push("acquisition");
  if (
    !isImmunopeptideContext(spec)
    && !spec.ptmTypes.length
    && !spec.specialThemes.length
    && !resolved.has("ptm_types")
    && !resolved.has("special_themes")
  ) {
    optional_missing.push("theme");
  }
  if (
    (!spec.labelingStrategy || ["any", "unknown"].includes(spec.labelingStrategy))
    && !resolved.has("labeling_strategy")
  ) {
    optional_missing.push("labeling");
  }
  return { required_missing, optional_missing, ready_for_confirm: required_missing.length === 0 };
}

export type GapReport = {
  requiredMissing: StrategySlot[];
  optionalMissing: StrategySlot[];
  readyForConfirm: boolean;
  suggestedFocus: StrategySlot | null;
};

export function assessGaps(spec: IntentSpec): GapReport {
  const requiredMissing: StrategySlot[] = [];
  const optionalMissing: StrategySlot[] = [];
  const resolved = new Set(spec.resolvedFields || []);

  if (!spec.taskType) requiredMissing.push("task");
  if (
    spec.targetProjectCount == null
    && (spec.quotaFlexibility === "fixed" || !spec.coverageMode)
  ) requiredMissing.push("coverage");

  // Theme: required when user has no domain substance at all (will fail ready via hasDomainSubstance)
  if (!isImmunopeptideContext(spec) && spec.ptmTypes.length === 0 && spec.specialThemes.length === 0) {
    // optional — browse-all proteomics is allowed
  }

  if (
    !spec.species.length
    && spec.speciesPolicy === "open"
    && !resolved.has("species")
    && !resolved.has("species_policy")
  ) {
    optionalMissing.push("species");
  } else if (!spec.species.length && spec.speciesPolicy !== "open") {
    // policy prefer/include without list → ask
    if (spec.speciesPolicy === "include_only" || spec.speciesPolicy === "prefer") {
      optionalMissing.push("species");
    } else {
      optionalMissing.push("species");
    }
  }

  if (
    (!spec.acquisitionMode || spec.acquisitionMode === "unknown")
    && !resolved.has("acquisition_mode")
  ) {
    optionalMissing.push("acquisition");
  }
  if (
    (!spec.labelingStrategy || spec.labelingStrategy === "any" || spec.labelingStrategy === "unknown")
    && !resolved.has("labeling_strategy")
  ) {
    optionalMissing.push("labeling");
  }

  const readyForConfirm =
    requiredMissing.length === 0 && hasDomainSubstance(spec) && Boolean(spec.taskType);

  const suggestedFocus: StrategySlot | null =
    requiredMissing[0] ||
    (isImmunopeptideContext(spec) && !spec.species.length ? "species" : null) ||
    optionalMissing[0] ||
    null;

  return { requiredMissing, optionalMissing, readyForConfirm, suggestedFocus };
}

/** Option catalog for a semantic slot (guidance for the agent, not a forced form step). */
export function optionsForSlot(slot: StrategySlot, spec: IntentSpec): GrillQuestion | null {
  switch (slot) {
    case "task":
      return questionQ1(spec);
    case "horizon":
      return questionQ2(spec);
    case "species":
      return questionQ3(spec);
    case "acquisition":
      return questionQ4(spec);
    case "theme":
      return questionQ5(spec);
    case "labeling":
      return questionQ6(spec);
    case "coverage":
      return questionQ7(spec);
    case "instrument":
      return questionQ8();
    default:
      return null;
  }
}

/**
 * @deprecated Q-order driver — kept only as catalog fallback via optionsForSlot.
 * Prefer assessGaps + LLM next_focus (D1).
 */

export function nextQuestion(spec: IntentSpec): GrillQuestion | null {
  if (!spec.answered.Q1) return questionQ1(spec);
  if (!spec.answered.Q3) return questionQ3(spec);
  if (!spec.answered.Q4) return questionQ4(spec);
  if (shouldAskQ5(spec) && !spec.answered.Q5) return questionQ5(spec);
  if (shouldAskQ6(spec) && !spec.answered.Q6) return questionQ6(spec);
  if (!spec.answered.Q7) return questionQ7(spec);
  if (shouldAskQ8(spec) && !spec.answered.Q8) return questionQ8();
  if (shouldAskQ9(spec) && !spec.answered.Q9) return questionQ9(spec);
  if (shouldAskQ10(spec) && !spec.answered.Q10) return questionQ10();
  return null;
}

function questionQ1(spec: IntentSpec): GrillQuestion {
  const blob = lower(spec.originalPrompt);
  const immuno = isImmunopeptideContext(spec);
  // Immunopeptide is MHC ligandome data — NOT classical "unknown PTM de novo" by default.
  const rec = /rt|保留/.test(blob)
    ? "rt_prediction"
    : immuno
      ? /de\s*novo|denovo/.test(blob)
        ? "denovo"
        : "browse_only"
      : /phospho|ptm/.test(blob)
        ? "ptm_denovo"
        : /de\s*novo|denovo/.test(blob)
          ? "denovo"
          : "browse_only";
  const why = immuno
    ? "免疫肽是 MHC 呈递肽（常非胰酶肽）；多数人先摸清人源候选，再决定库搜 / de novo。不要默认当成 PTM de novo。"
    : "不同任务对证据与筛选标准不同，例如 RT 预测与 de novo 的要求不一样。";
  return {
    id: "Q1",
    prompt: "这些数据主要用来做什么？（下游任务）",
    why,
    options: immuno
      ? [
          {
            id: "browse_only",
            label: "先摸清免疫肽公共数据、任务先不定",
            recommended: rec === "browse_only",
            reason: "最稳：先人源 HLA 配体候选清单，再决定下游",
          },
          {
            id: "denovo",
            label: "de novo / 序列解读（非常规肽）",
            recommended: rec === "denovo",
            reason: "免疫肽常非胰酶肽，序列解读是常见下游",
          },
          { id: "psm_scoring", label: "PSM/库搜打分" },
          {
            id: "rt_prediction",
            label: "RT 预测（肽保留时间模型）",
            recommended: rec === "rt_prediction",
          },
          { id: "fragment_intensity_prediction", label: "碎片强度预测" },
          { id: "chimeric_interpretation", label: "嵌合谱解释" },
          {
            id: "ptm_denovo",
            label: "PTM de novo（经典修饰发现，通常不是免疫肽主线）",
            recommended: false,
          },
          { id: "other", label: "其它（请一句话说明）" },
        ]
      : [
          {
            id: "rt_prediction",
            label: "RT 预测",
            recommended: rec === "rt_prediction",
            reason: "提到了保留时间/RT",
          },
          { id: "fragment_intensity_prediction", label: "碎片强度预测" },
          { id: "psm_scoring", label: "PSM 打分" },
          {
            id: "denovo",
            label: "de novo",
            recommended: rec === "denovo",
          },
          {
            id: "ptm_denovo",
            label: "PTM de novo",
            recommended: rec === "ptm_denovo",
          },
          { id: "chimeric_interpretation", label: "嵌合谱解释" },
          {
            id: "browse_only",
            label: "先只找数据、任务未定",
            recommended: rec === "browse_only",
            reason: "默认稳妥起点",
          },
          { id: "other", label: "其它（请一句话说明）" },
        ],
    freeTextHint: immuno
      ? "例如：做人源 HLA-I 配体发现、先建候选清单、或做 de novo"
      : "也可直接描述任务",
  };
}

function questionQ2(spec?: IntentSpec): GrillQuestion {
  const immuno = spec ? isImmunopeptideContext(spec) : false;
  const browse = spec?.taskType === "browse_only" || !spec?.taskType;
  return {
    id: "Q2",
    prompt: "这次你希望系统帮你做到哪一步就停？",
    why: immuno
      ? "免疫肽检索可能命中很多项目；先定终点，避免自动滚进高成本处理。"
      : "明确终点，避免搜完自动进入高成本步骤。",
    options: [
      {
        id: "plan_only",
        label: "只对齐需求并给出搜索计划",
        reason: "还不想真搜时用",
      },
      {
        id: "candidates_only",
        label: immuno ? "找到免疫肽候选项目/文件就停" : "找到候选数据就停",
        recommended: true,
        reason: immuno && browse ? "先摸清单，再决定是否做表/训练" : "最常见的数据发现终点",
      },
      {
        id: "candidates_reviewed",
        label: immuno ? "候选后再帮我粗审一版（质量/相关性）" : "找到并再审查一遍候选",
      },
      {
        id: "ai_ready_table",
        label: "做到可训练的数据表（后续仍会再确认）",
      },
      { id: "pre_release", label: "做到发布前候选版本" },
      { id: "full_release", label: "走完正式发布（仍需人工确认）" },
    ],
  };
}

function questionQ3(spec: IntentSpec): GrillQuestion {
  const hasHuman =
    /human|人类|人源|智人/.test(lower(spec.originalPrompt)) || spec.species.includes("human");
  const immuno = isImmunopeptideContext(spec);
  const rec = immuno || hasHuman ? "human_hard" : "open";
  return {
    id: "Q3",
    prompt: immuno ? "免疫肽这次卡不卡人源？" : "对物种有没有要求？",
    why: immuno
      ? "HLA/免疫肽公共数据以 human 为主；物种策略会直接改过滤与排序。"
      : "物种是硬过滤还是软偏好，会显著改变检索与排序。",
    options: [
      {
        id: "open",
        label: immuno ? "不限物种（可能混入鼠源等）" : "没有，开放搜索",
        recommended: rec === "open",
        reason: rec === "open" ? "未指定时更稳妥" : undefined,
      },
      {
        id: "human_hard",
        label: immuno ? "只要 human（硬限制，免疫肽常用）" : "只要 human（硬限制）",
        recommended: rec === "human_hard",
        reason: immuno
          ? "HLA 背景与公共库都以人源为主"
          : hasHuman
            ? "你前面提到了人源/人类"
            : undefined,
      },
      {
        id: "human_prefer",
        label: "优先 human，可含其它",
        reason: immuno ? "想略放宽时用" : undefined,
      },
      {
        id: "broaden",
        label: "开放，并尽量覆盖更多物种",
      },
      { id: "other_species", label: "其它物种（请写出物种名）" },
    ],
  };
}

function questionQ4(spec: IntentSpec): GrillQuestion {
  const dda = DDA_HINT.test(spec.originalPrompt);
  const dia = DIA_HINT.test(spec.originalPrompt);
  const immuno = isImmunopeptideContext(spec);
  const recDda = dda || immuno || (!dda && !dia && spec.taskType === "rt_prediction");
  return {
    id: "Q4",
    prompt: immuno
      ? "免疫肽采集要以 DDA 为主吗？遇到 DDA/DIA 混杂项目怎么办？"
      : "采集方式要 DDA、DIA，还是都可以？混杂项目怎么处理？",
    why: immuno
      ? "经典免疫肽发现流多为 DDA；公共项目偶尔混 DIA，需要约定是否下钻到文件级。"
      : "公共库常有 DDA/DIA 混杂项目，需要事先约定策略。",
    options: [
      {
        id: "dda",
        label: immuno
          ? "只要 DDA；混杂项目进入文件级审查（免疫肽常用）"
          : "只要 DDA；混合项目进入文件级审查",
        recommended: recDda,
        reason: immuno ? "经典免疫肽组多为 DDA" : "RT/shotgun 训练常用",
      },
      { id: "dia", label: "只要 DIA", recommended: Boolean(dia && !dda) },
      {
        id: "any",
        label: "两者都可",
        recommended: !recDda && !(dia && !dda),
      },
      { id: "dda_strict", label: "只要 DDA；混合项目整项排除" },
    ],
  };
}

function questionQ5(spec: IntentSpec): GrillQuestion {
  return {
    id: "Q5",
    prompt: "修饰 / 特殊主题如何限定？（检测到相关线索）",
    why: "PTM、免疫肽、非胰酶、药物处理细胞系等会改变检索词与排除规则。",
    options: [
      {
        id: "keep_inferred",
        label: `沿用已识别：${[...spec.ptmTypes, ...spec.specialThemes].join(" / ") || "无明确限定"}`,
        recommended: true,
      },
      { id: "phospho", label: "磷酸化为主" },
      { id: "immunopeptide", label: "免疫肽 / HLA" },
      { id: "non_tryptic", label: "非胰酶切相关" },
      { id: "patient_vs_line", label: "需区分病人样本 vs 永生化细胞" },
      { id: "none", label: "不限定修饰/主题" },
    ],
    freeTextHint: "也可写出具体修饰或实验主题",
  };
}

function questionQ6(spec: IntentSpec): GrillQuestion {
  return {
    id: "Q6",
    prompt: "标记策略有什么要求？",
    why: "Label-free、等重标签、代谢标记和化学同位素标记的适用样本与定量偏差不同。",
    options: [
      {
        id: "label_free",
        label: "只要 label-free（硬限制）",
        recommended: spec.taskType === "rt_prediction" || LABEL_FREE_HINT.test(spec.originalPrompt),
        reason: "RT 场景更常见",
      },
      { id: "silac", label: "只要 / 需要 SILAC", recommended: SILAC_HINT.test(spec.originalPrompt) },
      { id: "tmt", label: "TMT（多重等重标签）" },
      { id: "itraq", label: "iTRAQ（等重标签）" },
      { id: "dimethyl", label: "二甲基标记（化学同位素）" },
      {
        id: "any",
        label: "不限标记策略",
        recommended: !LABEL_FREE_HINT.test(spec.originalPrompt) && spec.taskType !== "rt_prediction",
      },
    ],
  };
}

function questionQ7(spec: IntentSpec): GrillQuestion {
  const exhaustive = EXHAUSTIVE_HINT.test(`${spec.originalPrompt} ${spec.objective}`);
  return {
    id: "Q7",
    prompt: "选择数量模式：快速找够指定数量，还是尽可能搜全？",
    why: "快速模式会从核心词开始分页检索、去重并审查，达到指定的可用项目数立即停止；数量不够才继续下一页或下一个词。",
    options: [
      {
        id: "quick",
        label: "快速找够 20 个可用项目",
        recommended: !exhaustive,
        reason: "达到目标后立即停止，不必拉完所有检索词",
      },
      {
        id: "exhaustive",
        label: "尽可能搜全（不设候选池上限）",
        recommended: exhaustive,
        reason: "你提到了搜全/越多越好",
      },
    ],
    freeTextHint: "也可以直接说：只要 20 个 / 目标 50 个可用项目",
  };
}

function questionQ8(): GrillQuestion {
  return {
    id: "Q8",
    prompt: "仪器平台偏好？（默认按发布时间偏新仪器）",
    why: "专家约定：默认以新仪器为主，除非你明确要老仪器。",
    options: [
      { id: "newer", label: "尽量新仪器（默认，按发布时间）", recommended: true, reason: "专家默认：越新越好" },
      { id: "newer_with_legacy_floor", label: "新为主，老仪器约保留 20% 保底" },
      { id: "classic", label: "尽量经典/老仪器" },
      { id: "none", label: "无特殊偏好" },
    ],
  };
}

function questionQ9(spec: IntentSpec): GrillQuestion {
  return {
    id: "Q9",
    prompt: "有没有必须排除的数据类型？",
    why: "硬排除会直接过滤；常见如 IP-MS、交联蛋白。过小/过大文件不必纠结。",
    options: [
      {
        id: "none",
        label: "没有必须排除的",
        recommended: !IP_MS.test(spec.originalPrompt) && !CROSSLINK.test(spec.originalPrompt),
      },
      { id: "ip_ms", label: "排除 IP-MS / AP-MS", recommended: IP_MS.test(spec.originalPrompt) },
      { id: "crosslink", label: "排除交联蛋白 / XL-MS", recommended: CROSSLINK.test(spec.originalPrompt) },
      { id: "topdown", label: "排除 Top-down / intact" },
      { id: "custom", label: "其它排除（请写出）" },
    ],
  };
}

function questionQ10(): GrillQuestion {
  return {
    id: "Q10",
    prompt: "怎样算这轮发现「够用」？（成功标准，软目标）",
    why: "这些是覆盖检查，不会静默变成硬排除。",
    options: [
      {
        id: "default_diverse",
        label: "采用推荐：多厂商仪器 +（开放物种时）多物种 + LC 长短梯度",
        recommended: true,
        reason: "专家多样性标准",
      },
      { id: "count_only", label: "主要看数量是否接近目标" },
      { id: "custom", label: "自定义成功标准（请写出）" },
    ],
  };
}

function uniquePush(list: string[], item: string): string[] {
  const value = text(item);
  if (!value) return list;
  return list.includes(value) ? list : [...list, value];
}

function normalizeOptionKey(answer: string): string {
  const m = text(answer).match(/^([0-9]{1,2})[).、.\s]/);
  if (m) return m[1];
  return lower(answer);
}

export function applyAnswer(spec: IntentSpec, questionId: QuestionId, rawAnswer: string): IntentSpec {
  const answer = text(rawAnswer);
  const key = normalizeOptionKey(answer);
  let next = {
    ...spec,
    answered: { ...spec.answered },
    excludeRules: [...spec.excludeRules],
    successCriteria: [...spec.successCriteria],
    ptmTypes: [...spec.ptmTypes],
    specialThemes: [...spec.specialThemes],
    selectedSearchTerms: [...(spec.selectedSearchTerms || [])],
  };

  switch (questionId) {
    case "Q1": {
      const map: Record<string, TaskType> = {
        "1": "rt_prediction",
        rt_prediction: "rt_prediction",
        "rt 预测": "rt_prediction",
        rt: "rt_prediction",
        "2": "fragment_intensity_prediction",
        fragment_intensity_prediction: "fragment_intensity_prediction",
        "3": "psm_scoring",
        psm_scoring: "psm_scoring",
        "4": "denovo",
        denovo: "denovo",
        "de novo": "denovo",
        "5": "ptm_denovo",
        ptm_denovo: "ptm_denovo",
        "6": "chimeric_interpretation",
        chimeric_interpretation: "chimeric_interpretation",
        "7": "browse_only",
        browse_only: "browse_only",
        "先只找": "browse_only",
        "8": "other",
        other: "other",
      };
      const mapped = map[key] || map[lower(answer)];
      if (mapped) {
        next.taskType = mapped;
      } else if (/免疫肽|immunopeptid|hla|找数据|搜数据|浏览|只要数据/.test(answer)) {
        // Free-text data request ≠ option id; default browse_only and keep theme via absorb.
        next.taskType = "browse_only";
      } else {
        next.taskType = "other";
        next.notes = [next.notes, answer].filter(Boolean).join("；");
      }
      // Never write bare "7" / option tokens into objective.
      if (!isPollutedObjective(answer) && answer.length >= 2) {
        if (isPollutedObjective(next.objective) || !next.objective) next.objective = answer;
        if (!next.originalPrompt || isGreetingPrompt(next.originalPrompt)) next.originalPrompt = answer;
      }
      next = absorbFreeTextSignals(next, answer);
      next = markAnswered(next, "Q1");
      break;
    }
    case "Q2": {
      const map: Record<string, RunHorizon> = {
        "1": "plan_only",
        plan_only: "plan_only",
        "2": "candidates_only",
        candidates_only: "candidates_only",
        "候选": "candidates_only",
        "3": "candidates_reviewed",
        candidates_reviewed: "candidates_reviewed",
        "4": "ai_ready_table",
        ai_ready_table: "ai_ready_table",
        "5": "pre_release",
        pre_release: "pre_release",
        "6": "full_release",
        full_release: "full_release",
      };
      next.runHorizon = "candidates_reviewed";
      next = markAnswered(next, "Q2");
      break;
    }
    case "Q3": {
      if (key === "1" || key === "open" || /开放/.test(answer)) {
        next.species = [];
        next.speciesPolicy = "open";
        next.speciesCoverage = "none";
      } else if (
        key === "2" ||
        key === "human_hard" ||
        /只要.*human|硬限制|只要人|仅人类|仅人源/.test(answer)
      ) {
        next.species = ["human"];
        next.speciesPolicy = "include_only";
      } else if (
        key === "3" ||
        key === "human_prefer" ||
        /优先.*human|优先人|人源|人类|^人$|^human$/i.test(answer)
      ) {
        next.species = ["human"];
        next.speciesPolicy = "prefer";
      } else if (key === "4" || key === "broaden" || /更多物种/.test(answer)) {
        next.speciesPolicy = "open";
        next.speciesCoverage = "broaden";
      } else if (/^(人|人类|人源|human)$/i.test(answer.trim())) {
        next.species = ["human"];
        next.speciesPolicy = "prefer";
      } else {
        const hit = detectSpeciesSignals(answer);
        if (hit && hit.species.length) {
          next.species = [...hit.species];
          next.speciesPolicy = hit.policy;
        } else {
          const sp = answer.replace(/^其它物种[:：]?\s*/i, "").trim();
          next.species = [sp === "人" ? "human" : sp];
          next.speciesPolicy = "include_only";
        }
      }
      next = markAnswered(next, "Q3");
      break;
    }
    case "Q4": {
      if (key === "1" || key === "dda" || /混合项目进入文件级审查/.test(answer)) {
        next.acquisitionMode = "dda";
        next.mixedAcquisitionPolicy = "review_mixed";
      } else if (key === "2" || key === "dia" || /只要 dia/i.test(answer)) {
        next.acquisitionMode = "dia";
        next.mixedAcquisitionPolicy = "review_mixed";
      } else if (key === "3" || key === "any" || /两者都可/.test(answer)) {
        next.acquisitionMode = "unknown";
        next.mixedAcquisitionPolicy = "allow";
      } else if (key === "4" || key === "dda_strict" || /整项排除/.test(answer)) {
        next.acquisitionMode = "dda";
        next.mixedAcquisitionPolicy = "reject_mixed";
      } else {
        next.acquisitionMode = "unknown";
      }
      next = markAnswered(next, "Q4");
      break;
    }
    case "Q5": {
      if (key === "1" || key === "keep_inferred" || /沿用/.test(answer)) {
        // keep
      } else if (key === "2" || key === "phospho" || /磷酸化/.test(answer)) {
        next.ptmTypes = ["phospho"];
      } else if (key === "3" || key === "immunopeptide" || /免疫肽|hla/i.test(answer)) {
        next.ptmTypes = ["immunopeptide"];
      } else if (key === "4" || key === "non_tryptic" || /非胰酶/.test(answer)) {
        if (!next.specialThemes.includes("non_tryptic")) next.specialThemes.push("non_tryptic");
      } else if (key === "5" || key === "patient_vs_line" || /病人|永生化/.test(answer)) {
        if (!next.specialThemes.includes("patient_sample")) next.specialThemes.push("patient_sample");
        if (!next.specialThemes.includes("immortalized_cell_line")) next.specialThemes.push("immortalized_cell_line");
      } else if (key === "6" || key === "none" || /不限定/.test(answer)) {
        next.ptmTypes = [];
        next.specialThemes = [];
      } else {
        next.notes = [next.notes, `特殊主题: ${answer}`].filter(Boolean).join("；");
      }
      next = markAnswered(next, "Q5");
      break;
    }
    case "Q6": {
      if (key === "1" || key === "label_free" || /label-free|无标记/.test(answer)) {
        next.labelingStrategy = "label_free";
        next.labelingHard = true;
      } else if (key === "2" || key === "silac" || /silac/i.test(answer)) {
        next.labelingStrategy = "silac";
        next.labelingHard = true;
      } else if (key === "3" || key === "tmt" || /tmt/i.test(answer)) {
        next.labelingStrategy = "tmt";
        next.labelingHard = false;
      } else if (key === "4" || key === "itraq" || /itraq/i.test(answer)) {
        next.labelingStrategy = "itraq";
        next.labelingHard = false;
      } else if (key === "5" || key === "dimethyl" || /dimethyl|二甲基/i.test(answer)) {
        next.labelingStrategy = "dimethyl";
        next.labelingHard = false;
      } else {
        next.labelingStrategy = "any";
        next.labelingHard = false;
      }
      next = markAnswered(next, "Q6");
      break;
    }
    case "Q7": {
      const exact = extractTargetProjectCount(answer);
      if (exact != null) {
        next = applyTargetProjectCount(next, exact);
        break;
      }
      let mode: CoverageMode = "quick";
      if (key === "2" || key === "exhaustive" || /搜全|尽量/.test(answer)) mode = "exhaustive";
      else if (key === "1" || key === "quick" || key === "curated" || /快速|精选/.test(answer)) mode = "quick";
      if (mode === "quick") {
        next = applyTargetProjectCount(next, 20);
        break;
      }
      next.coverageMode = mode;
      const q = coverageQuota(mode);
      next.targetProjectCount = q.projects;
      next.maxCandidateProjects = q.pool;
      next.quotaFlexibility = mode === "exhaustive" ? "open_ended" : "recommended";
      next.timeBudget = "multi_round";
      next.onSafetyCeiling = "ask";
      next = markAnswered(next, "Q7");
      break;
    }
    case "Q8": {
      if (key === "1" || key === "newer" || /新仪器/.test(answer)) next.instrumentPreference = "newer";
      else if (key === "2" || key === "newer_with_legacy_floor" || /保底/.test(answer)) {
        next.instrumentPreference = "newer_with_legacy_floor";
        next.legacyFloorRatio = 0.2;
      } else if (key === "3" || key === "classic" || /老仪器|经典/.test(answer)) next.instrumentPreference = "classic";
      else next.instrumentPreference = "none";
      next = markAnswered(next, "Q8");
      break;
    }
    case "Q9": {
      if (key === "1" || key === "none" || /没有/.test(answer)) next.excludeRules = [];
      else if (key === "2" || key === "ip_ms" || /ip-ms/i.test(answer)) next.excludeRules = uniquePush(next.excludeRules, "IP-MS / AP-MS");
      else if (key === "3" || key === "crosslink" || /交联/.test(answer)) next.excludeRules = uniquePush(next.excludeRules, "交联蛋白 / XL-MS");
      else if (key === "4" || key === "topdown" || /top-down/i.test(answer)) next.excludeRules = uniquePush(next.excludeRules, "Top-down / intact protein");
      else next.excludeRules = uniquePush(next.excludeRules, answer);
      next = markAnswered(next, "Q9");
      break;
    }
    case "Q10": {
      if (key === "1" || key === "default_diverse" || /推荐|多厂商/.test(answer)) {
        next.successCriteria = defaultSuccessCriteria(next);
      } else if (key === "2" || key === "count_only" || /数量/.test(answer)) {
        next.successCriteria = next.targetProjectCount
          ? [`可用项目数接近目标约 ${next.targetProjectCount}`]
          : ["达到所选覆盖程度的数量目标"];
      } else {
        next.successCriteria = [answer];
      }
      next = markAnswered(next, "Q10");
      break;
    }
    default:
      break;
  }
  // Any free-text answer can still carry themes / species / counts.
  if (answer && !/^[0-9]{1,2}$/.test(answer.trim())) {
    next = absorbFreeTextSignals(next, answer);
  } else {
    next = sanitizeIntentObjective(next);
  }
  return next;
}

export function buildStrategyCard(spec: IntentSpec): StrategyCard {
  const coverage = COVERAGE_LABELS[spec.coverageMode] || "均衡";
  const quota = coverageQuota(spec.coverageMode || "balanced");
  const projects = spec.targetProjectCount ?? quota.projects;
  const pool = spec.maxCandidateProjects ?? quota.pool;
  const openEnded = spec.quotaFlexibility === "open_ended";
  const fixedQuota = spec.quotaFlexibility === "fixed";
  const hasFixedCount = fixedQuota && spec.targetProjectCount != null;
  const resolved = new Set(spec.resolvedFields || []);
  const hard: string[] = [];
  const soft: string[] = [];

  if (spec.speciesPolicy === "include_only" && spec.species.length) {
    hard.push(`物种必须为：${spec.species.join(", ")}`);
  } else if (spec.speciesPolicy === "exclude" && spec.species.length) {
    hard.push(`排除物种：${spec.species.join(", ")}`);
  } else if (spec.speciesPolicy === "prefer" && spec.species.length) {
    soft.push(`物种优先：${spec.species.join(", ")}`);
  } else if (spec.speciesCoverage === "broaden") {
    soft.push("尽量覆盖更多物种");
  } else {
    soft.push("物种开放");
  }

  if (spec.acquisitionMode === "dda" || spec.acquisitionMode === "dia") {
    hard.push(`采集方式：${spec.acquisitionMode.toUpperCase()}`);
  } else {
    soft.push("采集方式不限");
  }
  const showMixedAcquisitionPolicy = Boolean(spec.acquisitionMode)
    || resolved.has("mixed_acquisition_policy")
    || spec.mixedAcquisitionPolicy !== "review_mixed";
  if (showMixedAcquisitionPolicy) {
    if (spec.mixedAcquisitionPolicy === "reject_mixed") hard.push("混合采集项目整项排除");
    else if (spec.mixedAcquisitionPolicy === "review_mixed") soft.push("混合采集项目进入文件级审查");
    else soft.push("混合采集项目可保留");
  }

  const labelingLabel: Record<string, string> = {
    label_free: "label-free",
    tmt: "TMT",
    itraq: "iTRAQ",
    silac: "SILAC",
    dimethyl: "二甲基标记",
  };
  if (spec.labelingStrategy && !["any", "unknown"].includes(spec.labelingStrategy)) {
    const label = labelingLabel[spec.labelingStrategy] || spec.labelingStrategy;
    if (spec.labelingHard) hard.push(`标记方式必须为：${label}`);
    else soft.push(`标记方式优先：${label}`);
  }

  if (spec.ptmTypes.length) hard.push(`主题/修饰：${spec.ptmTypes.join(", ")}`);
  for (const theme of spec.specialThemes) soft.push(`特殊主题：${theme}`);
  for (const rule of spec.excludeRules) hard.push(`排除：${rule}`);

  if (spec.instrumentPreference === "newer") soft.push("仪器：偏新（按发布时间）");
  else if (spec.instrumentPreference === "classic") soft.push("仪器：偏经典/老");
  else if (spec.instrumentPreference === "newer_with_legacy_floor") {
    soft.push(`仪器：新为主，老仪器保底约 ${Math.round((spec.legacyFloorRatio || 0.2) * 100)}%`);
  }

  for (const c of spec.successCriteria) soft.push(`成功标准：${c}`);

  if (spec.targetProjectCount != null) {
    if (fixedQuota) hard.push(`固定数量目标：${spec.targetProjectCount} 个项目（待搜索核验）`);
    else if (spec.quotaFlexibility === "recommended") {
      soft.push(`目标项目数约 ${spec.targetProjectCount}（建议值）`);
    }
  } else if (fixedQuota) {
    hard.push("固定数量目标：待补齐（尚未搜索）");
  } else if (openEnded) {
    soft.push("数量目标：安全上限内尽可能多，不以默认配额提前停止");
  }
  for (const constraint of spec.scientificConstraints) {
    const rendered = `${constraint.label}（${constraint.operator} ${
      Array.isArray(constraint.value)
        ? constraint.value.join(", ")
        : String(constraint.value ?? "待核验")
    }）`;
    if (constraint.strength === "hard") hard.push(rendered);
    else soft.push(rendered);
  }

  const horizon = RUN_HORIZON_LABELS[spec.runHorizon] || "找到候选数据就停";
  const task = TASK_TYPE_LABELS[spec.taskType] || spec.taskType || "未指定";

  return {
    summaryLines: [
      `目标：${deriveObjective(spec)}`,
      `下游任务：${task}`,
      `本次终点：${horizon}`,
      `覆盖程度：${coverage}`,
      openEnded
        ? "目标规模：尽可能多（以质量收敛与服务器安全上限为边界）"
        : hasFixedCount
          ? `快速模式 · 固定数量目标：${projects} 个项目（待搜索核验）；达到即停`
          : fixedQuota
            ? "固定数量目标：待补齐（尚未搜索）"
            : `目标规模：约 ${projects} 个可用项目；候选池约 ${pool}`,
      `撞顶策略：${spec.onSafetyCeiling === "ask" ? "触达安全上限时询问你" : spec.onSafetyCeiling}`,
      `仓库：${spec.repository || "pride"}`,
    ],
    hardConstraints: hard.length ? hard : ["（无额外硬约束）"],
    softPreferences: soft.length ? soft : ["（无额外软偏好）"],
    targetQuota: openEnded
      ? "不设固定数量目标；持续补搜高质量项目，直到质量收敛或触达服务器安全上限。"
      : hasFixedCount
        ? `固定数量目标 ${projects} 个；搜索后才核验实际可用项目数。从核心词开始分页检索并逐批审查，达到目标立即停止，不足才继续下一页或下一个词。`
        : fixedQuota
          ? "固定数量目标尚未补齐；当前预览不表示已经找到或入选任何项目。"
          : `目标入选约 ${projects}；候选池约 ${pool}（可随搜全抬高，不可突破服务器安全上限）`,
    safetyNote: "安全上限（运行时间 / 轮次 / 磁盘 / 并发）由服务器保护，Agent 不会私自放宽。",
    confirmButtonLabel: `确认并开始：按上述策略搜索，${horizon}`,
  };
}

export function searchTermCandidates(spec: IntentSpec): string[] {
  const themes: string[] = [];
  const add = (value: string) => {
    const normalized = text(value).replaceAll("_", " ");
    if (
      normalized &&
      !themes.some((existing) => existing.toLocaleLowerCase() === normalized.toLocaleLowerCase())
    ) {
      themes.push(normalized);
    }
  };
  (spec.selectedSearchTerms || []).forEach(add);
  spec.specialThemes.forEach(add);
  if (
    isImmunopeptideContext(spec) &&
    !themes.some((theme) => /immunopept|HLA|MHC/i.test(theme))
  ) {
    add("immunopeptidomics");
  }
  if (isImmunopeptideContext(spec) || themes.some((theme) => /immunopept|HLA|MHC/i.test(theme))) {
    [
      "immunopeptidomics",
      "immunopeptidome",
      "HLA ligandome",
      "MHC ligandome",
      "HLA peptidome",
      "MHC peptidome",
      "HLA class I ligandome",
      "HLA class II ligandome",
      "immunopeptides",
      "HLA ligands",
      "MHC ligands",
      "MHC-associated peptides",
    ].forEach(add);
  }
  spec.ptmTypes
    .filter((theme) => theme !== "immunopeptide")
    .forEach(add);
  if (!themes.length) {
    const objective = text(spec.objective || spec.originalPrompt);
    const knownThemeTerms: Array<[RegExp, string]> = [
      [/phospho(?:proteom|peptid)|磷酸化/i, "phosphoproteomics"],
      [/glyco(?:proteom|peptid)|糖基化/i, "glycoproteomics"],
      [/ubiquiti(?:nom|nyl)|泛素化/i, "ubiquitinomics"],
      [/acetyl(?:proteom|ation)|乙酰化/i, "acetylproteomics"],
      [/metabolom|代谢组/i, "metabolomics"],
      [/proteogenom|蛋白基因组/i, "proteogenomics"],
      [/\bproteomics?\b|蛋白质组/i, "proteomics"],
    ];
    knownThemeTerms.forEach(([pattern, term]) => {
      if (pattern.test(objective)) add(term);
    });
  }
  return themes.slice(0, 100);
}

export function confirmedSearchThemes(spec: IntentSpec): string[] {
  const selected = normalizeSearchTerms(spec.selectedSearchTerms || []);
  return selected.length ? selected.slice(0, 100) : searchTermCandidates(spec);
}

export function normalizeSearchTerms(values: string[]): string[] {
  const seen = new Set<string>();
  const normalized: string[] = [];
  for (const value of values) {
    const term = String(value || "").trim().replace(/\s+/g, " ");
    const key = term.toLocaleLowerCase();
    if (!term || term.length > 240 || seen.has(key)) continue;
    seen.add(key);
    normalized.push(term);
  }
  return normalized;
}

export function reconcileSearchTermSelection(
  previous: string[],
  spec: IntentSpec,
): string[] {
  const candidates = searchTermCandidates(spec);
  const persisted = (spec.selectedSearchTerms || []).filter((term) => candidates.includes(term));
  if (persisted.length) return persisted;
  const stillRelevant = previous.filter((term) => candidates.includes(term));
  return stillRelevant.length ? stillRelevant : candidates;
}

export function toDiscoveryJobPayload(spec: IntentSpec): DiscoveryJobPayload {
  const intentText = [
    spec.originalPrompt,
    spec.objective,
    spec.notes,
    ...spec.successCriteria,
  ].filter(Boolean).join(" ");
  // Legacy sessions may already contain curated/20/80 from the old browse-only
  // recommendation. An explicit "all/exhaustive/no limit" commitment is the
  // stronger user instruction unless the user also fixed an exact count.
  const repairLegacyExhaustive =
    spec.quotaFlexibility !== "fixed" && EXHAUSTIVE_HINT.test(intentText);
  const effectiveCoverage: CoverageMode = repairLegacyExhaustive
    ? "exhaustive"
    : (spec.coverageMode || "balanced");
  const effectiveQuotaFlexibility = repairLegacyExhaustive
    ? "open_ended"
    : (spec.quotaFlexibility || "recommended");
  const quota = coverageQuota(effectiveCoverage);
  const immunopeptideContext = isImmunopeptideContext(spec);
  const openEnded =
    effectiveQuotaFlexibility === "open_ended" ||
    (
      effectiveQuotaFlexibility !== "fixed" &&
      (
        effectiveCoverage === "exhaustive" ||
        EXHAUSTIVE_HINT.test(intentText)
      )
    );
  // Soft ambition for selected projects; open-ended / 越多越好 is not a hard knife.
  const projects = openEnded
    ? Math.min(5000, Math.max(coverageQuota("exhaustive").projects, spec.targetProjectCount ?? quota.projects))
    : Math.min(5000, Math.max(1, spec.targetProjectCount ?? quota.projects));
  // Business candidate pool: open-ended uses high safety ceiling (not 600/1000 business cap).
  const rawPool = Math.max(projects, spec.maxCandidateProjects ?? quota.pool);
  const pool = openEnded
    ? Math.min(20000, Math.max(rawPool, 20000))
    : Math.min(20000, rawPool);

  const hardFields = ["repository"];
  if (spec.speciesPolicy === "include_only" || spec.speciesPolicy === "exclude") {
    hardFields.push("species", "species_policy");
  }
  if (spec.acquisitionMode === "dda" || spec.acquisitionMode === "dia") hardFields.push("acquisition_mode");
  if (spec.labelingHard) hardFields.push("labeling_strategy");
  if (immunopeptideContext) hardFields.push("goal");
  else if (spec.ptmTypes.length) hardFields.push("goal", "ptm_type", "ptm_types");
  if (spec.quotaFlexibility === "fixed") {
    hardFields.push("max_projects", "quota_flexibility");
  }
  for (const constraint of spec.scientificConstraints) {
    if (constraint.strength === "hard") hardFields.push(`constraint:${constraint.id}`);
  }

  const provenance: Record<string, string> = { repository: "user" };
  for (const field of hardFields) provenance[field] = "user";
  if (spec.labelingStrategy && !["any", "unknown"].includes(spec.labelingStrategy)) {
    provenance.labeling_strategy = spec.labelingHard ? "user" : "user_preference";
  }
  for (const constraint of spec.scientificConstraints) {
    provenance[`constraint:${constraint.id}`] = constraint.source;
  }

  let goal = "general";
  if (immunopeptideContext) goal = "immunopeptidomics";
  else if (spec.ptmTypes.length) goal = "ptm";

  let labeling = spec.labelingStrategy || "unknown";
  if (labeling === "any") labeling = "unknown";

  let speciesPolicy = spec.speciesPolicy;
  if (speciesPolicy === "prefer") speciesPolicy = "open";

  const diversity =
    spec.speciesCoverage === "broaden" ||
    effectiveCoverage === "exhaustive" ||
    spec.instrumentPreference === "newer_with_legacy_floor"
      ? "high"
      : "balanced";

  // Backend modeling task_type only accepts RT/denovo/... profiles.
  // Grill "browse_only" / "other" means data discovery without a fixed modeling task.
  const apiTaskType =
    !spec.taskType || spec.taskType === "browse_only" || spec.taskType === "other"
      ? ""
      : spec.taskType;
  const taskPromptNote =
    spec.taskType === "browse_only"
      ? "task=browse_only (data discovery only; modeling task undecided)"
      : spec.taskType === "other"
        ? `task=other (${spec.notes || "user-described"})`
        : spec.taskType
          ? `task=${spec.taskType}`
          : "";

  const promptParts = [
    spec.objective,
    spec.originalPrompt !== spec.objective ? spec.originalPrompt : "",
    taskPromptNote,
    spec.notes,
    spec.excludeRules.length ? `exclude: ${spec.excludeRules.join("; ")}` : "",
    spec.successCriteria.length ? `success: ${spec.successCriteria.join("; ")}` : "",
    spec.instrumentPreference ? `instruments=${spec.instrumentPreference}` : "",
  ].filter(Boolean);
  // Backend OpenAI Agents mode requires a non-empty discovery request/prompt.
  const prompt =
    promptParts.join("\n").trim() ||
    "Proteomics data discovery (grill-confirmed strategy; objective not captured in free text).";

  return {
    prompt,
    runtime: "openai_agents",
    source: spec.repository === "local" ? "local" : "remote",
    repository: spec.repository || "pride",
    local_dir: (spec.localDir || "").trim(),
    output_language: "zh-CN",
    constraints_enabled: hardFields.length > 1,
    goal,
    query_terms: confirmedSearchThemes(spec),
    task_type: apiTaskType,
    acquisition_mode: spec.acquisitionMode || "unknown",
    labeling_strategy: labeling,
    labeling_hard: spec.labelingHard,
    mixed_acquisition_policy: spec.mixedAcquisitionPolicy || "review_mixed",
    species: spec.species,
    species_policy: speciesPolicy,
    diversity_strategy: diversity,
    scale_mode: effectiveCoverage,
    ptm_types: spec.ptmTypes.filter((x) => x !== "immunopeptide"),
    max_projects: projects,
    max_candidate_projects: pool,
    continuous_discovery: openEnded || effectiveQuotaFlexibility === "fixed",
    partial_delivery_batch_size: 500,
    inspection_batch_size: 30,
    use_memory: true,
    save_memory: true,
    hard_constraint_fields: hardFields,
    constraint_provenance: provenance,
    idempotency_key: crypto.randomUUID(),
    grill_confirmed: spec.confirmed === true,
    run_horizon: "candidates_reviewed",
    quota_flexibility: effectiveQuotaFlexibility,
    quantity_scope: openEnded ? "portfolio" : "unspecified",
    portfolio_size_preference: openEnded
      ? "maximize_qualified_projects"
      : null,
    instrument_preference: spec.instrumentPreference || "none",
    legacy_floor_ratio: spec.legacyFloorRatio,
    exclude_rules: spec.excludeRules,
    success_criteria: spec.successCriteria,
    scientific_constraints: spec.scientificConstraints,
    on_safety_ceiling: spec.onSafetyCeiling || "ask",
    time_budget_preference: spec.timeBudget || "multi_round",
  };
}

/** True when user is asking for explanation rather than answering. */
export function isMetaOrConfusedPrompt(prompt: string): boolean {
  const value = text(prompt);
  if (!value) return false;
  // Pure numbered / short option-like answers are not meta.
  if (/^[0-9]{1,2}([).、.\s].*)?$/.test(value)) return false;
  return /什么意思|啥意思|是什么|不懂|不太懂|为什么|为何|解释|区别|推荐哪个|该选哪个|怎么选|教我|举例|能不能说|帮我理解|什么叫|我不是这个意思|不是这个意思|你写错|写错了|不对吧|感觉.*死板|一点都不像/i.test(
    value,
  );
}

/** User is complaining about / questioning the strategy card itself (not changing a field). */
export function isStrategyComplaintPrompt(prompt: string): boolean {
  const value = text(prompt);
  if (!value) return false;
  return /为什么目标|目标[是为：: ]*\d|你写的目标|目标怎么是|不是.*意思|写错|搞错了|卡片.*错|策略.*错|我没说.*\d/i.test(
    value,
  );
}

export function formatMetaReply(question: GrillQuestion): string {
  const rec = question.options.find((opt) => opt.recommended);
  const lines = [
    question.why || `我想确认：${question.prompt}`,
  ];
  if (rec) {
    lines.push(
      rec.reason
        ? `不确定的话，我更倾向 **${rec.label}**——${rec.reason}。随时能改。`
        : `不确定的话，可以先按 **${rec.label}** 走，随时能改。`,
    );
  }
  lines.push("直接说你的判断就行，不必背编号。");
  return lines.join("\n");
}

export function summarizeAnsweredField(spec: IntentSpec, questionId: QuestionId): string {
  switch (questionId) {
    case "Q1":
      return TASK_TYPE_LABELS[spec.taskType] || spec.taskType || "下游任务";
    case "Q2":
      return RUN_HORIZON_LABELS[spec.runHorizon] || spec.runHorizon || "本次终点";
    case "Q3":
      if (spec.speciesPolicy === "include_only" && spec.species.length) return `必须 ${spec.species.join(", ")}`;
      if (spec.speciesPolicy === "prefer" && spec.species.length) return `优先 ${spec.species.join(", ")}`;
      if (spec.speciesCoverage === "broaden") return "开放并尽量多物种";
      return "物种开放";
    case "Q4":
      return spec.acquisitionMode
        ? `${spec.acquisitionMode.toUpperCase()} / ${spec.mixedAcquisitionPolicy}`
        : "采集方式";
    case "Q5":
      return spec.ptmTypes.length || spec.specialThemes.length
        ? [...spec.ptmTypes, ...spec.specialThemes].join(", ")
        : "无特殊主题限制";
    case "Q6":
      return spec.labelingStrategy || "标记策略";
    case "Q7":
      if (spec.targetProjectCount) {
        const mode = COVERAGE_LABELS[spec.coverageMode] || "";
        return mode
          ? `${mode} · 约 ${spec.targetProjectCount} 个可用项目`
          : `约 ${spec.targetProjectCount} 个可用项目`;
      }
      return COVERAGE_LABELS[spec.coverageMode] || spec.coverageMode || "覆盖程度";
    case "Q8":
      return spec.instrumentPreference || "仪器偏好";
    case "Q9":
      return spec.excludeRules.length ? spec.excludeRules.join("；") : "无硬排除";
    case "Q10":
      return spec.successCriteria.length ? spec.successCriteria.join("；") : "成功标准";
    default:
      return "已记录";
  }
}

export function formatAnswerAcknowledgement(
  questionId: QuestionId,
  spec: IntentSpec,
  rawAnswer?: string,
): string {
  const summary = summarizeAnsweredField(spec, questionId);
  const brief = taskBrief(spec);
  const immuno = isImmunopeptideContext(spec);
  const snippets: Record<QuestionId, string> = {
    Q1: immuno
      ? `下游先按「${summary}」来理解——免疫肽场景下这会直接影响后面筛项目的口径。`
      : `好，下游任务记作「${summary}」。`,
    Q2: `终点清楚了：${summary}。到这一步我就停，不会偷偷往下跑。`,
    Q3: immuno
      ? `物种策略：${summary}。免疫肽公共库里人源往往最齐，其它物种当补充更稳。`
      : `物种策略：${summary}。`,
    Q4: `采集侧按「${summary}」处理混杂项目。`,
    Q5: `主题/修饰：${summary}。`,
    Q6: `标记策略：${summary}。`,
    Q7:
      spec.quotaFlexibility === "fixed" && spec.targetProjectCount
        ? `规模按你指定的约 ${spec.targetProjectCount} 个可用项目来，不再用档位默认。`
        : `覆盖规模：${summary}。`,
    Q8: `仪器偏好：${summary}。`,
    Q9: `排除规则：${summary}。`,
    Q10: `成功标准已记下：${summary}。`,
  };
  const base = snippets[questionId] || `已记下：${summary}。`;
  // keep short; next question body carries the real expertise
  if (rawAnswer && rawAnswer.trim().length > 24) {
    return `${base}（围绕「${brief}」继续）`;
  }
  return base;
}


export function intentSnapshotForLlm(spec: IntentSpec): Record<string, unknown> {
  return {
    objective: spec.objective,
    original_prompt: spec.originalPrompt,
    task_type: spec.taskType,
    run_horizon: "candidates_reviewed",
    species: spec.species,
    species_policy: spec.speciesPolicy,
    species_coverage: spec.speciesCoverage,
    acquisition_mode: spec.acquisitionMode,
    mixed_acquisition_policy: spec.mixedAcquisitionPolicy,
    labeling_strategy: spec.labelingStrategy,
    labeling_hard: spec.labelingHard,
    coverage_mode: spec.coverageMode,
    target_project_count: spec.targetProjectCount,
    max_candidate_projects: spec.maxCandidateProjects,
    quota_flexibility: spec.quotaFlexibility,
    time_budget: spec.timeBudget,
    on_safety_ceiling: spec.onSafetyCeiling,
    instrument_preference: spec.instrumentPreference,
    legacy_floor_ratio: spec.legacyFloorRatio,
    exclude_rules: spec.excludeRules,
    ptm_types: spec.ptmTypes,
    special_themes: spec.specialThemes,
    selected_search_terms: spec.selectedSearchTerms || [],
    success_criteria: spec.successCriteria,
    scientific_constraints: spec.scientificConstraints,
    notes: spec.notes,
    open_risks: spec.openRisks,
    repository: spec.repository,
    resolved_fields: spec.resolvedFields || [],
  };
}

export function pendingQuestionPayload(question: GrillQuestion): Record<string, unknown> {
  return {
    id: question.id,
    prompt: question.prompt,
    why: question.why,
    options: question.options.map((opt) => ({
      id: opt.id,
      label: opt.label,
      recommended: Boolean(opt.recommended),
      reason: opt.reason || "",
    })),
  };
}

/** Compact option footer — hints only; never the whole personality of the turn. */
export function formatOptionsFooter(question: GrillQuestion, spec: IntentSpec): string {
  if (!question.options.length) {
    return "直接说就行；拿不准可以说 **按推荐默认**。";
  }
  const rec = question.options.find((o) => o.recommended);
  const lines: string[] = [];
  if (rec) {
    lines.push(
      rec.reason
        ? `我更建议：**${rec.label}**（${rec.reason}）。你也可以点选或直接改口。`
        : `我更建议：**${rec.label}**。你也可以点选或直接改口。`,
    );
  } else {
    lines.push("你可以这样选（回序号或原话都行）：");
  }
  // Grill-me style: show the concrete menu so users can click mentally without form feel.
  // Cap at 7 to keep scannable; remaining still matchable by free text.
  const show = question.options.slice(0, 7);
  show.forEach((opt, index) => {
    const mark = opt.recommended ? "（推荐）" : "";
    lines.push(`${index + 1}. ${opt.label}${mark}`);
  });
  if (question.options.length > 7) {
    lines.push(`…还有 ${question.options.length - 7} 个方向，直接描述也行`);
  }
  lines.push("想先聊聊理清思路也可以；拿不准就说 **按推荐默认**。");
  return lines.join("\n");
}

/**
 * Local fallback question framing.
 * Prefer LLM phrasing when available; this path still injects task brief + recommendation reasons.
 */
export function formatQuestionMessage(question: GrillQuestion, spec: IntentSpec): string {
  const brief = taskBrief(spec);
  const lines: string[] = [];
  if (spec.inferred[question.id]) {
    lines.push("前面信息里我先做了个推断，不对你随时改。");
  }
  // One-breath expert ask — avoid form scaffolding.
  lines.push(`为「${brief}」：${question.prompt}`);
  if (question.why && question.why.length <= 60) lines.push(question.why);
  lines.push("");
  lines.push(formatOptionsFooter(question, spec));
  if (question.freeTextHint) lines.push(question.freeTextHint);
  return lines.join("\n");
}

/** Merge LLM natural reply with a compact options footer (avoids double-templating). */
export function composeNextQuestionBody(
  llmLead: string | null | undefined,
  question: GrillQuestion | null | undefined,
  spec: IntentSpec,
  fallback: string,
): string {
  const lead = String(llmLead || "").trim();
  if (!question) return lead || fallback;
  // Grill-me style: always surface concrete options unless the lead already numbered them.
  const hasNumberedOptions = /(?:^|\n)\s*[1-9]\s*[)）.、.]\s*\S+/m.test(lead);
  if (lead && hasNumberedOptions) {
    // Still ensure a soft escape hatch if model forgot it.
    if (!/按推荐默认|推荐默认/.test(lead)) {
      return `${lead}\n\n拿不准就说 **按推荐默认**。`;
    }
    return lead;
  }
  if (lead) {
    return [lead, "", formatOptionsFooter(question, spec)].join("\n");
  }
  return fallback || formatQuestionMessage(question, spec);
}

export function formatConfirmMessage(spec: IntentSpec): string {
  const countNote =
    spec.targetProjectCount
      ? spec.quotaFlexibility === "fixed"
        ? `固定数量目标：${spec.targetProjectCount} 个项目（待搜索核验）。`
        : `规模约 ${spec.targetProjectCount} 个项目。`
      : "";
  return [
    "策略已同步到右侧预览。",
    `当前目标：${deriveObjective(spec)}${countNote ? `；${countNote}` : ""}`,
    `待确认检索主题词：${confirmedSearchThemes(spec).join("；") || "尚未形成主题词"}`,
    "请先确认以上主题词。确认前可以继续用自然语言修改；准备好后说 **确认** 才会开始 PRIDE 搜索。",
  ].join("\n");
}

export function isDefaultsCommand(input: string): boolean {
  return /按推荐默认|推荐默认|用默认|全部默认|defaults?/i.test(text(input));
}

export type DiscoveryProgressEvent = {
  id: string;
  kind: "think" | "tool" | "action";
  text?: string;
  name?: string;
  status?: "running" | "ok" | "fallback" | "error";
  detail?: string;
};

const TOOL_TYPE_NAMES: Record<string, string> = {
  candidate_search_started: "仓库检索",
  candidate_search_completed: "仓库检索",
  candidate_inspection_started: "候选审查",
  candidate_inspection_completed: "候选审查",
  project_judgments_recorded: "项目评分",
  discovery_quality_audited: "质量审计",
  discovery_quality_repair_started: "自主修复",
  discovery_quality_repair_completed: "自主修复",
  manifest_selected: "最终选择",
  repository_request_started: "仓库 API",
  tool_call: "Agent 工具",
  tool_output: "Agent 工具",
  budget_requested: "预算治理",
  budget_granted: "预算治理",
  budget_denied: "预算治理",
  dynamic_search_stopped: "搜索收敛",
  round_value_evaluated: "本轮评估",
  job_message: "数据发现",
};

/** Infrastructure SDK chatter — never show as user-facing work trajectory. */
function isNoiseLog(log: Record<string, unknown>): boolean {
  const type = text(log.type || "");
  const message = text(log.message || log.summary || "");
  const actor = text(log.actor);
  if (/^sdk[_-]/i.test(type)) return true;
  if (/^sdk\b/i.test(message)) return true;
  if (/OpenAI Agents SDK/i.test(actor) && (!message || /^sdk\b/i.test(message))) return true;
  if (/sdk_(llm|tool|run|agent|item)/i.test(type + " " + message)) return true;
  // Empty / pure boilerplate
  if (!message && !text(log.reasoning_summary)) return true;
  if (/^round value evaluated\.?$/i.test(message)) return true;
  if (/^Discovery job (queued|started)\.?$/i.test(message)) return true;
  if (/^数据发现任务已(排队|开始)/.test(message)) return true;
  // SDK-ish lifecycle with no user value
  if (/^(run started|tool completed|tool started|run item)\.?$/i.test(message)) return true;
  if (type === "run_started" || type === "tool_completed" || type === "tool_started") return true;
  // Generic "Reason: OpenAI Agents SDK …" planning fluff
  if (/^Reason:\s*OpenAI Agents SDK/i.test(message)) return true;
  return false;
}

type InterpretedStep = {
  id: string;
  kind: "think" | "tool" | "action";
  name?: string;
  status?: "running" | "ok" | "fallback" | "error";
  detail?: string;
  text?: string;
  /** Collapse key: later events with same key replace earlier ones. */
  mergeKey?: string;
  /** Higher = more worth keeping when truncating. */
  signal: number;
};

function toolNameFromType(type: string, actor: string, fallback = "数据发现"): string {
  if (TOOL_TYPE_NAMES[type]) return TOOL_TYPE_NAMES[type];
  if (/Budget/i.test(actor)) return "预算治理";
  if (/Search|Repository/i.test(actor)) return "仓库检索";
  if (/Inspector/i.test(actor)) return "候选审查";
  if (actor && actor !== "Discovery Agent" && !/SDK|OpenAI/i.test(actor)) return actor;
  return fallback;
}

function isFailEvent(type: string, level: string, message: string): boolean {
  if (level === "error") return true;
  if (level === "warning" && /fail|error|拒绝|失败/i.test(message)) return true;
  if (/failed|invalid|rejected|error/i.test(type)) return true;
  if (/失败|错误/.test(message) && /fail|error|失败|取消|cancel/i.test(type + message)) return true;
  return false;
}

function compactLogDetail(message: string, max = 140): string {
  const one = message.replace(/\s+/g, " ").trim();
  if (one.length <= max) return one;
  return one.slice(0, max - 1) + "…";
}

/**
 * Turn a single backend log into a Codex-style step (Chinese, actionable).
 * Returns null for noise or unparseable fluff.
 */
function interpretDiscoveryLog(
  log: Record<string, unknown>,
  index: number,
  streaming: boolean,
  isLast: boolean,
): InterpretedStep | null {
  if (isNoiseLog(log)) return null;

  const type = text(log.type || "job_message");
  const level = text(log.level || "info").toLowerCase();
  const message = text(log.message || log.summary || "");
  const reasoning = text(log.reasoning_summary);
  const actor = text(log.actor);
  const seq = text(log.sequence || index);
  const fail = isFailEvent(type, level, message);

  // Structured high-signal event types first
  if (type === "candidate_search_started" || type === "candidate_search_completed") {
    const done = type.endsWith("completed");
    const observed = message.match(/Search observed\s+(\d+)\s+candidate project/i);
    const plans = message.match(/Searching repository with\s+(\d+)\s+query plan/i);
    let detail: string;
    if (observed) {
      detail = `检索到约 ${observed[1]} 个候选项目`;
    } else if (plans) {
      const sample = message.split(":").slice(1).join(":").split(";")[0]?.trim();
      detail = sample
        ? `按 ${plans[1]} 组检索计划搜索（如「${sample.slice(0, 40)}」）`
        : `按 ${plans[1]} 组检索计划搜索`;
    } else {
      detail = compactLogDetail(
        message
          .replace(/^Searching repository with/i, "按检索计划搜索")
          .replace(/^Searching PRIDE projects:\s*/i, "搜索关键词：")
          .replace(/^Search observed\s*/i, "检索到约 ")
          .replace(/Project search returned/i, "检索返回"),
      );
    }
    return {
      id: `search-${seq}`,
      kind: "tool",
      name: "仓库检索",
      status: fail ? "error" : done ? "ok" : streaming && isLast ? "running" : "ok",
      detail,
      mergeKey: "repo-search",
      signal: 8,
    };
  }

  if (type === "candidate_inspection_started" || type === "candidate_inspection_completed") {
    const done = type.endsWith("completed");
    const pxd = message.match(/PXD\d+/i)?.[0];
    const produced = message.match(
      /Inspection produced\s+(\d+)\s+selected project\(s\)\s+and\s+(\d+)\s+selected file\(s\)/i,
    );
    let detail: string;
    if (done && produced) {
      detail = `审查完成：入选约 ${produced[1]} 个项目、${produced[2]} 个文件`;
    } else if (done) {
      detail = compactLogDetail(
        message
          .replace(/^Inspection produced\s*/i, "审查完成：")
          .replace(/selected project\(s\)/gi, "个入选项目")
          .replace(/selected file\(s\)/gi, "个入选文件")
          .replace(/;\s*next action:.*$/i, ""),
      );
    } else if (pxd) {
      detail = `正在审查 ${pxd.toUpperCase()}`;
    } else {
      detail = compactLogDetail(message);
    }
    return {
      id: `inspect-${seq}`,
      kind: "tool",
      name: "候选审查",
      status: fail ? "error" : done ? "ok" : streaming && isLast ? "running" : "ok",
      detail,
      mergeKey: done ? "inspect-batch-done" : "inspect-live",
      signal: done ? 9 : 7,
    };
  }

  if (type === "project_judgments_recorded") {
    return {
      id: `judge-${seq}`,
      kind: "tool",
      name: "项目评分",
      status: fail ? "error" : "ok",
      detail: compactLogDetail(message || "已记录逐项目评分、证据引用和约束判断"),
      mergeKey: "project-judgments",
      signal: 9,
    };
  }

  if (type === "discovery_quality_audited") {
    const ready = /\bready\b/i.test(message) && !/repair_required|blocked/i.test(message);
    return {
      id: `quality-${seq}`,
      kind: ready ? "action" : "think",
      text: ready
        ? `质量门已通过：${compactLogDetail(message)}`
        : `质量审计发现仍需处理：${compactLogDetail(message)}`,
      mergeKey: "quality-audit",
      signal: 10,
    };
  }

  if (type === "discovery_quality_repair_started" || type === "repair_attempt_started") {
    return {
      id: `repair-start-${seq}`,
      kind: "action",
      text: `修复尝试已开始：${compactLogDetail(message)}`,
      mergeKey: "quality-repair",
      signal: 10,
    };
  }

  if (type === "discovery_quality_repair_completed" || type === "repair_attempt_finished") {
    return {
      id: `repair-done-${seq}`,
      kind: "action",
      text: "修复尝试结束，结果待审计",
      mergeKey: "quality-repair",
      signal: 10,
    };
  }

  const repairEventText: Partial<Record<string, string>> = {
    repair_progressed: "修复取得可验证进展，尚待 build-ready 参考审计；L1 可用清单仍可先用",
    repair_no_progress: "本次修复未产生可验证进展",
    repair_succeeded: "收到修复成功事件；最终 build-ready 仅作参考，以可用文件清单为准交付",
    repair_incomplete: "修复尝试结束；build-ready 未签发（参考），不足会写入结果说明",
    repair_blocked: "自动加深未完成；不足点写入结果说明，不阻挡下载可用文件清单",
    build_ready_succeeded: "收到 build-ready 成功事件（L2 参考）；L1 仍以可用文件清单交付",
    blocked_with_progress: "已有进展；build-ready 未签发（参考），可用文件清单仍可送入批量",
  };
  if (repairEventText[type]) {
    const includeMessage = !["repair_succeeded", "build_ready_succeeded"].includes(type);
    return {
      id: `repair-result-${seq}`,
      kind: type === "repair_no_progress" ? "think" : "action",
      text: `${repairEventText[type]}${includeMessage && message ? `：${compactLogDetail(message)}` : ""}`,
      mergeKey: type === "build_ready_succeeded" ? "build-ready" : "quality-repair-result",
      signal: 10,
    };
  }

  if (/^(repair_|build_ready_)/.test(type)) {
    return {
      id: `unknown-repair-${seq}`,
      kind: "think",
      text: `收到未识别的修复事件 ${type}，已忽略其状态声明`,
      mergeKey: "quality-repair-unknown",
      signal: 8,
    };
  }

  if (type === "manifest_selected") {
    return {
      id: `manifest-selected-${seq}`,
      kind: "action",
      text: `最终清单已锁定：${compactLogDetail(message)}`,
      mergeKey: "select-done",
      signal: 10,
    };
  }

  if (/^budget_/.test(type)) {
    return {
      id: `budget-${seq}`,
      kind: "tool",
      name: "预算治理",
      status: /denied|fail/i.test(type) ? "fallback" : "ok",
      detail: compactLogDetail(message || type),
      mergeKey: "budget",
      signal: 6,
    };
  }

  if (type === "dynamic_search_stopped") {
    return {
      id: `stop-${seq}`,
      kind: "action",
      text: compactLogDetail(message || "搜索已按策略收敛停止"),
      signal: 8,
    };
  }

  if (type === "round_value_evaluated" && message && !/^round value evaluated/i.test(message)) {
    return {
      id: `round-${seq}`,
      kind: "think",
      text: compactLogDetail(message),
      signal: 5,
    };
  }

  // Free-form job_message → phase detection (Codex: show what work is happening)
  const m = message;

  // Observe/Reason/Act
  if (/^观察：|^Observe:/i.test(m)) {
    return { id: `obs-${seq}`, kind: "think", text: compactLogDetail(m), signal: 4 };
  }
  if (/^推理|^Reason:/i.test(m)) {
    return { id: `reason-${seq}`, kind: "think", text: compactLogDetail(m), signal: 4 };
  }
  if (/^执行：|^Act:/i.test(m)) {
    return { id: `act-${seq}`, kind: "action", text: compactLogDetail(m), signal: 5 };
  }

  // PRIDE search
  let hit = m.match(/Searching PRIDE projects:\s*(.+?)(?:\s*\(|$)/i);
  if (hit) {
    const q = hit[1].replace(/['"]/g, "").trim();
    return {
      id: `pride-q-${seq}`,
      kind: "tool",
      name: "仓库检索",
      status: streaming && isLast ? "running" : "ok",
      detail: `搜索「${q}」`,
      mergeKey: "repo-search",
      signal: 8,
    };
  }
  hit = m.match(/Project search returned\s+(\d+)\s+hit\(s\)\s+for\s+'([^']+)'/i);
  if (hit) {
    return {
      id: `pride-hit-${seq}`,
      kind: "tool",
      name: "仓库检索",
      status: "ok",
      detail: `「${hit[2]}」命中约 ${hit[1]} 条`,
      mergeKey: "repo-search",
      signal: 7,
    };
  }
  hit = m.match(/Deduped to\s+(\d+)\s+candidate project/i);
  if (hit) {
    return {
      id: `dedupe-${seq}`,
      kind: "tool",
      name: "候选去重",
      status: "ok",
      detail: `去重后约 ${hit[1]} 个项目`,
      mergeKey: "dedupe",
      signal: 8,
    };
  }

  // Project inspection pipeline
  hit = m.match(/Inspecting project\s+(PXD\d+)/i);
  if (hit) {
    return {
      id: `insp-${seq}`,
      kind: "tool",
      name: "候选审查",
      status: streaming && isLast ? "running" : "ok",
      detail: `正在审查 ${hit[1].toUpperCase()}`,
      mergeKey: "inspect-live",
      signal: 7,
    };
  }
  hit = m.match(/Listing files for\s+(PXD\d+)/i);
  if (hit) {
    return {
      id: `list-${seq}`,
      kind: "tool",
      name: "拉取文件清单",
      status: streaming && isLast ? "running" : "ok",
      detail: `${hit[1].toUpperCase()} 拉文件列表`,
      mergeKey: "inspect-live",
      signal: 6,
    };
  }
  hit = m.match(/(PXD\d+):\s*fetched\s+(\d+)\s+file record/i);
  if (hit) {
    return {
      id: `fetch-${seq}`,
      kind: "tool",
      name: "拉取文件清单",
      status: "ok",
      detail: `${hit[1].toUpperCase()} 取回 ${hit[2]} 条文件记录`,
      mergeKey: "inspect-live",
      signal: 6,
    };
  }
  hit = m.match(/(PXD\d+):\s*checking SDRF/i);
  if (hit) {
    return {
      id: `sdrf-${seq}`,
      kind: "tool",
      name: "读元数据",
      status: streaming && isLast ? "running" : "ok",
      detail: `${hit[1].toUpperCase()} 核对 SDRF / 元数据`,
      mergeKey: "inspect-live",
      signal: 6,
    };
  }
  hit = m.match(/(PXD\d+):\s*(?:downloading SDRF|loaded\s+(\d+)\s+SDRF)/i);
  if (hit) {
    const n = hit[2];
    return {
      id: `sdrf2-${seq}`,
      kind: "tool",
      name: "读元数据",
      status: "ok",
      detail: n
        ? `${hit[1].toUpperCase()} 载入 ${n} 行 SDRF`
        : `${hit[1].toUpperCase()} 下载 SDRF`,
      mergeKey: "inspect-live",
      signal: 5,
    };
  }
  hit = m.match(/(PXD\d+):\s*kept\s+(\d+)\s+file candidate/i);
  if (hit) {
    return {
      id: `kept-${seq}`,
      kind: "tool",
      name: "候选审查",
      status: "ok",
      detail: `${hit[1].toUpperCase()} 保留 ${hit[2]} 个文件候选`,
      mergeKey: "inspect-kept",
      signal: 7,
    };
  }
  hit = m.match(/(PXD\d+):\s*no usable\b/i);
  if (hit) {
    return {
      id: `skip-${seq}`,
      kind: "tool",
      name: "候选审查",
      status: "fallback",
      detail: `${hit[1].toUpperCase()} 过滤后无可用谱图/峰列表文件`,
      mergeKey: "inspect-kept",
      signal: 6,
    };
  }
  hit = m.match(/Search observed\s+(\d+)\s+candidate project/i);
  if (hit) {
    return {
      id: `obs-cand-${seq}`,
      kind: "tool",
      name: "仓库检索",
      status: "ok",
      detail: `检索观察到约 ${hit[1]} 个候选项目`,
      mergeKey: "repo-search",
      signal: 8,
    };
  }
  hit = m.match(/Inspecting\s+(\d+)\s+candidate project/i);
  if (hit) {
    return {
      id: `insp-batch-${seq}`,
      kind: "tool",
      name: "候选审查",
      status: streaming && isLast ? "running" : "ok",
      detail: `开始逐项审查约 ${hit[1]} 个候选`,
      mergeKey: "inspect-batch",
      signal: 8,
    };
  }

  // Selection / finish
  if (/Running diversity-aware selection/i.test(m) || /多样性/.test(m)) {
    return {
      id: `div-${seq}`,
      kind: "tool",
      name: "多样性筛选",
      status: streaming && isLast ? "running" : "ok",
      detail: "按策略做多样性入选…",
      mergeKey: "select",
      signal: 9,
    };
  }
  hit = m.match(/Selected\s+(\d+)\s+project\(s\),\s*(\d+)\s+file\(s\)/i);
  if (hit) {
    return {
      id: `sel-${seq}`,
      kind: "tool",
      name: "入选汇总",
      status: "ok",
      detail: `入选约 ${hit[1]} 个项目、${hit[2]} 个文件`,
      mergeKey: "select-done",
      signal: 10,
    };
  }
  if (/Cancel requested/i.test(m)) {
    return {
      id: `canceling-${seq}`,
      kind: "action",
      text: "正在取消…当前网络请求结束后会停下",
      signal: 9,
    };
  }
  if (/Discovery cancelled/i.test(m) || /已取消/.test(m)) {
    return {
      id: `cancelled-${seq}`,
      kind: "action",
      text: "本轮搜索已取消",
      signal: 9,
    };
  }

  // Chinese messages already good
  if (/[\u4e00-\u9fff]/.test(m) && m.length >= 4) {
    const name = toolNameFromType(type, actor);
    let status: InterpretedStep["status"] = fail ? "error" : "ok";
    if (!fail && streaming && isLast && /正在|检索|检查|排队/.test(m)) status = "running";
    return {
      id: `zh-${seq}`,
      kind: "tool",
      name,
      status,
      detail: compactLogDetail(m),
      mergeKey: name,
      signal: 6,
    };
  }

  // Reasoning only
  if (reasoning && reasoning.length > 8) {
    return {
      id: `think-${seq}`,
      kind: "think",
      text: compactLogDetail(reasoning, 180),
      signal: 5,
    };
  }

  // Generic fallback only if message looks operational (not "sdk …")
  if (m.length >= 12 && !/^sdk\b/i.test(m)) {
    const name = toolNameFromType(type, actor);
    if (name === "Agent 编排") return null;
    // Skip English boilerplate that survived noise filter but isn't user-facing
    if (!/[\u4e00-\u9fff]/.test(m) && !/PXD\d+/i.test(m) && m.split(/\s+/).length <= 6) {
      return null;
    }
    return {
      id: `gen-${seq}`,
      kind: "tool",
      name,
      status: fail ? "error" : streaming && isLast ? "running" : "ok",
      detail: compactLogDetail(m),
      signal: 3,
    };
  }

  return null;
}

/**
 * Codex-style progress: meaningful tools/thinks only — no raw SDK stream spam.
 */
export function humanizeJobProgress(job: {
  status?: unknown;
  error?: unknown;
  logs?: Array<Record<string, unknown>>;
  record?: Record<string, unknown>;
}): {
  summary: string;
  humanSteps: string[];
  progressEvents: DiscoveryProgressEvent[];
  rawLogCount: number;
  headline: string;
} {
  const logs = Array.isArray(job.logs) ? job.logs : [];
  const record = (job.record || {}) as Record<string, unknown>;
  const attemptFinishedWithoutAudit = logs.some((log) =>
    ["discovery_quality_repair_completed", "repair_attempt_finished"].includes(text(log.type)),
  );
  const status = honestDiscoveryStatus(
    text(job.status || "queued"),
    record,
    attemptFinishedWithoutAudit,
  );
  const jobError = text(job.error);
  const projectCount = Number(record.project_count || 0);
  const fileCount = Number(record.file_count || 0);
  const recordSummary =
    record.summary && typeof record.summary === "object" && !Array.isArray(record.summary)
      ? (record.summary as Record<string, unknown>)
      : {};
  const selectedProjectCount = Number(
    recordSummary.selected_projects ?? recordSummary.delivery_eligible_projects ?? 0,
  );
  const streaming = status === "queued" || status === "running";

  // Prefer full log when moderate size; otherwise head (incl. search phase) + live tail
  const window =
    logs.length <= 320
      ? logs
      : [...logs.slice(0, 90), ...logs.slice(-200)];
  const merged = new Map<string, InterpretedStep>();
  const ordered: InterpretedStep[] = [];
  let inspectCount = 0;

  window.forEach((raw, index) => {
    const log = (raw || {}) as Record<string, unknown>;
    const isLast = index === window.length - 1;
    const step = interpretDiscoveryLog(log, index, streaming, isLast);
    if (!step) return;

    // Count only real per-project inspections, not list/SDRF micro-steps
    if (/^正在审查 PXD\d+/i.test(step.detail || "")) {
      inspectCount += 1;
      if (step.detail && inspectCount > 1) {
        step.detail = `${step.detail}（本段已审查约 ${inspectCount} 个）`;
      }
    }

    if (step.mergeKey) {
      const prev = merged.get(step.mergeKey);
      if (prev) {
        // Replace in ordered list
        const idx = ordered.findIndex((s) => s.id === prev.id);
        if (idx >= 0) ordered[idx] = step;
        else ordered.push(step);
      } else {
        ordered.push(step);
      }
      merged.set(step.mergeKey, step);
    } else {
      ordered.push(step);
    }
  });

  // Codex: short trajectory — keep high-signal anchors + recent work
  const toEvent = (s: InterpretedStep): DiscoveryProgressEvent => {
    if (s.kind === "think") return { id: s.id, kind: "think" as const, text: s.text };
    if (s.kind === "action") return { id: s.id, kind: "action" as const, text: s.text };
    return {
      id: s.id,
      kind: "tool" as const,
      name: s.name || "数据发现",
      status: s.status || "ok",
      detail: s.detail,
    };
  };

  const pinKeys = new Set([
    "repo-search",
    "dedupe",
    "inspect-batch",
    "select",
    "select-done",
    "inspect-batch-done",
  ]);
  const anchors = ordered.filter((s) => (s.signal >= 8 && !!s.mergeKey) || (s.mergeKey && pinKeys.has(s.mergeKey)));
  const recent = ordered.filter((s) => s.signal >= 5).slice(-8);
  const picked: InterpretedStep[] = [];
  const seenIds = new Set<string>();
  // Phase pins first (search → … → select), then recent work
  for (const s of [...anchors, ...recent]) {
    if (seenIds.has(s.id)) continue;
    seenIds.add(s.id);
    picked.push(s);
  }
  const orderIndex = new Map(ordered.map((s, i) => [s.id, i]));
  picked.sort((a, b) => (orderIndex.get(a.id) ?? 0) - (orderIndex.get(b.id) ?? 0));

  // Cap: never drop pinned phase keys when trimming
  let kept = picked;
  if (kept.length > 12) {
    const pinned = kept.filter((s) => s.mergeKey && pinKeys.has(s.mergeKey));
    const rest = kept.filter((s) => !(s.mergeKey && pinKeys.has(s.mergeKey)));
    const room = Math.max(0, 12 - pinned.length);
    kept = [...pinned, ...rest.slice(-room)];
    kept.sort((a, b) => (orderIndex.get(a.id) ?? 0) - (orderIndex.get(b.id) ?? 0));
  }
  let progressEvents: DiscoveryProgressEvent[] = kept.map(toEvent);

  // Collapse only near-duplicate end summaries (入选汇总 vs 审查完成)
  const collapsed: DiscoveryProgressEvent[] = [];
  const isEndSummary = (ev: DiscoveryProgressEvent) =>
    ev.kind === "tool" &&
    (ev.name === "入选汇总" ||
      (ev.name === "候选审查" && /审查完成/.test(ev.detail || "")));
  for (const ev of progressEvents) {
    const prev = collapsed[collapsed.length - 1];
    if (prev && isEndSummary(prev) && isEndSummary(ev)) {
      // Keep the more specific line
      collapsed[collapsed.length - 1] =
        (ev.detail?.length || 0) >= (prev.detail?.length || 0) ? ev : prev;
      continue;
    }
    collapsed.push(ev);
  }
  progressEvents = collapsed;

  if (status === "queued" && progressEvents.length === 0) {
    progressEvents.push({
      id: "queued",
      kind: "tool",
      name: "任务调度",
      status: "running",
      detail: "已排队，马上按策略开始搜…",
    });
  }

  if (streaming && progressEvents.length === 0) {
    progressEvents.push({
      id: "boot",
      kind: "tool",
      name: "策略检索",
      status: "running",
      detail: "正在按已确认策略检索公开蛋白质组项目…",
    });
  }

  if (projectCount > 0) {
    const metricsId = "metrics-projects";
    if (!progressEvents.some((e) => e.id === metricsId)) {
      progressEvents.push({
        id: metricsId,
        kind: "action",
        text:
          `候选累计约 ${projectCount} 个项目` +
          (fileCount > 0 ? `、${fileCount} 个文件` : ""),
      });
    }
  }

  if (status === "completed") {
    progressEvents.push({
      id: "done",
      kind: "action",
      text: `本轮完成：约 ${projectCount} 个项目、${fileCount} 个文件可供继续处理。`,
    });
  }
  if (status === "failed") {
    progressEvents.push({
      id: "fail",
      kind: "tool",
      name: "数据发现",
      status: "error",
      detail: jobError ? `运行失败：${jobError}` : "运行失败，可展开本消息中的技术轨迹查看失败阶段",
    });
  }
  if (status === "blocked") {
    progressEvents.push({
      id: "quality-blocked",
      kind: "action",
      text: `搜索已结束：约 ${projectCount} 个候选项目。已验证的可交付文件可下载并送入「批量参数规划」；未决文件单列等待复核。`,
    });
  }
  if (status === "cancelled") {
    progressEvents.push({ id: "cancel", kind: "action", text: "已取消本次搜索。" });
  }

  // Final consecutive dedupe
  const deduped: DiscoveryProgressEvent[] = [];
  for (const ev of progressEvents) {
    const prev = deduped[deduped.length - 1];
    if (
      prev &&
      prev.kind === ev.kind &&
      prev.name === ev.name &&
      prev.detail === ev.detail &&
      prev.text === ev.text
    ) {
      continue;
    }
    deduped.push(ev);
  }

  const humanSteps = deduped
    .map((ev) => {
      if (ev.kind === "tool") return `${ev.name}${ev.detail ? " · " + ev.detail : ""}`;
      return ev.text || "";
    })
    .filter(Boolean)
    .slice(-12);

  const lastLive =
    [...deduped].reverse().find((e) => e.kind === "tool" && e.status === "running") ||
    [...deduped].reverse().find((e) => e.kind === "tool" || e.kind === "action");

  const headline =
    lastLive?.kind === "tool"
      ? `${lastLive.name}${lastLive.detail ? " · " + lastLive.detail : ""}`
      : lastLive?.text ||
        (streaming
          ? projectCount > 0
            ? `已看到约 ${projectCount} 个候选，还在往下筛…`
            : "正在按策略检索公开项目…"
          : "数据发现");

  const summary =
    status === "completed"
      ? `搜完了：约 ${projectCount} 个项目` + (fileCount ? `、${fileCount} 个文件` : "") + "。"
      : status === "failed"
        ? jobError
          ? `这轮没跑通：${jobError}`
          : "这轮数据发现失败。"
        : status === "blocked"
          ? `搜索已结束：约 ${projectCount} 个候选` +
            (fileCount ? `、${fileCount} 个文件线索` : "") +
            `。已保留可用文件列表，可继续批量参数规划与标准化构建。`
        : status === "cancelled"
          ? "已按你的要求停掉这轮搜索。"
          : projectCount > 0
            ? `进行中 · 已整理约 ${projectCount} 个候选`
            : `进行中 · ${headline.slice(0, 72)}`;

  return {
    summary,
    humanSteps,
    progressEvents: deduped.slice(-12),
    rawLogCount: logs.length,
    headline,
  };
}


export function formatDoneMessage(
  job: { status?: string; error?: string; record?: Record<string, unknown> | null },
  spec: IntentSpec,
): string {
  const record = (job.record || {}) as Record<string, unknown>;
  const projects = Number(record.project_count || 0);
  const files = Number(record.file_count || 0);
  if (!businessCompletionAllowsSuccess(record)) {
    const usable =
      Number((record as { usable_files?: number }).usable_files) ||
      Number((record.summary as Record<string, unknown> | undefined)?.usable_files) ||
      files;
    return [
      `本轮搜索与审查已结束：约 **${projects}** 个候选项目` +
        (usable ? `，约 **${usable}** 条验证通过的可交付文件` : "") +
        "。",
      "已为你保留可交付文件清单（可下载 batch_inputs_usable / 数据清单 CSV）。项目级同质证据继承会明确标注，未决文件不会混入清单。",
      "建议：下载「可用批量输入」→ 打开「批量处理」粘贴或点「送入批量参数规划」→ 运行参数推断 / 准备输入包。",
    ].join("\n");
  }
  const immuno = isImmunopeptideContext(spec);
  const lines = [
    immuno
      ? `免疫肽这轮搜完了：约 **${projects}** 个项目` + (files ? `、**${files}** 个文件线索` : "") + "。"
      : `这轮数据发现完成：约 **${projects}** 个项目` + (files ? `、**${files}** 个文件` : "") + "。",
    "审查报告、项目评分理由和下载入口已在右侧「发现结果」中就绪。",
    "你可以继续问某个 PXD 是否适合下游任务，或切到单文件、批量和 AI-ready 工作台。",
  ];
  return lines.join("\n");
}


export function detectNextStepCommand(input: string): "single" | "batch" | "ai-ready" | null {
  const value = lower(input);
  if (/单文件|single/.test(value)) return "single";
  if (/批量|batch/.test(value)) return "batch";
  if (/ai-?ready|训练表|建表/.test(value)) return "ai-ready";
  return null;
}
