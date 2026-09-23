from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from agent.discovery.models import DiscoveryEvidence
from agent.inference.mzml_metadata import infer_instrument_family_from_name


UNKNOWN = "unknown"

FRAGMENTATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("EThcD", re.compile(r"\bethcd\b|electron transfer/higher[- ]energy", re.IGNORECASE)),
    ("ETD", re.compile(r"\betd\b|electron transfer dissociation", re.IGNORECASE)),
    ("HCD", re.compile(r"\bhcd\b|higher[- ]energy collisional", re.IGNORECASE)),
    ("CID", re.compile(r"\bcid\b|collision[- ]induced dissociation", re.IGNORECASE)),
)

LC_MINUTES_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:min|mins|minute|minutes)\b", re.IGNORECASE)


@dataclass(frozen=True)
class DiscoveryFeatureSummary:
    instrument_names: list[str] = field(default_factory=list)
    instrument_families: list[str] = field(default_factory=list)
    laboratory_names: list[str] = field(default_factory=list)
    instrument_generation_score: float | None = None
    instrument_generation_label: str | None = None
    fragmentation_methods: list[str] = field(default_factory=list)
    tissue: str | None = None
    enzyme: str | None = None
    isolation_window: float | None = None
    resolution: float | None = None
    collision_energy: float | None = None
    scan_range: str | None = None
    lc_gradient: str | None = None
    lc_gradient_minutes: float | None = None
    evidence: list[DiscoveryEvidence] = field(default_factory=list)


def _entry_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        parts = []
        for key in ("name", "value", "accession", "description"):
            if value.get(key):
                parts.append(str(value[key]))
        return " ".join(parts)
    if isinstance(value, list):
        return " ".join(_entry_text(item) for item in value)
    return str(value)


def _dedupe(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _known(values: Iterable[str]) -> list[str]:
    return [value for value in _dedupe(values) if value.casefold() != UNKNOWN]


def _project_text_fields(project: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        ("title", _entry_text(project.get("title"))),
        ("description", _entry_text(project.get("projectDescription"))),
        ("sampleProcessing", _entry_text(project.get("sampleProcessingProtocol"))),
        ("dataProcessing", _entry_text(project.get("dataProcessingProtocol"))),
        ("keywords", _entry_text(project.get("keywords"))),
        ("experimentTypes", _entry_text(project.get("experimentTypes"))),
        ("instruments", _entry_text(project.get("instruments"))),
        ("laboratory", _entry_text(project.get("laboratory") or project.get("laboratoryName"))),
        ("institution", _entry_text(project.get("institution") or project.get("institutionName"))),
    ]


def _file_text_fields(file_record: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        ("file_name", _entry_text(file_record.get("fileName") or file_record.get("name"))),
        ("file_record", _entry_text(file_record)),
    ]


def _sdrf_text_fields(rows: list[dict[str, Any]], *, limit: int = 500) -> list[tuple[str, str]]:
    fields: list[tuple[str, str]] = []
    for row in rows[:limit]:
        for column, value in row.items():
            text = _entry_text(value).strip()
            if not text:
                continue
            normalized = column.casefold()
            if any(token in normalized for token in ("instrument", "fragment", "dissociation", "collision", "gradient", "chromatography", "acquisition", "laborator", "institution", "institute", "organization", "organisation", "center", "centre", "tissue", "organism part", "enzyme", "protease", "isolation window", "resolution", "scan range")):
                fields.append((f"sdrf:{column}", text))
    return fields


_LABORATORY_KEYS: tuple[str, ...] = (
    "laboratory",
    "laboratoryName",
    "laboratory_name",
    "laboratories",
    "lab",
    "labName",
    "lab_name",
    "institution",
    "institutionName",
    "institution_name",
    "institute",
    "center",
    "centre",
    "organization",
    "organisation",
)


def _laboratory_names_from_project(project: dict[str, Any]) -> tuple[list[str], list[DiscoveryEvidence]]:
    """Extract only explicitly named submitting laboratories/institutions.

    Publication authors and free-text descriptions are intentionally excluded:
    they are not reliable ownership evidence for a file-level split.
    """

    values: list[str] = []
    evidence: list[DiscoveryEvidence] = []
    for key in _LABORATORY_KEYS:
        value = project.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            entries = value
        else:
            entries = [value]
        for entry in entries:
            text = _entry_text(entry).strip()
            if not text:
                continue
            values.append(text)
            evidence.append(
                DiscoveryEvidence(field=key, source="laboratory", text=text, weight=8)
            )
    # PRIDE commonly stores the submitting institution on submitter / lab-PI
    # objects rather than on a top-level laboratory field.  Affiliation is an
    # explicit repository field; person names and free-text author lists are
    # intentionally not promoted to laboratories.
    for parent_key in ("labPIs", "submitters"):
        entries = project.get(parent_key) or []
        if not isinstance(entries, list):
            entries = [entries]
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            affiliation = _entry_text(entry.get("affiliation")).strip()
            if not affiliation:
                continue
            values.append(affiliation)
            evidence.append(
                DiscoveryEvidence(
                    field=f"{parent_key}.affiliation",
                    source="laboratory",
                    text=affiliation,
                    weight=8,
                )
            )
    return _dedupe(values), evidence


def _laboratory_names_from_fields(fields: list[tuple[str, str]]) -> tuple[list[str], list[DiscoveryEvidence]]:
    values: list[str] = []
    evidence: list[DiscoveryEvidence] = []
    tokens = ("laborator", "institution", "institute", "organization", "organisation", "center", "centre", "lab")
    for field_name, text in fields:
        if not any(token in field_name.casefold() for token in tokens):
            continue
        for value in re.split(r"[;|,]\s*", text):
            cleaned = value.strip()
            if not cleaned:
                continue
            values.append(cleaned)
            evidence.append(
                DiscoveryEvidence(field=field_name, source="laboratory", text=cleaned, weight=8)
            )
    return _dedupe(values), evidence


def _instrument_names_from_project(project: dict[str, Any]) -> tuple[list[str], list[DiscoveryEvidence]]:
    values = []
    for entry in project.get("instruments", []) or []:
        text = _entry_text(entry).strip()
        if text:
            values.append(text)
    names = _dedupe(values)
    evidence = [
        DiscoveryEvidence(field="instruments", source="instrument", text=name, weight=8)
        for name in names
    ]
    return names, evidence


# Scores are deliberately coarse generations, not a claim that one platform is
# universally better.  They make a user preference such as "仪器尽可能新"
# observable and auditable.  Unknown instruments remain unknown rather than
# being ranked from project publication date.
INSTRUMENT_GENERATIONS: tuple[tuple[re.Pattern[str], float, str], ...] = (
    (re.compile(r"orbitrap\s+astral|astral", re.I), 1.00, "current"),
    (re.compile(r"tims?tof\s+(?:ultra|scp)|astral\s+zoom", re.I), 0.95, "current"),
    (re.compile(r"orbitrap\s+(?:eclipse|exploris\s*480|exploris\s*240)", re.I), 0.86, "recent"),
    (re.compile(r"tims?tof\s+(?:pro\s*2|ht)", re.I), 0.84, "recent"),
    (re.compile(r"orbitrap\s+fusion\s+lumos|q\s*exactive\s*hf[- ]?x", re.I), 0.70, "modern"),
    (re.compile(r"tims?tof\s+pro\b", re.I), 0.68, "modern"),
    (re.compile(r"q\s*exactive\s*hf\b|orbitrap\s+fusion\b", re.I), 0.55, "established"),
    (re.compile(r"q\s*exactive\b|orbitrap\s+elite", re.I), 0.36, "legacy"),
    (re.compile(r"orbitrap\s+(?:velos|xl)|ltq\s+orbitrap", re.I), 0.18, "legacy"),
)


def instrument_generation(names: Iterable[str]) -> tuple[float | None, str | None]:
    matches: list[tuple[float, str]] = []
    for name in names:
        for pattern, score, label in INSTRUMENT_GENERATIONS:
            if pattern.search(str(name or "")):
                matches.append((score, label))
                break
    if not matches:
        return None, None
    score, label = max(matches, key=lambda item: item[0])
    return score, label


def _instrument_names_from_protocols(fields: list[tuple[str, str]]) -> tuple[list[str], list[DiscoveryEvidence]]:
    """Recover specific models mentioned in protocols when PRIDE CV is generic."""

    names: list[str] = []
    evidence: list[DiscoveryEvidence] = []
    for field_name, text in fields:
        if not text:
            continue
        for pattern, _score, _label in INSTRUMENT_GENERATIONS:
            for match in pattern.finditer(text):
                name = " ".join(match.group(0).split()).strip()
                if not name:
                    continue
                names.append(name)
                evidence.append(
                    DiscoveryEvidence(
                        field=field_name,
                        source="instrument_protocol",
                        text=name,
                        weight=9,
                    )
                )
    return _dedupe(names), evidence


def _instrument_names_from_fields(fields: list[tuple[str, str]]) -> tuple[list[str], list[DiscoveryEvidence]]:
    values: list[str] = []
    evidence: list[DiscoveryEvidence] = []
    for field_name, text in fields:
        if "instrument" not in field_name.casefold():
            continue
        for value in re.split(r"[;|,]\s*", text):
            cleaned = value.strip()
            if not cleaned:
                continue
            values.append(cleaned)
            evidence.append(DiscoveryEvidence(field=field_name, source="instrument", text=cleaned, weight=10))
    return _dedupe(values), evidence


def _instrument_families(names: list[str]) -> list[str]:
    return _known(infer_instrument_family_from_name(name) for name in names)


def _fragmentation_from_fields(fields: list[tuple[str, str]]) -> tuple[list[str], list[DiscoveryEvidence]]:
    methods: list[str] = []
    evidence: list[DiscoveryEvidence] = []
    for field_name, text in fields:
        if not text:
            continue
        for method, pattern in FRAGMENTATION_PATTERNS:
            if not pattern.search(text):
                continue
            methods.append(method)
            evidence.append(DiscoveryEvidence(field=field_name, source="fragmentation", text=method, weight=7))
    return _dedupe(methods), evidence


def _lc_gradient_from_fields(fields: list[tuple[str, str]]) -> tuple[str | None, float | None, list[DiscoveryEvidence]]:
    best_text: str | None = None
    best_minutes: float | None = None
    evidence: list[DiscoveryEvidence] = []
    for field_name, text in fields:
        if not text:
            continue
        field_has_gradient = any(token in field_name.casefold() for token in ("gradient", "chromatography"))
        text_has_gradient = bool(re.search(r"\bgradient\b|chromatograph|nano[- ]?lc|uplc|hplc", text, re.IGNORECASE))
        if not field_has_gradient and not text_has_gradient:
            continue
        minutes_match = LC_MINUTES_RE.search(text)
        minutes = float(minutes_match.group(1)) if minutes_match else None
        excerpt = re.sub(r"\s+", " ", text).strip()
        if len(excerpt) > 180:
            excerpt = excerpt[:177].rstrip() + "..."
        best_text = excerpt
        best_minutes = minutes
        evidence.append(DiscoveryEvidence(field=field_name, source="lc_gradient", text=excerpt, weight=5))
        break
    return best_text, best_minutes, evidence


def _named_text_from_fields(
    fields: list[tuple[str, str]],
    *name_tokens: str,
) -> tuple[str | None, list[DiscoveryEvidence]]:
    for field_name, text in fields:
        if text and any(token in field_name.casefold() for token in name_tokens):
            value = re.sub(r"\s+", " ", text).strip()
            return value, [
                DiscoveryEvidence(
                    field=field_name,
                    source="method_metadata",
                    text=value[:240],
                    weight=7,
                )
            ]
    return None, []


def _named_number_from_fields(
    fields: list[tuple[str, str]],
    *name_tokens: str,
) -> tuple[float | None, list[DiscoveryEvidence]]:
    text, evidence = _named_text_from_fields(fields, *name_tokens)
    if text is None:
        return None, []
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    return (float(match.group(0)), evidence) if match else (None, evidence)


def merge_feature_summaries(*summaries: DiscoveryFeatureSummary) -> DiscoveryFeatureSummary:
    names = _dedupe(name for summary in summaries for name in summary.instrument_names)
    families = _known(family for summary in summaries for family in summary.instrument_families)
    laboratories = _dedupe(name for summary in summaries for name in summary.laboratory_names)
    fragmentations = _dedupe(method for summary in summaries for method in summary.fragmentation_methods)
    lc_summary = next((summary for summary in summaries if summary.lc_gradient), None)
    tissue_summary = next((summary for summary in summaries if summary.tissue), None)
    enzyme_summary = next((summary for summary in summaries if summary.enzyme), None)
    isolation_summary = next((summary for summary in summaries if summary.isolation_window is not None), None)
    resolution_summary = next((summary for summary in summaries if summary.resolution is not None), None)
    collision_summary = next((summary for summary in summaries if summary.collision_energy is not None), None)
    scan_summary = next((summary for summary in summaries if summary.scan_range), None)
    evidence = [item for summary in summaries for item in summary.evidence]
    generation_candidates = [
        (summary.instrument_generation_score, summary.instrument_generation_label)
        for summary in summaries
        if summary.instrument_generation_score is not None
    ]
    generation_score, generation_label = (
        max(generation_candidates, key=lambda item: float(item[0] or 0.0))
        if generation_candidates
        else (None, None)
    )
    return DiscoveryFeatureSummary(
        instrument_names=names,
        instrument_families=families,
        laboratory_names=laboratories,
        instrument_generation_score=generation_score,
        instrument_generation_label=generation_label,
        fragmentation_methods=fragmentations,
        tissue=tissue_summary.tissue if tissue_summary else None,
        enzyme=enzyme_summary.enzyme if enzyme_summary else None,
        isolation_window=isolation_summary.isolation_window if isolation_summary else None,
        resolution=resolution_summary.resolution if resolution_summary else None,
        collision_energy=collision_summary.collision_energy if collision_summary else None,
        scan_range=scan_summary.scan_range if scan_summary else None,
        lc_gradient=lc_summary.lc_gradient if lc_summary else None,
        lc_gradient_minutes=lc_summary.lc_gradient_minutes if lc_summary else None,
        evidence=evidence,
    )


def extract_project_features(project: dict[str, Any], sdrf_rows: list[dict[str, Any]] | None = None) -> DiscoveryFeatureSummary:
    fields = _project_text_fields(project)
    if sdrf_rows:
        fields.extend(_sdrf_text_fields(sdrf_rows))

    project_names, project_name_evidence = _instrument_names_from_project(project)
    project_laboratories, project_laboratory_evidence = _laboratory_names_from_project(project)
    sdrf_names, sdrf_name_evidence = _instrument_names_from_fields(fields)
    sdrf_laboratories, sdrf_laboratory_evidence = _laboratory_names_from_fields(fields)
    protocol_names, protocol_name_evidence = _instrument_names_from_protocols(fields)
    names = _dedupe([*project_names, *sdrf_names, *protocol_names])
    families = _instrument_families(names)
    generation_score, generation_label = instrument_generation(names)
    fragmentation, fragmentation_evidence = _fragmentation_from_fields(fields)
    lc_gradient, lc_minutes, lc_evidence = _lc_gradient_from_fields(fields)
    tissue, tissue_evidence = _named_text_from_fields(fields, "tissue", "organism part")
    enzyme, enzyme_evidence = _named_text_from_fields(fields, "enzyme", "protease")
    isolation_window, isolation_evidence = _named_number_from_fields(fields, "isolation window")
    resolution, resolution_evidence = _named_number_from_fields(fields, "resolution")
    collision_energy, collision_evidence = _named_number_from_fields(fields, "collision energy")
    scan_range, scan_evidence = _named_text_from_fields(fields, "scan range")

    return DiscoveryFeatureSummary(
        instrument_names=names,
        instrument_families=families,
        laboratory_names=_dedupe([*project_laboratories, *sdrf_laboratories]),
        instrument_generation_score=generation_score,
        instrument_generation_label=generation_label,
        fragmentation_methods=fragmentation,
        tissue=tissue,
        enzyme=enzyme,
        isolation_window=isolation_window,
        resolution=resolution,
        collision_energy=collision_energy,
        scan_range=scan_range,
        lc_gradient=lc_gradient,
        lc_gradient_minutes=lc_minutes,
        evidence=(
            project_name_evidence
            + sdrf_name_evidence
            + protocol_name_evidence
            + project_laboratory_evidence
            + sdrf_laboratory_evidence
            + fragmentation_evidence
            + lc_evidence
            + tissue_evidence
            + enzyme_evidence
            + isolation_evidence
            + resolution_evidence
            + collision_evidence
            + scan_evidence
        ),
    )


def extract_file_features(
    file_record: dict[str, Any],
    project_features: DiscoveryFeatureSummary,
    matched_sdrf_rows: list[dict[str, Any]] | None = None,
    *,
    inherit_homogeneous_project_instrument: bool = False,
) -> DiscoveryFeatureSummary:
    fields = _file_text_fields(file_record)
    if matched_sdrf_rows:
        fields.extend(_sdrf_text_fields(matched_sdrf_rows))

    # Instruments are file-owned when SDRF/file fields identify them. If a
    # project has exactly one instrument family, inheriting its model is safe:
    # every raw file belongs to that homogeneous acquisition context. A
    # multi-family project remains unknown until file-level evidence resolves it.
    sdrf_names, sdrf_name_evidence = _instrument_names_from_fields(fields)
    sdrf_laboratories, sdrf_laboratory_evidence = _laboratory_names_from_fields(fields)
    names = _dedupe(sdrf_names)
    families = _instrument_families(names) if names else []
    inherited_instrument_evidence: list[DiscoveryEvidence] = []
    if (
        inherit_homogeneous_project_instrument
        and not names
        and len(project_features.instrument_families) == 1
        and project_features.instrument_names
    ):
        names = list(project_features.instrument_names)
        families = list(project_features.instrument_families)
        inherited_instrument_evidence = [
            DiscoveryEvidence(
                field="instruments",
                source="project_instrument_inherited",
                text=name,
                weight=6,
            )
            for name in names
        ]
    generation_score, generation_label = instrument_generation(names)
    fragmentation, fragmentation_evidence = _fragmentation_from_fields(fields)
    if not fragmentation:
        fragmentation = list(project_features.fragmentation_methods)
    lc_gradient, lc_minutes, lc_evidence = _lc_gradient_from_fields(fields)
    if lc_gradient is None:
        lc_gradient = project_features.lc_gradient
        lc_minutes = project_features.lc_gradient_minutes
    tissue, tissue_evidence = _named_text_from_fields(fields, "tissue", "organism part")
    enzyme, enzyme_evidence = _named_text_from_fields(fields, "enzyme", "protease")
    isolation_window, isolation_evidence = _named_number_from_fields(fields, "isolation window")
    resolution, resolution_evidence = _named_number_from_fields(fields, "resolution")
    collision_energy, collision_evidence = _named_number_from_fields(fields, "collision energy")
    scan_range, scan_evidence = _named_text_from_fields(fields, "scan range")

    return DiscoveryFeatureSummary(
        instrument_names=names,
        instrument_families=families,
        laboratory_names=_dedupe([*sdrf_laboratories, *project_features.laboratory_names]),
        instrument_generation_score=generation_score,
        instrument_generation_label=generation_label,
        fragmentation_methods=fragmentation,
        tissue=tissue or project_features.tissue,
        enzyme=enzyme or project_features.enzyme,
        isolation_window=isolation_window if isolation_window is not None else project_features.isolation_window,
        resolution=resolution if resolution is not None else project_features.resolution,
        collision_energy=collision_energy if collision_energy is not None else project_features.collision_energy,
        scan_range=scan_range or project_features.scan_range,
        lc_gradient=lc_gradient,
        lc_gradient_minutes=lc_minutes,
        evidence=(
            sdrf_name_evidence
            + sdrf_laboratory_evidence
            + inherited_instrument_evidence
            + fragmentation_evidence
            + lc_evidence
            + tissue_evidence
            + enzyme_evidence
            + isolation_evidence
            + resolution_evidence
            + collision_evidence
            + scan_evidence
        ),
    )


def lc_gradient_bucket(minutes: float | None) -> str:
    if minutes is None:
        return UNKNOWN
    if minutes < 30:
        return "<30min"
    if minutes < 60:
        return "30-60min"
    if minutes < 120:
        return "60-120min"
    return ">=120min"


def diversity_tags(
    *,
    species: list[str],
    instrument_families: list[str],
    fragmentation_methods: list[str],
    lc_gradient_minutes: float | None,
    modification_scope: str | None = None,
    immunopeptide_scope: str | None = None,
    labeling_strategy: str | None = None,
) -> list[str]:
    tags: list[str] = []
    for item in species or [UNKNOWN]:
        tags.append(f"species:{item or UNKNOWN}")
    tags.append(f"ptm:{modification_scope or UNKNOWN}")
    if immunopeptide_scope:
        tags.append(f"domain:{immunopeptide_scope}")
    tags.append(f"labeling:{labeling_strategy or UNKNOWN}")
    for item in instrument_families or [UNKNOWN]:
        tags.append(f"instrument:{item or UNKNOWN}")
    for item in fragmentation_methods or [UNKNOWN]:
        tags.append(f"fragmentation:{item or UNKNOWN}")
    tags.append(f"lc:{lc_gradient_bucket(lc_gradient_minutes)}")
    return tags
