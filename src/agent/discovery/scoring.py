from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from agent.discovery.features import DiscoveryFeatureSummary, diversity_tags
from agent.discovery.models import (
    DatasetRequest,
    DiscoveredFile,
    DiscoveredProject,
    DiscoveryEvidence,
    EvidenceLevel,
    FileRole,
    SdrfMatchStatus,
)
from agent.discovery.ontology import (
    interpret_immunopeptide_metadata,
    labeling_aliases,
    labeling_from_text,
    interpret_ptm_metadata,
    is_immunopeptidomics_goal,
    normalize_labeling_strategy,
    normalize_ptm_type,
    normalize_species_values,
    species_aliases,
    species_from_text,
)
from agent.discovery.validity import (
    assess_file_validity,
    assess_project_validity,
    normalize_acquisition_mode,
)
from agent.pride.client import PrideClient


DDA_TERMS = ("dda", "data dependent", "data-dependent", "shotgun")
DIA_TERMS = (
    "dia",
    "swath",
    "data independent",
    "data-independent",
    "diapasef",
    "dia-pasef",
)
TARGETED_TERMS = (
    "prm",
    "srm",
    "mrm",
    "targeted acquisition",
    "targeted proteomics",
    "parallel reaction monitoring",
    "selected reaction monitoring",
    "multiple reaction monitoring",
)
ACQUISITION_TERMS = {
    "dda": DDA_TERMS,
    "dia": DIA_TERMS,
    "targeted": TARGETED_TERMS,
}
RAW_FILE_SUFFIXES = (".raw", ".mzml", ".mzxml", ".mgf", ".wiff", ".d")
RAW_DIRECTORY_SUFFIXES = (".d.zip", ".d.tar", ".d.tar.gz")
RESULT_FILE_SUFFIXES = (
    ".tsv",
    ".txt",
    ".csv",
    ".xlsx",
    ".xls",
    ".pdf",
    ".mzid",
    ".html",
    ".json",
)
RESULT_NAME_TOKENS = (
    "search result",
    "search_result",
    "protein report",
    "peptide report",
    "psm",
    "summary",
    "result",
)
DERIVED_RAW_NAME_TOKENS = (
    ".mzid",
    ".mzidentml",
    ".pep.xml",
    ".pepxml",
    "protein_groups",
    "protein-group",
    "peptide_report",
    "protein_report",
    "psm_report",
    "search_result",
    "search result",
)

@dataclass(frozen=True)
class ProjectScore:
    project_score: float
    confidence: float
    needs_review: bool
    excluded: bool
    species: list[str]
    canonical_species: list[str]
    organism_taxon_id: list[str]
    acquisition_mode: str | None
    ptm_type: str | None
    ptm_subtype: str | None
    ptm_evidence_terms: list[str]
    ptm_enrichment_methods: list[str]
    semantic_metadata_confidence: float
    semantic_interpretation_trace: list[str]
    modification_scope: str | None
    immunopeptide_scope: str | None
    hla_class: list[str]
    hla_alleles: list[str]
    immunopeptide_evidence_terms: list[str]
    immunopeptide_enrichment_methods: list[str]
    immunopeptide_metadata_confidence: float
    labeling_strategy: str | None
    evidence: list[DiscoveryEvidence]
    exclusion_reason: str | None = None


@dataclass(frozen=True)
class FileRoleDecision:
    role: FileRole
    file_type: str | None
    reasons: list[str]


def _project_accession(project: dict[str, Any]) -> str:
    return str(project.get("accession") or project.get("projectAccession") or "")


def _project_title(project: dict[str, Any]) -> str | None:
    value = project.get("title")
    return str(value) if value else None


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


def _dedupe_evidence(values: list[DiscoveryEvidence]) -> list[DiscoveryEvidence]:
    result: list[DiscoveryEvidence] = []
    seen: set[tuple[str, str, str]] = set()
    for item in values:
        key = (item.field, item.source, item.text)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _project_fields(project: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        ("title", _entry_text(project.get("title"))),
        ("description", _entry_text(project.get("projectDescription"))),
        ("sampleProcessing", _entry_text(project.get("sampleProcessingProtocol"))),
        ("dataProcessing", _entry_text(project.get("dataProcessingProtocol"))),
        ("keywords", _entry_text(project.get("keywords"))),
        ("experimentTypes", _entry_text(project.get("experimentTypes"))),
        ("organisms", _entry_text(project.get("organisms"))),
    ]


def _combined_project_text(project: dict[str, Any]) -> str:
    return " ".join(text for _, text in _project_fields(project) if text)


def _term_pattern(term: str) -> re.Pattern[str]:
    if re.fullmatch(r"[a-z0-9]+", term.casefold()):
        return re.compile(rf"(?<![a-z0-9]){re.escape(term.casefold())}(?![a-z0-9])")
    return re.compile(re.escape(term.casefold()))


def _matches(text: str, term: str) -> bool:
    return bool(_term_pattern(term).search(text.casefold()))


def _has_source(evidence: list[DiscoveryEvidence], source: str) -> bool:
    return any(item.source == source for item in evidence)


def _keyword_evidence(fields: list[tuple[str, str]], terms: tuple[str, ...], source: str, weight: float) -> list[DiscoveryEvidence]:
    evidence: list[DiscoveryEvidence] = []
    seen: set[tuple[str, str]] = set()
    for field, text in fields:
        if not text:
            continue
        for term in terms:
            if not _matches(text, term):
                continue
            key = (field, term.casefold())
            if key in seen:
                continue
            seen.add(key)
            evidence.append(DiscoveryEvidence(field=field, source=source, text=term, weight=weight))
    return evidence


def _acquisition_evidence_by_mode(
    fields: list[tuple[str, str]],
    *,
    weight: float,
) -> dict[str, list[DiscoveryEvidence]]:
    return {
        mode: _keyword_evidence(fields, terms, "acquisition", weight)
        for mode, terms in ACQUISITION_TERMS.items()
    }


def _select_observed_acquisition_mode(
    evidence_by_mode: dict[str, list[DiscoveryEvidence]],
    requested_mode: str,
) -> str | None:
    observed_modes = {mode for mode, evidence in evidence_by_mode.items() if evidence}
    if len(observed_modes) == 1:
        return next(iter(observed_modes))
    if requested_mode in observed_modes:
        return requested_mode
    return None


def _requested_species_aliases(request: DatasetRequest) -> dict[str, tuple[str, ...]]:
    aliases: dict[str, tuple[str, ...]] = {}
    for species in request.species:
        aliases[species] = species_aliases(species)
    return aliases


def _structured_project_species(project: dict[str, Any]) -> tuple[list[str], list[str], list[DiscoveryEvidence]]:
    """Prefer official structured organisms/TaxIDs over free-text abstract hits."""
    structured_values: list[str] = []
    evidence: list[DiscoveryEvidence] = []
    for key in (
        "organisms",
        "organism",
        "species",
        "speciesList",
        "taxonomies",
        "taxonomy",
        "organismTaxonomies",
    ):
        raw = project.get(key)
        if raw is None:
            continue
        items = raw if isinstance(raw, list) else [raw]
        for item in items:
            if isinstance(item, dict):
                for field in (
                    "name",
                    "organism",
                    "scientificName",
                    "commonName",
                    "taxon",
                    "taxId",
                    "taxonomyId",
                    "accession",
                    "id",
                ):
                    value = item.get(field)
                    if value is not None and str(value).strip():
                        structured_values.append(str(value).strip())
                        evidence.append(
                            DiscoveryEvidence(
                                field=f"structured:{key}.{field}",
                                source="species",
                                text=str(value).strip(),
                                weight=28,
                            )
                        )
            elif str(item).strip():
                structured_values.append(str(item).strip())
                evidence.append(
                    DiscoveryEvidence(
                        field=f"structured:{key}",
                        source="species",
                        text=str(item).strip(),
                        weight=28,
                    )
                )
    canonical, taxa = normalize_species_values(structured_values)
    if not canonical and taxa:
        # TaxIDs alone may not reverse-map if only numeric strings were provided.
        for taxon in taxa:
            if str(taxon) == "9606":
                canonical.append("human")
            elif str(taxon) == "10090":
                canonical.append("mouse")
    return sorted(set(canonical)), sorted(set(taxa)), evidence


def _matched_species(project: dict[str, Any], request: DatasetRequest) -> tuple[list[str], list[DiscoveryEvidence]]:
    fields = _project_fields(project)
    matches: list[str] = []
    evidence: list[DiscoveryEvidence] = []

    structured_species, structured_taxa, structured_evidence = _structured_project_species(project)
    if structured_species:
        matches.extend(structured_species)
        evidence.extend(structured_evidence)
    else:
        # Fall back to free-text only when structured organisms are absent.
        detected_species, _detected_taxa = species_from_text(_combined_project_text(project))
        for species in detected_species:
            aliases = species_aliases(species)
            alias = next((item for item in aliases if _matches(_combined_project_text(project), item)), species)
            matches.append(species)
            evidence.append(DiscoveryEvidence(field="semantic_metadata", source="species", text=alias, weight=8))

    for requested, aliases in _requested_species_aliases(request).items():
        for field, text in fields:
            if not text:
                continue
            alias = next((alias for alias in aliases if _matches(text, alias)), None)
            if alias is None:
                continue
            matches.append(requested)
            evidence.append(DiscoveryEvidence(field=field, source="species", text=alias, weight=20))
            break

    # Keep structured TaxIDs visible even when free text adds noise.
    if structured_taxa:
        for taxon in structured_taxa:
            evidence.append(
                DiscoveryEvidence(field="structured:taxon", source="species", text=str(taxon), weight=30)
            )
    return sorted(set(matches)), evidence


def score_project(project: dict[str, Any], request: DatasetRequest) -> ProjectScore:
    fields = _project_fields(project)
    combined_text = _combined_project_text(project)
    requested_ptm = normalize_ptm_type(request.ptm_type)
    requested_acquisition = normalize_acquisition_mode(request.acquisition_mode)
    requested_labeling = normalize_labeling_strategy(request.labeling_strategy)
    requested_immunopeptidomics = is_immunopeptidomics_goal(request.goal)
    general_goal = str(request.goal or "").casefold() == "general"
    ptm_semantic = interpret_ptm_metadata(combined_text, requested_ptm)
    immuno_semantic = interpret_immunopeptide_metadata(combined_text)
    general_evidence = _keyword_evidence(fields, tuple(request.query_terms), "general_query", 12) if general_goal else []
    ptm_evidence = [
        DiscoveryEvidence(field="semantic_metadata", source="ptm", text=term, weight=16)
        for term in ptm_semantic.evidence_terms
    ]
    immuno_evidence = [
        DiscoveryEvidence(field="semantic_metadata", source="immunopeptidomics", text=term, weight=18)
        for term in immuno_semantic.evidence_terms
    ]
    acquisition_evidence_by_mode = _acquisition_evidence_by_mode(fields, weight=8)
    observed_acquisition_modes = {
        mode for mode, evidence in acquisition_evidence_by_mode.items() if evidence
    }
    acquisition_mode = _select_observed_acquisition_mode(
        acquisition_evidence_by_mode,
        requested_acquisition,
    )
    acquisition_evidence = [
        item
        for mode_evidence in acquisition_evidence_by_mode.values()
        for item in mode_evidence
    ]
    mixed_acquisition_evidence = [
        DiscoveryEvidence(
            field=item.field,
            source="mixed_acquisition",
            text=item.text,
            weight=-20,
        )
        for item in acquisition_evidence
    ] if len(observed_acquisition_modes) > 1 else []
    detected_labeling = labeling_from_text(combined_text)
    labeling_evidence = (
        _keyword_evidence(fields, labeling_aliases(detected_labeling), "labeling", 8)
        if detected_labeling
        else []
    )
    labeling_matches_request = (
        requested_labeling != "unknown"
        and detected_labeling is not None
        and normalize_labeling_strategy(detected_labeling) == requested_labeling
        and bool(labeling_evidence)
    )
    species, species_evidence = _matched_species(project, request)
    canonical_species, taxon_ids = normalize_species_values(species)

    score = 0.0
    if general_evidence:
        score += min(45.0, 18.0 + 5.0 * len({item.text.casefold() for item in general_evidence}))
    if immuno_evidence:
        score += min(52.0, 24.0 + 6.0 * len({item.text.casefold() for item in immuno_evidence}))
    if ptm_evidence:
        score += min(50.0, 20.0 + 6.0 * len({item.text.casefold() for item in ptm_evidence}))
    if species:
        score += 20.0
    matching_acquisition_evidence = acquisition_evidence_by_mode.get(
        requested_acquisition,
        [],
    )
    if matching_acquisition_evidence:
        score += 10.0
    elif requested_acquisition == "unknown" and acquisition_evidence:
        score += 5.0
    if labeling_matches_request:
        score += 8.0

    populated_fields = sum(1 for _, text in fields if text.strip())
    score += min(10.0, populated_fields * 1.5)

    hard_acquisition = (
        request.is_hard_constraint("acquisition_mode")
        and requested_acquisition != "unknown"
    )
    excluded = bool(
        hard_acquisition
        and observed_acquisition_modes
        and requested_acquisition not in observed_acquisition_modes
    )
    exclusion_reason = (
        (
            f"Observed acquisition modes {sorted(observed_acquisition_modes)} "
            f"conflict with requested {requested_acquisition}."
        )
        if excluded
        else None
    )
    labeling_missing = (
        request.is_hard_constraint("labeling_strategy")
        and requested_labeling != "unknown"
        and not labeling_matches_request
    )
    species_required = request.species_policy == "include_only"
    # Prefer structured organisms/TaxIDs over free-text abstract mentions.
    structured_species, structured_taxa, _structured_species_evidence = _structured_project_species(project)
    effective_species = structured_species or canonical_species
    requested_canonical, requested_taxa = normalize_species_values(request.species)
    needs_species_review = False
    if species_required and requested_canonical:
        allowed = set(requested_canonical)
        observed = set(effective_species)
        if observed and observed.isdisjoint(allowed):
            excluded = True
            exclusion_reason = (
                f"Structured species {sorted(observed)} do not include required "
                f"{sorted(allowed)}."
            )
        elif structured_taxa and requested_taxa and set(structured_taxa).isdisjoint(set(requested_taxa)):
            excluded = True
            exclusion_reason = (
                f"Structured TaxIDs {sorted(structured_taxa)} do not include required "
                f"{sorted(requested_taxa)}."
            )
        elif not observed and not structured_taxa:
            # No usable species evidence under a strict filter.
            needs_species_review = True
    elif species_required and not species:
        needs_species_review = True
    ptm_required = not general_goal and not requested_immunopeptidomics and requested_ptm != "unknown_ptm"
    immuno_missing = requested_immunopeptidomics and not immuno_evidence
    needs_review = (
        (ptm_required and not ptm_evidence)
        or immuno_missing
        or needs_species_review
        or (
            hard_acquisition
            and requested_acquisition not in observed_acquisition_modes
        )
        or labeling_missing
    )
    confidence = max(0.0, min(1.0, score / 90.0))
    evidence = (
        general_evidence
        + immuno_evidence
        + ptm_evidence
        + species_evidence
        + acquisition_evidence
        + labeling_evidence
        + mixed_acquisition_evidence
    )
    return ProjectScore(
        project_score=round(score, 2),
        confidence=round(confidence, 3),
        needs_review=needs_review or excluded,
        excluded=excluded,
        species=effective_species or species,
        canonical_species=effective_species or canonical_species,
        organism_taxon_id=structured_taxa or taxon_ids,
        acquisition_mode=acquisition_mode,
        ptm_type=ptm_semantic.canonical if ptm_evidence else None,
        ptm_subtype=";".join(ptm_semantic.subtypes) or None,
        ptm_evidence_terms=list(ptm_semantic.evidence_terms),
        ptm_enrichment_methods=list(ptm_semantic.enrichment_methods),
        semantic_metadata_confidence=ptm_semantic.confidence,
        semantic_interpretation_trace=list(ptm_semantic.trace),
        modification_scope=ptm_semantic.canonical if ptm_evidence else None,
        immunopeptide_scope=immuno_semantic.scope if immuno_evidence else None,
        hla_class=list(immuno_semantic.hla_classes),
        hla_alleles=list(immuno_semantic.hla_alleles),
        immunopeptide_evidence_terms=list(immuno_semantic.evidence_terms),
        immunopeptide_enrichment_methods=list(immuno_semantic.enrichment_methods),
        immunopeptide_metadata_confidence=immuno_semantic.confidence,
        labeling_strategy=detected_labeling,
        evidence=evidence,
        exclusion_reason=exclusion_reason,
    )


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def _project_completeness(score: ProjectScore, features: DiscoveryFeatureSummary) -> float:
    checks = [
        bool(score.ptm_type) or bool(score.immunopeptide_scope) or _has_source(score.evidence, "general_query"),
        bool(score.species),
        bool(score.acquisition_mode),
        bool(features.instrument_families),
        bool(features.fragmentation_methods),
    ]
    return round(sum(1 for item in checks if item) / len(checks), 3)


def _file_completeness(
    *,
    file: DiscoveredFile | None = None,
    download_url: str | None = None,
    expected_size_bytes: int | None = None,
    project: DiscoveredProject,
    features: DiscoveryFeatureSummary,
) -> float:
    checks = [
        bool(download_url if file is None else file.download_url),
        (expected_size_bytes if file is None else file.expected_size_bytes) is not None,
        bool(project.species),
        bool(project.acquisition_mode),
        bool(project.ptm_type) or bool(project.immunopeptide_scope) or _has_source(project.evidence, "general_query"),
        bool(features.instrument_families),
        bool(features.fragmentation_methods),
    ]
    return round(sum(1 for item in checks if item) / len(checks), 3)


def _trust_score(confidence: float, completeness: float, memory_prior: float, needs_review: bool) -> float:
    penalty = 0.05 if needs_review else 0.0
    return round(_clamp(confidence * 0.8 + completeness * 0.2 + memory_prior - penalty), 3)


def _trust_score_after_validity(trust_score: float, status: str, reasons: list[str]) -> float:
    if status == "exclude":
        return 0.0

    penalty = 0.0
    if status == "needs_review":
        penalty += 0.18
    elif status == "weak_keep":
        penalty += 0.06

    reason_penalties = {
        "project_level_ptm_evidence": 0.04,
        "missing_fragmentation": 0.03,
        "missing_instrument": 0.03,
        "missing_download_url": 0.08,
        "missing_file_size": 0.03,
        "file_name_species_conflict": 0.25,
        "converted_peaklist": 0.04,
        "project_level_evidence_only": 0.06,
        "sdrf_no_file_match": 0.04,
        "mixed_acquisition_project": 0.08,
        "needs_file_level_acquisition_confirmation": 0.12,
    }
    penalty += sum(reason_penalties.get(reason, 0.0) for reason in reasons)
    return round(_clamp(trust_score - penalty), 3)


def _file_context_from_evidence(
    *,
    project_evidence: list[DiscoveryEvidence],
    file_evidence: list[DiscoveryEvidence],
    sdrf_match_status: SdrfMatchStatus,
) -> tuple[EvidenceLevel, int, int, list[str]]:
    file_count = len(file_evidence)
    project_count = len(project_evidence)
    warnings: list[str] = []

    if sdrf_match_status == "no_sdrf":
        warnings.append("sdrf_missing")
    elif sdrf_match_status == "no_file_match":
        warnings.append("sdrf_no_file_match")

    if file_count and project_count:
        level: EvidenceLevel = "mixed"
    elif file_count:
        level = "file"
    elif project_count:
        level = "project"
        warnings.append("project_level_evidence_only")
    else:
        level = "unknown"
        warnings.append("no_evidence")

    if level == "mixed" and not any(item.field.startswith("sdrf:") for item in file_evidence):
        warnings.append("file_name_or_record_evidence_only")

    return level, file_count, project_count, sorted(set(warnings))


def build_discovered_project(
    project: dict[str, Any],
    request: DatasetRequest,
    score: ProjectScore,
    *,
    features: DiscoveryFeatureSummary | None = None,
    memory_prior: float = 0.0,
    memory_feedback: dict[str, Any] | None = None,
) -> DiscoveredProject:
    feature_summary = features or DiscoveryFeatureSummary()
    completeness = _project_completeness(score, feature_summary)
    trust = _trust_score(score.confidence, completeness, memory_prior, score.needs_review)
    goal = str(request.goal or "").casefold()
    ptm_goal = goal == "ptm" or bool(request.ptm_types)
    # Immunopeptidomics is not itself a PTM.  Do not promote incidental words
    # such as "acetylation" from a long protocol into the project's primary
    # scientific scope unless the user explicitly requested a PTM dimension.
    request_ptm = request.ptm_type if ptm_goal else None
    request_modification_scope = (
        request.modification_scope or normalize_ptm_type(request.ptm_type)
        if ptm_goal
        else None
    )
    observed_ptm = score.ptm_type if ptm_goal else None
    observed_modification_scope = score.modification_scope if ptm_goal else None
    project_accession = _project_accession(project)
    project_source_url = (
        f"https://www.ebi.ac.uk/pride/archive/projects/{project_accession}"
        if request.repository == "pride" and project_accession
        else None
    )
    project_model = DiscoveredProject(
        repository=request.repository,
        project_accession=project_accession,
        project_source_url=project_source_url,
        native_accession=str(project.get("accession") or project.get("projectAccession") or "") or None,
        px_accession=str(project.get("projectAccession") or project.get("accession") or "") or None,
        project_title=_project_title(project),
        project_description=_entry_text(project.get("projectDescription")) or None,
        species=score.species,
        species_policy=request.species_policy,
        canonical_species=score.canonical_species,
        organism_taxon_id=score.organism_taxon_id,
        acquisition_mode=score.acquisition_mode,
        ptm_type=observed_ptm or request_ptm,
        modification_scope=observed_modification_scope or request_modification_scope,
        ptm_subtype=score.ptm_subtype,
        ptm_evidence_terms=score.ptm_evidence_terms if ptm_goal else [],
        ptm_enrichment_methods=score.ptm_enrichment_methods if ptm_goal else [],
        semantic_metadata_confidence=score.semantic_metadata_confidence if ptm_goal else 0.0,
        semantic_interpretation_trace=score.semantic_interpretation_trace if ptm_goal else [],
        immunopeptide_scope=score.immunopeptide_scope or request.immunopeptide_scope,
        hla_class=score.hla_class or request.hla_class,
        hla_alleles=score.hla_alleles or request.hla_alleles,
        immunopeptide_evidence_terms=score.immunopeptide_evidence_terms or request.immunopeptide_evidence_terms,
        immunopeptide_enrichment_methods=score.immunopeptide_enrichment_methods or request.immunopeptide_enrichment_methods,
        immunopeptide_metadata_confidence=max(score.immunopeptide_metadata_confidence, request.immunopeptide_metadata_confidence),
        labeling_strategy=score.labeling_strategy,
        project_score=score.project_score,
        confidence=score.confidence,
        trust_score=trust,
        evidence_completeness=completeness,
        memory_prior=round(memory_prior, 3),
        memory_feedback=memory_feedback or {},
        needs_review=score.needs_review,
        evidence=_dedupe_evidence(score.evidence + feature_summary.evidence),
        instrument_names=feature_summary.instrument_names,
        instrument_families=feature_summary.instrument_families,
        laboratory_names=feature_summary.laboratory_names,
        instrument_generation_score=feature_summary.instrument_generation_score,
        instrument_generation_label=feature_summary.instrument_generation_label,
        project_publication_date=str(project.get("publicationDate") or "") or None,
        project_submission_date=str(project.get("submissionDate") or "") or None,
        fragmentation_methods=feature_summary.fragmentation_methods,
        lc_gradient=feature_summary.lc_gradient,
        lc_gradient_minutes=feature_summary.lc_gradient_minutes,
        diversity_tags=diversity_tags(
            species=score.species,
            instrument_families=feature_summary.instrument_families,
            fragmentation_methods=feature_summary.fragmentation_methods,
            lc_gradient_minutes=feature_summary.lc_gradient_minutes,
            modification_scope=observed_modification_scope or request_modification_scope,
            immunopeptide_scope=score.immunopeptide_scope or request.immunopeptide_scope,
            labeling_strategy=score.labeling_strategy,
        ),
        raw_metadata=project,
    )
    validity = assess_project_validity(project_model, request)
    project_model = project_model.model_copy(
        update={
            "validity_status": validity.status,
            "validity_reasons": validity.reasons,
            "needs_review": project_model.needs_review or validity.needs_review,
        }
    )
    from agent.discovery.calibration import load_active_calibration, score_project_with_calibration

    active_calibration = load_active_calibration()
    if active_calibration is None:
        return project_model
    calibrated = score_project_with_calibration(project_model, active_calibration)
    return project_model.model_copy(
        update={
            "calibrated_project_score": calibrated["score"],
            "calibration_version": calibrated["version_id"],
            "calibration_components": calibrated["components"],
        }
    )


def file_type_for_name(file_name: str) -> str | None:
    lower = file_name.casefold()
    for suffix in RAW_DIRECTORY_SUFFIXES:
        if lower.endswith(suffix):
            return ".d"
    for suffix in RAW_FILE_SUFFIXES:
        if lower.endswith(suffix):
            return suffix
    return None


def classify_file_role(file_name: str) -> FileRoleDecision:
    lower = file_name.casefold()
    file_type = file_type_for_name(file_name)
    reasons: list[str] = []

    if any(token in lower for token in DERIVED_RAW_NAME_TOKENS) or lower.endswith(".mzid"):
        return FileRoleDecision("search_result", file_type or ".mzid", ["derived_identification_token"])

    if lower.endswith((".sdrf.tsv", ".sdrf.txt")) or "sdrf" in lower:
        return FileRoleDecision("metadata", file_type, ["sdrf_metadata"])

    if lower.endswith(RESULT_FILE_SUFFIXES):
        table_suffixes = (".tsv", ".txt", ".csv", ".xlsx", ".xls", ".pdf", ".html", ".json")
        role: FileRole = "report_table" if lower.endswith(table_suffixes) else "search_result"
        return FileRoleDecision(role, file_type, ["result_or_report_suffix"])

    if file_type == ".mgf":
        return FileRoleDecision("converted_peaklist", file_type, ["peaklist_suffix"])

    if file_type == ".mzml":
        return FileRoleDecision("converted_peaklist", file_type, ["open_spectrum_mzml"])

    if file_type is not None:
        if file_type == ".d" and any(lower.endswith(suffix) for suffix in RAW_DIRECTORY_SUFFIXES):
            reasons.append("vendor_directory_archive")
        else:
            reasons.append("raw_acquisition_suffix")
        return FileRoleDecision("raw_acquisition", file_type, reasons)

    return FileRoleDecision("unknown", None, ["unsupported_or_unknown_suffix"])


def is_supported_raw_file(file_name: str) -> bool:
    return classify_file_role(file_name).role in {"raw_acquisition", "converted_peaklist"}


def is_result_or_report_file(file_name: str) -> bool:
    role = classify_file_role(file_name).role
    if role in {"search_result", "metadata", "report_table"}:
        return True
    lower = file_name.casefold()
    return role == "unknown" and any(token in lower for token in RESULT_NAME_TOKENS)


def _file_level_labeling(file_name: str, project_labeling: str | None) -> str | None:
    lower = str(file_name or "").casefold()
    if "silac" in lower:
        return "SILAC"
    if re.search(r"(^|[_\-\s.])tmt([_\-\s.]|$)|tmt\d+", lower):
        return "TMT"
    if "itraq" in lower:
        return "iTRAQ"
    if re.search(r"label[\s\-_]?free|lfq", lower):
        return "label_free"
    return project_labeling


def _file_name(file_record: dict[str, Any]) -> str:
    return str(file_record.get("fileName") or file_record.get("name") or "")


def _file_size(file_record: dict[str, Any]) -> int | None:
    for key in ("fileSizeBytes", "fileSize", "size"):
        value = file_record.get(key)
        if value in (None, ""):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _transfer_method(download_url: str | None) -> str | None:
    if not download_url:
        return None
    lower = download_url.casefold()
    if lower.startswith("ftp://"):
        return "ftp"
    if lower.startswith(("http://", "https://")):
        return "https"
    return "unknown"


def score_file(
    file_record: dict[str, Any],
    project: DiscoveredProject,
    request: DatasetRequest,
    *,
    features: DiscoveryFeatureSummary | None = None,
    memory_prior: float = 0.0,
    memory_feedback: dict[str, Any] | None = None,
    sdrf_match_status: SdrfMatchStatus = "not_checked",
) -> DiscoveredFile | None:
    file_name = _file_name(file_record)
    role = classify_file_role(file_name)
    if not file_name or role.file_type is None or role.role not in {"raw_acquisition", "converted_peaklist"}:
        return None

    fields = [("file_name", file_name), ("file_record", _entry_text(file_record))]
    project_evidence = list(project.evidence)
    requested_ptm = normalize_ptm_type(project.ptm_type or request.ptm_type)
    general_goal = str(request.goal or "").casefold() == "general"
    ptm_goal = str(request.goal or "").casefold() == "ptm" or bool(request.ptm_types)
    file_semantic = (
        interpret_ptm_metadata(" ".join(text for _, text in fields), requested_ptm)
        if ptm_goal
        else interpret_ptm_metadata("", None)
    )
    file_immuno = interpret_immunopeptide_metadata(" ".join(text for _, text in fields))
    general_file_evidence = _keyword_evidence(fields, tuple(request.query_terms), "general_query", 6) if general_goal else []
    file_evidence = [
        DiscoveryEvidence(field="file_name", source="ptm", text=term, weight=8)
        for term in file_semantic.evidence_terms
    ]
    immuno_file_evidence = [
        DiscoveryEvidence(field="file_name", source="immunopeptidomics", text=term, weight=9)
        for term in file_immuno.evidence_terms
    ]
    requested_acquisition = normalize_acquisition_mode(request.acquisition_mode)
    acquisition_evidence_by_mode = _acquisition_evidence_by_mode(fields, weight=4)
    observed_file_acquisition_modes = {
        mode for mode, evidence in acquisition_evidence_by_mode.items() if evidence
    }
    file_acquisition_evidence = [
        item
        for mode_evidence in acquisition_evidence_by_mode.values()
        for item in mode_evidence
    ]
    file_acquisition_mode = _select_observed_acquisition_mode(
        acquisition_evidence_by_mode,
        requested_acquisition,
    ) or project.acquisition_mode
    requested_labeling = normalize_labeling_strategy(request.labeling_strategy)
    detected_labeling = labeling_from_text(" ".join(text for _, text in fields))
    labeling_evidence = (
        _keyword_evidence(fields, labeling_aliases(detected_labeling), "labeling", 4)
        if detected_labeling
        else []
    )
    labeling_matches_request = (
        requested_labeling != "unknown"
        and detected_labeling is not None
        and normalize_labeling_strategy(detected_labeling) == requested_labeling
        and bool(labeling_evidence)
    )
    if (
        request.is_hard_constraint("acquisition_mode")
        and requested_acquisition != "unknown"
        and observed_file_acquisition_modes
        and requested_acquisition not in observed_file_acquisition_modes
    ):
        return None

    score = 40.0
    if role.file_type in {".raw", ".mzml"}:
        score += 10.0
    if role.role == "converted_peaklist":
        score -= 6.0
    if general_file_evidence:
        score += min(16.0, 6.0 + 3.0 * len(general_file_evidence))
    if file_evidence:
        score += min(20.0, 8.0 + 4.0 * len(file_evidence))
    if immuno_file_evidence:
        score += min(22.0, 10.0 + 4.0 * len(immuno_file_evidence))
    matching_acquisition_evidence = acquisition_evidence_by_mode.get(
        requested_acquisition,
        [],
    )
    if matching_acquisition_evidence:
        score += 5.0
    elif requested_acquisition == "unknown" and file_acquisition_evidence:
        score += 2.0
    if labeling_matches_request:
        score += 5.0

    download_url = PrideClient.first_download_url(file_record)
    expected_size_bytes = _file_size(file_record)
    # File review is owned by file evidence. Inheriting a project-level mixed
    # acquisition flag makes it impossible for file-level DDA/DIA/targeted
    # evidence to resolve the exact asset, defeating review_mixed semantics.
    needs_review = not download_url or expected_size_bytes is None
    confidence = max(0.0, min(1.0, (project.project_score + score) / 150.0))
    feature_summary = features or DiscoveryFeatureSummary()
    # Conditional project→file method inheritance (Wave A / Q2):
    # Fragmentation may be inherited when the file is empty and the project is
    # not mixed_acquisition. Instrument names/families stay file-owned (SDRF or
    # file fields only) — project-level instruments must not be broadcast onto
    # every file as if they were per-file evidence.
    project_method_inherited = False
    mixed_acquisition_on_file = any(
        item.source == "mixed_acquisition" for item in feature_summary.evidence
    ) or any(item.source == "mixed_acquisition" for item in project_evidence)
    inherited_instrument_names = list(feature_summary.instrument_names)
    inherited_instrument_families = list(feature_summary.instrument_families)
    inherited_fragmentation = list(feature_summary.fragmentation_methods)
    inherited_generation_score = feature_summary.instrument_generation_score
    inherited_generation_label = feature_summary.instrument_generation_label
    if (
        not mixed_acquisition_on_file
        and not inherited_fragmentation
        and list(project.fragmentation_methods or [])
    ):
        inherited_fragmentation = list(project.fragmentation_methods)
        project_method_inherited = True
    completeness = _file_completeness(
        download_url=download_url,
        expected_size_bytes=expected_size_bytes,
        project=project,
        features=feature_summary,
    )
    trust = _trust_score(confidence, completeness, memory_prior, needs_review)
    inheritance_evidence: list[DiscoveryEvidence] = []
    if project_method_inherited:
        inheritance_evidence.append(
            DiscoveryEvidence(
                field="project_method",
                source="project_method_inherited",
                text="fragmentation_from_project",
                weight=2,
            )
        )
    file_context_evidence = (
        general_file_evidence
        + immuno_file_evidence
        + file_evidence
        + file_acquisition_evidence
        + labeling_evidence
        + feature_summary.evidence
        + inheritance_evidence
    )
    evidence_level, file_level_count, project_level_count, evidence_warnings = _file_context_from_evidence(
        project_evidence=project_evidence,
        file_evidence=file_context_evidence,
        sdrf_match_status=sdrf_match_status,
    )

    diversity = diversity_tags(
        species=project.species,
        instrument_families=inherited_instrument_families,
        fragmentation_methods=inherited_fragmentation,
        lc_gradient_minutes=feature_summary.lc_gradient_minutes,
        modification_scope=project.modification_scope,
        immunopeptide_scope=file_immuno.scope if immuno_file_evidence else project.immunopeptide_scope,
        labeling_strategy=_file_level_labeling(file_name, detected_labeling or project.labeling_strategy),
    )
    if project_method_inherited and "project_method_inherited" not in diversity:
        diversity = [*diversity, "project_method_inherited"]

    file_model = DiscoveredFile(
        repository=request.repository,
        project_accession=project.project_accession,
        project_source_url=project.project_source_url,
        native_accession=project.native_accession,
        px_accession=project.px_accession,
        file_accession_or_path=str(
            file_record.get("fileAccession")
            or file_record.get("accession")
            or file_record.get("path")
            or file_record.get("filePath")
            or file_name
        ),
        project_title=project.project_title,
        file_name=file_name,
        download_url=download_url,
        transfer_method=_transfer_method(download_url),
        file_type=role.file_type,
        file_role=role.role,
        file_role_reasons=role.reasons,
        sdrf_match_status=sdrf_match_status,
        evidence_level=evidence_level,
        file_level_evidence_count=file_level_count,
        project_level_evidence_count=project_level_count,
        evidence_warnings=evidence_warnings,
        expected_size_bytes=expected_size_bytes,
        species=project.species,
        species_policy=request.species_policy,
        canonical_species=project.canonical_species,
        organism_taxon_id=project.organism_taxon_id,
        acquisition_mode=file_acquisition_mode,
        ptm_type=project.ptm_type,
        ptm_subtype=file_semantic.subtypes[0] if file_semantic.subtypes else project.ptm_subtype,
        ptm_evidence_terms=list(dict.fromkeys([*project.ptm_evidence_terms, *file_semantic.evidence_terms])),
        ptm_enrichment_methods=list(dict.fromkeys([*project.ptm_enrichment_methods, *file_semantic.enrichment_methods])),
        semantic_metadata_confidence=max(project.semantic_metadata_confidence, file_semantic.confidence),
        semantic_interpretation_trace=list(dict.fromkeys([*project.semantic_interpretation_trace, *file_semantic.trace])),
        modification_scope=project.modification_scope,
        immunopeptide_scope=file_immuno.scope if immuno_file_evidence else project.immunopeptide_scope,
        hla_class=list(dict.fromkeys([*project.hla_class, *file_immuno.hla_classes])),
        hla_alleles=list(dict.fromkeys([*project.hla_alleles, *file_immuno.hla_alleles])),
        immunopeptide_evidence_terms=list(dict.fromkeys([*project.immunopeptide_evidence_terms, *file_immuno.evidence_terms])),
        immunopeptide_enrichment_methods=list(dict.fromkeys([*project.immunopeptide_enrichment_methods, *file_immuno.enrichment_methods])),
        immunopeptide_metadata_confidence=max(project.immunopeptide_metadata_confidence, file_immuno.confidence),
        labeling_strategy=_file_level_labeling(file_name, detected_labeling or project.labeling_strategy),
        project_score=project.project_score,
        file_score=round(score, 2),
        confidence=round(confidence, 3),
        trust_score=trust,
        evidence_completeness=completeness,
        memory_prior=round(memory_prior, 3),
        memory_feedback=memory_feedback or {},
        needs_review=needs_review,
        evidence=_dedupe_evidence(project_evidence + file_context_evidence),
        instrument_names=inherited_instrument_names,
        instrument_families=inherited_instrument_families,
        laboratory_names=list(feature_summary.laboratory_names or project.laboratory_names),
        instrument_generation_score=inherited_generation_score,
        instrument_generation_label=inherited_generation_label,
        fragmentation_methods=inherited_fragmentation,
        tissue=feature_summary.tissue,
        enzyme=feature_summary.enzyme,
        isolation_window=feature_summary.isolation_window,
        resolution=feature_summary.resolution,
        collision_energy=feature_summary.collision_energy,
        scan_range=feature_summary.scan_range,
        lc_gradient=feature_summary.lc_gradient,
        lc_gradient_minutes=feature_summary.lc_gradient_minutes,
        diversity_tags=diversity,
        raw_record=file_record,
    )
    validity = assess_file_validity(file_model, request)
    return file_model.model_copy(
        update={
            "validity_status": validity.status,
            "validity_reasons": validity.reasons,
            "needs_review": file_model.needs_review or validity.needs_review,
            "trust_score": _trust_score_after_validity(file_model.trust_score, validity.status, validity.reasons),
        }
    )
