from __future__ import annotations

from collections import Counter, defaultdict

from agent.dataset_construction.identities import identity_values
from agent.dataset_construction.models import (
    DatasetCatalog,
    LeakageAudit,
    LeakageFinding,
    ObservationRecord,
    SplitPlan,
)
from agent.dataset_construction.splitting import PROTOCOL_IDENTITIES


_ALWAYS_CHECKED = (
    "file_family_id",
    "sample_id",
    "subject_id",
    "technical_replicate_id",
    "fraction_id",
    "tmt_plex_id",
)

_REPORT_ONLY = (
    "project_id",
    "lab_id",
    "instrument_id",
    "organism_id",
    "peptide",
    "modified_peptide",
    "protein_ids",
    "protein_family_ids",
    "modification_classes",
    "acquisition_id",
    "gradient_id",
    "search_workflow_id",
)

_VALID_SPLITS = {"train", "validation", "test"}


def _manifest_integrity_finding(
    catalog: DatasetCatalog,
    plan: SplitPlan,
) -> LeakageFinding:
    catalog_ids = [row.observation_id for row in catalog.observations]
    allocation_ids = [row.observation_id for row in plan.allocations]
    catalog_set = set(catalog_ids)
    allocation_set = set(allocation_ids)
    duplicates = sorted(
        observation_id
        for observation_id, count in Counter(allocation_ids).items()
        if count > 1
    )
    missing = sorted(catalog_set - allocation_set)
    unknown = sorted(allocation_set - catalog_set)
    invalid_split_rows = sorted(
        row.observation_id
        for row in plan.allocations
        if row.split not in _VALID_SPLITS
    )
    empty_component_rows = sorted(
        row.observation_id for row in plan.allocations if not row.component_id.strip()
    )
    represented_splits = {
        row.split for row in plan.allocations if row.split in _VALID_SPLITS
    }
    absent_splits = sorted(_VALID_SPLITS - represented_splits) if plan.status == "ready" else []
    affected = sorted(
        set(duplicates + missing + unknown + invalid_split_rows + empty_component_rows)
    )
    evidence = [
        *(f"duplicate:{value}" for value in duplicates),
        *(f"missing:{value}" for value in missing),
        *(f"unknown:{value}" for value in unknown),
        *(f"invalid_split:{value}" for value in invalid_split_rows),
        *(f"empty_component:{value}" for value in empty_component_rows),
        *(f"absent_split:{value}" for value in absent_splits),
    ]
    return LeakageFinding(
        dimension="manifest_integrity",
        requirement="complete_unique_valid_allocation",
        status="fail" if evidence else "pass",
        overlap_count=len(evidence),
        missing_count=len(missing),
        affected_identities=evidence,
        affected_observation_ids=affected,
        severity="critical" if evidence else "info",
    )


def _finding(
    *,
    catalog: DatasetCatalog,
    split_by_observation: dict[str, str],
    dimension: str,
    enforce_zero_overlap: bool,
    require_complete: bool,
    identity_policy: dict[str, object],
) -> LeakageFinding:
    identity_splits: dict[str, set[str]] = defaultdict(set)
    identity_observations: dict[str, set[str]] = defaultdict(set)
    missing = 0
    for row in catalog.observations:
        split = split_by_observation.get(row.observation_id)
        if not split:
            continue
        values = identity_values(row, dimension, identity_policy)
        if not values:
            missing += 1
            continue
        for identity in values:
            identity_splits[identity].add(split)
            identity_observations[identity].add(row.observation_id)
    overlaps = sorted(
        identity for identity, splits in identity_splits.items() if len(splits) > 1
    )
    affected = sorted(
        observation_id
        for identity in overlaps
        for observation_id in identity_observations[identity]
    )
    if overlaps and enforce_zero_overlap:
        status = "fail"
    elif overlaps:
        status = "reported_overlap"
    elif require_complete and missing:
        status = "inconclusive"
    else:
        status = "pass"
    return LeakageFinding(
        dimension=dimension,
        requirement="zero_overlap" if enforce_zero_overlap else "report_only",
        status=status,
        overlap_count=len(overlaps),
        missing_count=missing,
        affected_identities=overlaps,
        affected_observation_ids=affected,
        severity="critical" if status == "fail" else "warning" if status in {"inconclusive", "reported_overlap"} else "info",
    )


def audit_split(catalog: DatasetCatalog, plan: SplitPlan) -> LeakageAudit:
    """Recompute forbidden overlaps independently from the split planner."""

    holdout = PROTOCOL_IDENTITIES[plan.requested_protocol]
    split_by_observation = {
        allocation.observation_id: allocation.split for allocation in plan.allocations
    }
    checked_dimensions = list(_ALWAYS_CHECKED)
    if holdout not in checked_dimensions:
        checked_dimensions.append(holdout)
    zero_overlap_dimensions = (
        {holdout}
        if plan.requested_protocol == "row_random_control"
        else {*_ALWAYS_CHECKED, holdout}
    )
    required_complete_dimensions = (
        {holdout}
        if plan.requested_protocol == "row_random_control"
        else {"file_family_id", holdout}
    )
    findings = [_manifest_integrity_finding(catalog, plan)]
    findings.extend(
        [
        _finding(
            catalog=catalog,
            split_by_observation=split_by_observation,
            dimension=dimension,
            enforce_zero_overlap=dimension in zero_overlap_dimensions,
            require_complete=dimension in required_complete_dimensions,
            identity_policy=plan.identity_policy,
        )
        for dimension in checked_dimensions
        ]
    )
    findings.extend(
        _finding(
            catalog=catalog,
            split_by_observation=split_by_observation,
            dimension=dimension,
            enforce_zero_overlap=False,
            require_complete=False,
            identity_policy=plan.identity_policy,
        )
        for dimension in _REPORT_ONLY
        if dimension not in checked_dimensions
    )
    if plan.status in {"inconclusive", "infeasible"} and not plan.allocations:
        holdout_finding = next(row for row in findings if row.dimension == holdout)
        if plan.status == "inconclusive":
            holdout_finding.status = "inconclusive"
            holdout_finding.severity = "warning"
            holdout_finding.missing_count = max(
                holdout_finding.missing_count,
                plan.missing_identity_count,
            )
        return LeakageAudit(
            protocol=plan.requested_protocol,
            status=plan.status,
            findings=findings,
        )
    if any(row.status == "fail" for row in findings):
        status = "fail"
    elif any(row.status == "inconclusive" for row in findings):
        status = "inconclusive"
    else:
        status = "pass"
    return LeakageAudit(
        protocol=plan.requested_protocol,
        status=status,
        findings=findings,
    )
