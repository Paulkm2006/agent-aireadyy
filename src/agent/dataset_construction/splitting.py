from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable, Iterable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from agent.dataset_construction.identities import identity_values
from agent.dataset_construction.models import (
    DatasetCatalog,
    ObservationRecord,
    SplitAllocation,
    SplitPlan,
    SplitPolicy,
    SplitSuite,
)


PROTOCOL_IDENTITIES: dict[str, str] = {
    "row_random_control": "observation_id",
    "file_disjoint": "file_family_id",
    "project_disjoint": "project_id",
    "lab_disjoint": "lab_id",
    "instrument_disjoint": "instrument_id",
    "organism_disjoint": "organism_id",
    "peptide_disjoint": "peptide",
    "protein_family_disjoint": "protein_family_ids",
    "modification_disjoint": "modification_classes",
    "acquisition_disjoint": "acquisition_id",
}

_SPLIT_NAMES = ("train", "validation", "test")


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


def _union_matching(
    rows: list[ObservationRecord],
    union_find: _UnionFind,
    identity: Callable[[ObservationRecord], Iterable[str]],
) -> None:
    first_seen: dict[str, str] = {}
    for row in rows:
        for value in identity(row):
            previous = first_seen.setdefault(value, row.observation_id)
            union_find.union(previous, row.observation_id)


def _base_components(rows: list[ObservationRecord]) -> _UnionFind:
    """Build scientific must-link components shared by every protocol."""

    union_find = _UnionFind(row.observation_id for row in rows)
    for field in (
        "file_family_id",
        "sample_id",
        "subject_id",
        "technical_replicate_id",
        "fraction_id",
        "tmt_plex_id",
    ):
        _union_matching(rows, union_find, lambda row, field=field: identity_values(row, field))
    return union_find


def _protocol_groups(
    rows: list[ObservationRecord],
    protocol: str,
    policy: SplitPolicy,
) -> tuple[dict[str, list[ObservationRecord]], int]:
    holdout_field = PROTOCOL_IDENTITIES[protocol]
    if protocol == "row_random_control":
        return {row.observation_id: [row] for row in rows}, 0
    union_find = _base_components(rows)
    missing = sum(not identity_values(row, holdout_field, policy) for row in rows)
    if missing:
        return {}, missing
    _union_matching(
        rows,
        union_find,
        lambda row: identity_values(row, holdout_field, policy),
    )
    groups: dict[str, list[ObservationRecord]] = defaultdict(list)
    for row in rows:
        groups[union_find.find(row.observation_id)].append(row)
    return dict(groups), 0


def _stable_order_key(seed: int, component_id: str) -> str:
    return hashlib.sha256(f"{seed}:{component_id}".encode("utf-8")).hexdigest()


def _allocate_groups(
    groups: dict[str, list[ObservationRecord]],
    ratios: tuple[float, float, float],
    seed: int,
) -> tuple[list[SplitAllocation], str]:
    total = sum(len(rows) for rows in groups.values())
    ordered_ids = sorted(groups, key=lambda value: _stable_order_key(seed, value))
    group_count = len(ordered_ids)
    split_count = len(_SPLIT_NAMES)
    assignment_count = group_count * split_count
    variable_count = assignment_count + split_count * 2
    objective = np.zeros(variable_count, dtype=float)
    # The seed changes the stable group ordering; the tiny cost only resolves
    # equivalent optima without changing the primary size-balance objective.
    for position in range(group_count):
        for split_index in range(split_count):
            objective[position * split_count + split_index] = (
                ((position + 1) * (split_index + 1) % 17) * 1e-8
            )
    objective[assignment_count:] = 1.0
    integrality = np.zeros(variable_count, dtype=int)
    integrality[:assignment_count] = 1
    lower_bounds = np.zeros(variable_count, dtype=float)
    upper_bounds = np.full(variable_count, np.inf, dtype=float)
    upper_bounds[:assignment_count] = 1.0
    constraint_rows: list[np.ndarray] = []
    constraint_lower: list[float] = []
    constraint_upper: list[float] = []
    for position in range(group_count):
        row = np.zeros(variable_count, dtype=float)
        start = position * split_count
        row[start : start + split_count] = 1.0
        constraint_rows.append(row)
        constraint_lower.append(1.0)
        constraint_upper.append(1.0)
    for split_index in range(split_count):
        row = np.zeros(variable_count, dtype=float)
        row[split_index:assignment_count:split_count] = 1.0
        constraint_rows.append(row)
        constraint_lower.append(1.0)
        constraint_upper.append(np.inf)
    for split_index, ratio in enumerate(ratios):
        row = np.zeros(variable_count, dtype=float)
        for position, component_id in enumerate(ordered_ids):
            row[position * split_count + split_index] = len(groups[component_id])
        positive_deviation = assignment_count + split_index * 2
        negative_deviation = positive_deviation + 1
        row[positive_deviation] = -1.0
        row[negative_deviation] = 1.0
        target = total * ratio
        constraint_rows.append(row)
        constraint_lower.append(target)
        constraint_upper.append(target)
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(
            np.vstack(constraint_rows),
            np.asarray(constraint_lower),
            np.asarray(constraint_upper),
        ),
        options={"time_limit": 60.0, "mip_rel_gap": 0.0},
    )
    status_name = "optimal" if result.success else f"highs_status_{result.status}"
    if not result.success or result.x is None:
        return [], status_name
    assignments = {
        component_id: _SPLIT_NAMES[
            int(
                np.argmax(
                    result.x[
                        position * split_count : (position + 1) * split_count
                    ]
                )
            )
        ]
        for position, component_id in enumerate(ordered_ids)
    }
    return [
        SplitAllocation(
            observation_id=row.observation_id,
            component_id=component_id,
            split=assignments[component_id],
        )
        for component_id, members in sorted(groups.items())
        for row in sorted(members, key=lambda item: item.observation_id)
    ], status_name


def _distribution(
    allocations: list[SplitAllocation],
) -> tuple[dict[str, int], dict[str, float]]:
    counts = {name: 0 for name in _SPLIT_NAMES}
    for allocation in allocations:
        counts[allocation.split] += 1
    total = sum(counts.values())
    return counts, {
        name: count / total if total else 0.0 for name, count in counts.items()
    }


def _validate_ratios(ratios: tuple[float, float, float]) -> None:
    if len(ratios) != 3 or any(value <= 0 for value in ratios):
        raise ValueError("ratios must contain three positive values")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("ratios must sum to 1.0")


def plan_split_suite(
    catalog: DatasetCatalog,
    *,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    policy: SplitPolicy | None = None,
) -> SplitSuite:
    """Plan every supported holdout protocol without silent fallback."""

    _validate_ratios(ratios)
    policy = policy or SplitPolicy()
    protocols: dict[str, SplitPlan] = {}
    for protocol, holdout_field in PROTOCOL_IDENTITIES.items():
        groups, missing = _protocol_groups(catalog.observations, protocol, policy)
        identity_policy = policy.model_dump(mode="json")
        if missing:
            protocols[protocol] = SplitPlan(
                requested_protocol=protocol,
                resolved_protocol=protocol,
                status="inconclusive",
                holdout_identity=holdout_field,
                missing_identity_count=missing,
                reasons=[f"missing_required_identity:{holdout_field}"],
                identity_policy=identity_policy,
            )
            continue
        if len(groups) < len(_SPLIT_NAMES):
            protocols[protocol] = SplitPlan(
                requested_protocol=protocol,
                resolved_protocol=protocol,
                status="infeasible",
                holdout_identity=holdout_field,
                group_count=len(groups),
                reasons=["fewer_than_three_independent_groups"],
                identity_policy=identity_policy,
            )
            continue
        allocations, solver_status = _allocate_groups(groups, ratios, seed)
        if not allocations:
            protocols[protocol] = SplitPlan(
                requested_protocol=protocol,
                resolved_protocol=protocol,
                status="infeasible",
                holdout_identity=holdout_field,
                group_count=len(groups),
                reasons=[f"solver_status:{solver_status}"],
                solver="scipy_highs_milp",
                solver_status=solver_status,
                target_ratios=dict(zip(_SPLIT_NAMES, ratios, strict=True)),
                identity_policy=identity_policy,
            )
            continue
        actual_counts, actual_ratios = _distribution(allocations)
        protocols[protocol] = SplitPlan(
            requested_protocol=protocol,
            resolved_protocol=protocol,
            status="ready",
            holdout_identity=holdout_field,
            allocations=allocations,
            group_count=len(groups),
            solver="scipy_highs_milp",
            solver_status=solver_status,
            target_ratios=dict(zip(_SPLIT_NAMES, ratios, strict=True)),
            actual_counts=actual_counts,
            actual_ratios=actual_ratios,
            identity_policy=identity_policy,
        )
    return SplitSuite(ratios=ratios, seed=seed, protocols=protocols, policy=policy)
