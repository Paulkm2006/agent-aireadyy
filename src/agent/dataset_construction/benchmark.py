from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Iterable

from agent.dataset_construction.identities import identity_values
from agent.dataset_construction.models import (
    DatasetCatalog,
    ObservationRecord,
    SplitSuite,
)


DENOVO_BENCHMARK_PROFILE = "diverse_denovo_v1"
DEFAULT_DENOVO_BENCHMARK_POLICY: dict[str, Any] = {
    "profile": DENOVO_BENCHMARK_PROFILE,
    "required_search_engines": ["fragpipe", "sage"],
    "max_q_value": 0.01,
    "max_spectra_per_modified_peptide": 10,
    "min_peak_count": 10,
    "required_instrument_groups": [
        "q_exactive",
        "orbitrap_fusion",
        "orbitrap_eclipse",
        "timstof",
        "sciex",
    ],
    "required_fragmentation_groups": ["hcd", "cid", "etd_or_ethcd"],
    "required_gradient_buckets": ["short", "long"],
    "required_enzymes": ["trypsin", "gluc", "aspn", "chymotrypsin"],
    "min_distinct_ptms": 3,
    "target_species_groups": ["animal", "plant", "microorganism"],
    "max_species_fraction": 0.6,
    "max_ptm_fraction": 0.7,
}


def denovo_benchmark_policy(task_spec: dict[str, Any] | None) -> dict[str, Any] | None:
    task_spec = task_spec or {}
    if str(task_spec.get("benchmark_profile") or "").strip() != DENOVO_BENCHMARK_PROFILE:
        return None
    if str(task_spec.get("task_type") or "").strip().casefold() != "denovo":
        raise ValueError(f"{DENOVO_BENCHMARK_PROFILE} requires task_type=denovo")
    overrides = task_spec.get("benchmark_policy") or {}
    if not isinstance(overrides, dict):
        raise ValueError("task_spec.benchmark_policy must be a JSON object")
    return {**DEFAULT_DENOVO_BENCHMARK_POLICY, **overrides}


def _tokens(values: Iterable[Any]) -> set[str]:
    return {
        " ".join(str(value or "").strip().casefold().replace("-", " ").split())
        for value in values
        if str(value or "").strip()
    }


def _contains(values: set[str], *needles: str) -> bool:
    return any(needle in value for value in values for needle in needles)


def _instrument_groups(rows: list[ObservationRecord]) -> set[str]:
    values = _tokens(
        value
        for row in rows
        for value in (row.instrument_id, row.instrument_vendor)
    )
    groups: set[str] = set()
    if _contains(values, "q exactive"):
        groups.add("q_exactive")
    if _contains(values, "orbitrap fusion", "fusion lumos"):
        groups.add("orbitrap_fusion")
    if _contains(values, "orbitrap eclipse", "eclipse tribrid"):
        groups.add("orbitrap_eclipse")
    if _contains(values, "timstof", "tims tof"):
        groups.add("timstof")
    if _contains(values, "sciex", "tripletof"):
        groups.add("sciex")
    return groups


def _fragmentation_groups(rows: list[ObservationRecord]) -> set[str]:
    values = _tokens(
        row.fragmentation_method or row.acquisition_id for row in rows
    )
    groups: set[str] = set()
    if any(re.search(r"(?<![a-z])hcd(?![a-z])", value) for value in values):
        groups.add("hcd")
    if _contains(values, "cid"):
        groups.add("cid")
    if _contains(values, "etd", "ethcd"):
        groups.add("etd_or_ethcd")
    return groups


def _gradient_bucket(minutes: float | None) -> str:
    if minutes is None:
        return "unknown"
    if minutes <= 45:
        return "short"
    if minutes >= 90:
        return "long"
    return "medium"


def _enzyme_groups(rows: list[ObservationRecord]) -> set[str]:
    values = _tokens(row.enzyme for row in rows)
    groups: set[str] = set()
    if _contains(values, "trypsin"):
        groups.add("trypsin")
    if _contains(values, "gluc", "glu c", "glu-c"):
        groups.add("gluc")
    if _contains(values, "aspn", "asp n", "asp-n"):
        groups.add("aspn")
    if _contains(values, "chymotrypsin"):
        groups.add("chymotrypsin")
    return groups


def _species_group(value: str) -> str:
    text = value.casefold()
    if any(token in text for token in ("arabidopsis", "oryza", "rice", "plant", "zea mays", "wheat", "triticum", "ncbi:3702")):
        return "plant"
    if any(token in text for token in ("bacter", "microb", "archaea", "escherichia", "e. coli", "ecoli", "bacillus", "ncbi:562")):
        return "microorganism"
    if any(token in text for token in ("yeast", "saccharomyces", "fung", "ncbi:4932")):
        return "microorganism"
    if text:
        return "animal"
    return "unknown"


def _quality_order(row: ObservationRecord) -> tuple[Any, ...]:
    return (
        -(row.fragment_coverage if row.fragment_coverage is not None else -1.0),
        row.q_value if row.q_value is not None else float("inf"),
        -(row.psm_probability if row.psm_probability is not None else -1.0),
        -(row.psm_score if row.psm_score is not None else -1.0),
        -(row.total_ion_current if row.total_ion_current is not None else -1.0),
        -(row.peak_count if row.peak_count is not None else -1),
        row.observation_id,
    )


def curate_denovo_benchmark(
    catalog: DatasetCatalog,
    task_spec: dict[str, Any] | None,
) -> DatasetCatalog:
    """Apply strict de novo benchmark QC without mutating the source Batch."""

    policy = denovo_benchmark_policy(task_spec)
    if policy is None:
        return catalog
    required_engines = _tokens(policy["required_search_engines"])
    min_peak_count = int(policy["min_peak_count"])
    filter_counts: Counter[str] = Counter()
    candidates: list[ObservationRecord] = []
    for row in catalog.observations:
        if "dda" not in row.acquisition_id.casefold():
            filter_counts["non_dda_or_unknown_acquisition"] += 1
            continue
        if row.peak_count is not None and row.peak_count < min_peak_count:
            filter_counts["low_peak_count"] += 1
            continue
        if not required_engines.issubset(_tokens(row.search_engines)):
            filter_counts["missing_fragpipe_sage_consensus"] += 1
            continue
        if any(
            engine not in row.engine_q_values
            or float(row.engine_q_values[engine]) > float(policy["max_q_value"])
            for engine in required_engines
        ):
            filter_counts["engine_q_value_above_threshold"] += 1
            continue
        candidates.append(row)

    grouped: dict[str, list[ObservationRecord]] = defaultdict(list)
    for row in candidates:
        identity = (row.modified_peptide or row.peptide).strip().casefold()
        grouped[identity].append(row)

    cap = int(policy["max_spectra_per_modified_peptide"])
    retained: list[ObservationRecord] = []
    frequency_distribution: Counter[int] = Counter()
    for identity in sorted(grouped):
        group = sorted(grouped[identity], key=_quality_order)
        frequency_distribution[len(group)] += 1
        for rank, row in enumerate(group[:cap], start=1):
            retained.append(
                row.model_copy(
                    update={
                        "peptide_frequency": len(group),
                        "representative_rank": rank,
                    }
                )
            )
        filter_counts["peptidoform_frequency_cap"] += max(0, len(group) - cap)

    report = benchmark_coverage_report(retained, policy=policy)
    report.update(
        {
            "profile": DENOVO_BENCHMARK_PROFILE,
            "rows_in": len(catalog.observations),
            "rows_out": len(retained),
            "filter_counts": dict(sorted(filter_counts.items())),
            "peptidoform_frequency_distribution": {
                str(key): value for key, value in sorted(frequency_distribution.items())
            },
            "policy": policy,
        }
    )
    return catalog.model_copy(
        update={
            "observations": retained,
            "warnings": [
                *catalog.warnings,
                *(f"benchmark_filter:{key}:{value}" for key, value in sorted(filter_counts.items()) if value),
                *report["warnings"],
            ],
            "curation_report": report,
        }
    )


def benchmark_coverage_report(
    rows: list[ObservationRecord],
    *,
    policy: dict[str, Any],
) -> dict[str, Any]:
    instruments = _instrument_groups(rows)
    fragmentation = _fragmentation_groups(rows)
    gradients = {_gradient_bucket(row.lc_gradient_minutes) for row in rows}
    enzymes = _enzyme_groups(rows)
    ptms = _tokens(value for row in rows for value in row.modification_classes)
    ptms -= {"", "none", "unmodified", "no modification"}
    ptm_distribution = Counter(
        value
        for row in rows
        for value in _tokens(row.modification_classes)
        if value not in {"", "none", "unmodified", "no modification"}
    )
    species = Counter(row.organism_id.strip() or "unknown" for row in rows)
    species_groups = {_species_group(value) for value in species if value != "unknown"}
    total = sum(species.values())
    largest_species_fraction = max(species.values(), default=0) / total if total else 0.0
    ptm_total = sum(ptm_distribution.values())
    largest_ptm_fraction = (
        max(ptm_distribution.values(), default=0) / ptm_total if ptm_total else 0.0
    )

    requirements = {
        "instrument_groups": {
            "required": list(policy["required_instrument_groups"]),
            "observed": sorted(instruments),
        },
        "fragmentation_groups": {
            "required": list(policy["required_fragmentation_groups"]),
            "observed": sorted(fragmentation),
        },
        "gradient_buckets": {
            "required": list(policy["required_gradient_buckets"]),
            "observed": sorted(gradients - {"unknown"}),
        },
        "enzymes": {
            "required": list(policy["required_enzymes"]),
            "observed": sorted(enzymes),
        },
        "ptm_types": {
            "required_minimum": int(policy["min_distinct_ptms"]),
            "observed": sorted(ptms),
        },
    }
    blocking: list[str] = []
    for name in ("instrument_groups", "fragmentation_groups", "gradient_buckets", "enzymes"):
        missing = sorted(set(requirements[name]["required"]) - set(requirements[name]["observed"]))
        requirements[name]["missing"] = missing
        if missing:
            blocking.append(f"missing_{name}:{','.join(missing)}")
    if len(ptms) < int(policy["min_distinct_ptms"]):
        blocking.append(
            f"insufficient_ptm_types:{len(ptms)}<{int(policy['min_distinct_ptms'])}"
        )

    warnings: list[str] = []
    missing_species_groups = sorted(set(policy["target_species_groups"]) - species_groups)
    if missing_species_groups:
        warnings.append(f"missing_target_species_groups:{','.join(missing_species_groups)}")
    if largest_species_fraction > float(policy["max_species_fraction"]):
        warnings.append(
            f"single_species_fraction_above_target:{largest_species_fraction:.6f}"
        )
    if largest_ptm_fraction > float(policy["max_ptm_fraction"]):
        warnings.append(f"single_ptm_fraction_above_target:{largest_ptm_fraction:.6f}")
    return {
        "status": "blocked" if blocking else "pass",
        "blocking_issues": blocking,
        "warnings": warnings,
        "requirements": requirements,
        "species_distribution": dict(sorted(species.items())),
        "species_groups": sorted(species_groups),
        "largest_species_fraction": largest_species_fraction,
        "ptm_distribution": dict(sorted(ptm_distribution.items())),
        "largest_ptm_fraction": largest_ptm_fraction,
    }


def benchmark_overlap_report(
    catalog: DatasetCatalog,
    suite: SplitSuite,
) -> dict[str, Any]:
    """Report split leakage and cross-species sequence overlap for benchmark review."""

    by_id = {row.observation_id: row for row in catalog.observations}
    protocol_reports: dict[str, Any] = {}
    for protocol, plan in suite.protocols.items():
        split_ids = {
            split: {
                allocation.observation_id
                for allocation in plan.allocations
                if allocation.split == split
            }
            for split in ("train", "validation", "test")
        }
        dimension_reports: dict[str, Any] = {}
        for dimension in (
            "peptide",
            "modified_peptide",
            "protein_family_ids",
            "project_id",
        ):
            values_by_split = {
                split: {
                    value
                    for observation_id in observation_ids
                    if observation_id in by_id
                    for value in identity_values(
                        by_id[observation_id], dimension, plan.identity_policy
                    )
                }
                for split, observation_ids in split_ids.items()
            }
            dimension_reports[dimension] = {
                "train_test": _overlap_metrics(
                    values_by_split["train"], values_by_split["test"]
                ),
                "train_validation": _overlap_metrics(
                    values_by_split["train"], values_by_split["validation"]
                ),
                "validation_test": _overlap_metrics(
                    values_by_split["validation"], values_by_split["test"]
                ),
            }
        protocol_reports[protocol] = {
            "status": plan.status,
            "dimensions": dimension_reports,
            "cross_species_train_test_peptide_overlap": _cross_species_split_overlap(
                by_id,
                split_ids["train"],
                split_ids["test"],
                suite.policy,
            ),
        }

    peptides_by_species: dict[str, set[str]] = defaultdict(set)
    for row in catalog.observations:
        species = row.organism_id.strip() or "unknown"
        peptides_by_species[species].update(identity_values(row, "peptide", suite.policy))
    species_overlap: dict[str, Any] = {}
    species_names = sorted(peptides_by_species, key=str.casefold)
    for index, left in enumerate(species_names):
        for right in species_names[index + 1 :]:
            species_overlap[f"{left}__{right}"] = _overlap_metrics(
                peptides_by_species[left], peptides_by_species[right]
            )
    human_species = next(
        (name for name in species_names if name.casefold() in {"human", "homo sapiens", "ncbi:9606"}),
        None,
    )
    mouse_species = next(
        (name for name in species_names if name.casefold() in {"mouse", "mus musculus", "ncbi:10090"}),
        None,
    )
    return {
        "protocols": protocol_reports,
        "cross_species_peptide_overlap": species_overlap,
        "human_mouse_peptide_overlap": (
            _overlap_metrics(
                peptides_by_species[human_species],
                peptides_by_species[mouse_species],
            )
            if human_species and mouse_species
            else {"status": "not_available"}
        ),
    }


def _cross_species_split_overlap(
    by_id: dict[str, ObservationRecord],
    train_ids: set[str],
    test_ids: set[str],
    policy: Any,
) -> dict[str, Any]:
    train: dict[str, set[str]] = defaultdict(set)
    test: dict[str, set[str]] = defaultdict(set)
    for observation_id in train_ids:
        row = by_id.get(observation_id)
        if row is not None:
            train[row.organism_id.strip() or "unknown"].update(
                identity_values(row, "peptide", policy)
            )
    for observation_id in test_ids:
        row = by_id.get(observation_id)
        if row is not None:
            test[row.organism_id.strip() or "unknown"].update(
                identity_values(row, "peptide", policy)
            )
    return {
        f"{train_species}__{test_species}": _overlap_metrics(
            train[train_species], test[test_species]
        )
        for train_species in sorted(train, key=str.casefold)
        for test_species in sorted(test, key=str.casefold)
    }


def _overlap_metrics(left: set[str], right: set[str]) -> dict[str, Any]:
    overlap = left & right
    union = left | right
    return {
        "left_count": len(left),
        "right_count": len(right),
        "overlap_count": len(overlap),
        "jaccard": len(overlap) / len(union) if union else 0.0,
        "smaller_set_fraction": len(overlap) / min(len(left), len(right))
        if left and right
        else 0.0,
    }
