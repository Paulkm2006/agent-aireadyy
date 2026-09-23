# Leakage-aware dataset construction

This subsystem is downstream of the existing Discovery and Batch pipeline. It
does not replace, rewrite, or mutate either one. An existing Batch summary and
its Parquet outputs are treated as immutable source artifacts.

## Data unit and source boundaries

The physical input is still a file. The model observation is a task-level row:
one spectrum plus the label that the selected task learns from. Every
observation retains its project, source file family, sample, subject,
replicate, fraction, TMT plex, laboratory, instrument, organism, acquisition,
scan number, precursor, MS/MS peaks, peptide, modification, and source-row
identities when available.

Multi-task Batch output is filtered to the `task_spec.task_type` before any
split is planned. Each observation also carries a typed learning target and
its source: peptide/modified peptide, retention time plus unit, fragment target
payload fingerprint, or target/decoy label. Release-blocking contracts validate
the appropriate target for de novo, PTM de novo, RT, fragment-intensity and PSM
scoring tasks; a task cannot accidentally train from another task's Parquet.

The default must-link graph prevents these identities from being divided:

- source file family;
- biological sample and subject;
- technical replicate;
- fraction;
- TMT plex.

`row_random_control` deliberately bypasses that graph. It exists only to
measure how much a conventional row-random benchmark overestimates model
performance. It is not the recommended production split.

## Supported protocols

Every run plans all protocols and records its actual result. A protocol is
never silently replaced by a weaker split.

| Protocol | Required holdout identity | Intended question |
| --- | --- | --- |
| `row_random_control` | observation | Optimistic comparison control |
| `file_disjoint` | file family | Generalization to unseen files |
| `project_disjoint` | project | Generalization to unseen studies |
| `lab_disjoint` | laboratory | Generalization to unseen laboratories |
| `instrument_disjoint` | instrument | Generalization to unseen instruments |
| `organism_disjoint` | taxon | Cross-organism generalization |
| `peptide_disjoint` | peptide identity | De novo peptide generalization |
| `protein_family_disjoint` | reported protein family | Homology-aware generalization; inconclusive when family IDs are unavailable |
| `modification_disjoint` | PTM class or peptidoform | PTM generalization |
| `acquisition_disjoint` | acquisition profile | DDA/DIA or fragmentation generalization |

Each protocol reports one of:

- `ready`: an optimized three-way allocation exists;
- `inconclusive`: required identity metadata is missing;
- `infeasible`: fewer than three independent groups exist or the solver cannot
  produce a valid allocation.

SciPy/HiGHS MILP is the in-process solver. OR-Tools is installed as a separate
worker extra because its native CP-SAT runtime must not destabilize the web or
Discovery process.

## Identity policy

The policy is versioned and frozen into each manifest.

- Peptides default to I/L-equivalent identity and may use exact identity.
- Modifications may be held out by controlled class or exact modified peptide.
- Instrument, organism, and acquisition identities have explicit policy levels.
- Planner and independent auditor use the same canonical identity owner.

## Release gates

Publishing is blocked unless all of the following are true:

1. Pandera validates unique, non-empty observation and source identities.
2. The task label policy passes. For de novo-style peptide tasks, the default
   requires a peptide, a q-value, and `q_value <= 0.01`. A task spec can make
   these rules stricter or explicitly configure another task.
3. Every `ready` protocol passes an independent audit.
4. A split manifest contains each catalog observation exactly once, contains no
   unknown observation, uses only `train`, `validation`, and `test`, has a
   component ID, and represents all three splits.
5. Required leakage identities have zero cross-split overlap.

The auditor also reports, without hiding them, overlaps in project, lab,
instrument, organism, peptide, modified peptide, protein family, PTM class,
acquisition, gradient, and search workflow.

## Immutable release

A successful release contains:

```text
catalog/observations.parquet
catalog/benchmark_index.parquet
identity_ledger/assertions.parquet
identity_ledger/summary.json
split_manifests/<protocol>.parquet
audits/<protocol>.json
validation/catalog_contract.json
validation/benchmark_quality_report.json
provenance/prov.json
task_spec_snapshot.json
release_manifest.json
ro-crate-metadata.json
checksums.sha256
```

The release ID is unique in SQL and a non-empty output directory is never
overwritten. The Operations Alembic migration adds release, protocol, audit,
and allocation tables without removing or changing existing Discovery tables.

The input reader accepts either one aggregate Batch summary at the selected
root or multiple per-item summaries below a product Batch root. Relative
Parquet paths are resolved from the summary that declared them; repeated
artifacts and repeated observation identities are detected instead of being
silently counted twice.

## Durable product workflow

The Web UI, REST API and OpenAI Agents SDK use the same idempotent submission
service and Operations SQL state machine. A Huey worker records phase events
for ingestion, contract validation, identity-ledger construction, planning,
independent audit and immutable publication. Jobs can be safely cancelled at
phase boundaries and resumed. A heartbeat distinguishes long solver or I/O
work from a dead worker, and release registration is transactional.

```http
POST /api/ops/dataset-construction/jobs
GET  /api/ops/jobs/{job_id}
GET  /api/ops/jobs/{job_id}/events/page
POST /api/ops/jobs/{job_id}/cancel
POST /api/ops/jobs/{job_id}/resume
GET  /api/ops/jobs/{job_id}/artifacts/{artifact_key}
```

The Carbon UI is available inside **AI-ready 构建** and reports all ten
protocol and audit statuses rather than collapsing them into one success flag.

## One product environment

Discovery, Web/API, OpenAI Agents SDK, and dataset construction use one
project-local Conda environment:

```powershell
.\scripts\setup-project-conda.ps1
conda activate .\.conda-env
```

The declarative equivalent is `environment.yml`. Both install every product
extra into the same Python 3.13 environment. Existing `.venv` and other Conda
environments are not deleted.

## Commands

Preview every protocol without publishing:

```powershell
pride-dataset preview --batch-dir E:\path\to\batch
```

Build an immutable release:

```powershell
pride-dataset release `
  --batch-dir E:\path\to\batch `
  --output-dir E:\path\to\release `
  --release-id denovo-v1 `
  --task-spec E:\path\to\task-spec.json
```

The OpenAI Agents SDK specialist exposes inspect, preview, durable submission,
and job-status tools. Submission is marked `needs_approval=True`. Dataset
correctness is enforced by deterministic code and independent audit rather
than by model judgment.

For a diversity-controlled de novo release, set
`"benchmark_profile": "diverse_denovo_v1"` in the task spec. This opt-in
profile requires DDA FragPipe/Sage consensus evidence, applies the
representative-spectrum cap, forces exact peptidoform identity for modification
splits, and blocks release when the requested instrument, fragmentation,
gradient, enzyme, or PTM coverage is absent. Ordinary task specs retain the
standard backward-compatible construction path.
