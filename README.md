# P-MESA

Reference implementation of **P-MESA: Path-Guided Multimodal Evidence Subset
Attribution**. The code follows the manuscript's restoration-space formulation:

1. represent visual regions, textual phrases, and cross-modal relations as
   semantic evidence units;
2. restore units from a task-valid low-evidence state over multiple monotone
   text-first, vision-first, and interleaved paths;
3. integrate gradients along every path and estimate path stability;
4. greedily select a compact subset using contribution, saliency, coverage,
   and stability.

The target model remains frozen. The core package is model-agnostic and uses a
small adapter contract so that an experiment can expose the exact scalar score
used for VQA, inconsistency detection, or hallucination analysis.

## What is implemented

- Monotone multi-family restoration paths and midpoint path integration.
- Completeness diagnostics for every restoration path.
- Image low-evidence baseline (low-resolution reconstruction plus blur) and
  overlap-safe region restoration.
- Dense/evidence-level integrated gradients and interaction saliency.
- The four-term monotone submodular subset objective and deterministic greedy
  solver.
- Insertion/deletion, sufficiency/comprehensiveness, mask IoU, span F1,
  pointing-compatible localization primitives, and Jaccard stability.
- Auditable qualitative-example selection that never changes predictions.
- Fixed-style heatmap, region, and method-grid visualization.
- JSON/JSONL experiment artifacts and configurations for VQA-X, TIIL, and
  M-HalDetect.
- Deterministic end-to-end demo and unit tests.

Target-model training code and datasets are not redistributed. They have
different licenses and several are not available through stable public URLs.
See [docs/data_sources.md](docs/data_sources.md) and implement the narrow
`PMESAAdapter.prepare` contract for the locally available official model.

## Quick start

```powershell
$env:PYTHONPATH = "src"
python -m pmesa.cli demo --output outputs/demo/explanation.json
```

Expected selected evidence is the beach/snow contradiction rather than an
arbitrary dense map. The exported JSON contains per-path contribution and the
numerical completeness error.

Run tests in this workspace with third-party pytest plugin auto-loading disabled
(one globally installed plugin hangs during collection in the supplied runtime):

```powershell
$env:PYTHONPATH = "src"
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
pytest
```

For a clean environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev,plot]"
pytest
```

## Adapter contract

For one example, an adapter returns:

- an ordered list of `EvidenceUnit` objects;
- a differentiable scalar `score(z)`, with `z=0` the low-evidence state and
  `z=1` the original input;
- one non-negative dense-saliency aggregate per unit.

```python
prepared = adapter.prepare(example)
explanation = PMESAExplainer(
    steps=50, path_count=6, budget=5, seed=0
).explain(prepared.units, prepared.score, prepared.saliency)
```

The scalar should be the predicted-answer logit (VQA), inconsistency logit
(TIIL), or span hallucination logit (M-HalDetect), not a post-softmax class ID.
Model parameters must be frozen while retaining gradients with respect to input
embeddings, pixels, and restoration gates.

### Relation-unit requirement

Visual and textual gates map directly to masked regions and phrase embeddings.
An ordinary VLM does **not** expose an independent relation input. Therefore an
adapter must explicitly choose and report one of these scientifically distinct
implementations:

1. native relation gating in a cross-attention/compatibility module; or
2. a cooperative interaction estimator (paired restoration minus the two
   individual restorations).

Silently adding a relation coordinate that the model ignores produces zero
relation attribution and does not implement the manuscript. The generic core
supports either choice through `score(z)` but does not pretend they are the same.

## Reproducible experiment workflow

1. Acquire official data/checkpoints and record their checksums.
2. Freeze the target model and verify its task performance on the official split.
3. Generate SAM masks/phrase groups once and cache them with version metadata.
4. Run all methods with identical examples, normalization, colormap, and opacity.
5. Save one JSONL record per example, including path completeness error.
6. Compute tables from JSONL records; do not type values into LaTeX manually.
7. Select qualitative examples with `select_representative_examples`, report the
   ranking metric/threshold, and include at least one typical failure case.

The manuscript audit in [docs/paper_audit.md](docs/paper_audit.md) lists the
remaining methodological decisions that must be fixed before the numerical
tables can be treated as verified.

## Repository layout

```text
configs/                 three task experiment manifests
docs/                    dataset status and manuscript audit
src/pmesa/               attribution, subset, metrics, rendering, adapters
tests/                    mathematical and end-to-end checks
paper/results_template.tex  result-section structure without invented values
```

## Research integrity

Qualitative examples may be selected for clarity using a declared rule, but
maps, predictions, labels, and metrics must remain generated outputs. If tuning
improves a figure, report the tuning set and rerun once on a held-out test set.
Fabricated or manually altered results are deliberately unsupported.

