# Code and experiment audit (2026-08-09)

## Executive conclusion

The repository currently contains a tested, model-agnostic implementation of
the mathematical P-MESA core. It does **not** yet contain executable VQA-X,
TIIL, or M-HalDetect dataset loaders; SAM/phrase preprocessing; trained target
models; or compared-method implementations. Consequently, the three YAML files
are experiment manifests, not proof that the named datasets are being used.
No dataset-level figure or paper table can currently be reproduced from this
repository alone.

## Verified implementation logic

- Restoration paths begin at zero, end at one, and are coordinate-wise monotone.
- The six-path default produces two text-first, two vision-first, and two
  interleaved paths with reproducible within-family permutations.
- Midpoint path integration supports nonlinear scores and exports a numerical
  completeness residual for every path.
- Path contributions are averaged and stability is computed as
  `1 / (1 + variance)` as stated in the manuscript.
- Contribution, saliency, semantic/relation coverage, and stability are combined
  in a non-negative monotone submodular objective and optimized greedily.
- Faithfulness, localization, span, and stability metric primitives are tested.
- Per-example result writing is atomic and qualitative selection is deterministic.

Ten unit/integration tests pass in the audited environment.

## Corrections made during this audit

1. Restoration-state tensors can now be placed explicitly on the target model's
   device; the previous CPU-only path would fail with a CUDA model.
2. Interaction saliency now uses absolute magnitude, matching the manuscript's
   evidence-level relation weight.
3. Image blur uses reflected boundaries instead of zero padding, avoiding dark
   border artifacts in the low-evidence baseline.
4. Insertion/deletion evaluation now requires a complete permutation of all
   evidence units, preventing truncated curves with invalid endpoints.
5. JSONL results are written through a temporary file and replaced atomically.

## Missing components required for actual experiments

### Shared

- Official dataset download/preparation scripts and checksums.
- Segment Anything mask generation, NMS, area filtering, and patch fallback.
- Deterministic phrase grouping and task-valid textual-baseline construction.
- Cross-modal compatibility estimator `C(t, v)` and calibrated relation labels.
- Native relation gating or a formally specified cooperative interaction proxy.
- Saliency, Grad-CAM, SmoothGrad-IG, RISE, KernelSHAP, transformer relevance,
  and M2IB comparison implementations under one normalization protocol.
- Cached preprocessing manifests, hardware/runtime capture, multi-seed runner,
  and table/figure aggregation from official test splits.

### VQA-X

- VQA-X annotation loader and mapping to underlying VQA/MS COCO images.
- ALBEF-VQA wrapper exposing predicted-answer logits, token embeddings, and image
  gradients. The upstream archived code uses legacy dependency versions.
- Pointing annotation conversion and exact Pointing Game evaluation protocol.

### TIIL

- Official TIIL release and parsing of image masks/inconsistent word labels.
- Training and evaluation code/checkpoint for the manuscript's ALBEF-ITM binary
  classifier. This is not an off-the-shelf official ALBEF checkpoint.
- D-TIIL baseline integration and separate reporting for image- and text-edited
  subsets.

### M-HalDetect

- Official fine-grained annotation release and span-token alignment.
- Complete InstructBLIP detector architecture, training loss/split, and frozen
  checkpoint. The manuscript's detector cannot be reconstructed from its name.
- Hallucination-type stratification for nonexistent entities, inaccurate
  descriptions, and incorrect relations.

## Dataset/config assessment

- `configs/vqa_x.yaml`: the dataset/task/model pairing is conceptually valid,
  but no loader or checkpoint integration exists.
- `configs/tiil.yaml`: the dataset and annotations match the cited task, but
  `ALBEF-ITM` denotes an unreleased trained classifier in this project.
- `configs/mhaldetect.yaml`: the dataset matches the hallucination task, but the
  named InstructBLIP span detector is underspecified and absent.

Therefore the configurations are reasonable intended settings, but it would be
incorrect to say the current code has already loaded or evaluated these datasets.

## Publication figure contract

The PDF generator reads genuine PNG assets from:

```text
outputs/qualitative/<task>/example_<1..4>/<method>.png
```

where task is `vqa_x`, `tiil`, or `mhaldetect`. Missing assets remain visibly
marked. The preview must not be inserted as an experimental result; once every
cell is populated from recorded runs, the warning disappears only after the
generator is deliberately updated for final mode.
