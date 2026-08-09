# Manuscript-to-code audit

This audit separates what the equations define from what an implementation must
still decide. Until these items are resolved and experiments are rerun, the
specific numerical values already present in the manuscript are unverified.

## Blocking specification gaps

1. **Relation restoration.** Equations treat relation units as independent
   restoration coordinates, but ALBEF/InstructBLIP inputs contain only image and
   text. Define native cross-attention gating or a cooperative-interaction
   estimator and update the proof/algorithm accordingly.
2. **Text baseline construction.** “Replace detailed descriptions with general
   expressions” is not an algorithm. Publish deterministic task-specific rules
   or a fixed baseline table. LLM-generated baselines introduce another model
   and stochasticity.
3. **Semantic factors and coverage weights.** The object/attribute/relation/
   number/location taxonomy, `w_gk`, `w_rk`, and relation caps are not specified.
   These choices directly affect the selected subset.
4. **Relation compatibility.** `C(t_i,v_j)` is called grounding, conflict, or
   absence-of-support depending on task, but its estimator, calibration, range,
   and sign are missing.
5. **Task target models.** The TIIL ALBEF classifier and M-HalDetect span detector
   are newly trained models, yet architecture/training/checkpoints are absent.
6. **Baseline methods.** “Aggregate over the same evidence units and take top-K”
   needs a rule for multimodal baselines that output only visual maps and for
   signed scores.
7. **Stability scaling.** With equal objective weights, min-max normalization can
   let a nearly irrelevant but path-invariant unit outrank a high-contribution
   interaction whose attribution legitimately varies by restoration order.
   Report component scales and include a stability-weight sensitivity ablation.

## Mathematical/reporting corrections

- The stated path-attribution cost `O(LN)` omits the integration-point count and
  forward/backward cost. With simultaneous input gradients it is approximately
  `O(L M C_f)` for `L` paths and `M` quadrature points, plus restoration cost;
  it is not one independent evaluation per coordinate.
- Completeness should be reported numerically per path because quadrature and
  non-smooth restoration operators introduce approximation error.
- The same symbol `R` is used for the candidate relation-unit set and the set of
  relation types. Rename one to avoid ambiguity.
- Two sensitivity figures currently reference the identical file
  `common/sensitivity_path_number.pdf`; the budget plot needs its own file.
- The budget-sensitivity caption says only sensitivity to `K`, while the next
  figure is for path count `L`; filenames and labels should match.
- The Results subsections for all three tasks are empty, while later tables
  contain precise averages. Every value should be generated from saved per-item
  records with mean, standard deviation, split size, and seeds.
- The appendix claims distributed training on 8 Ascend-910 nodes with MindSpore,
  whereas target methods are described as PyTorch ALBEF/InstructBLIP/SAM. Clarify
  which components were trained in which framework; otherwise the environment is
  internally inconsistent.

## Qualitative figure protocol

Use a predeclared selection policy rather than manual outcome alteration:

1. Fix the test split and compute results for every eligible example.
2. Stratify by task-relevant category (VQA answer type; TIIL manipulation type;
   M-HalDetect hallucination type).
3. Rank within each stratum by a declared P-MESA metric such as faithfulness,
   subject to correct target-model prediction.
4. Select at most one example per source image and include a median-quality or
   failure example alongside strong examples.
5. Render every method with the same normalization, colormap, resolution, and
   opacity, and show unmodified text predictions.
6. Publish the selected example IDs and selection script.

This produces persuasive figures because the evidence is clear and auditable,
not because outputs were changed after inspection.
