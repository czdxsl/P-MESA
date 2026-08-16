# P-MESA

Official implementation of **P-MESA: Path-Guided Multimodal Evidence Subset Attribution**.

P-MESA explains a frozen vision-language model with primitive visual and textual evidence, derived cross-modal relations, multiple monotonic restoration paths, and a compact submodular evidence subset.

## Installation

Linux with Python 3.10 and CUDA 11.8 is recommended.

```bash
conda env create -f environment.yml
conda activate pmesa
pip install -e '.[experiments,dev]'
pip install 'git+https://github.com/facebookresearch/segment-anything.git'
```

Target-model checkpoints and datasets are not redistributed. Put them at the paths declared in `configs/` or update those paths locally.

## Configuration

The three manuscript tasks use:

- VQA-X with ALBEF-VQA and the predicted-answer pre-softmax logit;
- TIIL with ALBEF-ITM and the image-text inconsistency score;
- M-HalDetect with an InstructBLIP-based detector and span hallucination probability.

The default method settings match the manuscript: 50 integration points, six restoration paths, `K=5`, three relation candidates per textual unit, equal objective weights, and three random seeds.

```bash
python scripts/validate_setup.py configs/vqa_x.yaml
python scripts/validate_setup.py configs/tiil.yaml
python scripts/validate_setup.py configs/mhaldetect.yaml
```

Add `--check-artifacts` to verify local dataset and checkpoint paths.

Run a task with the target-model integration that loads your official checkpoint and dataset split:

```bash
python scripts/run_experiment.py configs/vqa_x.yaml \
  --integration integrations.vqax:build \
  --device cuda \
  --output results/vqa_x
```

The runner executes all seeds declared in the task configuration and writes one JSONL file per seed.

## Model integration

Implement `PMESAAdapter.prepare` for the target model. It must return:

- visual and textual primitive evidence;
- relation units whose `endpoints` reference primitive unit IDs;
- one differentiable scalar score accepting only primitive restoration gates;
- one saliency value per primitive or relation unit.

The integration module passed to `run_experiment.py` also supplies the official dataset iterator, example IDs, categories, and task metrics through the `ExperimentIntegration` interface.

The core never creates a synthetic relation input. Relation contribution is computed from the four endpoint restoration states in Eq. (5).

```python
from pmesa import PMESAExplainer

prepared = adapter.prepare(example)
explanation = PMESAExplainer(
    steps=50,
    path_count=6,
    budget=5,
    seed=17,
).explain(prepared.units, prepared.score, prepared.saliency)
```

SAM mask filtering, phrase construction, Eq. (1) multimodal saliency, sparse relation construction, target scores, evidence matching, metrics, and JSON serialization are provided under `src/pmesa/`.
`GatedMultimodalScore` combines image-region and text-phrase gates into the partially restored input evaluated by the frozen target model.

## Verification

```bash
pytest -q
pmesa demo --output outputs/demo/explanation.json
```

## License

[MIT](LICENSE)
