# P-MESA

Implementation of **P-MESA: Path-Guided Multimodal Evidence Subset
Attribution**.

P-MESA integrates evidence contributions along multiple restoration paths and
selects a compact multimodal evidence subset for each prediction.

## Installation

```bash
conda env create -f environment.yml
conda activate pmesa
pip install -e .[experiments,dev]
pytest -q
```

## Demo

```bash
pmesa demo --output results/demo/explanation.json
```

## Data

Expected dataset locations:

```text
data/vqa_x/
data/tiil/
data/mhaldetect/
```

VQA-X uses VQA v2 annotations and MS-COCO 2014 images. M-HalDetect uses its
released span annotations and COCO image identifiers. TIIL must be obtained
from the D-TIIL authors. Dataset and checkpoint sources are listed in
[docs/METHOD_SOURCES.md](docs/METHOD_SOURCES.md).

## Qualitative experiments

Run VQA-X:

```bash
python scripts/run_qualitative.py \
  --dataset vqax \
  --annotations data/explanation_dataset_test.json \
  --image-dir data/vqax_images \
  --indices 19506,19507,19508,19510,19512,19513,19514,19511 \
  --output results/vqax
```

Run M-HalDetect:

```bash
python scripts/run_qualitative.py \
  --dataset mhaldetect \
  --annotations data/mhaldetect/val_raw.json \
  --image-dir data/mhaldetect/images \
  --indices 4,12,23,31,57,79,83,0 \
  --output results/mhaldetect
```

Select four examples for each task:

```bash
for task in vqax mhaldetect; do
  python scripts/select_qualitative_examples.py \
    results/${task}/manifest.json \
    --output results/${task}/selection.json
done
```

Build the comparison PDF:

```bash
python scripts/build_qualitative_pdf.py \
  --root results \
  --output results/pmesa_qualitative_results.pdf
```

The included qualitative runner uses frozen CLIP ViT-B/32. Task experiments
defined in `configs/` require the corresponding ALBEF or InstructBLIP
checkpoint and expose the predicted-answer, inconsistency, or hallucination
logit through `PMESAAdapter`.

## Outputs

```text
results/<task>/<example_id>/
results/<task>/manifest.json
results/<task>/selection.json
```

Each example directory contains the input, method visualizations, and raw
attribution arrays. Data, checkpoints, model caches, and results are excluded
from version control.

## License

This project is released under the [MIT License](LICENSE).
