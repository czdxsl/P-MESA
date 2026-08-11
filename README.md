# P-MESA

Code for **P-MESA: Path-Guided Multimodal Evidence Subset Attribution**.

## Setup

```bash
conda env create -f environment.yml
conda activate pmesa
pip install -e '.[experiments,dev]'
huggingface-cli download Salesforce/blip-vqa-base
```

## VQA-X

Place the VQA-X test annotations at `data/explanation_dataset_test.json`, then download the selected COCO images:
```bash
python scripts/prepare_vqax.py \
  --annotations data/explanation_dataset_test.json \
  --indices 19511,19513,19521,19527,19510,19532,19512,19519
```

Run the experiment:

```bash
PYTHONPATH=src python scripts/run_vqax.py \
  --annotations data/explanation_dataset_test.json \
  --image-dir data/vqax_images \
  --indices 19511,19513,19521,19527,19510,19532,19512,19519 \
  --output output/qualitative/vqax
```

## M-HalDetect

```bash
git clone https://github.com/hendryx-scale/mhal-detect third_party/mhal-detect

python scripts/prepare_mhaldetect.py \
  --annotations third_party/mhal-detect/train_raw.json \
  --output data/mhaldetect/images \
  --max-images 700

PYTHONPATH=src python scripts/train_mhaldetect.py \
  --annotations third_party/mhal-detect/train_raw.json \
  --image-dir data/mhaldetect/images \
  --max-images 700 \
  --checkpoint checkpoints/mhaldetect_span_head.pt \
  --feature-cache results/mhaldetect/features.pt

python scripts/prepare_mhaldetect.py \
  --annotations third_party/mhal-detect/val_raw.json \
  --output data/mhaldetect/images \
  --indices 13,58,347,324,222,725,346,327

PYTHONPATH=src python scripts/run_mhaldetect.py \
  --annotations third_party/mhal-detect/val_raw.json \
  --image-dir data/mhaldetect/images \
  --checkpoint checkpoints/mhaldetect_span_head.pt \
  --indices 13,58,347,324,222,725,346,327 \
  --output output/qualitative/mhaldetect
```

## TIIL

Place TIIL under `data/tiil/`, then run:

```bash
PYTHONPATH=src python scripts/run_tiil.py \
  --consistent data/tiil/consistent.json \
  --inconsistent data/tiil/inconsistent.json \
  --data-root data/tiil \
  --indices 6630,3887,5511,3674,944,343,1734,5905 \
  --output output/qualitative/tiil
```

## Figures

```bash
for task in vqax mhaldetect tiil; do
  python scripts/select_qualitative_examples.py \
    output/qualitative/${task}/manifest.json \
    --output output/qualitative/${task}/selection.json \
    --count 8
done

python scripts/build_qualitative_pdf.py \
  --root output/qualitative \
  --output output/pdf/pmesa_qualitative_results.pdf
```

Each input and method panel is stored as an individual `960x600` JPEG. Raw
attribution arrays, manifests, and the combined PDF are written to `output/`.
Dataset and model sources are listed in
[`docs/METHOD_SOURCES.md`](docs/METHOD_SOURCES.md).

Qualitative baselines use 32-step trapezoidal Integrated Gradients and
SmoothGrad-IG with 8 noise samples and 16 integration steps. RISE uses 48 masks
for these qualitative panels; increase its mask budget for aggregate benchmark
reporting.

## Test

```bash
pytest -q
```

## License

[MIT](LICENSE)
