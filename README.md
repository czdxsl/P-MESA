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

Place the VQA-X test annotations at `data/explanation_dataset_test.json` and
the corresponding COCO images in `data/vqax_images/`, then run:

```bash
PYTHONPATH=src python scripts/run_vqax.py \
  --annotations data/explanation_dataset_test.json \
  --image-dir data/vqax_images \
  --indices 19510,19512,19513,19511 \
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
  --indices 23,213,327,159

PYTHONPATH=src python scripts/run_mhaldetect.py \
  --annotations third_party/mhal-detect/val_raw.json \
  --image-dir data/mhaldetect/images \
  --checkpoint checkpoints/mhaldetect_span_head.pt \
  --indices 23,213,327,159 \
  --output output/qualitative/mhaldetect
```

## TIIL

Place TIIL under `data/tiil/`, then run:

```bash
PYTHONPATH=src python scripts/run_tiil.py \
  --consistent data/tiil/consistent.json \
  --inconsistent data/tiil/inconsistent.json \
  --data-root data/tiil \
  --indices 6630,5511,3887,944 \
  --output output/qualitative/tiil
```

## Figures

```bash
for task in vqax mhaldetect tiil; do
  python scripts/select_qualitative_examples.py \
    output/qualitative/${task}/manifest.json \
    --output output/qualitative/${task}/selection.json
done

python scripts/build_qualitative_pdf.py \
  --root output/qualitative \
  --output output/pdf/pmesa_qualitative_results.pdf
```

Each input and method panel is stored as an individual `960x600` JPEG. Raw
attribution arrays, manifests, and the combined PDF are written to `output/`.
Dataset and model sources are listed in
[`docs/METHOD_SOURCES.md`](docs/METHOD_SOURCES.md).

## Test

```bash
pytest -q
```

## License

[MIT](LICENSE)
