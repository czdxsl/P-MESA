# Dataset and target-model status

Checked 2026-08-09 against primary paper/project sources.

## VQA-X / ALBEF-VQA

- VQA-X introduces human textual and visual explanations for VQA decisions.
  The paper describes human visual pointing annotations, so a localization
  metric is possible after converting the released annotations to masks.
- Images come from the underlying VQA/MS COCO data and are subject to their
  respective terms.
- The official Salesforce ALBEF repository provides VQA code, downstream JSON,
  and a fine-tuned VQA checkpoint. The repository was archived on 2025-05-01
  and pins old dependencies (`torch 1.8`, `transformers 4.8.1`, `timm 0.4.9`).

Sources:

- https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/2708.pdf
- https://github.com/salesforce/ALBEF
- https://visualqa.org/download.html

## TIIL / ALBEF-ITM

- The ICLR 2024 paper describes 14K consistent/inconsistent text-image pairs
  based on Visual News, with manually selected word- and pixel-level
  inconsistency annotations.
- The paper says its project page provides source code and data, but a stable
  repository/download URL was not exposed by the indexed official pages during
  this audit. Obtain the authors' official release and record the exact version.
- `ALBEF-ITM` in the P-MESA manuscript is a new binary classifier, not an
  off-the-shelf checkpoint from the official ALBEF repository. Its training
  split, class balance, optimizer, selection criterion, and checkpoint checksum
  must be released for reproducibility.

Sources:

- https://proceedings.iclr.cc/paper_files/paper/2024/hash/73ba81c7b25134a559c8a9c39ec1a4c3-Abstract-Conference.html
- https://arxiv.org/abs/2404.18033

## M-HalDetect / InstructBLIP detector

- The AAAI 2024 paper describes roughly 16K fine-grained annotations on VQA
  examples, including nonexistent entities, inaccurate descriptions, and
  inaccurate relations.
- The official arXiv page does not expose a direct public dataset/code download.
  Do not replace it with a similarly named third-party dataset without changing
  the manuscript.
- The manuscript's “InstructBLIP-based fine-grained detector” is a trained
  target model whose architecture, span tokenization, loss, split, and checkpoint
  are currently unspecified. Those artifacts are prerequisites for reproduction.

Source:

- https://arxiv.org/abs/2308.06394

## Required local manifest

For every run, save these fields next to results:

```yaml
dataset_name: ...
dataset_release_url: ...
dataset_checksum: ...
split_checksum: ...
target_model_commit: ...
checkpoint_checksum: ...
sam_model_and_checksum: ...
tokenizer_name_and_revision: ...
seed: ...
hardware: ...
software_versions: ...
```

