# Method sources

## VQA-X

VQA-X provides textual and visual explanations for VQA v2 examples. Images are
from MS-COCO. The qualitative runner uses the VQA-finetuned BLIP checkpoint.

- https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/2708.pdf
- https://visualqa.org/download.html
- https://huggingface.co/Salesforce/blip-vqa-base

## TIIL

TIIL contains consistent and inconsistent image-text pairs with word- and
pixel-level annotations. The qualitative runner uses CLIP similarity between
the minimal original and falsified phrases and evaluates against the released
segmentation masks.

- https://github.com/Mingzhen-Huang/D-TIIL
- https://proceedings.iclr.cc/paper_files/paper/2024/hash/73ba81c7b25134a559c8a9c39ec1a4c3-Abstract-Conference.html
- https://huggingface.co/openai/clip-vit-base-patch32

## M-HalDetect

M-HalDetect provides fine-grained accurate, inaccurate, and analysis spans for
multimodal responses. `scripts/train_mhaldetect.py` trains the image-conditioned
span head used by the qualitative runner.

- https://github.com/hendryx-scale/mhal-detect
- https://arxiv.org/abs/2308.06394
