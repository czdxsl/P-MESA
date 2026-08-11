from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas


TASKS = {
    "vqax": {
        "title": "VQA-X: predicted-answer attribution",
        "methods": (("input", "Input"), ("gradcam", "Grad-CAM"), ("attention", "Attention"),
                    ("ig", "Integrated Gradients"), ("smoothgrad_ig", "SmoothGrad-IG"),
                    ("rise", "RISE"), ("pmesa", "P-MESA")),
    },
    "mhaldetect": {
        "title": "M-HalDetect: inaccurate-span attribution",
        "methods": (("input", "Input"), ("hallucination", "Positive evidence"),
                    ("counterevidence", "Counterevidence"), ("ig", "Integrated Gradients"),
                    ("smoothgrad_ig", "SmoothGrad-IG"), ("rise", "RISE"), ("pmesa", "P-MESA")),
    },
    "tiil": {
        "title": "TIIL: fine-grained inconsistency attribution",
        "methods": (("input", "Input"), ("ground_truth", "Ground truth"),
                    ("patch_similarity", "Patch similarity"), ("gradcam", "Grad-CAM"),
                    ("ig", "Integrated Gradients"), ("smoothgrad_ig", "SmoothGrad-IG"),
                    ("rise", "RISE"), ("pmesa", "P-MESA")),
    },
}


def fit(canvas: Canvas, text: str, x: float, y: float, width: float, size: float, bold: bool = False) -> None:
    font = "Helvetica-Bold" if bold else "Helvetica"
    while size > 5 and stringWidth(text, font, size) > width:
        size -= 0.35
    while len(text) > 4 and stringWidth(text, font, size) > width:
        text = text[:-4].rstrip() + "..."
    canvas.setFont(font, size)
    canvas.drawCentredString(x, y, text)


def reader(path: Path) -> ImageReader:
    image = Image.open(path).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90, optimize=True)
    buffer.seek(0)
    return ImageReader(buffer)


def draw_image(canvas: Canvas, source: ImageReader, x: float, y: float, width: float, height: float) -> None:
    source_width, source_height = source.getSize()
    scale = min(width / source_width, height / source_height)
    drawn_width, drawn_height = source_width * scale, source_height * scale
    canvas.setFillColor(colors.white)
    canvas.rect(x, y, width, height, fill=1, stroke=0)
    canvas.drawImage(source, x + (width - drawn_width) / 2, y + (height - drawn_height) / 2,
                     drawn_width, drawn_height, mask="auto")


def heading(row: dict, task: str) -> tuple[str, str]:
    if task == "vqax":
        delta = row["score"] - row["baseline_score"]
        return f"Q: {row['question']}", f"A: {row['predicted_answer']} | score gain {delta:.3f}"
    if task == "mhaldetect":
        return row["span"], f"inaccurate probability {row['hallucination_probability']:.3f}"
    energy = row["localization"]["pmesa"]["energy_in_mask"]
    return f"{row['original_phrase']} -> {row['inconsistent_phrase']}", f"margin {row['consistency_margin']:.3f} | mask energy {energy:.3f}"


def build(root: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    page_width, page_height = landscape(A4)
    canvas = Canvas(str(output), pagesize=(page_width, page_height), pageCompression=1)
    canvas.setTitle("P-MESA qualitative results")

    pages = [
        (task, spec, offset)
        for task, spec in TASKS.items()
        for offset in (0, 4)
    ]
    for page_number, (task, spec, offset) in enumerate(pages, 1):
        selection = json.loads((root / task / "selection.json").read_text(encoding="utf-8"))
        manifest = json.loads((root / task / "manifest.json").read_text(encoding="utf-8"))
        examples = {row["example_id"]: row for row in manifest["examples"]}
        ids = [row["example_id"] for row in selection["selected"]][offset:offset + 4]
        methods = spec["methods"]

        margin, label_width, gap = 22, 106, 5
        canvas.setFillColor(colors.HexColor("#111827"))
        canvas.setFont("Helvetica-Bold", 13)
        canvas.drawString(margin, page_height - 21, spec["title"])
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#4B5563"))
        canvas.drawRightString(page_width - margin, page_height - 21, f"Examples {offset + 1}-{offset + len(ids)} | {page_number}/{len(pages)}")

        grid_x = margin + label_width
        cell_width = (page_width - grid_x - margin - gap * 3) / 4
        grid_top, grid_bottom = page_height - 83, 38
        cell_height = (grid_top - grid_bottom - gap * (len(methods) - 1)) / len(methods)

        for column, example_id in enumerate(ids):
            x = grid_x + column * (cell_width + gap)
            first, second = heading(examples[example_id], task)
            fit(canvas, example_id, x + cell_width / 2, page_height - 38, cell_width, 7.2, True)
            fit(canvas, first, x + cell_width / 2, page_height - 51, cell_width, 5.8)
            fit(canvas, second, x + cell_width / 2, page_height - 63, cell_width, 5.8)

        for method_index, (method, label) in enumerate(methods):
            y = grid_top - (method_index + 1) * cell_height - method_index * gap
            ours = method == "pmesa"
            canvas.setFillColor(colors.HexColor("#E8F1FF") if ours else colors.HexColor("#F7F8FA"))
            canvas.setStrokeColor(colors.HexColor("#78A9E6") if ours else colors.HexColor("#D6DAE0"))
            canvas.roundRect(margin, y, label_width - 7, cell_height, 3, fill=1, stroke=1)
            canvas.setFillColor(colors.HexColor("#0B57A4") if ours else colors.HexColor("#1F2937"))
            fit(canvas, label, margin + (label_width - 7) / 2, y + cell_height / 2 - 2,
                label_width - 14, 7, ours)
            for column, example_id in enumerate(ids):
                x = grid_x + column * (cell_width + gap)
                draw_image(canvas, reader(root / task / example_id / f"{method}.jpg"),
                           x, y, cell_width, cell_height)
                canvas.setStrokeColor(colors.HexColor("#C7CCD3"))
                canvas.rect(x, y, cell_width, cell_height, fill=0, stroke=1)

        canvas.setFillColor(colors.HexColor("#4B5563"))
        canvas.setFont("Helvetica", 5.8)
        if task == "vqax":
            canvas.drawString(margin, 22, "Target: BLIP VQA predicted-answer log probability. Selection: correct answer, score gain >= 0.02, completeness error <= 1e-3.")
        elif task == "mhaldetect":
            training = manifest["training"]
            canvas.drawString(margin, 22, f"Target: BLIP span-detector logit. Training: {training['train_images']} images, {training['train_spans']} spans; held-out F1 {training['best_f1']:.3f}.")
        else:
            canvas.drawString(margin, 22, "Target: CLIP similarity margin between the original and falsified phrases. Selection requires positive margin, correct pointing, and positive mask-energy gain.")
        canvas.drawString(margin, 13, "Baselines: IG-32; SmoothGrad-IG 8x16; RISE-48. P-MESA: two paths, 6x6 grid, six regions. Global per-method p99 scaling.")
        canvas.showPage()
    canvas.save()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("output/pdf/pmesa_qualitative_results.pdf"))
    args = parser.parse_args()
    build(args.root, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
