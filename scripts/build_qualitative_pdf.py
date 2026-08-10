"""Build a qualitative comparison PDF."""

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


TASKS = (
    ("vqax", "VQA-X: answer-rationale attribution"),
    ("mhaldetect", "M-HalDetect: annotated hallucinated-span attribution"),
)
METHODS = (
    ("input", "Input"),
    ("saliency", "Saliency"),
    ("ig", "Integrated Gradients"),
    ("smoothgrad_ig", "SmoothGrad-IG"),
    ("rise", "RISE"),
    ("kernelshap", "KernelSHAP"),
    ("pmesa", "P-MESA (top-6)"),
)


def fit(canvas: Canvas, text: str, x: float, y: float, width: float, size: float, *, bold: bool = False) -> None:
    font = "Helvetica-Bold" if bold else "Helvetica"
    while size > 5 and stringWidth(text, font, size) > width:
        size -= 0.4
    while len(text) > 4 and stringWidth(text, font, size) > width:
        text = text[:-4].rstrip() + "..."
    canvas.setFont(font, size)
    canvas.drawCentredString(x, y, text)


def cropped_reader(path: Path) -> ImageReader:
    image = Image.open(path).convert("RGB")
    image = image.crop((0, 42, image.width, image.height))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88, optimize=True)
    buffer.seek(0)
    return ImageReader(buffer)


def draw_contain(canvas: Canvas, source: ImageReader, x: float, y: float, width: float, height: float) -> None:
    sw, sh = source.getSize()
    scale = min(width / sw, height / sh)
    dw, dh = sw * scale, sh * scale
    canvas.setFillColor(colors.white)
    canvas.rect(x, y, width, height, fill=1, stroke=0)
    canvas.drawImage(source, x + (width - dw) / 2, y + (height - dh) / 2, dw, dh, mask="auto")


def build(root: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    page_w, page_h = landscape(A4)
    canvas = Canvas(str(output), pagesize=(page_w, page_h), pageCompression=1)
    canvas.setTitle("P-MESA qualitative results")
    for page_no, (task, title) in enumerate(TASKS, 1):
        selection = json.loads((root / task / "selection.json").read_text(encoding="utf-8"))
        manifest = json.loads((root / task / "manifest.json").read_text(encoding="utf-8"))
        by_id = {row["example_id"]: row for row in manifest["examples"]}
        ids = [row["example_id"] for row in selection["selected"]]

        margin, label_w, gap = 24, 108, 5
        canvas.setFillColor(colors.HexColor("#111827"))
        canvas.setFont("Helvetica-Bold", 13)
        canvas.drawString(margin, page_h - 22, title)
        canvas.setFont("Helvetica", 7.2)
        canvas.setFillColor(colors.HexColor("#4B5563"))
        canvas.drawRightString(page_w - margin, page_h - 21, f"Frozen CLIP ViT-B/32 | {page_no}/2")

        canvas.setFillColor(colors.HexColor("#EEF6FF"))
        canvas.setStrokeColor(colors.HexColor("#9CC5F3"))
        canvas.roundRect(margin, page_h - 43, page_w - 2 * margin, 14, 3, fill=1, stroke=1)
        canvas.setFillColor(colors.HexColor("#174A7E"))
        fit(canvas, "All panels are computed outputs; no heatmap or prediction was manually altered.", page_w / 2, page_h - 39, page_w - 2 * margin - 10, 7.2, bold=True)

        grid_x = margin + label_w
        cell_w = (page_w - grid_x - margin - gap * 3) / 4
        top_y, bottom = page_h - 93, 35
        cell_h = (top_y - bottom - gap * (len(METHODS) - 1)) / len(METHODS)

        for col, example_id in enumerate(ids):
            x = grid_x + col * (cell_w + gap)
            row = by_id[example_id]
            canvas.setFillColor(colors.HexColor("#111827"))
            fit(canvas, example_id, x + cell_w / 2, page_h - 57, cell_w, 7.5, bold=True)
            fit(canvas, row["target_span"][:92], x + cell_w / 2, page_h - 69, cell_w, 6.0)
            metric = row["selected_subset_delta_fraction"]
            fit(canvas, f"subset/full |delta| = {metric:.2f}", x + cell_w / 2, page_h - 80, cell_w, 5.8)

        for method_index, (method, label) in enumerate(METHODS):
            y = top_y - (method_index + 1) * cell_h - method_index * gap
            is_ours = method == "pmesa"
            canvas.setFillColor(colors.HexColor("#E8F1FF") if is_ours else colors.HexColor("#F7F8FA"))
            canvas.setStrokeColor(colors.HexColor("#78A9E6") if is_ours else colors.HexColor("#D6DAE0"))
            canvas.roundRect(margin, y, label_w - 7, cell_h, 3, fill=1, stroke=1)
            canvas.setFillColor(colors.HexColor("#0B57A4") if is_ours else colors.HexColor("#1F2937"))
            fit(canvas, label, margin + (label_w - 7) / 2, y + cell_h / 2 - 2, label_w - 14, 7.1, bold=is_ours)
            for col, example_id in enumerate(ids):
                x = grid_x + col * (cell_w + gap)
                source = cropped_reader(root / task / example_id / f"{method}.png")
                draw_contain(canvas, source, x, y, cell_w, cell_h)
                canvas.setStrokeColor(colors.HexColor("#C7CCD3"))
                canvas.rect(x, y, cell_w, cell_h, fill=0, stroke=1)

        canvas.setFillColor(colors.HexColor("#4B5563"))
        canvas.setFont("Helvetica", 5.9)
        canvas.drawString(margin, 19, "Model: openai/clip-vit-base-patch32 (frozen). Grid: 7x7. P-MESA: four seeded restoration paths; six selected patches.")
        canvas.drawString(margin, 11, "Selection: |full - blur score| >= 0.015 and path completeness error <= 0.06, then rank by subset/full delta fraction. Global per-method p99 scale across 8 candidates.")
        canvas.showPage()
    canvas.save()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/pmesa_qualitative_results.pdf"))
    args = parser.parse_args()
    build(args.root, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
