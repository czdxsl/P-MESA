"""Build a publication-style method-by-example qualitative comparison PDF.

Real assets are loaded from:
  outputs/qualitative/<task>/<example>/<method>.png

Missing assets are rendered as explicit placeholders. This makes the PDF safe as
a layout preview while keeping the exact same generator for final results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.utils import ImageReader


TASKS = [
    ("vqa_x", "VQA-X - Visual Question Answering", [
        ("input", "Input"), ("saliency", "Saliency"), ("gradcam", "Grad-CAM"),
        ("ig", "Integrated Gradients"), ("smoothgrad_ig", "SmoothGrad-IG"),
        ("rise", "RISE"), ("m2ib", "M2IB"), ("pmesa", "P-MESA (subset)"),
    ]),
    ("tiil", "TIIL - Image-Text Inconsistency Detection", [
        ("input", "Input"), ("ig", "Integrated Gradients"), ("rise", "RISE"),
        ("kernelshap", "KernelSHAP"), ("chefer", "Transformer Relevance"),
        ("m2ib", "M2IB"), ("d_tiil", "D-TIIL"), ("pmesa", "P-MESA (subset)"),
    ]),
    ("mhaldetect", "M-HalDetect - Multimodal Hallucination Analysis", [
        ("input", "Input"), ("saliency", "Saliency"),
        ("ig", "Integrated Gradients"), ("smoothgrad_ig", "SmoothGrad-IG"),
        ("rise", "RISE"), ("kernelshap", "KernelSHAP"),
        ("m2ib", "M2IB"), ("pmesa", "P-MESA (subset)"),
    ]),
]

EXAMPLES = ["Example 1", "Example 2", "Example 3", "Example 4"]


def fit_text(canvas: Canvas, text: str, x: float, y: float, max_width: float, size: float = 8) -> None:
    while size > 5 and stringWidth(text, "Helvetica", size) > max_width:
        size -= 0.5
    canvas.setFont("Helvetica", size)
    canvas.drawCentredString(x, y, text)


def draw_placeholder(canvas: Canvas, x: float, y: float, width: float, height: float) -> None:
    canvas.setFillColor(colors.HexColor("#F4F5F7"))
    canvas.setStrokeColor(colors.HexColor("#CBD0D8"))
    canvas.rect(x, y, width, height, fill=1, stroke=1)
    canvas.setStrokeColor(colors.HexColor("#D9DDE3"))
    canvas.line(x, y, x + width, y + height)
    canvas.line(x, y + height, x + width, y)
    canvas.setFillColor(colors.HexColor("#697386"))
    fit_text(canvas, "RESULT ASSET NOT AVAILABLE", x + width / 2, y + height / 2 - 3, width - 8, 6.5)


def draw_image(canvas: Canvas, path: Path, x: float, y: float, width: float, height: float) -> None:
    image = ImageReader(str(path))
    source_w, source_h = image.getSize()
    scale = min(width / source_w, height / source_h)
    draw_w, draw_h = source_w * scale, source_h * scale
    canvas.setFillColor(colors.white)
    canvas.rect(x, y, width, height, fill=1, stroke=0)
    canvas.drawImage(image, x + (width - draw_w) / 2, y + (height - draw_h) / 2,
                     draw_w, draw_h, preserveAspectRatio=True, mask="auto")


def expected_assets(assets: Path) -> list[Path]:
    return [
        assets / task_key / f"example_{col}" / f"{method_key}.png"
        for task_key, _, methods in TASKS
        for method_key, _ in methods
        for col in range(1, len(EXAMPLES) + 1)
    ]


def build(output: Path, assets: Path, *, final: bool = False) -> None:
    missing = [path for path in expected_assets(assets) if not path.exists()]
    if final and missing:
        preview = "\n  ".join(str(path) for path in missing[:10])
        suffix = f"\n  ... and {len(missing) - 10} more" if len(missing) > 10 else ""
        raise SystemExit(f"Final mode requires every result asset; missing {len(missing)}:\n  {preview}{suffix}")
    output.parent.mkdir(parents=True, exist_ok=True)
    page_w, page_h = landscape(A4)
    canvas = Canvas(str(output), pagesize=(page_w, page_h), pageCompression=1)
    canvas.setTitle("P-MESA Qualitative Comparison Layout Preview")
    margin_x, top, bottom = 24, 50, 30
    label_w, gap = 116, 5
    grid_x = margin_x + label_w
    cell_w = (page_w - grid_x - margin_x - gap * (len(EXAMPLES) - 1)) / len(EXAMPLES)
    method_count = 8
    cell_h = (page_h - top - bottom - 24 - gap * (method_count - 1)) / method_count

    for page_index, (task_key, task_title, methods) in enumerate(TASKS, start=1):
        canvas.setFillColor(colors.HexColor("#111827"))
        canvas.setFont("Helvetica-Bold", 13)
        canvas.drawString(margin_x, page_h - 22, task_title)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#6B7280"))
        canvas.drawRightString(page_w - margin_x, page_h - 21, f"Qualitative comparison - page {page_index}/3")

        canvas.setFillColor(colors.white if final else colors.HexColor("#FFF4D6"))
        canvas.setStrokeColor(colors.white if final else colors.HexColor("#E8B84A"))
        canvas.roundRect(margin_x, page_h - 43, page_w - 2 * margin_x, 14, 3, fill=1, stroke=1)
        if not final:
            canvas.setFillColor(colors.HexColor("#7A4E00"))
            fit_text(canvas, "LAYOUT PREVIEW - NO DATASET OR BASELINE OUTPUTS ARE PRESENT IN THE REPOSITORY",
                     page_w / 2, page_h - 39, page_w - 2 * margin_x - 10, 7)

        header_y = page_h - top - 7
        canvas.setFillColor(colors.HexColor("#374151"))
        for col, example in enumerate(EXAMPLES):
            x = grid_x + col * (cell_w + gap)
            fit_text(canvas, example, x + cell_w / 2, header_y, cell_w, 8)

        for row, (method_key, method_label) in enumerate(methods):
            y = page_h - top - 24 - (row + 1) * cell_h - row * gap
            is_pmesa = method_key == "pmesa"
            canvas.setFillColor(colors.HexColor("#EAF2FF") if is_pmesa else colors.HexColor("#F8FAFC"))
            canvas.setStrokeColor(colors.HexColor("#8DB7F4") if is_pmesa else colors.HexColor("#E1E5EA"))
            canvas.roundRect(margin_x, y, label_w - 6, cell_h, 3, fill=1, stroke=1)
            canvas.setFillColor(colors.HexColor("#0B57A4") if is_pmesa else colors.HexColor("#1F2937"))
            fit_text(canvas, method_label, margin_x + (label_w - 6) / 2, y + cell_h / 2 - 3,
                     label_w - 14, 7.5)
            for col, example in enumerate(EXAMPLES, start=1):
                x = grid_x + (col - 1) * (cell_w + gap)
                asset = assets / task_key / f"example_{col}" / f"{method_key}.png"
                if asset.exists():
                    draw_image(canvas, asset, x, y, cell_w, cell_h)
                    canvas.setStrokeColor(colors.HexColor("#D1D5DB"))
                    canvas.rect(x, y, cell_w, cell_h, fill=0, stroke=1)
                else:
                    draw_placeholder(canvas, x, y, cell_w, cell_h)

        canvas.setFillColor(colors.HexColor("#6B7280"))
        canvas.setFont("Helvetica", 6.5)
        canvas.drawString(margin_x, 12,
            "Final figure must use fixed normalization, colormap, opacity, sample IDs, and unmodified model outputs.")
        canvas.showPage()
    canvas.save()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, default=Path("outputs/qualitative"))
    parser.add_argument("--output", type=Path,
                        default=Path("output/pdf/pmesa_qualitative_layout_preview.pdf"))
    parser.add_argument("--final", action="store_true",
                        help="require all assets and remove the preview warning")
    args = parser.parse_args()
    build(args.output, args.assets, final=args.final)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
