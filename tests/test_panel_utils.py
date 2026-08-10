import sys
from pathlib import Path

from PIL import Image


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from panel_utils import PANEL_SIZE, fit_panel


def test_panel_is_landscape_and_fixed_size():
    source = Image.new("RGB", (200, 400), "red")
    panel = fit_panel(source)
    assert panel.size == PANEL_SIZE == (960, 600)
    assert panel.width > panel.height
