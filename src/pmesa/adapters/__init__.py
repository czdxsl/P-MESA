"""Target-model adapter contracts."""

from .base import AdapterOutput, PMESAAdapter
from .gated import GatedMultimodalScore

__all__ = ["AdapterOutput", "GatedMultimodalScore", "PMESAAdapter"]
