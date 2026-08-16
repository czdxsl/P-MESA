"""P-MESA public API."""

__version__ = "1.0.0"

from .attribution import PathAttribution, path_integrated_gradients, relation_path_attribution
from .adapters import GatedMultimodalScore
from .config import PMESAConfig, load_config
from .construction import textual_evidence_from_groups, visual_evidence_from_sam
from .evidence import EvidenceKind, EvidenceUnit
from .explainer import Explanation, PMESAExplainer
from .saliency import multimodal_saliency
from .subset import ObjectiveWeights, greedy_select
from .targets import length_normalized_log_likelihood

__all__ = [
    "EvidenceKind",
    "EvidenceUnit",
    "Explanation",
    "GatedMultimodalScore",
    "ObjectiveWeights",
    "PMESAConfig",
    "PMESAExplainer",
    "PathAttribution",
    "greedy_select",
    "path_integrated_gradients",
    "relation_path_attribution",
    "load_config",
    "length_normalized_log_likelihood",
    "multimodal_saliency",
    "textual_evidence_from_groups",
    "visual_evidence_from_sam",
    "__version__",
]
