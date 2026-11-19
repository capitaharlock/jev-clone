"""asimovia-decision: a byte-level decision model trained from random weights.

from asimovia import load_model, predict

model = load_model("checkpoints/asimovia-decision-v1")
predict(model, [row], "cpu")[0]   # {"choice": ..., "probabilities": ...}
"""

from .model import ByteTokenizer, DecisionModel, ModelConfig, load_model, predict, save

__version__ = "1.0.0"

__all__ = [
    "ByteTokenizer",
    "DecisionModel",
    "ModelConfig",
    "__version__",
    "load_model",
    "predict",
    "save",
]
