"""Lazy vision exports; importing the package does not load PyTorch."""
from importlib import import_module

_EXPORTS = {
    "SpatialTokenAdapter": ".yolo_tokenizer",
    "YoloVisualTokenizer": ".yolo_tokenizer",
    "YoloWorldFeatureExtractor": ".yolo_world_extractor",
    "YoloWorldFeatures": ".yolo_world_extractor",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
