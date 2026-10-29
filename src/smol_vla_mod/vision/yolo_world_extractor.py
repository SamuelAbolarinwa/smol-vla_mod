"""YOLO-World Dense Feature Extractor for Smol-VLAxYOLO Architecture v2.

This module extracts dense visual feature maps from a YOLOv8-Worldv2 model for use in:
1. Policy Spatial Token Adapter (from text-conditioned PAFPN neck layers 15, 18, 21).
2. Auxiliary Depth Head (from unconditioned visual backbone layers 4, 6, 9).

Pinned source reference:
- Repository: third_party/ultralytics
- Commit SHA: aa1f778a75a610f7dd2aa4762df1c25f6a3063d7
- Version: 8.4.189
- Config: cfg/models/v8/yolov8-worldv2.yaml
"""

from __future__ import annotations

import copy
from typing import Any, NamedTuple

try:
    import torch
    from torch import Tensor, nn
except ImportError:
    # Allow imports in environments without PyTorch for packaging/metadata inspection
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[assignment,misc]
    nn = Any  # type: ignore[assignment]


# Expected routing specification ('from' indices) for YOLOv8s-Worldv2 modules 0..21
EXPECTED_ROUTING: dict[int, int | list[int]] = {
    0: -1,
    1: -1,
    2: -1,
    3: -1,
    4: -1,          # Backbone Stage 3 (P3/8)
    5: -1,
    6: -1,          # Backbone Stage 4 (P4/16)
    7: -1,
    8: -1,
    9: -1,          # Backbone Stage 5 (P5/32, post-SPPF)
    10: -1,         # Top-down Upsample (P5 -> P4)
    11: [-1, 6],    # Concat with Backbone Stage 4
    12: -1,         # Top-down C2fAttn (P4)
    13: -1,         # Top-down Upsample (P4 -> P3)
    14: [-1, 4],    # Concat with Backbone Stage 3
    15: -1,         # Neck Stage 3 (P3/8)
    16: 15,         # Bottom-up Conv from Layer 15 (P3 -> P4)
    17: [-1, 12],   # Concat with Top-down Neck Stage 4
    18: -1,         # Neck Stage 4 (P4/16)
    19: -1,         # Bottom-up Conv (P4 -> P5)
    20: [-1, 9],    # Concat with Backbone Stage 5
    21: -1,         # Neck Stage 5 (P5/32)
}

EXPECTED_MODULE_TYPES: dict[int, str] = {
    0: "Conv",
    1: "Conv",
    2: "C2f",
    3: "Conv",
    4: "C2f",
    5: "Conv",
    6: "C2f",
    7: "Conv",
    8: "C2f",
    9: "SPPF",
    10: "Upsample",
    11: "Concat",
    12: "C2fAttn",
    13: "Upsample",
    14: "Concat",
    15: "C2fAttn",
    16: "Conv",
    17: "Concat",
    18: "C2fAttn",
    19: "Conv",
    20: "Concat",
    21: "C2fAttn",
}

BACKBONE_TAP_INDICES: tuple[int, int, int] = (4, 6, 9)
NECK_TAP_INDICES: tuple[int, int, int] = (15, 18, 21)
ATTN_MODULE_INDICES: frozenset[int] = frozenset({12, 15, 18, 21})


class YoloWorldFeatures(NamedTuple):
    """Named output container for YOLO-World feature representations.

    Attributes:
        neck: Tuple of 3 multi-scale feature maps from neck layers 15, 18, 21
            (P3/8, P4/16, P5/32) modulated by task language.
            Expected small-model channels: (128, 256, 512).
            Expected spatial strides: (8, 16, 32).
        backbone: Optional tuple of 3 multi-scale feature maps from backbone layers
            4, 6, 9 (P3/8, P4/16, P5/32) before language conditioning.
            Expected small-model channels: (128, 256, 512).
            Expected spatial strides: (8, 16, 32).
            None if return_backbone=False.
    """

    neck: tuple[Tensor, Tensor, Tensor]
    backbone: tuple[Tensor, Tensor, Tensor] | None = None


class YoloWorldFeatureExtractor(nn.Module if torch is not None else object):  # type: ignore[misc]
    """Extracts dense feature maps from a YOLOv8-Worldv2 model graph (modules 0..21).

    This extractor isolates the visual backbone and language-guided neck of YOLO-World,
    discarding the object detection head (layer 22 WorldDetect), CLIP text model,
    cached offline vocabulary, predictor, and training wrappers.

    Parameter Sharing and Ownership:
    - By default (`share_parameters=True`), the extractor directly adopts the module
      instances from the supplied source model. In PyTorch, child modules maintain no
      circular strong references to their parent container. Consequently, the caller
      can safely delete or let the outer `WorldModel` go out of scope, releasing
      unused components (detection head, criterion, cached text embeddings) while the
      extractor retains the active parameter tensors.
    - If `share_parameters=False`, modules are deep-copied into independent parameter
      objects.

    Trainability and Freezing:
    - By default, all 22 modules are trainable.
    - Calling `freeze_backbone()` sets `requires_grad=False` on backbone parameters
      (modules 0..9) and forces backbone BatchNorm layers to remain in evaluation mode
      even when `extractor.train()` is called.
    - Calling `unfreeze_backbone()` restores `requires_grad=True` and synchronizes
      backbone modules with the extractor's current training mode.
    """

    def __init__(self, source_model: Any, *, share_parameters: bool = True) -> None:
        """Initialize the feature extractor from an already-created Ultralytics WorldModel.

        Args:
            source_model: An initialized Ultralytics WorldModel instance or a module
                exposing the underlying sequence of graph layers.
            share_parameters: If True (default), reuse the module and parameter instances
                from the source model. If False, deep-copy modules 0..21.

        Raises:
            TypeError: If PyTorch is not available or source_model is not recognized.
            ValueError: If the source model does not match the pinned YOLOv8s-Worldv2 graph.
        """
        if torch is None:
            raise TypeError("PyTorch must be installed to instantiate YoloWorldFeatureExtractor.")

        super().__init__()

        raw_layers = self._extract_raw_layers(source_model)
        self._validate_graph(raw_layers)

        retained = raw_layers[:22]
        if not share_parameters:
            retained = copy.deepcopy(retained)

        self.layers = nn.ModuleList(retained)
        self._backbone_frozen: bool = False
        self.requires_grad_(True)
        self.train(True)

    @staticmethod
    def _extract_raw_layers(source_model: Any) -> list[nn.Module]:
        """Extract the sequence of graph layers from various model container types."""
        if hasattr(source_model, "model") and isinstance(source_model.model, (torch.nn.Sequential, torch.nn.ModuleList)):
            return list(source_model.model)
        elif hasattr(source_model, "model") and hasattr(source_model.model, "model") and isinstance(source_model.model.model, (torch.nn.Sequential, torch.nn.ModuleList)):
            return list(source_model.model.model)
        elif isinstance(source_model, (torch.nn.Sequential, torch.nn.ModuleList)):
            return list(source_model)
        elif isinstance(source_model, list) and all(isinstance(m, nn.Module) for m in source_model):
            return list(source_model)
        raise TypeError(
            f"Expected source_model to be a WorldModel or contain a sequence of layers, got {type(source_model).__name__}"
        )

    @classmethod
    def _validate_graph(cls, raw_layers: list[nn.Module]) -> None:
        """Validate that the supplied module sequence conforms to the pinned YOLOv8s-Worldv2 graph."""
        if len(raw_layers) < 22:
            raise ValueError(
                f"YOLOv8-Worldv2 requires at least 22 modules (0..21), but source model has {len(raw_layers)}."
            )

        if len(raw_layers) > 23:
            raise ValueError("Expected exactly 22 retained layers or 23 layers including WorldDetect.")

        # Validate module types and routing for modules 0..21
        for i in range(22):
            m = raw_layers[i]
            actual_type = m.__class__.__name__
            expected_type = EXPECTED_MODULE_TYPES[i]
            if actual_type != expected_type:
                raise ValueError(
                    f"Graph mismatch at layer {i}: expected module type '{expected_type}', got '{actual_type}'."
                )

            actual_routing = getattr(m, "f", None)
            expected_routing = EXPECTED_ROUTING[i]
            if actual_routing != expected_routing:
                raise ValueError(
                    f"Graph mismatch at layer {i}: expected routing 'from'={expected_routing}, got '{actual_routing}'."
                )

        # Validate the output convolutions at all six public taps. Other scales
        # share the same routing and class names, but violate our channel contract.
        for index, channels in ((4, 128), (6, 256), (9, 512),
                                (15, 128), (18, 256), (21, 512)):
            conv = getattr(getattr(raw_layers[index], "cv2", None), "conv", None)
            actual = getattr(conv, "out_channels", None)
            if actual != channels:
                raise ValueError(f"Layer {index} output channels: expected {channels}, got {actual}.")

        if len(raw_layers) == 23 and raw_layers[22].__class__.__name__ != "WorldDetect":
            raise ValueError("Expected layer 22 to be WorldDetect.")

    def freeze_backbone(self) -> None:
        """Freeze the visual backbone (modules 0..9).

        Sets requires_grad=False on all backbone parameters. Backbone BatchNorm
        modules are switched to eval mode and will not update running statistics
        under extractor.train().
        """
        self._backbone_frozen = True
        for m in self.layers[:10]:
            for p in m.parameters():
                p.requires_grad = False
            m.eval()

    def unfreeze_backbone(self) -> None:
        """Unfreeze the visual backbone (modules 0..9).

        Restores requires_grad=True on all backbone parameters. Synchronizes
        backbone module modes with the current extractor training mode.
        """
        self._backbone_frozen = False
        for m in self.layers[:10]:
            for p in m.parameters():
                p.requires_grad = True
        if self.training:
            for m in self.layers[:10]:
                m.train(True)
        else:
            for m in self.layers[:10]:
                m.eval()

    @property
    def is_backbone_frozen(self) -> bool:
        """Return True if the visual backbone (modules 0..9) is currently frozen."""
        return self._backbone_frozen

    def train(self, mode: bool = True) -> YoloWorldFeatureExtractor:
        """Set the training mode for the extractor.

        If the visual backbone is frozen, its modules (and specifically BatchNorm layers)
        are preserved in eval mode even when mode=True.
        """
        super().train(mode)
        if self._backbone_frozen and mode:
            for m in self.layers[:10]:
                m.eval()
        return self

    def forward(
        self,
        images: Tensor,
        text_embeddings: Tensor,
        *,
        return_backbone: bool = False,
    ) -> YoloWorldFeatures:
        """Execute the YOLO-Worldv2 dense feature extraction graph (modules 0..21).

        Args:
            images: Floating RGB tensor [N, 3, H, W] in range [0, 1].
                H and W must be positive integer multiples of 32.
            text_embeddings: Floating tensor [N, 1, 512].
                One pooled, projected, L2-normalized full-instruction embedding per image.
            return_backbone: If True, also return unconditioned visual backbone maps
                from layers 4, 6, 9. Default False.

        Returns:
            YoloWorldFeatures named tuple containing:
                - neck: tuple of 3 feature maps (P3, P4, P5) from layers 15, 18, 21.
                - backbone: tuple of 3 feature maps (P3, P4, P5) from layers 4, 6, 9
                  if return_backbone=True; otherwise None.

        Raises:
            TypeError: If inputs are not floating-point torch.Tensors.
            ValueError: If dimensions, shapes, batch sizes, or devices do not match contract.
        """
        # Validate images
        if not isinstance(images, torch.Tensor):
            raise TypeError(f"Expected images to be a torch.Tensor, got {type(images).__name__}")
        if not images.is_floating_point():
            raise TypeError(f"Expected floating point image tensor, got dtype {images.dtype}")
        if images.ndim != 4:
            raise ValueError(f"Expected 4D image tensor [N, 3, H, W], got shape {list(images.shape)}")

        n_img, c_img, h_img, w_img = images.shape
        if n_img <= 0:
            raise ValueError("Expected a positive batch size.")
        if c_img != 3:
            raise ValueError(f"Expected 3 RGB image channels, got {c_img} in shape {list(images.shape)}")
        if h_img <= 0 or h_img % 32 != 0:
            raise ValueError(f"Image height must be a positive multiple of 32, got H={h_img}")
        if w_img <= 0 or w_img % 32 != 0:
            raise ValueError(f"Image width must be a positive multiple of 32, got W={w_img}")

        # Validate text embeddings
        if not isinstance(text_embeddings, torch.Tensor):
            raise TypeError(f"Expected text_embeddings to be a torch.Tensor, got {type(text_embeddings).__name__}")
        if not text_embeddings.is_floating_point():
            raise TypeError(f"Expected floating point text_embeddings, got dtype {text_embeddings.dtype}")
        if text_embeddings.ndim != 3:
            raise ValueError(
                f"Expected 3D text_embeddings tensor [N, 1, 512], got shape {list(text_embeddings.shape)}"
            )

        n_txt, k_txt, d_txt = text_embeddings.shape
        if k_txt != 1:
            raise ValueError(
                f"Expected single pooled instruction embedding (K=1), got K={k_txt} in shape {list(text_embeddings.shape)}"
            )
        if d_txt != 512:
            raise ValueError(
                f"Expected 512-dimensional text embeddings, got dimension {d_txt} in shape {list(text_embeddings.shape)}"
            )

        # Batch and device contract
        if n_img != n_txt:
            raise ValueError(
                f"Batch size mismatch: images have N={n_img}, text_embeddings have N={n_txt}. "
                "One instruction per image observation is strictly required without implicit broadcasting."
            )
        if images.device != text_embeddings.device:
            raise ValueError(
                f"Device mismatch: images on {images.device}, text_embeddings on {text_embeddings.device}"
            )

        # Execute routing for modules 0..21
        y: list[Tensor | None] = [None] * 22
        x: Any = images

        for i, m in enumerate(self.layers):
            f = m.f
            if f != -1:
                x = y[f] if isinstance(f, int) else [x if j == -1 else y[j] for j in f]  # type: ignore[index]

            if i in ATTN_MODULE_INDICES:
                x = m(x, text_embeddings)
            else:
                x = m(x)

            y[i] = x

        neck = (y[15], y[18], y[21])
        backbone = (y[4], y[6], y[9]) if return_backbone else None

        return YoloWorldFeatures(neck=neck, backbone=backbone)  # type: ignore[arg-type]
