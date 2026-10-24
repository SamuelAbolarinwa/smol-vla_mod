"""Standalone YOLO visual tokenizer for SmolVLA integration.

This module provides:
1. SpatialTokenAdapter: Multi-scale pooling, coordinate-augmented projection, and MLP tokenization.
2. YoloVisualTokenizer: Owned YOLO11n backbone extraction (modules 0..10) and token generation.
"""

from __future__ import annotations

from typing import Any, Sequence
import torch
from torch import Tensor, nn


def _is_autocast_active() -> bool:
    """Check if PyTorch automatic mixed precision (autocast) is currently active."""
    try:
        return torch.is_autocast_enabled()
    except (AttributeError, TypeError):
        try:
            return torch.is_autocast_enabled("cuda") or torch.is_autocast_enabled("cpu")
        except Exception:
            return False


def _inspect_module_out_channels(module: nn.Module) -> int | None:
    """Inspect a module to extract its output channel count if statically exposed.

    Supports standard PyTorch Conv2d, Ultralytics Conv, C3k2, C2f, SPPF, etc.
    """
    if hasattr(module, "out_channels") and isinstance(module.out_channels, int):
        return module.out_channels
    if hasattr(module, "c2") and isinstance(module.c2, int):
        return module.c2
    conv_layer = getattr(module, "conv", None)
    if conv_layer is not None and hasattr(conv_layer, "out_channels") and isinstance(conv_layer.out_channels, int):
        return conv_layer.out_channels
    cv2_layer = getattr(module, "cv2", None)
    if cv2_layer is not None:
        if hasattr(cv2_layer, "out_channels") and isinstance(cv2_layer.out_channels, int):
            return cv2_layer.out_channels
        cv2_conv = getattr(cv2_layer, "conv", None)
        if cv2_conv is not None and hasattr(cv2_conv, "out_channels") and isinstance(cv2_conv.out_channels, int):
            return cv2_conv.out_channels
    return None


class SpatialTokenAdapter(nn.Module):
    """Adapts multi-scale 2D visual feature maps into a 1D sequence of visual tokens.

    Takes 3 feature maps (ordered fine to coarse), adaptively pools each to a fixed grid size,
    projects each to a uniform channel dimension with 1x1 convolutions, appends 2D spatial
    coordinate channels [-1, 1], flattens in row-major order, and applies an MLP.

    Output tokens are returned unscaled (without output LayerNorm, L2 normalisation,
    or sqrt(hidden_size) scaling).
    """

    def __init__(
        self,
        feature_channels: Sequence[int] = (128, 128, 256),
        hidden_size: int = 960,
        grid_size: int = 8,
        projection_channels: int = 64,
        mlp_hidden_size: int = 512,
    ) -> None:
        super().__init__()
        if len(feature_channels) != 3:
            raise ValueError(
                f"Expected exactly 3 feature channel specifications (fine, mid, coarse), "
                f"got {len(feature_channels)}: {feature_channels}"
            )
        if any(c <= 0 for c in feature_channels):
            raise ValueError(f"All feature channels must be positive, got {feature_channels}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if grid_size <= 0:
            raise ValueError(f"grid_size must be positive, got {grid_size}")
        if projection_channels <= 0:
            raise ValueError(f"projection_channels must be positive, got {projection_channels}")
        if mlp_hidden_size <= 0:
            raise ValueError(f"mlp_hidden_size must be positive, got {mlp_hidden_size}")

        self.feature_channels = tuple(feature_channels)
        self.hidden_size = hidden_size
        self.grid_size = grid_size
        self.projection_channels = projection_channels
        self.mlp_hidden_size = mlp_hidden_size

        # Spatial pooling to target grid resolution
        self.pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))

        # Separate trainable 1x1 convolutions per feature map
        self.projections = nn.ModuleList([
            nn.Conv2d(in_channels=c, out_channels=projection_channels, kernel_size=1, bias=True)
            for c in self.feature_channels
        ])

        # Feature channels (3 * projection_channels) + 2 coordinate channels (x, y)
        in_mlp_dim = 3 * projection_channels + 2

        self.norm = nn.LayerNorm(in_mlp_dim)
        self.fc1 = nn.Linear(in_mlp_dim, mlp_hidden_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_hidden_size, hidden_size)

    def get_coordinate_grid(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Generate normalized 2D coordinate grid channels [B, 2, grid_size, grid_size].

        - x increases linearly from -1.0 to 1.0 across columns (width).
        - y increases linearly from -1.0 to 1.0 across rows (height).
        - Generated on the specified device and dtype.
        - Broadcast across the batch dimension.
        """
        y = torch.linspace(-1.0, 1.0, self.grid_size, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, self.grid_size, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        # Stack coordinates: channel 0 is x, channel 1 is y
        coords = torch.stack([grid_x, grid_y], dim=0)  # [2, grid_size, grid_size]
        return coords.unsqueeze(0).expand(batch_size, -1, -1, -1)

    def forward(self, feature_maps: tuple[Tensor, Tensor, Tensor] | list[Tensor]) -> Tensor:
        """Forward pass adapting three feature maps into spatial tokens.

        Args:
            feature_maps: Tuple or list of 3 feature tensors [B, C_i, H_i, W_i],
                          ordered fine to coarse.

        Returns:
            Tensor of shape [B, grid_size * grid_size, hidden_size] (e.g. [B, 64, 960]).
        """
        if not isinstance(feature_maps, (tuple, list)):
            raise TypeError(
                f"Expected feature_maps to be a tuple or list of 3 tensors, got {type(feature_maps).__name__}"
            )
        if len(feature_maps) != 3:
            raise ValueError(
                f"Expected exactly 3 feature maps (fine, mid, coarse), got {len(feature_maps)}"
            )

        first_feat = feature_maps[0]
        if not isinstance(first_feat, torch.Tensor):
            raise TypeError(f"Feature map 0 must be a torch.Tensor, got {type(first_feat).__name__}")
        if first_feat.ndim != 4:
            raise ValueError(
                f"Feature map 0 must be 4D [B, C, H, W], got shape {first_feat.shape}"
            )

        batch_size = first_feat.shape[0]
        device = first_feat.device
        dtype = first_feat.dtype

        projected_maps: list[Tensor] = []
        for i, (feat, exp_c, proj) in enumerate(
            zip(feature_maps, self.feature_channels, self.projections, strict=True)
        ):
            if not isinstance(feat, torch.Tensor):
                raise TypeError(f"Feature map {i} must be a torch.Tensor, got {type(feat).__name__}")
            if feat.ndim != 4:
                raise ValueError(
                    f"Feature map {i} must be 4D [B, C, H, W], got shape {feat.shape}"
                )
            if feat.shape[0] != batch_size:
                raise ValueError(
                    f"Batch size mismatch across feature maps: map 0 has batch size {batch_size}, "
                    f"but map {i} has batch size {feat.shape[0]}"
                )
            if feat.shape[1] != exp_c:
                raise ValueError(
                    f"Channel count mismatch for feature map {i}: expected {exp_c} channels, "
                    f"got {feat.shape[1]} (tensor shape {feat.shape})"
                )
            if feat.device != device:
                raise ValueError(
                    f"Device mismatch across feature maps: map 0 is on {device}, but map {i} is on {feat.device}"
                )
            if feat.dtype != dtype:
                raise ValueError(
                    f"Dtype mismatch across feature maps: map 0 has {dtype}, but map {i} has {feat.dtype}"
                )

            # Adaptive pool to [grid_size, grid_size] and 1x1 convolution projection
            pooled = self.pool(feat)
            proj_feat = proj(pooled)
            projected_maps.append(proj_feat)

        # Concatenate projected feature maps along channel dimension: [B, 3 * proj_c, grid_size, grid_size]
        f_concat = torch.cat(projected_maps, dim=1)

        # Append coordinate channels: [B, 3 * proj_c + 2, grid_size, grid_size]
        coords = self.get_coordinate_grid(batch_size, device=device, dtype=dtype)
        f_with_coords = torch.cat([f_concat, coords], dim=1)

        # Flatten spatial dimensions in row-major order: [B, C, H, W] -> [B, H*W, C]
        b, c, h, w = f_with_coords.shape
        tokens = f_with_coords.permute(0, 2, 3, 1).reshape(b, h * w, c)

        # Differentiable MLP token projection
        x = self.norm(tokens)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        # Return unscaled tokens (no output LayerNorm, L2 norm, or sqrt(hidden_size))
        return x


class YoloVisualTokenizer(nn.Module):
    """Standalone YOLO visual tokenizer for SmolVLA.

    Extracts and owns only modules 0 through 10 of a supplied YOLO DetectionModel.
    Executes the backbone directly as PyTorch modules (without calling predict(), NMS,
    or detection postprocessing). Feature maps from modules 4, 6, and 10 are collected
    and mapped into visual tokens via SpatialTokenAdapter.

    By default, the YOLO backbone is frozen (parameters requires_grad=False) and maintained
    in eval mode (including its BatchNorm layers) even when tokenizer.train() is invoked.
    """

    def __init__(
        self,
        detection_model: Any,
        selected_modules: tuple[int, int, int] = (4, 6, 10),
        expected_channels: tuple[int, int, int] = (128, 128, 256),
        expected_strides: tuple[int, int, int] = (8, 16, 32),
        grid_size: int = 8,
        hidden_size: int = 960,
        projection_channels: int = 64,
        mlp_hidden_size: int = 512,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.selected_modules = tuple(selected_modules)
        self.expected_channels = tuple(expected_channels)
        self.expected_strides = tuple(expected_strides)
        self.grid_size = grid_size
        self.hidden_size = hidden_size
        self.projection_channels = projection_channels
        self.mlp_hidden_size = mlp_hidden_size

        if len(self.selected_modules) != 3:
            raise ValueError(
                f"Expected exactly 3 selected module indices, got {len(self.selected_modules)}: {self.selected_modules}"
            )
        if len(self.expected_channels) != 3:
            raise ValueError(
                f"Expected exactly 3 expected channel counts, got {len(self.expected_channels)}: {self.expected_channels}"
            )
        if len(self.expected_strides) != 3:
            raise ValueError(
                f"Expected exactly 3 expected strides, got {len(self.expected_strides)}: {self.expected_strides}"
            )

        # Extract modules from supplied YOLO DetectionModel
        if hasattr(detection_model, "model") and isinstance(
            detection_model.model, (nn.Sequential, nn.ModuleList, list, tuple)
        ):
            raw_modules = list(detection_model.model)
        elif isinstance(detection_model, (nn.Sequential, nn.ModuleList)):
            raw_modules = list(detection_model)
        elif isinstance(detection_model, (list, tuple)):
            raw_modules = list(detection_model)
        else:
            raise TypeError(
                f"detection_model must be an Ultralytics DetectionModel or an nn.Sequential/nn.ModuleList, "
                f"got {type(detection_model).__name__}"
            )

        max_module_idx = max(self.selected_modules)
        if len(raw_modules) <= max_module_idx:
            raise ValueError(
                f"Supplied YOLO model must contain at least {max_module_idx + 1} modules (0 through {max_module_idx}), "
                f"but only found {len(raw_modules)} modules."
            )

        # Construct owned backbone containing only modules 0 through max_module_idx (e.g. 0..10)
        # Note: Neither the full DetectionModel nor neck/head modules are retained.
        backbone_layers: list[nn.Module] = []
        for idx in range(max_module_idx + 1):
            m = raw_modules[idx]
            if not isinstance(m, nn.Module):
                raise TypeError(f"Module at index {idx} is not an nn.Module (type: {type(m).__name__})")

            # Validate sequential routing: modules 0..10 must take input sequentially from the previous module
            if hasattr(m, "f"):
                f = m.f
                is_sequential = (f == -1) or (f == idx - 1) or (f == [-1])
                if not is_sequential:
                    raise ValueError(
                        f"Backbone module {idx} does not use sequential routing (routing f={f}). "
                        f"Expected sequential routing (f=-1)."
                    )
            backbone_layers.append(m)

        # Validate static module output channels if exposed by the modules
        for sel_idx, exp_c in zip(self.selected_modules, self.expected_channels, strict=True):
            mod = backbone_layers[sel_idx]
            ch = _inspect_module_out_channels(mod)
            if ch is not None and ch != exp_c:
                raise ValueError(
                    f"Module {sel_idx} static output channels ({ch}) do not match expected ({exp_c}) "
                    f"for the configured YOLO backbone (e.g. YOLO11n)."
                )

        self.backbone = nn.ModuleList(backbone_layers)

        # Initialize spatial token adapter
        self.adapter = SpatialTokenAdapter(
            feature_channels=self.expected_channels,
            hidden_size=hidden_size,
            grid_size=grid_size,
            projection_channels=projection_channels,
            mlp_hidden_size=mlp_hidden_size,
        )

        # Freezing state management
        self._backbone_frozen: bool = False
        if freeze_backbone:
            self.freeze_backbone()

    @property
    def is_backbone_frozen(self) -> bool:
        """Return True if the YOLO backbone parameters are frozen."""
        return self._backbone_frozen

    def freeze_backbone(self) -> None:
        """Freeze all YOLO backbone parameters and lock backbone modules into eval mode."""
        self._backbone_frozen = True
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def unfreeze_backbone(self) -> None:
        """Unfreeze YOLO backbone parameters and restore training mode if tokenizer is training."""
        self._backbone_frozen = False
        for p in self.backbone.parameters():
            p.requires_grad = True
        if self.training:
            self.backbone.train()
        else:
            self.backbone.eval()

    def train(self, mode: bool = True) -> YoloVisualTokenizer:
        """Set training mode.

        When the backbone is frozen, its parameters and BatchNorm layers are kept in eval mode
        even when the tokenizer is switched to training mode.
        """
        super().train(mode)
        if self._backbone_frozen:
            self.backbone.eval()
        return self

    def forward(self, image: Tensor) -> Tensor:
        """Forward pass converting preprocessed SmolVLA camera images to visual tokens.

        Args:
            image: Float tensor of shape [B, 3, H, W] normalized in [-1.0, 1.0] (SmolVLA convention).

        Returns:
            Unscaled visual tokens of shape [B, grid_size * grid_size, hidden_size] (e.g. [B, 64, 960]).
        """
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Expected image to be a torch.Tensor, got {type(image).__name__}")
        if image.ndim != 4:
            raise ValueError(f"Expected 4D image tensor [B, C, H, W], got shape {image.shape}")
        if image.shape[1] != 3:
            raise ValueError(f"Expected 3 RGB channels at dim 1, got {image.shape[1]} channels")
        if not image.is_floating_point():
            raise TypeError(f"Expected floating point image tensor, got dtype {image.dtype}")

        # Recover [0.0, 1.0] pixel range from SmolVLA [-1.0, 1.0] representation.
        # Do not apply another resize, letterbox, RGB/BGR conversion, or ImageNet normalisation.
        # Do not silently clamp invalid input.
        rgb = (image + 1.0) * 0.5

        # Maintain consistent parameter/input dtype outside autocast without forcing FP16 or BF16
        backbone_param = next(self.backbone.parameters(), None)
        if backbone_param is not None and rgb.dtype != backbone_param.dtype:
            if not _is_autocast_active():
                rgb = rgb.to(dtype=backbone_param.dtype)

        b, c, h, w = rgb.shape

        # Execute backbone directly as sequential PyTorch modules
        x = rgb
        features: dict[int, Tensor] = {}
        for i, module in enumerate(self.backbone):
            x = module(x)
            if i in self.selected_modules:
                features[i] = x

        collected_maps = tuple(features[i] for i in self.selected_modules)

        # Validate collected feature map channels and spatial strides against input dimensions
        for idx, feat, exp_c, exp_stride in zip(
            self.selected_modules, collected_maps, self.expected_channels, self.expected_strides, strict=True
        ):
            if feat.shape[1] != exp_c:
                raise ValueError(
                    f"Backbone module {idx} channel mismatch: expected {exp_c}, got {feat.shape[1]}"
                )
            exp_h = h // exp_stride
            exp_w = w // exp_stride
            if feat.shape[2] != exp_h or feat.shape[3] != exp_w:
                raise ValueError(
                    f"Backbone module {idx} spatial size mismatch: expected stride {exp_stride} "
                    f"({exp_h}x{exp_w}), got {feat.shape[2]}x{feat.shape[3]} for input {h}x{w}"
                )

        # Adapter maps multi-scale features to visual tokens
        tokens = self.adapter(collected_maps)
        return tokens
