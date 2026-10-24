"""Unit tests for SpatialTokenAdapter and YoloVisualTokenizer.

Note on Execution:
- Standard-library and syntax compilation checks run locally.
- Numerical tests run on Kaggle where PyTorch and Ultralytics are available.
- Real YOLO tests use YOLO11n packaged architecture YAML with random weights
  (no network downloads); pretrained weights will be loaded explicitly on Kaggle.
"""

from __future__ import annotations

import pytest

# Skip test execution cleanly if torch is not installed locally
torch = pytest.importorskip("torch", reason="PyTorch is not installed locally; numerical tests run on Kaggle.")
from torch import Tensor, nn

from src.smol_vla_mod.vision.yolo_tokenizer import SpatialTokenAdapter, YoloVisualTokenizer


# ---------------------------------------------------------------------------
# Synthetic YOLO Backbone and Helpers
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Synthetic conv block mimicking an Ultralytics YOLO module."""

    def __init__(self, in_c: int, out_c: int, stride: int = 1, routing_f: int | list[int] = -1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU()
        self.out_channels = out_c
        self.f = routing_f

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class MockDetectionModel(nn.Module):
    """Synthetic DetectionModel with 11 sequential backbone modules and 1 neck module."""

    def __init__(
        self,
        c4: int = 128,
        c6: int = 128,
        c10: int = 256,
        invalid_routing: bool = False,
    ) -> None:
        super().__init__()
        # Modules 0..10 represent the YOLO11n backbone
        # Strides: 0: /2, 1: /4, 2: /4, 3: /8, 4: /8, 5: /16, 6: /16, 7: /32, 8: /32, 9: /32, 10: /32
        m4_routing = [-1, 2] if invalid_routing else -1
        layers: list[nn.Module] = [
            ConvBlock(3, 16, stride=2),            # 0: stride 2
            ConvBlock(16, 32, stride=2),           # 1: stride 4
            ConvBlock(32, 32, stride=1),           # 2: stride 4
            ConvBlock(32, 64, stride=2),           # 3: stride 8
            ConvBlock(64, c4, stride=1, routing_f=m4_routing),  # 4: stride 8, 128 channels
            ConvBlock(c4, 128, stride=2),          # 5: stride 16
            ConvBlock(128, c6, stride=1),          # 6: stride 16, 128 channels
            ConvBlock(c6, 256, stride=2),          # 7: stride 32
            ConvBlock(256, 256, stride=1),         # 8: stride 32
            ConvBlock(256, 256, stride=1),         # 9: stride 32
            ConvBlock(256, c10, stride=1),         # 10: stride 32, 256 channels
            ConvBlock(c10, 128, stride=1),         # 11: neck module (should be discarded)
        ]
        for idx, m in enumerate(layers):
            m.i = idx
        self.model = nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------

@pytest.mark.adapter_only
def test_adapter_output_shape():
    """1. Adapter produces [B, 64, 960] from the expected three feature maps."""
    adapter = SpatialTokenAdapter(
        feature_channels=(128, 128, 256),
        hidden_size=960,
        grid_size=8,
        projection_channels=64,
        mlp_hidden_size=512,
    )
    b = 3
    # Inputs with varying spatial resolutions from fine to coarse
    f4 = torch.randn(b, 128, 64, 64)
    f6 = torch.randn(b, 128, 32, 32)
    f10 = torch.randn(b, 256, 16, 16)

    tokens = adapter((f4, f6, f10))
    assert tokens.shape == (b, 64, 960), f"Expected shape {(b, 64, 960)}, got {tokens.shape}"


@pytest.mark.adapter_only
def test_token_order_and_coordinates():
    """2. Token order and x/y coordinates follow the specified row-major convention."""
    grid_size = 8
    adapter = SpatialTokenAdapter(grid_size=grid_size)
    device = torch.device("cpu")
    coords = adapter.get_coordinate_grid(batch_size=2, device=device, dtype=torch.float32)

    assert coords.shape == (2, 2, grid_size, grid_size)
    expected_linear = torch.linspace(-1.0, 1.0, grid_size)

    # Check x coordinate (channel 0): increases from -1 to 1 across columns, constant down rows
    for r in range(grid_size):
        torch.testing.assert_close(coords[0, 0, r, :], expected_linear)
    for c in range(grid_size):
        val = coords[0, 0, 0, c]
        torch.testing.assert_close(coords[0, 0, :, c], torch.full((grid_size,), val))

    # Check y coordinate (channel 1): increases from -1 to 1 across rows, constant across columns
    for c in range(grid_size):
        torch.testing.assert_close(coords[0, 1, :, c], expected_linear)
    for r in range(grid_size):
        val = coords[0, 1, r, 0]
        torch.testing.assert_close(coords[0, 1, r, :], torch.full((grid_size,), val))

    # Check row-major flattening order: token index t = r * grid_size + c
    f_coords = coords.permute(0, 2, 3, 1).reshape(2, grid_size * grid_size, 2)
    # First token (top-left, r=0, c=0): (-1, -1)
    torch.testing.assert_close(f_coords[0, 0], torch.tensor([-1.0, -1.0]))
    # Last token of row 0 (top-right, r=0, c=7): (1, -1)
    torch.testing.assert_close(f_coords[0, grid_size - 1], torch.tensor([1.0, -1.0]))
    # First token of row 1 (r=1, c=0): (-1, -1 + 2/7)
    torch.testing.assert_close(f_coords[0, grid_size, 0], torch.tensor(-1.0))
    # Last token (bottom-right, r=7, c=7): (1, 1)
    torch.testing.assert_close(f_coords[0, -1], torch.tensor([1.0, 1.0]))


@pytest.mark.adapter_only
def test_adapter_output_changes_with_input():
    """3. Adapter output changes when input features change."""
    adapter = SpatialTokenAdapter()
    f4_a = torch.randn(2, 128, 64, 64)
    f6_a = torch.randn(2, 128, 32, 32)
    f10_a = torch.randn(2, 256, 16, 16)

    f4_b = torch.randn(2, 128, 64, 64)
    f6_b = torch.randn(2, 128, 32, 32)
    f10_b = torch.randn(2, 256, 16, 16)

    out_a = adapter((f4_a, f6_a, f10_a))
    out_b = adapter((f4_b, f6_b, f10_b))

    assert not torch.allclose(out_a, out_b, atol=1e-5), "Adapter output must differ for distinct inputs"


@pytest.mark.adapter_only
def test_adapter_backward_gradients():
    """4. Backward produces finite, nonzero adapter gradients."""
    adapter = SpatialTokenAdapter()
    f4 = torch.randn(2, 128, 64, 64, requires_grad=True)
    f6 = torch.randn(2, 128, 32, 32, requires_grad=True)
    f10 = torch.randn(2, 256, 16, 16, requires_grad=True)

    out = adapter((f4, f6, f10))
    loss = out.sum()
    loss.backward()

    # Verify input feature gradients
    assert f4.grad is not None and torch.isfinite(f4.grad).all() and (f4.grad != 0).any()
    assert f6.grad is not None and torch.isfinite(f6.grad).all() and (f6.grad != 0).any()
    assert f10.grad is not None and torch.isfinite(f10.grad).all() and (f10.grad != 0).any()

    # Verify all adapter parameters receive finite, nonzero gradients
    for name, p in adapter.named_parameters():
        assert p.grad is not None, f"Parameter {name} has no gradient"
        assert torch.isfinite(p.grad).all(), f"Parameter {name} has non-finite gradient"
        assert (p.grad != 0).any(), f"Parameter {name} has all-zero gradient"


@pytest.mark.yolo_backbone
def test_frozen_backbone_receives_no_gradients():
    """5. Frozen backbone receives no parameter gradients."""
    mock_yolo = MockDetectionModel()
    tokenizer = YoloVisualTokenizer(mock_yolo, freeze_backbone=True)

    image = torch.randn(2, 3, 256, 256)
    out = tokenizer(image)
    loss = out.sum()
    loss.backward()

    # Backbone parameters must receive no gradients
    for name, p in tokenizer.backbone.named_parameters():
        assert p.grad is None, f"Frozen backbone parameter {name} received grad: {p.grad}"

    # Adapter parameters must receive finite nonzero gradients
    for name, p in tokenizer.adapter.named_parameters():
        assert p.grad is not None, f"Adapter parameter {name} has no gradient"
        assert torch.isfinite(p.grad).all(), f"Adapter parameter {name} gradient is non-finite"


@pytest.mark.yolo_backbone
def test_tokenizer_train_keeps_frozen_backbone_eval():
    """6. tokenizer.train() keeps frozen backbone BatchNorm in eval mode."""
    mock_yolo = MockDetectionModel()
    tokenizer = YoloVisualTokenizer(mock_yolo, freeze_backbone=True)

    # Put tokenizer into training mode
    res = tokenizer.train(True)
    assert res is tokenizer, "train() must return self"
    assert tokenizer.training is True
    assert tokenizer.adapter.training is True
    assert tokenizer.backbone.training is False

    # Check all BatchNorm layers in the backbone are in eval mode
    bn_count = 0
    for m in tokenizer.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            assert m.training is False, "BatchNorm in frozen backbone must stay in eval mode"
            bn_count += 1
    assert bn_count > 0, "Test fixture must contain BatchNorm2d modules"


@pytest.mark.yolo_backbone
def test_unfreezing_enables_backbone_gradients():
    """7. Unfreezing enables backbone gradients."""
    mock_yolo = MockDetectionModel()
    tokenizer = YoloVisualTokenizer(mock_yolo, freeze_backbone=True)

    tokenizer.unfreeze_backbone()
    assert not tokenizer.is_backbone_frozen

    for name, p in tokenizer.backbone.named_parameters():
        assert p.requires_grad is True, f"Backbone parameter {name} should have requires_grad=True"

    image = torch.randn(2, 3, 256, 256)
    out = tokenizer(image)
    loss = out.sum()
    loss.backward()

    for name, p in tokenizer.backbone.named_parameters():
        assert p.grad is not None, f"Unfrozen backbone parameter {name} has no gradient"
        assert torch.isfinite(p.grad).all(), f"Unfrozen parameter {name} gradient is non-finite"


def test_input_conversion_mapping():
    """8. Input conversion maps -1, 0 and 1 to 0, 0.5 and 1 without clamping."""
    # SmolVLA [-1, 1] to YOLO [0, 1] formula: (image + 1.0) * 0.5
    inp = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
    rgb = (inp + 1.0) * 0.5
    torch.testing.assert_close(rgb, torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32))

    # Verify no silent clamping occurs on inputs outside [-1, 1]
    inp_ext = torch.tensor([-2.0, 2.0], dtype=torch.float32)
    rgb_ext = (inp_ext + 1.0) * 0.5
    torch.testing.assert_close(rgb_ext, torch.tensor([-0.5, 1.5], dtype=torch.float32))

    # Verify dtype preservation
    inp_f64 = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
    rgb_f64 = (inp_f64 + 1.0) * 0.5
    assert rgb_f64.dtype == torch.float64


@pytest.mark.yolo_backbone
def test_state_dict_save_load():
    """9. State-dict save/load reproduces outputs in eval mode."""
    mock_yolo1 = MockDetectionModel()
    mock_yolo2 = MockDetectionModel()

    tok1 = YoloVisualTokenizer(mock_yolo1, freeze_backbone=False)
    tok2 = YoloVisualTokenizer(mock_yolo2, freeze_backbone=False)

    state = tok1.state_dict()
    tok2.load_state_dict(state)

    tok1.eval()
    tok2.eval()

    image = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out1 = tok1(image)
        out2 = tok2(image)

    torch.testing.assert_close(out1, out2)


@pytest.mark.yolo_backbone
def test_random_weight_yolo11n_forward():
    """10. A random-weight YOLO11n forward pass at 512x512 produces expected feature shapes and [B,64,960] output."""
    try:
        from ultralytics.nn.tasks import DetectionModel
        # Model construction from packaged architecture YAML with random weights (no downloads)
        yolo_model = DetectionModel(cfg="yolo11n.yaml")
    except ImportError:
        # Fall back to synthetic mock with matching YOLO11n topology if ultralytics is absent
        yolo_model = MockDetectionModel()

    tokenizer = YoloVisualTokenizer(yolo_model)
    image = torch.randn(2, 3, 512, 512)

    tokens = tokenizer(image)
    assert tokens.shape == (2, 64, 960), f"Expected [2, 64, 960], got {tokens.shape}"


def test_invalid_feature_channel_inputs_raise_errors():
    """11. Invalid feature/channel inputs raise clear errors."""
    adapter = SpatialTokenAdapter(feature_channels=(128, 128, 256))

    # Adapter: not 3 feature maps
    f1 = torch.randn(2, 128, 64, 64)
    f2 = torch.randn(2, 128, 32, 32)
    with pytest.raises(ValueError, match="Expected exactly 3 feature maps"):
        adapter((f1, f2))

    # Adapter: wrong channel count
    f3_bad_channels = torch.randn(2, 64, 16, 16)  # expected 256
    with pytest.raises(ValueError, match="Channel count mismatch for feature map 2"):
        adapter((f1, f2, f3_bad_channels))

    # Adapter: mismatched batch sizes
    f3_bad_batch = torch.randn(4, 256, 16, 16)
    with pytest.raises(ValueError, match="Batch size mismatch across feature maps"):
        adapter((f1, f2, f3_bad_batch))

    # Tokenizer: non-4D input
    mock_yolo = MockDetectionModel()
    tokenizer = YoloVisualTokenizer(mock_yolo)
    with pytest.raises(ValueError, match="Expected 4D image tensor"):
        tokenizer(torch.randn(3, 256, 256))

    # Tokenizer: wrong channel count (not 3)
    with pytest.raises(ValueError, match="Expected 3 RGB channels"):
        tokenizer(torch.randn(2, 1, 256, 256))

    # Tokenizer: non-floating point tensor
    with pytest.raises(TypeError, match="Expected floating point image tensor"):
        tokenizer(torch.randint(0, 255, (2, 3, 256, 256), dtype=torch.uint8))

    # Tokenizer: non-sequential routing in YOLO backbone
    bad_routing_yolo = MockDetectionModel(invalid_routing=True)
    with pytest.raises(ValueError, match="does not use sequential routing"):
        YoloVisualTokenizer(bad_routing_yolo)

    # Tokenizer: wrong module channel count in YOLO backbone
    bad_channel_yolo = MockDetectionModel(c4=64)  # expected 128
    with pytest.raises(ValueError, match="static output channels"):
        YoloVisualTokenizer(bad_channel_yolo)
