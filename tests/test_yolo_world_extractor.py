"""Real-graph unit and integration tests for YoloWorldFeatureExtractor.

Test Organization:
- Lightweight Validation Tests: Test graph validation, input contracts, and error handling
  using the real random-weight YOLOv8-Worldv2 topology.
- Real Graph Tests: Marked with `@pytest.mark.real_yolo`, these instantiate the actual
  Ultralytics `WorldModel` from packaged `yolov8s-worldv2.yaml` with random weights
  (no network downloads, no pretrained checkpoints).
- CUDA Tests: Marked with `@pytest.mark.cuda`, exercising GPU forward/backward execution.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest

# Skip test suite cleanly if torch is not installed locally
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from smol_vla_mod.vision.yolo_world_extractor import (
    EXPECTED_ROUTING,
    EXPECTED_MODULE_TYPES,
    YoloWorldFeatureExtractor,
    YoloWorldFeatures,
)


# All extractor tests use the real graph. Missing dependencies are errors.
pytestmark = pytest.mark.real_yolo


def build_test_graph(corrupt_layer_idx=None, corrupt_routing=False,
                     corrupt_type=False, include_head=True):
    model = _get_real_ultralytics_world_model()
    layers = list(model.model)
    if not include_head:
        layers = layers[:22]
    if corrupt_layer_idx is not None:
        if corrupt_routing:
            layers[corrupt_layer_idx].f = -999
        if corrupt_type:
            layers[corrupt_layer_idx] = nn.Identity()
            layers[corrupt_layer_idx].f = EXPECTED_ROUTING[corrupt_layer_idx]
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Section 1: Lightweight Validation & Input Contract Tests
# ---------------------------------------------------------------------------

def test_extractor_retains_exactly_22_modules_without_head():
    """Check I: Exactly 22 graph modules are retained; no WorldDetect or CLIP module is registered."""
    synthetic_model = build_test_graph(include_head=True)
    assert len(synthetic_model) == 23

    extractor = YoloWorldFeatureExtractor(synthetic_model)
    assert len(extractor.layers) == 22

    # Verify no detection head or CLIP is registered
    for m in extractor.modules():
        assert "Detect" not in m.__class__.__name__
    for name, _ in extractor.named_modules():
        assert "clip" not in name.lower()
    assert not hasattr(extractor, "clip_model")
    assert not hasattr(extractor, "txt_feats")


def test_extractor_rejects_insufficient_layers():
    """Extractor rejects model sequences with fewer than 22 modules."""
    short_model = nn.Sequential(*list(build_test_graph())[:15])
    with pytest.raises(ValueError, match="requires at least 22 modules"):
        YoloWorldFeatureExtractor(short_model)


def test_extractor_rejects_unsupported_routing():
    """Extractor rejects module sequences with incorrect routing connections."""
    bad_routing_model = build_test_graph(corrupt_layer_idx=11, corrupt_routing=True)
    with pytest.raises(ValueError, match="expected routing 'from'=\\[-1, 6\\]"):
        YoloWorldFeatureExtractor(bad_routing_model)


def test_extractor_rejects_unsupported_module_type():
    """Extractor rejects module sequences with incorrect module classes."""
    bad_type_model = build_test_graph(corrupt_layer_idx=9, corrupt_type=True)
    with pytest.raises(ValueError, match="expected module type 'SPPF'"):
        YoloWorldFeatureExtractor(bad_type_model)


def test_extractor_input_contract_validations():
    """Check K: Invalid batch sizes, guide shapes, image shapes, and input types produce clear errors."""
    synthetic_model = build_test_graph()
    extractor = YoloWorldFeatureExtractor(synthetic_model)

    valid_images = torch.rand(2, 3, 256, 256)
    valid_guides = F.normalize(torch.randn(2, 1, 512), dim=-1)

    # 1. Non-tensor inputs
    with pytest.raises(TypeError, match="Expected images to be a torch.Tensor"):
        extractor("not_a_tensor", valid_guides)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Expected text_embeddings to be a torch.Tensor"):
        extractor(valid_images, "not_a_tensor")  # type: ignore[arg-type]

    # 2. Non-floating point inputs
    with pytest.raises(TypeError, match="Expected floating point image tensor"):
        extractor(torch.randint(0, 255, (2, 3, 256, 256), dtype=torch.uint8), valid_guides)
    with pytest.raises(TypeError, match="Expected floating point text_embeddings"):
        extractor(valid_images, torch.randint(0, 10, (2, 1, 512), dtype=torch.int64))

    # 3. Image shape violations
    with pytest.raises(ValueError, match="Expected 4D image tensor"):
        extractor(torch.rand(3, 256, 256), valid_guides)
    with pytest.raises(ValueError, match="Expected 3 RGB image channels"):
        extractor(torch.rand(2, 1, 256, 256), valid_guides)
    with pytest.raises(ValueError, match="Image height must be a positive multiple of 32"):
        extractor(torch.rand(2, 3, 250, 256), valid_guides)
    with pytest.raises(ValueError, match="Image width must be a positive multiple of 32"):
        extractor(torch.rand(2, 3, 256, 250), valid_guides)

    # 4. Text guide shape violations
    with pytest.raises(ValueError, match="Expected 3D text_embeddings tensor"):
        extractor(valid_images, torch.randn(2, 512))
    with pytest.raises(ValueError, match="Expected single pooled instruction embedding \\(K=1\\)"):
        extractor(valid_images, torch.randn(2, 77, 512))
    with pytest.raises(ValueError, match="Expected 512-dimensional text embeddings"):
        extractor(valid_images, torch.randn(2, 1, 256))

    # 5. Batch size mismatch
    with pytest.raises(ValueError, match="Batch size mismatch"):
        extractor(valid_images, F.normalize(torch.randn(3, 1, 512), dim=-1))


def test_extractor_freezing_and_unfreezing_behavior():
    """Check H (lightweight): Frozen-backbone mode, BatchNorm behavior, and unfreezing work."""
    synthetic_model = build_test_graph()
    extractor = YoloWorldFeatureExtractor(synthetic_model)

    assert not extractor.is_backbone_frozen

    # 1. Freeze backbone
    extractor.freeze_backbone()
    assert extractor.is_backbone_frozen

    for m in extractor.layers[:10]:
        for p in m.parameters():
            assert not p.requires_grad
        assert not m.training

    # 2. train() mode must preserve backbone eval mode
    extractor.train(True)
    for m in extractor.layers[:10]:
        assert not m.training
    for m in extractor.layers[10:]:
        assert m.training

    # 3. Unfreeze backbone
    extractor.unfreeze_backbone()
    assert not extractor.is_backbone_frozen

    for m in extractor.layers[:10]:
        for p in m.parameters():
            assert p.requires_grad
        assert m.training


# ---------------------------------------------------------------------------
# Section 2: Real Ultralytics Graph Tests (Executed on Kaggle GPU)
# ---------------------------------------------------------------------------

def _get_real_ultralytics_world_model() -> Any:
    """Instantiate a real Ultralytics WorldModel using packaged yolov8s-worldv2.yaml with random weights."""
    try:
        from ultralytics.nn.tasks import WorldModel
    except ImportError:
        pytest.fail("Ultralytics is required for real-graph tests but is not installed.")

    # Random-weight model instantiation from packaged YAML; strictly no network downloads
    model = WorldModel(cfg="yolov8s-worldv2.yaml", nc=1, verbose=False)
    return model


@pytest.mark.real_yolo
def test_real_graph_shapes_and_finite_outputs():
    """Check A: Expected channels, spatial shapes and finite outputs at 256x256 and 512x512."""
    world_model = _get_real_ultralytics_world_model()
    extractor = YoloWorldFeatureExtractor(world_model)
    extractor.eval()

    for res in (256, 512):
        images = torch.rand(2, 3, res, res)
        guides = F.normalize(torch.randn(2, 1, 512), dim=-1)

        with torch.no_grad():
            features = extractor(images, guides, return_backbone=True)

        assert features.backbone is not None
        assert len(features.neck) == 3
        assert len(features.backbone) == 3

        # Neck shapes: strides 8, 16, 32; channels 128, 256, 512
        p3_n, p4_n, p5_n = features.neck
        assert p3_n.shape == (2, 128, res // 8, res // 8)
        assert p4_n.shape == (2, 256, res // 16, res // 16)
        assert p5_n.shape == (2, 512, res // 32, res // 32)
        assert torch.isfinite(p3_n).all() and torch.isfinite(p4_n).all() and torch.isfinite(p5_n).all()

        # Backbone shapes: strides 8, 16, 32; channels 128, 256, 512
        p3_b, p4_b, p5_b = features.backbone
        assert p3_b.shape == (2, 128, res // 8, res // 8)
        assert p4_b.shape == (2, 256, res // 16, res // 16)
        assert p5_b.shape == (2, 512, res // 32, res // 32)
        assert torch.isfinite(p3_b).all() and torch.isfinite(p4_b).all() and torch.isfinite(p5_b).all()


@pytest.mark.real_yolo
def test_real_graph_numerical_agreement_with_original_world_model():
    """Check B: Numerical agreement with original WorldModel intermediate outputs for all six taps."""
    world_model = _get_real_ultralytics_world_model()
    world_model.eval()

    # Capture reference features from original WorldModel using temporary forward hooks
    captured_ref: dict[int, Tensor] = {}
    hook_handles = []

    def make_hook(layer_idx: int):
        def _hook(_module: nn.Module, _inputs: Any, output: Tensor):
            captured_ref[layer_idx] = output.detach().clone()
        return _hook

    target_taps = (4, 6, 9, 15, 18, 21)
    try:
        for idx in target_taps:
            h = world_model.model[idx].register_forward_hook(make_hook(idx))
            hook_handles.append(h)

        images = torch.rand(2, 3, 256, 256)
        guides = F.normalize(torch.randn(2, 1, 512), dim=-1)

        with torch.no_grad():
            _ = world_model.predict(images, txt_feats=guides)

    finally:
        for h in hook_handles:
            h.remove()

    # Verify all 6 taps were captured
    assert set(captured_ref.keys()) == set(target_taps)

    # Run feature extractor on identical inputs
    extractor = YoloWorldFeatureExtractor(world_model)
    extractor.eval()

    with torch.no_grad():
        features = extractor(images, guides, return_backbone=True)

    assert features.backbone is not None
    # Compare backbone taps (4, 6, 9)
    torch.testing.assert_close(features.backbone[0], captured_ref[4], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(features.backbone[1], captured_ref[6], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(features.backbone[2], captured_ref[9], rtol=1e-5, atol=1e-5)

    # Compare neck taps (15, 18, 21)
    torch.testing.assert_close(features.neck[0], captured_ref[15], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(features.neck[1], captured_ref[18], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(features.neck[2], captured_ref[21], rtol=1e-5, atol=1e-5)


@pytest.mark.real_yolo
def test_identical_images_with_different_guides():
    """Check C: Identical images with different guides keep backbone identical while neck changes."""
    world_model = _get_real_ultralytics_world_model()
    extractor = YoloWorldFeatureExtractor(world_model)
    extractor.eval()

    images = torch.rand(2, 3, 256, 256)
    guide1 = F.normalize(torch.randn(2, 1, 512), dim=-1)
    guide2 = F.normalize(torch.randn(2, 1, 512), dim=-1)

    with torch.no_grad():
        out1 = extractor(images, guide1, return_backbone=True)
        out2 = extractor(images, guide2, return_backbone=True)

    assert out1.backbone is not None and out2.backbone is not None

    # Backbone must be mathematically identical (zero language conditioning)
    for b1, b2 in zip(out1.backbone, out2.backbone):
        torch.testing.assert_close(b1, b2)

    # At least one neck feature map must change due to text conditioning
    differences = [not torch.allclose(n1, n2, rtol=0, atol=0) for n1, n2 in zip(out1.neck, out2.neck)]
    assert any(differences), "Text guide modulation produced no change across all neck feature maps."


@pytest.mark.real_yolo
def test_swapping_text_guides_swaps_neck_results():
    """Check D: Swapping text guides swaps the corresponding neck results for identical images."""
    world_model = _get_real_ultralytics_world_model()
    extractor = YoloWorldFeatureExtractor(world_model)
    extractor.eval()

    single_img = torch.rand(1, 3, 256, 256)
    images = single_img.repeat(2, 1, 1, 1)

    g0 = F.normalize(torch.randn(1, 1, 512), dim=-1)
    g1 = F.normalize(torch.randn(1, 1, 512), dim=-1)

    guides = torch.cat([g0, g1], dim=0)
    swapped_guides = torch.cat([g1, g0], dim=0)

    with torch.no_grad():
        out = extractor(images, guides)
        swapped_out = extractor(images, swapped_guides)

    for n_orig, n_swap in zip(out.neck, swapped_out.neck):
        torch.testing.assert_close(n_orig[0:1], n_swap[1:2], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(n_orig[1:2], n_swap[0:1], rtol=1e-5, atol=1e-5)


@pytest.mark.real_yolo
def test_batched_output_agrees_with_independent_samples():
    """Check E: Batched output agrees with independently processed samples in eval mode."""
    world_model = _get_real_ultralytics_world_model()
    extractor = YoloWorldFeatureExtractor(world_model)
    extractor.eval()

    images = torch.rand(2, 3, 256, 256)
    guides = F.normalize(torch.randn(2, 1, 512), dim=-1)

    with torch.no_grad():
        batched_out = extractor(images, guides, return_backbone=True)
        s0_out = extractor(images[0:1], guides[0:1], return_backbone=True)
        s1_out = extractor(images[1:2], guides[1:2], return_backbone=True)

    assert batched_out.backbone is not None
    assert s0_out.backbone is not None and s1_out.backbone is not None

    for i in range(3):
        torch.testing.assert_close(batched_out.neck[i][0:1], s0_out.neck[i], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(batched_out.neck[i][1:2], s1_out.neck[i], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(batched_out.backbone[i][0:1], s0_out.backbone[i], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(batched_out.backbone[i][1:2], s1_out.backbone[i], rtol=1e-5, atol=1e-5)


@pytest.mark.real_yolo
def test_gradient_flow_from_neck_loss_to_neck_and_backbone():
    """Check F: A loss on neck outputs produces finite, nonzero gradients in neck and backbone."""
    world_model = _get_real_ultralytics_world_model()
    extractor = YoloWorldFeatureExtractor(world_model)
    extractor.train()

    images = torch.rand(2, 3, 256, 256, requires_grad=True)
    guides = F.normalize(torch.randn(2, 1, 512), dim=-1)

    features = extractor(images, guides)
    loss = sum(m.sum() for m in features.neck)
    loss.backward()

    # Check neck gradient flow
    neck_grads = [p.grad for m in extractor.layers[10:] for p in m.parameters() if p.requires_grad]
    assert len(neck_grads) > 0
    assert all(g is not None and torch.isfinite(g).all() for g in neck_grads)
    assert any((g != 0).any() for g in neck_grads)

    # Check backbone gradient flow
    backbone_grads = [p.grad for m in extractor.layers[:10] for p in m.parameters() if p.requires_grad]
    assert len(backbone_grads) > 0
    assert all(g is not None and torch.isfinite(g).all() for g in backbone_grads)
    assert any((g != 0).any() for g in backbone_grads)


@pytest.mark.real_yolo
def test_gradient_flow_from_backbone_loss_does_not_update_neck():
    """Check G: A loss only on returned backbone features updates backbone without creating neck gradients."""
    world_model = _get_real_ultralytics_world_model()
    extractor = YoloWorldFeatureExtractor(world_model)
    extractor.train()
    extractor.zero_grad(set_to_none=True)

    images = torch.rand(2, 3, 256, 256)
    guides = F.normalize(torch.randn(2, 1, 512), dim=-1)

    features = extractor(images, guides, return_backbone=True)
    assert features.backbone is not None

    loss = sum(m.sum() for m in features.backbone)
    loss.backward()

    # Backbone parameters must have valid nonzero gradients
    backbone_grads = [p.grad for m in extractor.layers[:10] for p in m.parameters() if p.requires_grad]
    assert any(g is not None and (g != 0).any() for g in backbone_grads)

    # Neck parameters must have NO gradients (p.grad is None)
    neck_grads = [p.grad for m in extractor.layers[10:] for p in m.parameters()]
    assert all(g is None for g in neck_grads), "Neck parameters received gradients from pure backbone loss."


@pytest.mark.real_yolo
def test_frozen_backbone_gradient_and_batchnorm_isolation():
    """Check H: Frozen-backbone mode isolates backbone parameters and keeps BatchNorm in eval mode."""
    world_model = _get_real_ultralytics_world_model()
    extractor = YoloWorldFeatureExtractor(world_model)

    extractor.freeze_backbone()
    extractor.train()
    extractor.zero_grad(set_to_none=True)

    # Verify backbone BatchNorms are in eval mode under train()
    for m in extractor.layers[:10]:
        assert not m.training

    images = torch.rand(2, 3, 256, 256)
    guides = F.normalize(torch.randn(2, 1, 512), dim=-1)

    features = extractor(images, guides)
    loss = sum(m.sum() for m in features.neck)
    loss.backward()

    # Backbone parameters must receive no gradients
    backbone_grads = [p.grad for m in extractor.layers[:10] for p in m.parameters()]
    assert all(g is None for g in backbone_grads)

    # Neck parameters must receive valid gradients
    neck_grads = [p.grad for m in extractor.layers[10:] for p in m.parameters() if p.requires_grad]
    assert any(g is not None and (g != 0).any() for g in neck_grads)

    # Unfreeze and verify gradient restoration
    extractor.unfreeze_backbone()
    extractor.zero_grad(set_to_none=True)

    features2 = extractor(images, guides)
    loss2 = sum(m.sum() for m in features2.neck)
    loss2.backward()

    restored_bb_grads = [p.grad for m in extractor.layers[:10] for p in m.parameters() if p.requires_grad]
    assert any(g is not None and (g != 0).any() for g in restored_bb_grads)


@pytest.mark.real_yolo
def test_save_load_state_dict_checkpoint():
    """Check J: Save and load through state_dict reproduces identical outputs in eval mode."""
    world_model1 = _get_real_ultralytics_world_model()
    extractor1 = YoloWorldFeatureExtractor(world_model1)
    extractor1.eval()

    images = torch.rand(2, 3, 256, 256)
    guides = F.normalize(torch.randn(2, 1, 512), dim=-1)

    with torch.no_grad():
        out1 = extractor1(images, guides, return_backbone=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "yolo_world_extractor.pt"
        torch.save(extractor1.state_dict(), ckpt_path)

        # Build fresh extractor
        world_model2 = _get_real_ultralytics_world_model()
        extractor2 = YoloWorldFeatureExtractor(world_model2)
        extractor2.load_state_dict(torch.load(ckpt_path, weights_only=True))
        extractor2.eval()

        with torch.no_grad():
            out2 = extractor2(images, guides, return_backbone=True)

    assert out1.backbone is not None and out2.backbone is not None
    for n1, n2 in zip(out1.neck, out2.neck):
        torch.testing.assert_close(n1, n2)
    for b1, b2 in zip(out1.backbone, out2.backbone):
        torch.testing.assert_close(b1, b2)


@pytest.mark.cuda
def test_cuda_fp32_smoke():
    """Check L: CUDA FP32 forward and backward smoke test."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is unavailable; skipping CUDA smoke test.")

    world_model = _get_real_ultralytics_world_model().cuda()
    extractor = YoloWorldFeatureExtractor(world_model).cuda()
    extractor.train()

    images = torch.rand(2, 3, 256, 256, device="cuda", requires_grad=True)
    guides = F.normalize(torch.randn(2, 1, 512, device="cuda"), dim=-1)

    features = extractor(images, guides, return_backbone=True)
    assert features.backbone is not None

    loss = sum(m.sum() for m in features.neck) + sum(m.sum() for m in features.backbone)
    loss.backward()

    assert images.grad is not None and torch.isfinite(images.grad).all()


def test_rejects_wrong_scale_channels():
    model = _get_real_ultralytics_world_model()
    model.model[4].cv2.conv = nn.Conv2d(model.model[4].cv2.conv.in_channels, 64, 1)
    with pytest.raises(ValueError, match="output channels"):
        YoloWorldFeatureExtractor(model)


def test_default_trainability_and_mode_from_frozen_source():
    model = _get_real_ultralytics_world_model().eval()
    model.requires_grad_(False)
    extractor = YoloWorldFeatureExtractor(model)
    assert all(p.requires_grad for p in extractor.parameters())
    assert all(m.training for m in extractor.modules())


def test_empty_batch_rejected():
    extractor = YoloWorldFeatureExtractor(_get_real_ultralytics_world_model())
    with pytest.raises(ValueError, match="positive batch"):
        extractor(torch.empty(0, 3, 256, 256), torch.empty(0, 1, 512))


def test_copy_ownership():
    model = _get_real_ultralytics_world_model()
    shared = YoloWorldFeatureExtractor(model)
    copied = YoloWorldFeatureExtractor(model, share_parameters=False)
    assert shared.layers[0] is model.model[0]
    assert copied.layers[0] is not model.model[0]
    assert next(copied.parameters()).data_ptr() != next(shared.parameters()).data_ptr()
