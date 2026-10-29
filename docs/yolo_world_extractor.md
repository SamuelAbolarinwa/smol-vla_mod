# YOLO-World Dense Feature Extractor Documentation

**Module:** `src/smol_vla_mod/vision/yolo_world_extractor.py`
**Class:** `YoloWorldFeatureExtractor`
**Output Container:** `YoloWorldFeatures`
**Architecture Context:** Architecture v2 (YOLO-World Visual Pathway with Cross-Modal PAFPN Neck & Auxiliary Relative Depth Tap)

---

## 1. Pinned Source Reference

The extractor is verified and pinned against the following upstream repository revision:
- **Repository:** `third_party/ultralytics`
- **Git Commit SHA:** `aa1f778a75a610f7dd2aa4762df1c25f6a3063d7`
- **Package Version:** `8.4.189`
- **Model Topology:** `cfg/models/v8/yolov8-worldv2.yaml` (YOLOv8s-Worldv2)

---

## 2. Retained Graph & Feature Tap Contract

The feature extractor retains **modules 0 through 21** of the YOLOv8-Worldv2 model graph.
The object detection head (layer 22 `WorldDetect`), the CLIP model, cached offline vocabulary (`txt_feats`), and predictor wrappers are discarded.

| Stream | Tap Layer Indices | Module Classes | Small-Model Channels (`s`) | Spatial Strides | Role in Architecture v2 |
|---|---|---|---|---|---|
| **Neck** | `(15, 18, 21)` | `(C2fAttn, C2fAttn, C2fAttn)` | `(128, 256, 512)` | `(8, 16, 32)` | **Policy Pathway:** Multi-scale feature maps conditioned by the language prompt via `MaxSigmoidAttnBlock`. Input to the Spatial Token Adapter. |
| **Backbone** | `(4, 6, 9)` | `(C2f, C2f, SPPF)` | `(128, 256, 512)` | `(8, 16, 32)` | **Auxiliary Relative Depth Head:** Unconditioned visual features representing scene relative spatial structure. Tapped during training to enforce geometric grounding via shared gradients. |

---

## 3. Input & Output Contract

### 3.1 Forward Signature

```python
def forward(
    self,
    images: torch.Tensor,
    text_embeddings: torch.Tensor,
    *,
    return_backbone: bool = False,
) -> YoloWorldFeatures:
```

### 3.2 Input Specifications

1. **`images`:** Floating RGB tensor of shape `[N, 3, H, W]`.
   - Normalization: Preprocessed into `[0.0, 1.0]`.
   - Resolution: `H` and `W` must be positive integer multiples of 32 (e.g., 256×256, 512×512).
   - No implicit resizing, letterboxing, or BGR conversion is applied inside the extractor.
   - Batch dimension $N$ represents the total flattened observation count across batch and camera views.
2. **`text_embeddings`:** Floating tensor of shape `[N, 1, 512]`.
   - Pretrained compatibility: A single pooled, projected, L2-normalized instruction embedding vector per observation.
   - Batch alignment: $N$ must strictly match `images.shape[0]`. Implicit broadcasting across observations is prohibited.
3. **Devices & Dtypes:**
   - `images.device` and `text_embeddings.device` must match.
   - Dtype: Supports FP32, FP16, and BF16. Standard PyTorch autocast execution (`torch.autocast`) is natively supported.

### 3.3 Output Specification (`YoloWorldFeatures`)

```python
class YoloWorldFeatures(NamedTuple):
    neck: tuple[Tensor, Tensor, Tensor]
    backbone: tuple[Tensor, Tensor, Tensor] | None = None
```

- `neck`: Tuple of 3 tensors from layers 15, 18, 21 with shapes:
  - P3: `[N, 128, H/8, W/8]`
  - P4: `[N, 256, H/16, W/16]`
  - P5: `[N, 512, H/32, W/32]`
- `backbone`: If `return_backbone=True`, tuple of 3 tensors from layers 4, 6, 9 with shapes:
  - P3: `[N, 128, H/8, W/8]`
  - P4: `[N, 256, H/16, W/16]`
  - P5: `[N, 512, H/32, W/32]`
  Otherwise `None`.

---

## 4. Multi-Camera Observation Flattening Example

In robotic manipulation policies with $B$ trajectory episodes and $V$ camera viewpoints (e.g., wrist camera and external camera), camera flattening belongs to the policy wrapper:

```python
# Raw robot batch:
# images: [B, V, 3, H, W]
# task_instruction_embedding: [B, 1, 512] (one prompt per robot episode)

B, V, C, H, W = images.shape

# 1. Flatten images into observation batch N = B * V
flattened_images = images.view(B * V, C, H, W)

# 2. Replicate text guide V times per sample in matching observation order:
# Sample 0 gets cameras 0..V-1, Sample 1 gets cameras 0..V-1, etc.
flattened_guides = (
    task_instruction_embedding.unsqueeze(1)    # [B, 1, 1, 512]
    .expand(-1, V, -1, -1)                      # [B, V, 1, 512]
    .reshape(B * V, 1, 512)                     # [B * V, 1, 512]
)

# 3. Pass to YoloWorldFeatureExtractor
features = extractor(flattened_images, flattened_guides, return_backbone=True)
```

---

## 5. Parameter Ownership & Freezing Behavior

### 5.1 Parameter Ownership

When initialized with `share_parameters=True` (default):
- `self.layers` references the identical module and parameter objects as the supplied `WorldModel`.
- Because PyTorch child modules do not maintain strong references back to their parent container, the caller can safely delete or drop references to the source `WorldModel` (freeing its detection head and criterion) without dangling references or memory leaks.
- When `share_parameters=False`, modules 0..21 are deep-copied into independent parameter tensors.

### 5.2 Visual Backbone Freezing (`freeze_backbone` / `unfreeze_backbone`)

- **`extractor.freeze_backbone()`:**
  - Sets `requires_grad=False` for all parameters in modules 0..9.
  - Puts modules 0..9 in `.eval()` mode.
  - Overridden `extractor.train(mode)` guarantees that backbone BatchNorm layers remain in `.eval()` mode and do not update running statistics during policy fine-tuning.
  - Gradients are NOT unconditionally detached in forward; the computational graph remains intact.
- **`extractor.unfreeze_backbone()`:**
  - Restores `requires_grad=True` on all backbone parameters.
  - Synchronizes backbone module modes with `extractor.training`.

---

## 6. Why the Phase 1 YOLO11n Extractor Cannot Be Used Unchanged

1. **Absence of Cross-Modal Modulation:** YOLO11n is a purely visual backbone. It contains no text-guided attention mechanisms (`C2fAttn`, `MaxSigmoidAttnBlock`).
2. **Single Extraction Stage:** The Phase 1 extractor only tapped three backbone stages. Architecture v2 requires a dual-tap pipeline: unconditioned pre-text features (layers 4, 6, 9) for relative depth supervision and conditioned post-text features (layers 15, 18, 21) for policy action generation.
3. **Dynamic Attention Routing:** YOLO-Worldv2's neck contains top-down and bottom-up skip connections that must receive the per-sample text guide dynamically during the forward pass.

---

## 7. Requirement for Downstream Text Encoders

> [!IMPORTANT]
> The language module / text encoder in `src/smol_vla_mod/language/` must return standard **detached** PyTorch tensors (e.g. via `tensor.detach()`).
> It must **not** return tensors generated under `torch.inference_mode()`, because PyTorch forbids saving inference-mode tensors for backward gradient computation when consumed by downstream trainable layers (such as `C2fAttn`).
