# Upstream LeRobot SmolVLA Inspection Report

**Date / Timestamp:** 2026-10-24 08:22 UTC+1
**Candidate Upstream Checkout:** `third_party/lerobot`
**Upstream Repository:** `https://github.com/huggingface/lerobot.git`
**Inspected Commit SHA:** `d3793b7e36522321c1c455f0dc4826db3793dceb`
**Commit Message:** `feat(fastwam): add Faster-WAM (2.5-5.3x faster policy inference on A100) (#5049)`
**Exclusion Status:** `third_party/` is excluded via parent `.gitignore` (verified).

---

## 1. Verified Source Files and Key Symbols

All paths and symbols were located via filesystem and source inspection within `third_party/lerobot`:

| Component | File Path | Key Symbols |
|---|---|---|
| **SmolVLA Configuration** | `src/lerobot/policies/smolvla/configuration_smolvla.py` | [SmolVLAConfig](file:///home/abolarinwa-sa/Documents/Smol-VLAxYOLO/third_party/lerobot/src/lerobot/policies/smolvla/configuration_smolvla.py#L26) |
| **SmolVLA Policy & Flow Matching** | `src/lerobot/policies/smolvla/modeling_smolvla.py` | [SmolVLAPolicy](file:///home/abolarinwa-sa/Documents/Smol-VLAxYOLO/third_party/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py#L144), [VLAFlowMatching](file:///home/abolarinwa-sa/Documents/Smol-VLAxYOLO/third_party/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py#L519) |
| **VLM Backbone + Action Expert** | `src/lerobot/policies/smolvla/smolvlm_with_expert.py` | [SmolVLMWithExpertModel](file:///home/abolarinwa-sa/Documents/Smol-VLAxYOLO/third_party/lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py#L74), [apply_rope](file:///home/abolarinwa-sa/Documents/Smol-VLAxYOLO/third_party/lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py#L41) |
| **Processor & Pipelines** | `src/lerobot/policies/smolvla/processor_smolvla.py` | [make_smolvla_pre_post_processors](file:///home/abolarinwa-sa/Documents/Smol-VLAxYOLO/third_party/lerobot/src/lerobot/policies/smolvla/processor_smolvla.py#L33) |
| **VLA Utilities & Resizing** | `src/lerobot/policies/common/vla_utils.py` | [resize_with_pad](file:///home/abolarinwa-sa/Documents/Smol-VLAxYOLO/third_party/lerobot/src/lerobot/policies/common/vla_utils.py#L218), [make_att_2d_masks](file:///home/abolarinwa-sa/Documents/Smol-VLAxYOLO/third_party/lerobot/src/lerobot/policies/common/vla_utils.py#L65), [pad_vector](file:///home/abolarinwa-sa/Documents/Smol-VLAxYOLO/third_party/lerobot/src/lerobot/policies/common/vla_utils.py#L116) |
| **Flow Matching Primitives** | `src/lerobot/policies/common/flow_matching.py` | [make_flow_matching_inputs](file:///home/abolarinwa-sa/Documents/Smol-VLAxYOLO/third_party/lerobot/src/lerobot/policies/common/flow_matching.py#L71), [euler_integrate](file:///home/abolarinwa-sa/Documents/Smol-VLAxYOLO/third_party/lerobot/src/lerobot/policies/common/flow_matching.py#L143) |
| **Pretrained Base Class** | `src/lerobot/policies/pretrained.py` | [PreTrainedPolicy](file:///home/abolarinwa-sa/Documents/Smol-VLAxYOLO/third_party/lerobot/src/lerobot/policies/pretrained.py#L144) |
| **Package Metadata & Dependencies**| `pyproject.toml` | `[project.dependencies]`, `[project.optional-dependencies.smolvla]` |
| **Tests** | `tests/policies/smolvla/test_modeling_smolvla.py`<br>`tests/policies/smolvla/test_smolvla_rtc.py`<br>`tests/processor/test_smolvla_processor.py` | [test_prepare_images_normalizes_uint8_inputs_before_resizing](file:///home/abolarinwa-sa/Documents/Smol-VLAxYOLO/third_party/lerobot/tests/policies/smolvla/test_modeling_smolvla.py#L26), [test_smolvla_rtc_initialization](file:///home/abolarinwa-sa/Documents/Smol-VLAxYOLO/third_party/lerobot/tests/policies/smolvla/test_smolvla_rtc.py#L32), [test_make_smolvla_processor_basic](file:///home/abolarinwa-sa/Documents/Smol-VLAxYOLO/third_party/lerobot/tests/processor/test_smolvla_processor.py#L88) |
| **Tutorial Example & Docs** | `examples/tutorial/smolvla/using_smolvla_example.py`<br>`docs/source/smolvla.mdx` | [main](file:///home/abolarinwa-sa/Documents/Smol-VLAxYOLO/third_party/lerobot/examples/tutorial/smolvla/using_smolvla_example.py#L14) |

---

## 2. Visual Pipeline: Code Excerpts with Line Numbers

The forward path from input images to the language-backbone inputs is traced below:

### Step 2.1: Image Preprocessing and Camera Handling
**File:** [modeling_smolvla.py:385-433](file:///home/abolarinwa-sa/Documents/Smol-VLAxYOLO/third_party/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py#L385-L433)
```python
385:     def prepare_images(self, batch):
...
398:         # Preprocess image features present in the batch
399:         for key in present_img_keys:
400:             img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
401:             if img.dtype == torch.uint8:
402:                 img = img.to(torch.float32) / 255.0
403:             if self.config.resize_imgs_with_padding is not None:
404:                 # SmolVLA stores the target as (width, height); the shared helper expects (height, width).
405:                 img = resize_with_pad(
406:                     img,
407:                     self.config.resize_imgs_with_padding[1],
408:                     self.config.resize_imgs_with_padding[0],
409:                     pad_value=0,
410:                 )
411:
412:             # Normalize from range [0,1] to [-1,1] as expacted by siglip
413:             img = img * 2.0 - 1.0
...
421:             images.append(img)
422:             img_masks.append(mask)
...
426:         for num_empty_cameras in range(len(missing_img_keys)):
427:             if num_empty_cameras >= self.config.empty_cameras:
428:                 break
429:             img = torch.ones_like(img) * -1
430:             mask = torch.zeros_like(mask)
431:             images.append(img)
432:             img_masks.append(mask)
433:         return images, img_masks
```

### Step 2.2: Aspect-Preserving Padding
**File:** [vla_utils.py:218-242](file:///home/abolarinwa-sa/Documents/Smol-VLAxYOLO/third_party/lerobot/src/lerobot/policies/common/vla_utils.py#L218-L242)
```python
218: def resize_with_pad(img: torch.Tensor, height: int, width: int, *, pad_value: float) -> torch.Tensor:
219:     """Resize a (b, c, h, w) image without distortion, padding on the LEFT and TOP."""
...
232:     ratio = max(current_width / width, current_height / height)
233:     resized_height = int(current_height / ratio)
234:     resized_width = int(current_width / ratio)
235:     resized_img = F.interpolate(
236:         img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
237:     )
238:
239:     pad_height = max(0, height - resized_height)
240:     pad_width = max(0, width - resized_width)
241:     padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
242:     return padded_img
```

### Step 2.3: Vision Encoder and Connector Forward Pass
**File:** [smolvlm_with_expert.py:202-215](file:///home/abolarinwa-sa/Documents/Smol-VLAxYOLO/third_party/lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py#L202-L215)
```python
202:     def embed_image(self, image: torch.Tensor):
203:         patch_attention_mask = None
204:         # Get sequence from the vision encoder
205:         image_hidden_states = (
206:             self.get_vlm_model()
207:             .vision_model(
208:                 pixel_values=image.to(dtype=self.get_vlm_model().vision_model.dtype),
209:                 patch_attention_mask=patch_attention_mask,
210:             )
211:             .last_hidden_state
212:         )
213:         # Modality projection & resampling
214:         image_hidden_states = self.get_vlm_model().connector(image_hidden_states)
215:         return image_hidden_states
```

### Step 2.4: Prefix Token Assembly (Image + Text + State)
**File:** [modeling_smolvla.py:606-697](file:///home/abolarinwa-sa/Documents/Smol-VLAxYOLO/third_party/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py#L606-L697)
```python
606:     def embed_prefix(
607:         self, images, img_masks, lang_tokens, lang_masks, state: torch.Tensor = None
608:     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
...
614:         for _img_idx, (img, img_mask) in enumerate(zip(images, img_masks, strict=False)):
...
633:             img_emb = self.vlm_with_expert.embed_image(img)
...
636:             # Normalize image embeddings
637:             img_emb_dim = img_emb.shape[-1]
638:             img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)
639:
640:             bsize, num_img_embs = img_emb.shape[:2]
641:             img_mask = img_mask[:, None].expand(bsize, num_img_embs)
642:
643:             embs.append(img_emb)
644:             pad_masks.append(img_mask)
645:             att_masks += [0] * (num_img_embs)
...
661:         lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
664:         lang_emb = lang_emb * math.sqrt(lang_emb_dim)
666:         embs.append(lang_emb)
667:         pad_masks.append(lang_masks)
670:         att_masks += [0] * num_lang_embs
671:
672:         state_emb = self.state_proj(state)
673:         state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
674:         embs.append(state_emb)
678:         states_seq_len = state_emb.shape[1]
680:         pad_masks.append(state_mask)
683:         att_masks += [1] * (states_seq_len)
684:         embs = torch.cat(embs, dim=1)
685:         pad_masks = torch.cat(pad_masks, dim=1)
...
697:         return embs, pad_masks, att_masks
```

---

## 3. Tensor Dimensions

### 3.1 Dimensions Explicitly Established by Source and Config (Facts)
- **Image Input Resolution:** `(3, 512, 512)` (C, H, W) per camera via `SmolVLAConfig.resize_imgs_with_padding = (512, 512)`.
- **Pixel Value Range:** `[-1.0, 1.0]` (from `[0.0, 1.0]` scaled by `img * 2.0 - 1.0`).
- **Max State Dimension:** `32` (`SmolVLAConfig.max_state_dim = 32`). Smaller vectors are zero-padded to 32.
- **Max Action Dimension:** `32` (`SmolVLAConfig.max_action_dim = 32`). Smaller vectors are zero-padded to 32.
- **Action Chunk Size / Action Steps:** `50` (`SmolVLAConfig.chunk_size = 50`, `n_action_steps = 50`).
- **Observation Steps:** `1` (`SmolVLAConfig.n_obs_steps = 1`).
- **Flow Matching Integration Steps:** `10` (`SmolVLAConfig.num_steps = 10`).
- **Language Max Token Length:** `48` (`SmolVLAConfig.tokenizer_max_length = 48`).
- **VLM Layers Retained:** `16` (`SmolVLAConfig.num_vlm_layers = 16`, text model truncated to first 16 layers).
- **Expert Width Multiplier:** `0.75` (`SmolVLAConfig.expert_width_multiplier = 0.75`).
- **Self-Attention Interleave:** Every `2` layers (`SmolVLAConfig.self_attn_every_n_layers = 2`).
- **State Projection Layer:** `nn.Linear(32, text_config.hidden_size)`.
- **Action Input Projection:** `nn.Linear(32, expert_hidden_size)`.
- **Action Output Projection:** `nn.Linear(expert_hidden_size, 32)`.
- **Action-Time MLP:** `nn.Linear(expert_hidden_size * 2, expert_hidden_size)` -> `SiLU` -> `nn.Linear(expert_hidden_size, expert_hidden_size)`.

### 3.2 Dimensions Determined at Runtime (Facts)
- **Batch Size ($B$):** Configured at train/eval invocation.
- **Camera Count ($N_{cam}$):** Dictated by robot/dataset camera configuration in `config.image_features`.
- **Visual Token Count per Camera ($L_{img}$):** Determined by SmolVLM's SigLIP patch size (typically 14x14) and connector downsampling / resampling ratio on 512x512 inputs.
- **Language Token Count ($L_{lang}$):** Variable sequence length $\le 48$ depending on tokenized task string and padding mode (`"longest"` vs `"max_length"`).
- **VLM Hidden Size ($D_{vlm}$):** Determined by the loaded Hugging Face model config (`text_config.hidden_size`).
- **Expert Hidden Size ($D_{expert}$):** Calculated as `int(D_{vlm} * 0.75)`.
- **Prefix Sequence Length ($L_{prefix}$):** $N_{cam} \times L_{img} + L_{lang} + 1$ (plus optional start/end special tokens).

---

## 4. Upstream Dependency Constraints

From `third_party/lerobot/pyproject.toml`:
- **Python Version:** `requires-python = ">=3.12"` (local machine has Python 3.12.3).
- **PyTorch Stack:**
  - `torch>=2.7,<2.12.0`
  - `torchvision>=0.22.0,<0.27.0`
  - `numpy>=2.0.0,<2.3.0`
- **SmolVLA Policy Extra (`lerobot[smolvla]`):**
  - `transformers>=5.4.0,<5.6.0` (from `transformers-dep`)
  - `accelerate>=1.14.0,<2.0.0` (from `accelerate-dep`)
  - `num2words>=0.5.14,<0.6.0`
- **Vision & Robotics Core:**
  - `opencv-python-headless>=4.9.0,<4.14.0`
  - `Pillow>=10.0.0,<13.0.0`
  - `einops>=0.8.0,<0.9.0`
  - `safetensors>=0.4.3,<1.0.0`
  - `gymnasium>=1.1.1,<2.0.0`

---

## 5. Architectural Findings and Extension Points (Analysis & Interpretation)

1. **Config-Driven Architecture:**
   `SmolVLAConfig` uses standard `@PreTrainedConfig.register_subclass`. A subclass or extended config can register additional fields (e.g. YOLO checkpoint path, bounding box/feature dimensions, fusion modality flags).
2. **Preprocessing Interception Point (`prepare_images`):**
   `SmolVLAPolicy.prepare_images` is the direct ingress point for RGB cameras. It receives raw observation tensors and yields preprocessed tensors. Any dual-stream YOLO preprocessing (e.g. YOLO-specific resolution or format) can tap in here.
3. **Prefix Embedding Interception Point (`embed_prefix`):**
   `VLAFlowMatching.embed_prefix` builds the sequence `[img_tokens, lang_tokens, state_token]` and sets up bidirectional 2D attention masks. YOLO object tokens, bounding box tokens, or spatial feature tokens can be concatenated into this prefix sequence with minimal disturbance to the existing attention topology.
4. **VLM Wrapper Interception Point (`SmolVLMWithExpertModel`):**
   `embed_image` delegates directly to `vision_model` and `connector`. A parallel YOLO visual backbone can be plugged in alongside or after the connector.
5. **Decoupled Action Expert:**
   The action expert operates on flow matching noise tokens and cross-attends to the cached prefix KV representations. Changing or enriching the visual prefix does not alter the action expert's dimension contract ($32 \to D_{expert} \to 32$) as long as the prefix provides keys and values in the VLM embedding dimension.
