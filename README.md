# SmolVLA x YOLO: Compact Language-Conditioned Robot Policy

## 1. Project Objective

This project investigates and implements a compact, high-efficiency, language-conditioned robot manipulation policy. Moving beyond bulky vision-language-action foundation models that rely on generative LLM backbones, the new architecture (Architecture v2) combines:
1. An open-vocabulary **YOLO-World** visual pathway conditioned by language.
2. A **direct instruction embedding** path where the full language representation feeds directly into multimodal fusion.
3. Multi-scale **spatial visual tokenization** for fine-grained object interaction.
4. An **MLP robot state encoder** for proprioceptive conditioning.
5. A **multimodal fusion module** integrating visual, instruction, and proprioceptive features.
6. A continuous **flow-matching action expert** generating action chunks directly.
7. **Training-only auxiliary depth supervision** driven by a frozen teacher providing pseudo-depth targets, transferring geometric scene structure into shared visual representations while adding zero latency or parameters at inference.

Upstream SmolVLA is retained as a reference baseline and a candidate distillation teacher for later stages. The core policy itself operates without a generative language model backbone, targeting high-rate closed-loop deployment on edge robotics hardware.

---

## 2. Compute Plan & Division of Responsibilities

To balance local development constraints against deep learning workloads, work is strictly separated across three environments:

| Environment | Role & Scope | Constraints & Policy |
|---|---|---|
| **Local Machine** (Ubuntu 24.04, i7, 15 GB RAM, 9.4 GB free disk) | Codebase design, source implementation, AST compilation, packaging, linting, git maintenance. | No heavyweight ML dependencies (PyTorch, Transformers, Ultralytics) installed locally. No weights or datasets downloaded locally. |
| **Kaggle GPU** | Pretrained weights loading, dataset ingestion, numerical unit/regression tests, model training, and ablation experiments. | Handles full PyTorch / CUDA stacks, GPU training runs, and large dataset storage. |
| **Jetson Orin Nano** | Physical/edge deployment, TensorRT/ONNX compilation, inference latency profiling, and real-time control checks. | Final target platform for standalone deployment without depth teacher or auxiliary heads. |

---

## 3. High-Level Architecture (v2)

### Inference Pathway (Zero Depth Overhead)
```
RGB Image(s) ──► [Pretrained YOLO-World Vision Pathway] ──► Spatial Tokens ─┐
                         ▲                                                  │
Language Text ──► [Text Encoder] ─┬─────────────────────────────────────────┼──► [Multimodal Fusion] ──► [Flow-Matching Action Expert] ──► Action Chunks
                                  └─► Complete Instruction Embedding ───────┤
                                                                            │
Robot Proprioception ──────────────► [State MLP Encoder] ───────────────────┘
```

### Training-Only Auxiliary Depth Supervision
```
RGB Image ──► [Shared Visual Backbone Layers] ──┬──► [Language-Conditioned Vision Layers] ──► (Policy Pathway)
                                               │
                                               └──► [Auxiliary Depth Head] ──► Predicted Depth
                                                                                      ▲
RGB Image ─────────────────────────► [Frozen Depth Teacher] ──────────────────────────┴──► Loss (Updates Shared Visual Layers)
```
- **Shared Gradients:** Depth prediction gradients backpropagate directly into the early visual backbone layers before language conditioning, forcing features to capture 3D geometric layout.
- **Clean Deployment:** Both the frozen depth teacher and auxiliary depth head are detached during inference, leaving latency and memory completely unaffected.

---

## 4. Current State: Existing Implementation vs. Planned Components

| Component | Status | Source Location | Description |
|---|---|---|---|
| **YOLO11n Tokenizer & Adapter** | Preserved (Phase 1 Baseline) | `src/smol_vla_mod/vision/` | Multi-scale pooling, coordinate-augmented projection, and MLP tokenization for SmolVLA encoder replacement. |
| **Tokenizer Tests** | Preserved (Phase 1 Baseline) | `tests/test_yolo_tokenizer.py` | 11 unit tests covering shapes, coordinate grids, gradient flow, and validation. |
| **Packaging Scaffolding** | Complete (Phase 2 Migration) | `pyproject.toml`, `src/smol_vla_mod/` | Minimal setuptools src-layout package structure with lightweight subpackage initializers. |
| **YOLO-World Pathway** | Planned | `src/smol_vla_mod/vision/` | Source inspection, intermediate tap selection, and feature adapter implementation. |
| **Language Module** | Planned | `src/smol_vla_mod/language/` | Dual text representation: YOLO-World text conditioning and full instruction fusion vectors. |
| **Multimodal Fusion** | Planned | `src/smol_vla_mod/fusion/` | Integration of spatial tokens, complete instruction embeddings, and robot state. |
| **Flow-Matching Policy** | Planned | `src/smol_vla_mod/policy/` | Continuous action chunk diffusion/flow matching expert. |
| **Auxiliary Depth Module** | Planned | `src/smol_vla_mod/depth/` | Training-time lightweight depth prediction head and pseudo-depth teacher adapter. |
| **Data & Training Pipelines**| Planned | `src/smol_vla_mod/data/`, `training/` | LIBERO dataset loader, loss orchestration, and Kaggle training harnesses. |
| **Evaluation Harness** | Planned | `src/smol_vla_mod/evaluation/` | LIBERO-Object rollout and closed-loop benchmarking. |

---

## 5. Evaluation Benchmark & Planned Comparisons

### Benchmark Task
- **Target Suite:** **LIBERO-Object** manipulation benchmark.
- **Dataset Verification:** Dataset schema, observation keys (RGB cameras, joint states, gripper state), and action format remain to be verified directly against LeRobot / LIBERO dataset releases.

### Experimental Comparisons (No Claimed Results Ahead of Tests)
1. **SmolVLA Reference Baseline:** Upstream SmolVLA (pretrained `smolvla_base`) evaluated on LIBERO-Object.
2. **RGB-Only Policy (v2 without depth):** YOLO-World visual pathway + direct instruction fusion + flow-matching expert.
3. **RGB + Auxiliary Depth Policy (v2 with depth):** Same policy trained with auxiliary depth supervision from a frozen depth teacher.
4. **Latency & Parameter Profile:** Inference FPS and memory footprints measured across laptop (CPU/T2000), Kaggle (GPU), and Jetson Orin Nano.
