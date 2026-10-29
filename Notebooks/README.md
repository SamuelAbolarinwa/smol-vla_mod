# Notebook workflow

All notebooks are standalone Kaggle entry points. GPU and Internet are required except for the data-pipeline stage, where GPU is optional. Checkpoint inputs are previous notebook outputs attached through Kaggle.

| Notebook | Required input | Purpose |
|---|---|---|
| [validate_extractor_kaggle_standalone](validation/validate_extractor_kaggle_standalone.ipynb) | No attached checkpoint. GPU and Internet required. | Dense YOLO feature extractor |
| [validate_pretrained_clip_fp32](validation/validate_pretrained_clip_fp32.ipynb) | No attached checkpoint. Downloads pinned YOLO and CLIP weights. | Pretrained vision and instruction encoder |
| [validate_observation_fusion_kaggle](validation/validate_observation_fusion_kaggle.ipynb) | No dataset or checkpoint. GPU and Internet for installation. | Observation fusion |
| [validate_flow_policy_kaggle](validation/validate_flow_policy_kaggle.ipynb) | No attached dataset or checkpoint. GPU and Internet required. | Flow action expert and integrated policy |
| [libero_object_data_pipeline](data/libero_object_data_pipeline.ipynb) | Internet required; GPU optional. Downloads ten task HDF5 files. | LIBERO Object data preparation |
| [small_data_overfit_kaggle](training/small_data_overfit_kaggle.ipynb) | No attached checkpoint. GPU and Internet required. | Initial small data learning check |
| [extended_overfit_kaggle](training/extended_overfit_kaggle.ipynb) | Optional overfit_checkpoint.pt to resume; otherwise initialises a new run. | Extended small data learning check |
| [resume_overfit_sampling_comparison](training/resume_overfit_sampling_comparison.ipynb) | One overfit_checkpoint.pt from an extended run at step 3,000 or later. | Overfit continuation and sampling comparison |
| [libero_rgb_baseline_training](training/libero_rgb_baseline_training.ipynb) | No overfit checkpoint. GPU and Internet required. | RGB baseline training |
| [libero_closed_loop_pilot_v4](evaluation/libero_closed_loop_pilot_v4.ipynb) | RGB baseline notebook output containing baseline_best.pt. GPU and Internet required. | Closed loop baseline pilot |
| [libero_pickup_diagnostics](evaluation/libero_pickup_diagnostics.ipynb) | RGB baseline output containing baseline_best.pt. Prior rollout ZIP is unnecessary. | Pickup and action execution diagnosis |
| [libero_demonstration_replay](evaluation/libero_demonstration_replay.ipynb) | RGB baseline output containing baseline_best.pt. | Recorded demonstration replay |
| [libero_four_demo_fit](training/libero_four_demo_fit.ipynb) | Original RGB baseline output containing baseline_best.pt. | Four demonstration fitting |
| [libero_observation_audit](evaluation/libero_observation_audit.ipynb) | Completed four-demo-fit output containing diagnostic_fit.pt. | Stored and simulator observation comparison |
| [libero_gripper_settling_test](evaluation/libero_gripper_settling_test.ipynb) | Completed four-demo-fit output containing diagnostic_fit.pt. GPU and Internet required. | Gripper state isolation and settling comparison |

## Suggested progression

Component validation → data preparation → small-data learning checks → RGB baseline → closed-loop pilot → targeted diagnosis. Overfit continuation uses the preceding overfit checkpoint. The replay, pickup and four-demo-fit experiments use the original baseline output. Observation audit and settling test use the four-demo diagnostic checkpoint.

Archive versions are preserved for provenance and are not the recommended starting points. Cleaned notebooks retain executable code unchanged. This package does not claim a fresh GPU rerun.
