# RGB baseline training

## Purpose
RGB baseline training within the MLC Robotics language-conditioned YOLO policy programme.

## Environment and inputs
No overfit checkpoint. GPU and Internet required.

## Execution
The code cells run sequentially using Run All. Embedded installation code provisions the required project modules.

## Outputs
baseline_outputs/baseline_best.pt and latest checkpoint, configuration, normalisation and manifest.

## Evidence and interpretation
Fresh fusion and expert; pretrained YOLO and CLIP. Initial budget 10,000 updates, batch 4. Best recorded validation action MSE 0.6226 at step 8,000; mean-action baseline 0.9704.

Code: Notebooks/training/libero_rgb_baseline_training.ipynb
