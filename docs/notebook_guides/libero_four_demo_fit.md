# Four demonstration fitting

## Purpose
Four demonstration fitting within the MLC Robotics language-conditioned YOLO policy programme.

## Environment and inputs
Original RGB baseline output containing baseline_best.pt.

## Execution
The code cells run sequentially using Run All. Embedded installation code provisions the required project modules.

## Outputs
four_demo_outputs/diagnostic_fit.pt, fit metrics and four policy rollout videos.

## Evidence and interpretation
Fits all 610 timesteps; YOLO frozen, fusion and expert trained. Selected update 3,000: chunk MSE 0.0485, first-action MSE 0.0978. Subsequent rollouts 0/4 successful.

Code: Notebooks/training/libero_four_demo_fit.ipynb
