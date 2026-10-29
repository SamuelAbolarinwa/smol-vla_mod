# Pickup and action execution diagnosis

## Purpose
Pickup and action execution diagnosis within the MLC Robotics language-conditioned YOLO policy programme.

## Environment and inputs
RGB baseline output containing baseline_best.pt. Prior rollout ZIP is unnecessary.

## Execution
The code cells run sequentially using Run All. Embedded installation code provisions the required project modules.

## Outputs
Stored-observation audits, paired one-action versus four-action execution, videos and traces.

## Evidence and interpretation
One success in eight trials for each condition, involving the same salad-dressing initial state. No established benefit from four-action execution.

Code: Notebooks/evaluation/libero_pickup_diagnostics.ipynb
