# Closed loop baseline pilot

## Purpose
Closed loop baseline pilot within the MLC Robotics language-conditioned YOLO policy programme.

## Environment and inputs
RGB baseline notebook output containing baseline_best.pt. GPU and Internet required.

## Execution
The code cells run sequentially using Run All. Embedded installation code provisions the required project modules.

## Outputs
Fifty trials, task success records and rollout artefacts.

## Evidence and interpretation
Observed 1/50 successes, or 2 percent. Executes one action per replan, maximum 280 policy steps.

Code: Notebooks/evaluation/libero_closed_loop_pilot_v4.ipynb
