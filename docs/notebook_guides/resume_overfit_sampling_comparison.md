# Overfit continuation and sampling comparison

## Purpose
Overfit continuation and sampling comparison within the MLC Robotics language-conditioned YOLO policy programme.

## Environment and inputs
One overfit_checkpoint.pt from an extended run at step 3,000 or later.

## Execution
The code cells run sequentially using Run All. Embedded installation code provisions the required project modules.

## Outputs
3,000 additional updates; matched-noise comparisons with 10 and 20 Euler steps.

## Evidence and interpretation
Observed 6,000-step sampled MSE approximately 0.0263 versus mean-action baseline 1.0193.

Code: Notebooks/training/resume_overfit_sampling_comparison.ipynb
