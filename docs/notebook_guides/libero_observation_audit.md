# Stored and simulator observation comparison

## Purpose
Stored and simulator observation comparison within the MLC Robotics language-conditioned YOLO policy programme.

## Environment and inputs
Completed four-demo-fit output containing diagnostic_fit.pt.

## Execution
The code cells run sequentially using Run All. Embedded installation code provisions the required project modules.

## Outputs
observation_audit.zip with matched-noise predictions, hybrid inputs and comparison images.

## Evidence and interpretation
Completed 116 observations without errors. During recorded replay: stored inputs MSE 0.0332, live inputs 0.2212, live images plus stored state 0.0351. These are normalised first-action errors on sampled training positions.

Code: Notebooks/evaluation/libero_observation_audit.ipynb
