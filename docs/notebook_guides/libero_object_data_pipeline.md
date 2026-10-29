# LIBERO Object data preparation

## Purpose
LIBERO Object data preparation within the MLC Robotics language-conditioned YOLO policy programme.

## Environment and inputs
Internet required; GPU optional. Downloads ten task HDF5 files.

## Execution
The code cells run sequentially using Run All. Embedded installation code provisions the required project modules.

## Outputs
Episode manifest, training-only statistics and data-pipeline checks.

## Evidence and interpretation
450 training and 50 validation episodes in the baseline split. Dataset downloads can occupy several GB.

Code: Notebooks/data/libero_object_data_pipeline.ipynb
