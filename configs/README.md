# Configuration Directory

This directory stores experiment, architecture, and environment configuration files (YAML / JSON).

## Intended Layout
- `policy/`: Policy hyperparameters (YOLO-World backbone variant, spatial token adapter grid/channel settings, fusion dimensions, flow-matching steps).
- `data/`: Dataset paths, camera keys, state/action feature mappings, and normalization metadata (e.g. LIBERO-Object).
- `training/`: Optimizer settings, learning rate schedules, auxiliary depth loss weights, and compute dispatch profiles (e.g. Kaggle training configs).
- `deploy/`: Lightweight edge-inference configurations for Jetson Orin Nano deployment.
