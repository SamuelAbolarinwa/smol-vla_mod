# Language conditioned YOLO policy project handover

## Current architecture

The implemented RGB policy combines two camera views, a language instruction and eight robot-state values. A frozen CLIP ViT-B/32 text encoder supplies a 512-dimensional instruction embedding. The embedding conditions YOLOv8s-Worldv2 neck features and supplies one projected instruction token to observation fusion. YOLO modules 0 through 21 are retained; the detection head is removed.

Three visual scales per camera are pooled to 4 by 4 grids and projected to width 256. The resulting 96 visual tokens, one instruction token and one state token enter two transformer encoder layers. A custom two-layer transformer action expert cross-attends to this context. Conditional flow matching produces sixteen seven-dimensional actions using ten Euler integration steps. The main evaluation executes one action and replans. Actions represent translation, rotation and gripper control, with training normalisation inverted before execution.

There is no generative LLM in the current model. The fusion network and action expert were trained from random initialisation; they are not the pretrained SmolVLA modules. Their parameter counts are 2,016,768 and 2,246,919 respectively. These figures exclude YOLO and CLIP. Exact full-model and deployment counts still require checkpoint-level accounting. Depth, distillation and Jetson deployment remain future work.

## Training and evaluation record

The initial LIBERO Object split uses 450 training and 50 validation episodes across ten tasks. The RGB baseline uses pretrained vision and language components with newly trained fusion and action modules, 10,000 updates and batch size four. The backbone is frozen while the neck, fusion and action expert are trainable. The selected baseline checkpoint is from update 8,000, with sampled validation action MSE about 0.6226 versus a mean-action baseline of 0.9704.

The fifty-trial closed-loop pilot completed one task, a 2 percent success rate. The one-action versus four-action pickup comparison achieved one success in eight attempts in each condition. These small experiments do not establish reliable instruction following, generalisation or an efficiency advantage over SmolVLA.

Recorded actions succeeded on all four selected training demonstrations, while the baseline policy failed all four. Additional fitting of the fusion and action expert on all 610 timesteps reduced sampled chunk MSE to about 0.0485, but autonomous rollouts still failed all four. The fitted checkpoint is diagnostic, not a replacement for the general RGB baseline.

The observation audit compared 116 observations without execution errors. During recorded-action replay, mean normalised first-action MSE was 0.0332 with stored inputs, 0.2212 with live simulator inputs, 0.1997 with stored images plus live state, and 0.0351 with live images plus stored state. Direct state restoration produced a different pattern, with stored and live errors close to 0.096 and 0.094. The finding is therefore specific to the replay conditions and sampled observations.

The largest early normalised discrepancies occur in gripper state. Three fitted-policy traces closed the gripper within the first five control steps while still far from the object. This supports a state-sensitivity hypothesis, but does not yet isolate a controller bug, timing issue or learned shortcut. The gripper settling notebook tests gripper-only state substitutions and immediate versus settled policy control. Its latest supplied run stopped during installation because CUDA was unavailable; no settling success result is recorded.

## Runtime provenance

Ultralytics revision: `aa1f778a75a610f7dd2aa4762df1c25f6a3063d7`.

CLIP revision: `7ffa84b3bfa40c42ecc1c77147a855e69cb2dd40`.

LIBERO revision: `8f1084e3132a39270c3a13ebe37270a43ece2a01`.

Hugging Face LIBERO dataset revision: `f13aa24a3da8c43c7225569f28c562979fa0e35a`, repository `yifengzhu-hf/LIBERO-datasets`.

Simulator notebooks provision isolated Python 3.10, torch 2.6.0 with CUDA 12.4 wheels, torchvision 0.21.0, NumPy 1.26.4, robosuite 1.4.0, MuJoCo 2.3.7, bddl 1.0.1, gym 0.25.2 and numba 0.60.0. The notebooks contain the full installation recipe. This environment has not been validated for Jetson.

## Collaboration and next experiments

Each run should record its notebook hash, checkpoint hash, dataset and source revisions, seeds, settings, execution outcome and task metrics. Large outputs should remain external to Git; concise summaries and references belong in reports. Component checks, fitting error and closed-loop task success should be reported separately.

The next steps are to complete the gripper-state experiment, establish whether changing only the instruction changes the selected object in the same scene, and validate any correction on episodes beyond the four fitting demonstrations. Reusing SmolVLA pretraining is a separate comparison requiring feature-interface and robot-action adaptation. The current implementation should remain available as the compact baseline.
