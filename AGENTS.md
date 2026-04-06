# AGENTS.md - Object Theater VLA

## Project Overview

**Object Theater VLA** is a Vision-Language-Action (VLA) robotic system designed for zero-bias creative pedagogy. It implements a modular pipeline that combines:
- **SigLIP** for semantic text embeddings
- **V-JEPA** for vision representation learning
- **FAISS-based LEMB** for episodic memory retrieval
- **Diffusion Policy** for action sequence generation

### Active Compliance & Dynamic Memory Injection

The system supports real-time human interventions via force-threshold detection:
- Force sensor monitoring detects when a human physically guides the robot
- Intervention manager records the manual trajectory
- New trajectories are dynamically injected into the Server's Episodic Memory Buffer
- Memory injection happens without stopping the main simulation loop
- Keyboard device fallback available for manual triggering

## Repository Structure

```
object-theater-vla/
├── configs/          # Configuration management (dataclasses)
│   ├── device.py     # Global DEVICE variable (cuda/cpu)
│   ├── config.py     # Hyperparameter dataclasses
│   └── __init__.py
├── envs/             # Environment implementations
│   ├── robosuite_sandbox.py  # Robosuite wrapper with OSC_POSE
│   └── __init__.py
├── memory/           # Episodic memory
│   ├── lemb_core.py  # FAISS-based Localized Episodic Memory Buffer
│   └── __init__.py
├── models/           # ML models
│   ├── siglip_grounding.py   # SigLIP text encoder (768-dim)
│   ├── v_jepa_encoder.py     # V-JEPA encoder/predictor
│   ├── diffusion_policy.py   # Diffusion policy (16-step horizon)
│   └── __init__.py
├── scripts/          # Execution scripts
│   ├── 01_teleop_demonstrate.py     # Teleop demonstration recorder
│   ├── 02_autonomous_rollout.py     # Single-process autonomous pipeline
│   ├── 03_train_diffusion_policy.py # Diffusion policy training
│   ├── 03_server_brain.py           # ZeroMQ VLA inference server (GPU models)
│   └── 04_client_body.py            # ZeroMQ Robosuite client with intervention support
├── utils/            # Utility functions
│   ├── visualization.py  # Plotting, animation, visualization
│   ├── dataset.py        # HDF5 dataset loading
│   ├── train.py          # Training utilities
│   └── __init__.py
├── tests/            # Unit tests (placeholder)
│   └── __init__.py
├── requirements.txt  # Python dependencies
├── LICENSE           # MIT License
└── README.md         # Project documentation
```

## Core Components

### Environment (`envs/robosuite_sandbox.py`)
- **Robot**: Panda arm
- **Controller**: OSC_POSE (Operational Space Control)
- **Scene**: Tabletop with 3 manipulable objects (Box, Cylinder, Sphere)
- **Observations**: RGB camera (agentview), proprioceptive state, force-torque sensor
- **Action space**: 7-dim (dx, dy, dz, roll, pitch, yaw, gripper)
- **Force sensor**: `robot0_eef_force` observation (3D vector)

### Memory (`memory/lemb_core.py`)
- **FAISS index**: IndexFlatIP (cosine) or IndexFlatL2 (Euclidean)
- **Storage**: semantic_vector (768-dim), visual_state, action_trajectory
- **Methods**: `add_memory()`, `retrieve_closest_trajectory()`
- **Dynamic injection**: New trajectories can be added at runtime via ZeroMQ `add_memory` message

### SigLIP (`models/siglip_grounding.py`)
- Model: `google/siglip-base-patch16-224`
- Output: 768-dim normalized text embedding
- Methods: `encode_text()`, `encode_batch()`

### V-JEPA (`models/v_jepa_encoder.py`)
- Model: V-JEPA 2.1 ViT-Gigantic (2B params) via `torch.hub.load('facebookresearch/vjepa2', ...)`
- Dense patch-level feature extraction (not just CLS token)
- Frozen encoder — acts as a fixed perception engine
- Preprocessor: official `vjepa2_preprocessor`

### Diffusion Policy (`models/diffusion_policy.py`)
- Architecture: Conditional 1D UNet
- Conditioning: V-JEPA latent state
- Output: 16-step action sequence
- Diffusion steps: 1000

### Intervention Manager (`scripts/04_client_body.py`)
- Force-threshold detection on `robot0_eef_force` (default: 15N)
- Records manual guidance using robosuite keyboard device
- Compresses initial camera frame and sends `add_memory` payload to server
- Resumes autonomous rollout after memory injection
- CLI flag: `--no-intervention` to disable

## Configuration

All hyperparameters are in `configs/config.py` using dataclasses:

```python
@dataclass
class EnvironmentConfig:
    image_size: Tuple[int, int] = (224, 224)
    control_freq: int = 20
    horizon: int = 1000

@dataclass
class MemoryConfig:
    embedding_dim: int = 768
    retrieval_k: int = 3
    retrieval_alpha: float = 0.5

@dataclass
class ModelConfig:
    vjepa_latent_dim: int = 1024
    diffusion_action_dim: int = 7
    diffusion_num_timesteps: int = 1000

@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
```

## Usage Workflow

### 1. Record Demonstrations
```bash
python scripts/01_teleop_demonstrate.py --device keyboard
# Press 's' to start, 'e' to end, 'w' to write to HDF5
```

### 2. Train Diffusion Policy
```bash
python scripts/03_train_diffusion_policy.py \
    --dataset data/demonstrations/demonstrations_*.h5 \
    --num-epochs 100
```

### 3. Run Autonomous Rollout (single-process)
```bash
python scripts/02_autonomous_rollout.py \
    --task "grasp the red box" \
    --num-rollouts 5
```

### 3b. Run with Force-Threshold Intervention

Run the distributed Brain/Body system with intervention enabled (default):

```bash
# Terminal 1: Start the Brain
python scripts/03_server_brain.py --bind tcp://0.0.0.0:5555

# Terminal 2: Start the Body with intervention
python scripts/04_client_body.py \
    --server tcp://<server-ip>:5555 \
    --task "grasp the red box"
```

When the robot encounters resistance (EEF force > 15N), it automatically:
1. Yields to the human operator
2. Records the manual guidance trajectory
3. Injects the new memory into the Brain's episodic buffer
4. Resumes autonomous operation

To disable intervention:
```bash
python scripts/04_client_body.py \
    --server tcp://<server-ip>:5555 \
    --task "grasp the red box" \
    --no-intervention
```

### 4. Run Distributed Brain/Body (client-server)

This is the recommended architecture for production — GPU-resident models run
on the server, while the Robosuite simulation with 3D rendering runs locally.

**Terminal 1 — Start the Brain (GPU server):**
```bash
python scripts/03_server_brain.py --bind tcp://0.0.0.0:5555
```

**Terminal 2 — Start the Body (local client):**
```bash
python scripts/04_client_body.py \
    --server tcp://<server-ip>:5555 \
    --task "grasp the red box"
```

#### Force-Threshold Intervention

When the robot detects physical guidance (EEF force > 15N), it automatically:
1. Yields control to the human
2. Records the manual trajectory using the keyboard device
3. Compresses the initial camera frame
4. Sends `add_memory` payload to the Brain server
5. The server decodes the frame, extracts V-JEPA features, encodes the task with SigLIP, and adds to LEMB
6. The robot resumes autonomous rollout with updated memory

The intervention can be disabled with `--no-intervention` flag.

#### Asynchronous Action Chunking

To minimize network latency, the Brain/Body protocol uses asynchronous action chunking:
- The server computes and returns a full 16-action trajectory for each step request
- The client buffers actions locally and consumes them at native control_freq
- Background thread fetches the next chunk when buffer falls below `chunk_request_threshold` (default: 8)
- Zero-velocity fallback action ensures safe stop if network lag empties the buffer

Configuration parameter: `env.chunk_request_threshold` in `configs/config.py`

#### Brain ↔ Body Protocol

Communication uses ZeroMQ REQ/REP with `send_pyobj`/`recv_pyobj` (pickle):

| Direction | Message |
|---|---|
| Client → Server | `{"type": "init", "task": "..."}` |
| Server → Client | `{"status": "ready"}` |
| Client → Server | `{"type": "step", "image": <jpeg bytes>}` |
| Server → Client | `{"action_chunk": [[dx, dy, dz, roll, pitch, yaw, gripper], ...]}` |
| Client → Server | `{"type": "add_memory", "task": "...", "initial_image": ..., "action_trajectory": ...}` |
| Server → Client | `{"status": "memory_added_successfully"}` |

- Camera frames are JPEG-compressed with `cv2.imencode` to save bandwidth.
- The server computes and returns a full 16-action trajectory for each step request.
- The client buffers action chunks locally and consumes them at native control_freq.
- Background thread fetches the next chunk asynchronously when buffer falls below threshold.
- Memory injection uses the same protocol with longer timeout for V-JEPA + FAISS processing.

## Code Standards (Phase 3)

- **Type hints**: All functions use Python 3.10+ type hints
- **Docstrings**: Google-style for all classes
- **Config management**: No hardcoded magic numbers
- **Device agnosticism**: Global `DEVICE` variable used everywhere

## Testing Policy

**⚠️ No Runtime Testing Allowed Until Test Environment is Ready**

The `tests/` directory is currently a placeholder. Until proper test infrastructure is established:

- **Do NOT run runtime tests** (e.g., `python models/diffusion_policy.py`)
- **Do NOT execute scripts** that require external dependencies
- **Code changes should be syntactically correct and type-hint complete**

This policy ensures we maintain a clean development workflow without premature integration issues. Once the test environment is ready, this policy will be updated with proper testing guidelines.

## Dependencies

```bash
torch>=2.1.0
torchvision>=0.16.0
robosuite>=1.5.0
faiss-cpu>=1.8.0
transformers>=4.35.0
diffusers>=0.25.0
h5py>=3.10.0
numpy>=1.24.0
matplotlib>=3.7.0
PyYAML>=6.0
pyzmq>=25.0.0
opencv-python>=4.8.0
```

## Future Work

- Add unit tests
- Create Jupyter notebooks for experimentation
- Implement reward engineering for training
- Add more complex object arrangements

## Contact

- **Lead**: Avner (Avnerus-fbear)
- **Project**: Object Theater VLA
- **License**: MIT

---

**Last Updated**: 2026-04-06  
**Architecture**: Active Compliance + Dynamic Memory Injection
