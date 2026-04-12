# Object Theater VLA

A Vision-Language-Action (VLA) robotic system designed for zero-bias creative pedagogy.

## Installation

Using `uv` (recommended):

```bash
# Install all dependencies (default + all groups)
uv sync

# Or install specific groups:
uv sync --group server  # For running the Brain server
uv sync --group client  # For running the Body client
```

## Project Structure

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
│   ├── 02_train_diffusion_policy.py # Diffusion policy training
│   ├── 03_server_brain.py           # ZeroMQ VLA inference server (GPU models)
│   └── 04_client_body.py            # ZeroMQ Robosuite client with intervention support
├── utils/            # Utility functions
│   ├── visualization.py  # Plotting, animation, visualization
│   ├── dataset.py        # HDF5 dataset loading
│   ├── train.py          # Training utilities
│   └── __init__.py
├── tests/            # Unit tests (placeholder)
│   └── __init__.py
├── pyproject.toml    # Project metadata and dependencies (managed via uv)
├── LICENSE           # MIT License
└── README.md         # Project documentation
```

## Usage

### 1. Record Demonstrations

Record human demonstrations using keyboard or SpaceMouse:

```bash
uv run python scripts/01_teleop_demonstrate.py --device keyboard
```

Controls:
- `s` - Start recording (enter task description)
- `e` - End recording
- `w` - Write demonstrations to HDF5 file
- `q` - Quit

### Distributed Brain/Body Architecture

For production use, the pipeline is split across a client (Body) and server (Brain) using ZeroMQ. The GPU-resident models run on the server while the Robosuite simulation with 3D rendering runs locally.

#### Setup

**Install server dependencies:**
```bash
uv sync --group server
```

**Install client dependencies:**
```bash
uv sync --group client
```

**Terminal 1 — Start the Brain (GPU server):**
```bash
uv run python scripts/03_server_brain.py --bind tcp://0.0.0.0:5555
```

**Terminal 2 — Start the Body (local client):**
```bash
uv run python scripts/04_client_body.py \
    --server tcp://<server-ip>:5555 \
    --task "grasp the red box and place it on the left"
```

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

### 2. Train Diffusion Policy

Train the diffusion policy on collected demonstrations:

```bash
uv run python scripts/02_train_diffusion_policy.py \
    --dataset data/demonstrations/demonstrations_*.h5 \
    --num-epochs 100
```

## Configuration

Configuration is managed via dataclasses in `configs/config.py`. You can:

- Use default configuration
- Create custom config from YAML
- Override specific parameters

```python
from configs.config import Config

config = Config()
config.env.horizon = 2000
config.memory.retrieval_k = 5
```

### Configuration Options

All hyperparameters are defined in `configs/config.py` using dataclasses:

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

### Environment (`envs/robosuite_sandbox.py`)
- **Robot**: Panda arm
- **Controller**: OSC_POSE (Operational Space Control)
- **Scene**: Tabletop with 3 manipulable objects (Box, Cylinder, Sphere)
- **Observations**: RGB camera (agentview), proprioceptive state, force-torque sensor
- **Action space**: 7-dim (dx, dy, dz, roll, pitch, yaw, gripper)
- **Force sensor**: `robot0_eef_force` observation (3D vector)

### Memory (`memory/lemb_core.py`)
- **FAISS index**: IndexFlatIP (cosine) or IndexFlatL2 (Euclidean)
- **Storage**: semantic_vector (768-dim), visual_state, action_trajectory, task_label
- **Methods**: `add_memory()`, `retrieve_closest_trajectory()`, `get_all_task_labels()`
- **Dynamic injection**: New trajectories can be added at runtime via ZeroMQ `add_memory` message
- **Zero-Bias SLM**: Retrieves all task labels for RAG-based chat responses

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
- Architecture: Conditional 1D UNet with Cross-Attention
- **Tri-Modal Conditioning**:
  - **Visual**: V-JEPA dense feature maps via Cross-Attention
  - **Language**: SigLIP semantic embedding fused with time embedding
  - **Memory**: Historical trajectory via trajectory priming
- Output: 16-step action sequence
- Diffusion steps: 1000
- Fuses time + semantic embeddings for semantic guidance in cross-attention

### Zero-Bias SLM (`scripts/03_server_brain.py`)
- Model: Qwen2.5-7B-Instruct (7B params) via Hugging Face pipeline
- Zero-bias prompt: strictly grounded in Episodic Memory Buffer
- Returns "I don't know" for unknown queries, asks for physical demonstration
- **Grammar Parsing**: Extracts verbs/nouns for targeted memory routing
- Used for intervention task labeling and general robot conversation

## Design Principles

- **Type Hints**: All functions use Python 3.10+ type hints
- **Docstrings**: Google-style docstrings throughout
- **Config Management**: Dataclasses + YAML support
- **Device Agnosticism**: Global `DEVICE` variable for PyTorch tensors
- **Modularity**: Strict separation of concerns
- **Publication-Ready**: Clean, documented code suitable for research

## Dependencies

All dependencies are managed in `pyproject.toml` using `uv`:

- **Deep Learning Core**: torch, torchvision, triton-rocm (AMD ROCm optimized)
- **Neural Network Libraries**: transformers, accelerate, diffusers, sentencepiece, protobuf
- **Memory & Physics**: faiss-cpu, robosuite, h5py
- **Math & Core Utilities**: numpy, matplotlib, PyYAML, pyzmq, opencv-python
- **Test Tools**: pyright

### Dependency Groups

The project uses uv dependency groups to optimize installations:

- **server group**: Contains all GPU-heavy dependencies (torch, transformers, faiss, robosuite, etc.)
- **client group**: Contains minimal dependencies (pynput, robosuite) for running on WSL2

### Usage Workflow

**For Server (Brain):**
```bash
uv sync --group server
uv run python scripts/03_server_brain.py --bind tcp://0.0.0.0:5555
```

**For Client (Body):**
```bash
uv sync --group client
uv run python scripts/04_client_body.py --server tcp://<server-ip>:5555 --task "grasp the red box"
```

**For Development/Training Scripts:**
```bash
uv sync  # Installs default dependencies + all groups
uv run python scripts/01_teleop_demonstrate.py --device keyboard
uv run python scripts/02_train_diffusion_policy.py --num-epochs 100
```

## License

MIT License
