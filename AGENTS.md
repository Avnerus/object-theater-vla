# AGENTS.md - Object Theater VLA

## Project Overview

**Object Theater VLA** is a Vision-Language-Action (VLA) robotic system designed for zero-bias creative pedagogy. It implements a modular pipeline that combines:
- **SigLIP** for semantic text embeddings
- **V-JEPA** for vision representation learning
- **FAISS-based LEMB** for episodic memory retrieval
- **Diffusion Policy** for action sequence generation

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
│   ├── 01_teleop_demonstrate.py    # Teleop demonstration recorder
│   ├── 02_autonomous_rollout.py    # Autonomous execution pipeline
│   └── 03_train_diffusion_policy.py # Diffusion policy training
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
- **Observations**: RGB camera (agentview), proprioceptive state
- **Action space**: 7-dim (dx, dy, dz, roll, pitch, yaw, gripper)

### Memory (`memory/lemb_core.py`)
- **FAISS index**: IndexFlatIP (cosine) or IndexFlatL2 (Euclidean)
- **Storage**: semantic_vector (768-dim), visual_state, action_trajectory
- **Methods**: `add_memory()`, `retrieve_closest_trajectory()`

### SigLIP (`models/siglip_grounding.py`)
- Model: `google/siglip-base-patch16-224`
- Output: 768-dim normalized text embedding
- Methods: `encode_text()`, `encode_batch()`

### V-JEPA (`models/v_jepa_encoder.py`)
- Encoder: ViT-B with patch embedding
- Predictor: Transformer-based state prediction
- Latent dim: 1024
- Action dim: 7, Action horizon: 16

### Diffusion Policy (`models/diffusion_policy.py`)
- Architecture: Conditional 1D UNet
- Conditioning: V-JEPA latent state
- Output: 16-step action sequence
- Diffusion steps: 1000

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

### 3. Run Autonomous Rollout
```bash
python scripts/02_autonomous_rollout.py \
    --task "grasp the red box" \
    --num-rollouts 5
```

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
```

## Future Work

- Implement actual V-JEPA pre-trained weights
- Add unit tests
- Create Jupyter notebooks for experimentation
- Implement reward engineering for training
- Add more complex object arrangements

## Contact

- **Lead**: Avner (Avnerus-fbear)
- **Project**: Object Theater VLA
- **License**: MIT
