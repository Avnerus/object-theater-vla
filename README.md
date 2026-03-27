# Object Theater VLA

A Vision-Language-Action (VLA) robotic system designed for zero-bias creative pedagogy.

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
object-theater-vla/
├── configs/          # Configuration management
│   ├── device.py     # Global DEVICE variable (cuda/cpu)
│   ├── config.py     # Dataclass-based hyperparameters
│   └── __init__.py
├── envs/             # Environment implementations
│   ├── robosuite_sandbox.py  # Robosuite wrapper with OSC_POSE
│   └── __init__.py
├── memory/           # Episodic memory
│   ├── lemb_core.py  # FAISS-based Localized Episodic Memory Buffer
│   └── __init__.py
├── models/           # ML models
│   ├── siglip_grounding.py   # SigLIP text encoder
│   ├── v_jepa_encoder.py     # V-JEPA encoder/predictor
│   ├── diffusion_policy.py   # Diffusion policy
│   └── __init__.py
├── scripts/          # Execution scripts
│   ├── 01_teleop_demonstrate.py   # Teleop demonstration recorder
│   ├── 02_autonomous_rollout.py   # Autonomous execution pipeline
│   └── __init__.py
├── tests/            # Unit tests
│   └── __init__.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Usage

### Teleoperation Demonstration

Record human demonstrations using keyboard or SpaceMouse:

```bash
python scripts/01_teleop_demonstrate.py --device keyboard
```

Controls:
- `s` - Start recording (enter task description)
- `e` - End recording
- `w` - Write demonstrations to HDF5 file
- `q` - Quit

### Autonomous Rollout

Execute tasks using the trained VLA pipeline:

```bash
python scripts/02_autonomous_rollout.py \
    --task "grasp the red box and place it on the left" \
    --num-rollouts 5
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

## Core Components

### Environment (`envs/robosuite_sandbox.py`)
- Panda arm with OSC_POSE controller
- Tabletop scene with manipulable objects
- RGB camera observations
- Proprioceptive state output

### Memory (`memory/lemb_core.py`)
- FAISS-based vector database
- Cosine or Euclidean similarity search
- Stores: semantic_vector, visual_state, action_trajectory

### Models (`models/`)
- **SigLIP**: Text embedding generation (768-dim)
- **V-JEPA**: Vision representation + state prediction
- **Diffusion Policy**: Action sequence generation (16-step horizon)

## Design Principles

- **Type Hints**: All functions use Python 3.10+ type hints
- **Docstrings**: Google-style docstrings throughout
- **Config Management**: Dataclasses + YAML support
- **Device Agnosticism**: Global `DEVICE` variable for PyTorch tensors
- **Modularity**: Strict separation of concerns

## License

MIT License
