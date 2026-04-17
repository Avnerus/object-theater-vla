# AGENTS.md - Object Theater VLA

## Project Overview

**Object Theater VLA** is a Vision-Language-Action (VLA) robotic system designed for zero-bias creative pedagogy. It implements a modular pipeline that combines:
- **SigLIP** for semantic text embeddings
- **V-JEPA** for vision representation learning
- **FAISS-based LEMB** for episodic memory retrieval
- **Diffusion Policy** for action sequence generation
- **Zero-Bias SLM** for conversational interaction grounded in memory

### Active Compliance & Dynamic Memory Injection

The system supports real-time human interventions via force-threshold detection:
- Force sensor monitoring detects when a human physically guides the robot
- Intervention manager records the manual trajectory
- New trajectories are dynamically injected into the Server's Episodic Memory Buffer
- Memory injection happens without stopping the main simulation loop
- Keyboard device fallback available for manual triggering

### Milestone Chunking & Verification UX

The intervention pipeline now extracts discrete milestones from human guidance:
- Physics-based chunking using kinematic edge detectors (gripper switching, velocity threshold, contact detection)
- Minimum 15 steps enforced per chunk to prevent micro-chunks
- User intent prompt before takeover for accurate task labeling
- Quality Control (QC) gate: prompts for success verification before memory injection
- Sequential multi-chunk injection with success counting

### Optional Memory Consolidator

Background thread for automated memory scaffolding:
- Runs every 60 seconds when enabled via `--enable-consolidator` CLI flag
- Scans execution history for consecutive skill chains
- Uses SLM to generate macro-labels for fused skills
- Fuses two 16-step trajectories into 32-step macro-memories
- Tracks successful skill execution for consolidation

### Tri-Modal Diffusion

The diffusion policy uses three conditioning modalities:
- **Visual**: V-JEPA dense feature maps via Cross-Attention
- **Language**: SigLIP semantic embedding fused with time embedding
- **Memory**: Historical trajectory via trajectory priming

The UNet fuses time embedding with SigLIP semantic embedding, allowing cross-attention (Vision) to be semantically guided when generating actions.

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

## Core Components

### SLM Grammar Parser & Unified LEMB Routing (Recent Update - 2026-04-09)

The server brain now implements a grammar-based task decomposition system:

**Grammar Parsing (`scripts/03_server_brain.py`)**
- New `parse_grammar()` method uses the 8B-parameter SLM to extract verbs (actions) and nouns (objects)
- Returns structured JSON: `{"verb": "action_word", "nouns": ["object_1", "object_2"]}`
- Robust fallback: returns original text as verb if parsing fails

**Targeted Memory Retrieval**
- **Verb retrieval**: Generic action trajectory for trajectory priming
  - Stores in `self.priming_trajectory` (used by diffusion policy)
- **Noun retrieval**: Object-specific visual patches for cross-attention
  - Stores in `self.target_visual_patches` (future multimodal conditioning)

**Updated LEMB Return Signature (`memory/lemb_core.py`)**
- Changed `retrieve_closest_trajectory()` to return 4-tuple:
  `(memory_id, score, action_trajectory, visual_state)`
- Visual state now accessible for noun-based target conditioning

**Modified Flow**
1. Client sends task: `"grasp the red box"`
2. Server parses: `{"verb": "grasp", "nouns": ["box", "red"]}`
3. Verb query → priming trajectory for diffusion policy
4. Noun queries → visual patches for target conditioning
5. Diffusion policy generates actions with grammar-guided priming

### A* Latent Graph Search (The Hippocampus) (Recent Update - 2026-04-16)

The system implements high-level planning via visual latent space navigation:

**Visual Indexing (`memory/lemb_core.py`)**
- FAISS `visual_index` for spatial graph routing over V-JEPA states
- Dimension: 1664 (V-JEPA 2.1 ViT-Gigantic output)
- Supports nearest-neighbor search for entry/exit nodes

**A* Pathfinding (`EpisodicMemoryBuffer.find_latent_path()`)**
- Graph search algorithm over visual memory states
- Cost = spatial distance (1 - cosine similarity)
- Heuristic = direct distance to goal state
- Returns list of visual states representing the path

**Brain Server Integration (`scripts/03_server_brain.py`)**
- Abstract plan generation on first step: `self.milestone_queue`
- Milestone arrival recognition: similarity threshold > 0.95
- Step counter tracking: `self.step_counter`

**Modified Flow**
1. Task parsed into verb/nouns → priming trajectory + visual patches
2. Goal visual patch extracted from first noun
3. A* finds path from current → goal visual state
4. Milestone queue populated: `self.milestone_queue`
5. Each step checks milestone arrival via similarity
6. Queue pops when milestone reached (sim > 0.95)

### Memory Consolidator (Recent Update - 2026-04-16)

Background thread for automated memory scaffolding:

**Trajectory Fusion (`EpisodicMemoryBuffer.fuse_memories()`)**
- Combines two sequential 16-step memories into 32-step macro-memory
- Uses start visual state of first skill
- Auto-increments memory IDs for new macro-skills

**Consolidation Loop (`BrainServer._consolidation_loop()`)**
- Runs every 60 seconds (daemon thread)
- Grabs last two memories from `execution_history`
- Uses SLM to generate 2-3 word macro-label
- Fuses memories and clears history

**Brain Server Integration (`scripts/03_server_brain.py`)**
- Enabled via `--enable-consolidator` CLI flag
- Tracks successful skills: `self.execution_history`
- Runs background thread on initialization

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
uv run python scripts/01_teleop_demonstrate.py --device keyboard
# Press 's' to start, 'e' to end, 'w' to write to HDF5
```

### 2. Train Diffusion Policy
```bash
uv run python scripts/02_train_diffusion_policy.py \
    --dataset data/demonstrations/demonstrations_*.h5 \
    --num-epochs 100
```

### 3. Run with Force-Threshold Intervention

Run the distributed Brain/Body system with intervention enabled (default):

```bash
# Terminal 1: Start the Brain
uv run python scripts/03_server_brain.py --bind tcp://0.0.0.0:5555

# Terminal 2: Start the Body with intervention
uv run python scripts/04_client_body.py \
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
uv run python scripts/04_client_body.py \
    --server tcp://<server-ip>:5555 \
    --task "grasp the red box" \
    --no-intervention
```

### 4. Run Distributed Brain/Body (client-server)

This is the recommended architecture for production — GPU-resident models run
on the server, while the Robosuite simulation with 3D rendering runs locally.

**Terminal 1 — Start the Brain (GPU server):**
```bash
uv run python scripts/03_server_brain.py --bind tcp://0.0.0.0:5555
```

**Terminal 2 — Start the Body (local client):**
```bash
uv run python scripts/04_client_body.py \
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

### Type Checking with Pyright

All code must pass static type checking with Pyright before being considered complete:

```bash
uv run pyright models/<module_name>.py
```

**Requirements:**
- All functions must have proper type hints (Python 3.10+ syntax)
- Class attributes should be type-annotated
- Use `# type: ignore[misc]` for pyright limitations (e.g., iterating over `nn.ModuleList`)
- Use `typing.cast()` when pyright cannot infer types correctly

### Runtime Testing with uv

After type checking passes, verify functionality using `uv`:

```bash
# Run a module directly
uv run python -m models.<module_name>

# Run inline type tests
uv run python -c "import models.<module_name>; <test_code>"

# Add pacakges if needed
uv add <package>

# Test with both numpy arrays and torch tensors
uv run python -c "
import models.<module_name> as mod
import numpy as np
import torch

# Test with numpy
arr = np.array([...])
result = mod.function(arr)
print(f'Numpy test: {result.shape}')

# Test with torch
tensor = torch.tensor([...])
result = mod.function(tensor)
print(f'Torch test: {result.shape}')
"
```

### Documentation and API Reference

For library documentation and API reference:

1. First check the library's official documentation online
2. Use the `playwright-cli` skill for web browsing if needed:
   ```bash
   # In pi, use playwright-cli skill to browse docs
   ```
3. Check type stubs in `typeshed` or the library's source code

### Workflow Summary

1. **Type Check**: Run `uv run pyright <module>.py` - must pass with 0 errors
2. **Runtime Test**: Run `uv run python -m models.<module>` or inline tests
3. **Documentation**: Use playwright-cli skill to verify API behavior if unclear

### Examples

```bash
# Full workflow example
uv run pyright models/diffusion_policy.py  # Step 1: Type check
uv run python -m models.diffusion_policy   # Step 2: Run module tests
```

This ensures code quality through static analysis before runtime validation.

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

## Future Work

### Immediate (Phase 4)
- Implement SLM-based grammar parsing for task decomposition
- Add targeted LEMB retrieval using parsed verbs (action primitives) and nouns (object targets)
- Integrate priming trajectory from verb retrieval into diffusion policy
- Support visual patch extraction from noun retrieval for cross-attention conditioning
- Add `target_visual_patches` to server brain state for future multimodal conditioning

### Medium-Term
- Add unit tests
- Create Jupyter notebooks for experimentation
- Implement reward engineering for training
- Add more complex object arrangements
- Expand keyboard/SpaceMouse intervention triggers
- Support multi-language semantic conditioning
- Integrate speech synthesis for robot's voice output
- Add more SLM architectures for different performance/quality tradeoffs

### Long-Term
- Implement hierarchical task planning with grammar-based decomposition
- Add multi-objective optimization for action generation
- Support continuous learning with experience replay
- Extend to multi-robot coordination scenarios
- Integrate real-world hardware deployment pipeline

## Future Work: Pedagogical HRI & Advanced Intent Routing

To elevate the Object Theater from a reactive execution system to a proactive learning companion, future development will focus on two major architectural upgrades to the Server Brain:

### 1. The Curriculum Director (Pedagogical Orchestration)
* **Concept:** A background orchestration layer that bridges formal education (e.g., a primary school JSON curriculum) with the "Protégé Effect" (learning by teaching).
* **Mechanism:** Instead of lecturing the user, the Director monitors the live 3D scene via V-JEPA and secretly injects "Curiosity Goals" into the SLM's system prompt based on the curriculum objectives.
* **Result:** The robot acts as a curious student. If the curriculum dictates "Addition," the SLM asks the child, *"What happens if we push these blocks together?"* This prompts the child to physically demonstrate the concept, creating a zero-bias, child-led pedagogical loop.

### 2. The Cognitive Fork (Intent Classification)
* **Concept:** A zero-shot SLM routing layer that acts as the robot's prefrontal cortex, determining whether a user's input requires physical movement or conversation.
* **Mechanism:** Before processing text, the SLM classifies the input as either an `ACTION` (physical command) or `DIALOGUE` (conversation/teaching).
  * **Path A (Action):** The SLM acts as a subconscious grammar parser. It extracts verbs and nouns, queries the unified LEMB for generic trajectories and specific visual patches, and fires the Tri-Modal Diffusion Policy to move the arm.
  * **Path B (Dialogue):** Physical execution is paused. The SLM acts as the Conscious Protégé, checking the Curriculum Director's current goal and speaking back to the user.
* **Result:** Seamlessly unifies the system's physical Vision-Language-Action (VLA) capabilities with natural conversational AI, preventing conflicting server states and maintaining a fluid human-robot interaction.

## Contact

- **Lead**: Avner (Avnerus-fbear)
- **Project**: Object Theater VLA
- **License**: MIT

---

**Last Updated**: 2026-04-16  
**Architecture**: Zero-Bias SLM + Tri-Modal Diffusion + Active Compliance + Dynamic Memory Injection + SLM Grammar Parser + Unified LEMB Routing + A* Latent Graph Search (The Hippocampus) + Milestone Chunking & Verification UX + Optional Memory Consolidator  
**Dependencies**: Migrated from `requirements.txt` to `pyproject.toml` (managed via `uv`)
