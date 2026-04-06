"""
Server Brain — VLA Model Inference Daemon

ZeroMQ REP server that loads all heavy VLA models (SigLIP, V-JEPA, Diffusion
Policy, Episodic Memory) and serves action predictions to lightweight remote
clients over the network.

Protocol (pickle-serialised dicts):
    CLIENT → SERVER
        {"type": "init", "task": "<task description>"}
        {"type": "step", "image": <jpeg-encoded bytes>}
    SERVER → CLIENT
        {"status": "ready"}
        {"action_chunk": [[dx, dy, dz, roll, pitch, yaw, gripper], ...]}

Action chunking protocol
    The server computes a full 16-step action trajectory for each step request
    and returns the entire chunk as a list of lists. The client buffers these
    actions locally and consumes them at its native control_freq while fetching
    the next chunk asynchronously in the background.
"""

from typing import Any, Dict, List, Optional, Tuple
import pickle

import numpy as np
import torch
import zmq

from configs.device import DEVICE
from configs.config import Config, default_config

from memory.lemb_core import EpisodicMemoryBuffer
from models.siglip_grounding import SigLIPTextEncoder
from models.v_jepa_encoder import VJepaEncoder
from models.diffusion_policy import DiffusionPolicy


class BrainServer:
    """
    ZeroMQ-backed inference server for the Object Theater VLA pipeline.

    Holds every GPU-resident model and replies to step requests with the next
    action from a receding-horizon buffer.
    """

    def __init__(
        self,
        bind_address: str = "tcp://0.0.0.0:5555",
        config: Config = None,
        memory_buffer: Optional[EpisodicMemoryBuffer] = None,
        num_inference_steps: int = 20,
    ):
        """
        Initialise the Brain server.

        Args:
            bind_address: ZeroMQ bind string for the REP socket.
            config: Hyperparameter configuration.
            memory_buffer: Pre populated episodic memory (or None).
            num_inference_steps: Denoising steps for diffusion sampling.
        """
        self.config = config or default_config
        self.num_inference_steps = num_inference_steps

        # -------- ZeroMQ socket --------
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(bind_address)
        print(f"[Brain] Bound to {bind_address}")

        # -------- Models --------
        self._init_models()

        # -------- Memory --------
        self.memory = memory_buffer or EpisodicMemoryBuffer(
            embedding_dim=self.config.memory.embedding_dim,
            use_cosine_similarity=self.config.memory.use_cosine_similarity,
            max_memory_chunks=self.config.memory.max_memory_chunks,
        )

        # -------- State (reset on every "init") --------
        self.current_semantic_target: Optional[np.ndarray] = None
        self.step_counter: int = 0

    # ── Model bootstrapping ────────────────────────────────────────────

    def _init_models(self) -> None:
        """Load every VLA model and set to eval mode."""
        print("[Brain] Loading SigLIP text encoder …")
        self.siglip = SigLIPTextEncoder(
            model_name=self.config.model.siglip_model_name,
            device=DEVICE,
        )
        self.siglip.model.eval()

        print("[Brain] Loading V-JEPA 2.1 encoder …")
        self.vjepa_encoder = VJepaEncoder().to(DEVICE)
        self.vjepa_encoder.eval()

        print("[Brain] Loading Diffusion Policy …")
        self.diffusion_policy = DiffusionPolicy(
            latent_dim=self.config.model.vjepa_latent_dim,
            action_dim=self.config.model.diffusion_action_dim,
            action_horizon=self.config.model.diffusion_action_horizon,
            device=DEVICE,
        )
        self.diffusion_policy.model.eval()

        print("[Brain] All models loaded and set to eval mode.")

    # ── Inference helpers ───────────────────────────────────────────────

    def extract_semantic_target(self, text: str) -> np.ndarray:
        """Encode a natural-language task description into a 768-D embedding."""
        with torch.no_grad():
            embedding = self.siglip.encode_text(text, normalize=True)
        return embedding  # [768]

    def _decode_image(self, jpeg_bytes: bytes, image_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
        """
        Decode JPEG bytes into a V-JEPA-ready tensor.

        Returns tensor of shape [1, C, 1, H, W].
        """
        # lazy import — opencv may not be present on every system
        import cv2  # type: ignore[import-untyped]

        nparr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # [H, W, 3] (BGR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # resize if necessary
        if img_rgb.shape[:2] != image_size:
            img_rgb = cv2.resize(img_rgb, (image_size[1], image_size[0]))

        # → [C, H, W]
        tensor = torch.from_numpy(img_rgb).to(DEVICE).float().permute(2, 0, 1)
        # → [1, C, 1, H, W]
        tensor = tensor.unsqueeze(0).unsqueeze(2)
        return tensor

    def _extract_visual_state(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Forward-pass through V-JEPA. Returns [1, num_patches, latent_dim]."""
        with torch.no_grad():
            dense_features = self.vjepa_encoder.extract_features(image_tensor)
        return dense_features  # [1, num_patches, 1024]

    def _pool_visual_state(self, visual_state: torch.Tensor) -> np.ndarray:
        """Mean-pool dense features → [latent_dim] numpy array for FAISS."""
        pooled = visual_state.mean(dim=1)  # [1, latent_dim]
        return pooled.squeeze(0).cpu().numpy()

    def add_memory(
        self,
        task: str,
        initial_image_bytes: bytes,
        action_trajectory: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Add a new memory trajectory to the episodic buffer.

        Args:
            task: Natural-language description of the task.
            initial_image_bytes: JPEG-encoded initial camera frame.
            action_trajectory: Array of shape [horizon, action_dim].

        Returns:
            Status dictionary with success flag.
        """
        try:
            # 1. Decode image to tensor [1, C, 1, H, W]
            image_tensor = self._decode_image(
                initial_image_bytes,
                image_size=self.config.env.image_size,
            )

            # 2. Extract dense visual features using V-JEPA
            visual_state = self._extract_visual_state(image_tensor)  # [1, num_patches, latent_dim]

            # 3. Mean-pool to 1D for FAISS
            pooled_visual = self._pool_visual_state(visual_state)  # [latent_dim]

            # 4. Encode task using SigLIP
            semantic_vector = self.extract_semantic_target(task)  # [768]

            # 5. Add to episodic memory
            self.memory.add_memory(
                semantic_vector=semantic_vector,
                visual_state=pooled_visual,
                action_trajectory=action_trajectory,
            )

            return {"status": "memory_added_successfully"}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _query_memory(
        self,
        pooled_visual: np.ndarray,
        k: int = 3,
    ) -> Optional[np.ndarray]:
        """
        Query episodic memory and return the best action trajectory, or None.
        """
        results = self.memory.retrieve_closest_trajectory(
            query_state=pooled_visual.reshape(1, -1),
            target_semantic_vector=self.semantic_target,  # type: ignore[arg-type]
            k=k,
            alpha=self.config.memory.retrieval_alpha,
        )
        if results:
            _, _, traj = results[0]
            return traj  # [horizon, action_dim]
        return None

    def _plan(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Full re-plan: extract visual state, query memory, run diffusion.

        Returns a fresh action buffer of shape [1, horizon, action_dim].
        """
        visual_state = self._extract_visual_state(image_tensor)  # [1, patches, 1024]
        pooled = self._pool_visual_state(visual_state)

        memory_traj = self._query_memory(pooled)

        with torch.no_grad():
            actions = self.diffusion_policy.predict_action(
                visual_state.cpu().numpy(),  # dense conditioning
                semantic_condition=self.current_semantic_target,  # language conditioning
                num_inference_steps=self.num_inference_steps,
                memory_trajectory=memory_traj,
            )
        return actions  # [1, horizon, action_dim]

    # ── Main loop ───────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Enter the infinite request-reply loop.

        Handled message types:
            init  — receive a task, reset state, reply *ready*.
            step  — receive a JPEG frame, compute full action chunk, reply it.
        """
        print("[Brain] Waiting for connections …")
        while True:
            msg: Dict[str, Any] = self.socket.recv_pyobj()
            msg_type = msg.get("type")

            if msg_type == "init":
                task_text = msg.get("task", "")
                print(f"[Brain] init  →  task='{task_text}'")

                self.current_semantic_target = self.extract_semantic_target(task_text)
                self.step_counter = 0

                self.socket.send_pyobj({"status": "ready"})

            elif msg_type == "step":
                jpeg_bytes: bytes = msg["image"]

                # Decode image → V-JEPA tensor
                image_tensor = self._decode_image(
                    jpeg_bytes,
                    image_size=self.config.env.image_size,
                )

                # Plan full trajectory (stateless, runs every step)
                actions = self._plan(image_tensor)

                # Squeeze batch dimension and convert to Python list
                action_chunk = actions.squeeze(0).tolist()  # [horizon, action_dim]

                self.socket.send_pyobj({"action_chunk": action_chunk})

            elif msg_type == "add_memory":
                task: str = msg["task"]
                initial_image_bytes: bytes = msg["initial_image"]
                action_trajectory: np.ndarray = np.array(msg["action_trajectory"], dtype=np.float32)

                print(f"[Brain] add_memory → task='{task}', trajectory_shape={action_trajectory.shape}")

                # Ensure action_trajectory has correct shape [horizon, action_dim]
                expected_shape = (self.config.env.horizon, self.config.env.diffusion_action_dim)
                if action_trajectory.shape != expected_shape:
                    print(f"[Brain] WARNING: action_trajectory shape {action_trajectory.shape} != {expected_shape}")

                result = self.add_memory(
                    task=task,
                    initial_image_bytes=initial_image_bytes,
                    action_trajectory=action_trajectory,
                )

                self.socket.send_pyobj(result)

            else:
                self.socket.send_pyobj({"error": f"Unknown message type: {msg_type}"})

    def close(self) -> None:
        """Cleanly shut down the ZeroMQ socket."""
        print("[Brain] Shutting down …")
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.close()
        self.context.term()


# ── Entry point ─────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Brain — VLA model inference server")
    parser.add_argument("--bind", type=str, default="tcp://0.0.0.0:5555", help="ZeroMQ bind address")
    parser.add_argument("--inference-steps", type=int, default=20, help="Diffusion denoising steps")
    args = parser.parse_args()

    server = BrainServer(
        bind_address=args.bind,
        num_inference_steps=args.inference_steps,
    )

    try:
        server.run()
    except KeyboardInterrupt:
        print("\n[Brain] Interrupted by user")
    finally:
        server.close()
        print("Done!")


if __name__ == "__main__":
    main()
