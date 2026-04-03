"""
Client Body — Robosuite Simulation Frontend

ZeroMQ REQ client that runs the Robosuite environment with a local 3D
GUI (has_renderer=True) and streams compressed camera frames to the remote
*Brain* server for action inference.

No AI models are loaded here — this script is purely physics + rendering +
network I/O.

Protocol (pickle-serialised dicts):
    CLIENT → SERVER
        {"type": "init", "task": "<task description>"}
        {"type": "step", "image": <jpeg-encoded bytes>}
    SERVER → CLIENT
        {"status": "ready"}
        {"action": [dx, dy, dz, roll, pitch, yaw, gripper]}
"""

from typing import Any, Dict, Optional

import cv2  # type: ignore[import-untyped]
import numpy as np
import zmq

from configs.device import DEVICE  # noqa: F401 — kept for module import consistency
from configs.config import Config, default_config
from envs.robosuite_sandbox import RobosuiteSandbox


class BodyClient:
    """
    ZeroMQ REQ client wrapping a local Robosuite environment.

    Captures camera observations, compresses them as JPEG, sends to the
    Brain server, and applies the returned actions step-by-step.
    """

    def __init__(
        self,
        server_address: str = "tcp://localhost:5555",
        config: Config = None,
    ):
        """
        Initialise the Body client.

        Args:
            server_address: ZeroMQ address of the Brain server.
            config: Hyperparameter configuration.
        """
        self.config = config or default_config

        # -------- ZeroMQ socket --------
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(server_address)
        print(f"[Body] Connected to Brain at {server_address}")

        # -------- Robosuite environment --------
        self.env = RobosuiteSandbox(
            render_mode="human",  # shows local 3D GUI
            camera_name=self.config.env.camera_name,
            image_size=self.config.env.image_size,
            control_freq=self.config.env.control_freq,
            horizon=self.config.env.horizon,
        )

    # ── Helpers ─────────────────────────────────────────────────────────

    def _compress_frame(self, frame: np.ndarray, quality: int = 85) -> bytes:
        """
        Compress an RGB frame as JPEG.

        Args:
            frame: H×W×3 numpy array (uint8).
            quality: JPEG quality (0–100, higher = larger / better).

        Returns:
            Raw JPEG bytes ready for ZeroMQ transport.
        """
        # Convert RGB → BGR for cv2.imencode
        if frame.shape[-1] == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame

        success, encoded = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not success:
            raise RuntimeError("Failed to encode frame as JPEG")
        # Type narrowing: encoded is always ndarray[uint8] when success is True
        return encoded.tobytes()  # type: ignore[return-value]

    def _send_init(self, task: str) -> Dict[str, Any]:
        """Send the task description and wait for server readiness."""
        self.socket.send_pyobj({"type": "init", "task": task})
        reply: Dict[str, Any] = self.socket.recv_pyobj()
        return reply

    def _send_step(self, jpeg_bytes: bytes) -> Dict[str, Any]:
        """Send a compressed frame and receive the next action."""
        self.socket.send_pyobj({"type": "step", "image": jpeg_bytes})
        reply: Dict[str, Any] = self.socket.recv_pyobj()
        return reply

    # ── Execution loop ──────────────────────────────────────────────────

    def run_episode(
        self,
        task: str,
        max_steps: int = 200,
        jpeg_quality: int = 85,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run a single episode end-to-end.

        1. Prompt the Brain with the task string.
        2. Reset the physics engine.
        3. Stream frames, receive actions, step the simulation.

        Args:
            task: Natural-language task description.
            max_steps: Maximum simulation steps before auto-terminating.
            jpeg_quality: JPEG compression quality for camera frames.
            verbose: If *True*, print progress information.

        Returns:
            Dictionary with observations, actions, rewards, and metadata.
        """
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Running episode: '{task}'")
            print(f"{'=' * 60}")

        # ── 1. Initialise server-side models ──
        if verbose:
            print("[Body] Sending init request to Brain …")
        reply = self._send_init(task)
        if reply.get("status") != "ready":
            raise RuntimeError(f"Brain did not become ready: {reply}")
        if verbose:
            print("[Body] Brain is ready.")

        # ── 2. Reset environment ──
        if verbose:
            print("[Body] Resetting environment …")
        obs = self.env.reset()
        if verbose:
            print("[Body] Environment reset.")

        # ── 3. Simulation loop ──
        observations: list = []
        actions_executed: list = []
        rewards: list = []

        for step in range(max_steps):
            # Grab camera frame: typically under 'agentview_image' or 'robot0_agentview_image'
            frame: Optional[np.ndarray] = obs.get("agentview_image") or obs.get("robot0_agentview_image")
            if frame is None:
                raise KeyError("No camera observation found in observation dict")

            # Compress and send
            jpeg_bytes = self._compress_frame(frame, quality=jpeg_quality)
            reply = self._send_step(jpeg_bytes)

            if "error" in reply:
                print(f"[Body] Server error: {reply['error']}")
                break

            action = np.array(reply["action"], dtype=np.float32)
            action = np.clip(action, -1.0, 1.0)

            # Apply action to physics engine
            obs, reward, terminated, truncated, info = self.env.step(action)

            # Record
            observations.append(obs)
            actions_executed.append(action)
            rewards.append(reward)

            # Render the local 3D viewer
            self.env.render()

            if verbose and (step + 1) % 10 == 0:
                print(f"  Step {step + 1}: reward={reward:.4f}")

            # Control loop pacing
            import time
            time.sleep(1.0 / self.config.env.control_freq)

            if terminated or truncated:
                if verbose:
                    print(f"  Episode terminated at step {step + 1}")
                break

        # ── Statistics ──
        total_reward = float(sum(rewards))
        episode_length = len(observations)

        if verbose:
            print(f"\nResults:")
            print(f"  Total reward: {total_reward:.4f}")
            print(f"  Episode length: {episode_length} steps")

        return {
            "success": True,
            "task": task,
            "observations": observations,
            "actions_executed": actions_executed,
            "rewards": rewards,
            "total_reward": total_reward,
            "episode_length": episode_length,
        }

    def close(self) -> None:
        """Clean up resources."""
        self.env.close()
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.close()
        self.context.term()
        print("[Body] Shut down.")


# ── Entry point ─────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Body — Local Robosuite + networked Brain client")
    parser.add_argument(
        "--server",
        type=str,
        default="tcp://localhost:5555",
        help="Brain server ZeroMQ address",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task description to execute",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=85,
        help="JPEG compression quality (0-100)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    args = parser.parse_args()

    client = BodyClient(server_address=args.server)

    try:
        result = client.run_episode(
            task=args.task,
            max_steps=args.max_steps,
            jpeg_quality=args.jpeg_quality,
            verbose=args.verbose,
        )
        if result["success"]:
            print(f"\nEpisode completed successfully!  Total reward: {result['total_reward']:.4f}")
    except KeyboardInterrupt:
        print("\n[Body] Interrupted by user")
    finally:
        client.close()
        print("Done!")


if __name__ == "__main__":
    main()
