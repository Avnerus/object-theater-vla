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
        {"action_chunk": [[dx, dy, dz, roll, pitch, yaw, gripper], ...]}

Asynchronous action chunking
    The client receives full 16-step action chunks from the server and buffers
    them locally. Actions are consumed at the native control_freq while the
    next chunk is fetched asynchronously in a background thread to mask network
    latency.
"""

import sys
import select

from collections import deque
from threading import Thread, Lock
from typing import Any, Deque, Dict, List, Optional

import cv2  # type: ignore[import-untyped]
import numpy as np
import zmq

from configs.config import Config, default_config
from envs.robosuite_sandbox import RobosuiteSandbox


# ── Intervention Manager ──────────────────────────────────────────────────


class InterventionManager:
    """
    Handles force-threshold intervention detection and recording.

    Monitors EEF force sensor and keyboard input to detect when a human
    takes physical control of the robot. Records the manual trajectory
    and can inject it into the Brain's episodic memory.
    """

    def __init__(
        self,
        client: "BodyClient",
        force_threshold: float = 15.0,
        device: str = "keyboard",
    ):
        """
        Initialize the Intervention Manager.

        Args:
            client: Parent BodyClient instance.
            force_threshold: Force magnitude threshold for intervention trigger.
            device: Intervention control device ("keyboard" or "spacemouse").
        """
        self.client = client
        self.force_threshold = force_threshold
        self.device_type = device
        self._keyboard_device: Optional[Any] = None
        self._spacemouse_device: Optional[Any] = None
        self._takeover_active = False

        if device == "keyboard":
            try:
                from robosuite.devices import Keyboard
                # Standard initialization with env and movement sensitivities
                self._keyboard_device = Keyboard(self.client.env.env, pos_sensitivity=0.05, rot_sensitivity=0.05) 
                print("[InterventionManager] Keyboard device initialized.")
            except ImportError:
                print("[InterventionManager] WARNING: robosuite.devices.Keyboard not available.")
                self._keyboard_device = None

    def check_force_sensor(self, obs: Dict[str, np.ndarray]) -> bool:
        """
        Check if force threshold is exceeded.

        Args:
            obs: Current environment observation dict.

        Returns:
            True if force magnitude exceeds threshold.
        """
        eef_force = obs.get("robot0_eef_force", np.zeros(3))
        force_magnitude = np.linalg.norm(eef_force)
        return bool(force_magnitude > self.force_threshold)

    def start_takeover(self) -> None:
        """Initialize takeover mode."""
        self._takeover_active = True
        if self._keyboard_device is not None:
            # Reset internal state and enable the device
            self._keyboard_device._reset_internal_state()
            self._keyboard_device._reset_state = 0
            self._keyboard_device._enabled = True
            print("[InterventionManager] Keyboard device enabled.")
        print("INTERVENTION DETECTED: Yielding to human...")

    def stop_takeover(self) -> None:
        """End takeover mode."""
        self._takeover_active = False

    def read_takeover_action(self) -> Optional[np.ndarray]:
        """
        Read action from takeover device.

        Returns:
            Action array or None if device unavailable.
        """
        if self._keyboard_device is not None:
            state = self._keyboard_device.get_controller_state()
            print(f"Controller state: {state}")
            # State contains: pos, orn, grasp, reset
            # Build action array: [dx, dy, dz, roll, pitch, yaw, gripper]
            pos = state.get("dpos", np.zeros(3)) * 10.0
            orn = state.get("raw_drotation", np.zeros(3)) * 10.0
            grasp = state.get("grasp", 0)
            
            # Map device output to action format
            # pos → dx, dy, dz; orn → roll, pitch, yaw; grasp → gripper
            action = np.concatenate([pos, orn, [grasp]]).astype(np.float32)
            return action
        return None

    def record_intervention(
        self,
        initial_obs: Dict[str, np.ndarray],
        task_label: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Record a human demonstration during intervention and split into
        discrete milestones using kinematic rules.

        Args:
            initial_obs: Observation at intervention start.
            task_label: Pre-labeled task description from user intent.

        Returns:
            List of chunk dictionaries, each with initial_image,
            action_trajectory, and task, or None if intervention failed.
        """
        initial_image = initial_obs.get("agentview_image")
        if initial_image is None:
            return None

        chunks = []
        current_chunk_actions = []
        current_start_image = initial_image

        # Kinematic state trackers
        prev_gripper = -1.0
        prev_vel = 0.0
        was_in_contact = False

        print("[InterventionManager] Recording manual guidance...")
        print("  - Click the 3D window and use the keyboard to move")
        print("  - Type 'f' and press ENTER in this terminal to finish recording")

        while True:
            action = self.read_takeover_action()
            if action is None:
                action = np.zeros(7, dtype=np.float32)
                action[6] = -1.0

            obs, reward, terminated, truncated, info = self.client.env.step(action)
            self.client.env.render()

            if self.client._poll_terminal() == 'f':
                print("[InterventionManager] Human finished recording early.")
                break

            current_chunk_actions.append(action.tolist())

            # --- Physics Edge Detectors ---
            curr_gripper = action[6]
            curr_vel = np.linalg.norm(action[:3])
            force_mag = np.linalg.norm(obs.get("robot0_eef_force", np.zeros(3)))
            is_in_contact = force_mag > 15.0

            gripper_switched = abs(curr_gripper - prev_gripper) > 1.0
            came_to_stop = (curr_vel < 0.05) and (prev_vel >= 0.05)
            made_contact = is_in_contact and not was_in_contact

            # Chunk if a physical milestone is detected (minimum 15 steps to prevent micro-chunks)
            if (gripper_switched or came_to_stop or made_contact) and len(current_chunk_actions) >= 15:
                if current_start_image is not None:
                    chunks.append({
                        "initial_image": self.client._compress_frame(current_start_image),
                        "action_trajectory": np.array(current_chunk_actions, dtype=np.float32),
                        "task": task_label
                    })
                    current_chunk_actions = []
                    current_start_image = obs.get("agentview_image")

            # Update trackers
            prev_gripper = curr_gripper
            prev_vel = curr_vel
            was_in_contact = is_in_contact

            import time
            time.sleep(1.0 / self.client.config.env.control_freq)

            if terminated or truncated:
                break

        # Append the final chunk if it contains data
        if len(current_chunk_actions) > 0:
            if current_start_image is not None:
                chunks.append({
                    "initial_image": self.client._compress_frame(current_start_image),
                    "action_trajectory": np.array(current_chunk_actions, dtype=np.float32),
                    "task": task_label
                })

        # --- Quality Control (QC) Gate ---
        self.client._exit_raw_mode()
        print(f"\n[InterventionManager] Extracted {len(chunks)} physical milestones.")
        verify = input(f"Was this execution of '{task_label}' successful? (y/n): ").strip().lower()
        self.client._enter_raw_mode()

        if verify != 'y':
            print("[InterventionManager] Demonstration discarded. FAISS index protected.")
            return None

        return chunks


class BodyClient:
    """
    ZeroMQ REQ client wrapping a local Robosuite environment.

    Captures camera observations, compresses them as JPEG, sends to the
    Brain server, and applies the returned actions step-by-step using
    asynchronous action chunking.

    The client maintains an action buffer that is populated by a background
    thread fetching new chunks from the server. Actions are popped from the
    buffer at the native control_freq, ensuring smooth execution even under
    network latency.
    """

    def __init__(
        self,
        server_address: str = "tcp://localhost:5555",
        config: Optional[Config] = None,
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

        # -------- Action buffer and threading primitives --------
        self._action_queue: Deque[List[float]] = deque()
        self._buffer_lock = Lock()
        self._fetcher_thread: Optional[Thread] = None
        self._fetcher_running = False
        self._fetcher_lock = Lock()

        # -------- Safe fallback action (zero velocity) --------
        self._zero_action = np.zeros(self.config.env.diffusion_action_dim, dtype=np.float32)
        self._zero_action[-1] = -1.0  # gripper open

        # -------- Terminal State (for cbreak mode) --------
        self._fd = sys.stdin.fileno() if sys.platform != 'win32' else None
        self._old_termios = None

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
        """Send a compressed frame and receive the next action chunk."""
        self.socket.send_pyobj({"type": "step", "image": jpeg_bytes})
        reply: Dict[str, Any] = self.socket.recv_pyobj()
        return reply

    # ── Asynchronous action chunking ────────────────────────────────────

    def _fetch_action_chunk(self, jpeg_bytes: bytes) -> None:
        """
        Background thread method to fetch an action chunk from the server.

        This method is called from the main loop when the action buffer falls
        below the threshold. It blocks on network I/O but does not affect the
        simulation loop since it runs in a separate thread.

        Args:
            jpeg_bytes: Compressed camera frame to send for action prediction.
        """
        try:
            reply = self._send_step(jpeg_bytes)
            if "action_chunk" in reply:
                action_chunk = reply["action_chunk"]
                if isinstance(action_chunk, list) and len(action_chunk) > 0:
                    with self._buffer_lock:
                        self._action_queue.extend(action_chunk)
        except Exception as e:
            print(f"[Body] Error fetching action chunk: {e}")
        finally:
            with self._fetcher_lock:
                self._fetcher_running = False

    def _ensure_action_buffer(self, jpeg_frame: np.ndarray) -> None:
        """
        Ensure the action buffer has enough actions to avoid starvation.

        If the buffer length falls below the threshold, spawn a background
        thread to fetch the next chunk asynchronously.

        Args:
            jpeg_frame: Current camera frame to use for prediction request.
        """
        with self._buffer_lock:
            queue_len = len(self._action_queue)

        if queue_len <= self.config.env.chunk_request_threshold:
            with self._fetcher_lock:
                if not self._fetcher_running:
                    self._fetcher_running = True
                    jpeg_bytes = self._compress_frame(jpeg_frame)
                    self._fetcher_thread = Thread(
                        target=self._fetch_action_chunk,
                        args=(jpeg_bytes,),
                        daemon=True,
                    )
                    self._fetcher_thread.start()

    def _enter_raw_mode(self) -> None:
        """Put the terminal into cbreak mode for instant keystroke reading."""
        if self._fd is not None:
            import termios
            import tty
            import atexit  # <-- NEW: Import atexit
            
            # Save current settings
            self._old_termios = termios.tcgetattr(self._fd)
            
            # Register the safety net BEFORE changing the terminal state
            atexit.register(self._exit_raw_mode)
            
            # Drop into raw mode
            tty.setcbreak(self._fd)

    def _exit_raw_mode(self) -> None:
        """Restore the terminal to normal canonical mode."""
        if self._fd is not None and self._old_termios is not None:
            import termios
            try:
                # TCSANOW applies the change instantly
                termios.tcsetattr(self._fd, termios.TCSANOW, self._old_termios)
            except Exception:
                pass  # Swallow exceptions during teardown

    def _poll_terminal(self) -> str:
        """Read a single character instantly without toggling termios state."""
        if sys.platform == 'win32':
            import msvcrt
            if msvcrt.kbhit():
                try:
                    return msvcrt.getch().decode('utf-8').strip().lower()
                except Exception:
                    pass
            return ""
        
        import select
        i, _, _ = select.select([sys.stdin], [], [], 0.0)
        if i:
            return sys.stdin.read(1).lower()
        return ""

    # ── Execution loop ──────────────────────────────────────────────────

    def run_episode(
        self,
        task: str,
        max_steps: int = 200,
        jpeg_quality: int = 85,
        verbose: bool = True,
        intervention_enabled: bool = True,
    ) -> Dict[str, Any]:
        """
        Run a single episode end-to-end with asynchronous action chunking and optional intervention.

        1. Prompt the Brain with the task string.
        2. Reset the physics engine.
        3. Stream frames, receive action chunks, buffer locally.
        4. Consume actions from buffer while fetching next chunk asynchronously.
        5. Monitor for force-threshold intervention and record human guidance.

        Args:
            task: Natural-language task description.
            max_steps: Maximum simulation steps before auto-terminating.
            jpeg_quality: JPEG compression quality for camera frames.
            verbose: If *True*, print progress information.
            intervention_enabled: If *True*, enable force-threshold intervention mode.

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

        # ── 2. Initialize InterventionManager if enabled ──
        intervention_manager: Optional[InterventionManager] = None
        if intervention_enabled:
            intervention_manager = InterventionManager(self)
            print("[Body] InterventionManager initialized.")

        # ── 3. Reset environment ──
        if verbose:
            print("[Body] Resetting environment …")
        obs = self.env.reset()
        if verbose:
            print("[Body] Environment reset.")

        # ── 4. Simulation loop ──
        observations: list = []
        actions_executed: list = []
        rewards: list = []
        keyboard_triggered = False  # For keyboard 'T' intervention trigger
        
        self._enter_raw_mode()
        for step in range(max_steps):
            # Grab camera frame: typically under 'agentview_image' or 'robot0_agentview_image'
            frame: Optional[np.ndarray] = obs.get("agentview_image")
            if frame is None:
                raise KeyError("No camera observation found in observation dict")

            # Ensure action buffer has enough actions
            self._ensure_action_buffer(frame)

            # Poll the terminal for user commands
            term_input = self._poll_terminal()

            # Check for force-threshold intervention trigger
            if intervention_manager is not None:
                force_triggered = intervention_manager.check_force_sensor(obs)
                keyboard_triggered = (term_input == 't')

                if force_triggered or keyboard_triggered:
                    # 1. Pre-Label Intent UX
                    self._exit_raw_mode()
                    print("\n[Intervention] Pausing physics engine.")
                    task_label = input("What are we about to teach the robot? (e.g., 'drop the block'): ").strip()
                    if not task_label:
                        task_label = "intervention_demo"
                    self._enter_raw_mode()

                    # 2. Record & Extract Milestones
                    intervention_manager.start_takeover()
                    intervention_chunks = intervention_manager.record_intervention(obs, task_label=task_label)
                    intervention_manager.stop_takeover()

                    # 3. Dynamic Memory Injection (Sequential Milestones)
                    if intervention_chunks is not None:
                        print(f"[Body] Injecting {len(intervention_chunks)} milestones into Brain's episodic memory...")
                        self.socket.setsockopt(zmq.LINGER, -1)
                        
                        success_count = 0
                        for i, chunk in enumerate(intervention_chunks):
                            payload = {
                                "type": "add_memory",
                                "task": chunk["task"],
                                "initial_image": chunk["initial_image"],
                                "action_trajectory": chunk["action_trajectory"].tolist(),
                            }
                            self.socket.send_pyobj(payload)
                            reply = self.socket.recv_pyobj()
                            if reply.get("status") == "memory_added_successfully":
                                success_count += 1
                                
                        self.socket.setsockopt(zmq.LINGER, 0)
                        print(f"[Body] Successfully injected {success_count}/{len(intervention_chunks)} milestones. Resuming autonomy.")

                    # Flush the old action buffer so we don't execute stale movements
                    with self._buffer_lock:
                        self._action_queue.clear()
                        
                    if verbose:
                        print("[Body] Resuming autonomy from current physical state.")
                    continue

            # Check for chat trigger ('c' key)
            chat_triggered = (term_input == 'c')

            if chat_triggered:
                self._exit_raw_mode()  # Suspend raw mode so user can type normally
                print("Chat mode enabled. Type 'exit' to return to control.")
                user_msg = input("Talk to the robot: ").strip()
                if user_msg.lower() != "exit":
                    self.socket.send_pyobj({"type": "chat", "text": user_msg})
                    chat_reply = self.socket.recv_pyobj()
                    print(f"[Robot]: {chat_reply['reply']}\n")
                self._enter_raw_mode() # Resume raw mode

            # Get next action from buffer (or zero action if buffer is empty)
            with self._buffer_lock:
                if len(self._action_queue) > 0:
                    action_list = self._action_queue.popleft()
                else:
                    action_list = self._zero_action.tolist()

            action = np.array(action_list, dtype=np.float32)
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
                print(f"  Step {step + 1}: reward={reward:.4f}, buffer_len={len(self._action_queue)}")

            # Control loop pacing
            import time
            time.sleep(1.0 / self.config.env.control_freq)

            if terminated or truncated:
                if verbose:
                    print(f"  Episode terminated at step {step + 1}")
                break

        self._exit_raw_mode()  # Restore terminal to normal mode

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
    parser.add_argument(
        "--no-intervention",
        action="store_true",
        help="Disable force-threshold intervention mode",
    )
    args = parser.parse_args()

    client = BodyClient(server_address=args.server)

    try:
        result = client.run_episode(
            task=args.task,
            max_steps=args.max_steps,
            jpeg_quality=args.jpeg_quality,
            verbose=args.verbose,
            intervention_enabled=not args.no_intervention,
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
