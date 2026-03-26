"""
Autonomous Rollout Script for Object Theater VLA

Main inference loop that:
1. Resets environment
2. Extracts semantic target from user text via SigLIP
3. Queries LEMB for historical trajectory
4. Runs V-JEPA + Diffusion Policy
5. Executes predicted actions
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import time

from configs.device import DEVICE
from configs.config import Config, default_config

from envs.robosuite_sandbox import RobosuiteSandbox
from memory.lemb_core import EpisodicMemoryBuffer
from models.siglip_grounding import SigLIPTextEncoder
from models.v_jepa_encoder import VJepaPredictor, VJepaEncoder
from models.diffusion_policy import DiffusionPolicy


class AutonomousRollout:
    """
    Autonomous execution pipeline for Object Theater VLA.
    
    Implements the full inference loop:
    1. Semantic targeting via SigLIP
    2. Memory retrieval via LEMB
    3. State prediction via V-JEPA
    4. Action generation via Diffusion Policy
    5. Execution via Robosuite
    """
    
    def __init__(
        self,
        config: Config = None,
        env: RobosuiteSandbox = None,
        memory: EpisodicMemoryBuffer = None,
    ):
        """
        Initialize the autonomous rollout pipeline.
        
        Args:
            config: Configuration instance
            env: Pre-initialized Robosuite environment
            memory: Pre-populated EpisodicMemoryBuffer
        """
        self.config = config or default_config
        self.env = env or RobosuiteSandbox(
            render_mode="rgb_array",
            camera_name=self.config.env.camera_name,
            image_size=self.config.env.image_size,
        )
        
        self.memory = memory or EpisodicMemoryBuffer(
            embedding_dim=self.config.memory.embedding_dim,
            use_cosine_similarity=self.config.memory.use_cosine_similarity,
            max_memory_chunks=self.config.memory.max_memory_chunks,
        )
        
        # Initialize models
        self._init_models()
    
    def _init_models(self) -> None:
        """Initialize all models and move to device."""
        # SigLIP text encoder
        self.siglip = SigLIPTextEncoder(
            model_name=self.config.model.siglip_model_name,
            device=DEVICE,
        )
        
        # V-JEPA encoder and predictor
        self.vjepa_encoder = VJepaEncoder(
            image_size=self.config.env.image_size[0],
            embed_dim=self.config.model.vjepa_latent_dim,
        ).to(DEVICE)
        
        self.vjepa_predictor = VJepaPredictor(
            latent_dim=self.config.model.vjepa_latent_dim,
            action_dim=self.config.model.vjepa_action_dim,
            action_horizon=self.config.model.vjepa_action_horizon,
        ).to(DEVICE)
        
        # Diffusion policy
        self.diffusion_policy = DiffusionPolicy(
            latent_dim=self.config.model.vjepa_latent_dim,
            action_dim=self.config.model.diffusion_action_dim,
            action_horizon=self.config.model.vjepa_action_horizon,
            device=DEVICE,
        )
        
        # Ensure models are in eval mode
        self.siglip.model.eval()
        self.vjepa_encoder.eval()
        self.vjepa_predictor.eval()
        self.diffusion_policy.model.eval()
    
    def extract_semantic_target(self, text: str) -> np.ndarray:
        """
        Extract semantic embedding from text using SigLIP.
        
        Args:
            text: User-provided task description
        
        Returns:
            Normalized semantic embedding vector
        """
        with torch.no_grad():
            embedding = self.siglip.encode_text(text, normalize=True)
        return embedding
    
    def query_memory(
        self,
        visual_state: np.ndarray,
        semantic_target: np.ndarray,
    ) -> List[Tuple[int, float, np.ndarray]]:
        """
        Query LEMB for similar historical trajectories.
        
        Args:
            visual_state: Current visual state from environment
            semantic_target: Target semantic embedding
        
        Returns:
            List of (memory_id, score, action_trajectory) tuples
        """
        results = self.memory.retrieve_closest_trajectory(
            query_state=visual_state,
            target_semantic_vector=semantic_target,
            k=self.config.memory.retrieval_k,
            alpha=self.config.memory.retrieval_alpha,
        )
        return results
    
    def predict_next_state(
        self,
        current_visual_state: np.ndarray,
        retrieved_trajectory: np.ndarray,
    ) -> np.ndarray:
        """
        Predict next latent state using V-JEPA.
        
        Args:
            current_visual_state: Current image latent
            retrieved_trajectory: Retrieved action sequence
        
        Returns:
            Predicted next latent state
        """
        with torch.no_grad():
            current_tensor = torch.from_numpy(current_visual_state).to(DEVICE).float()
            action_tensor = torch.from_numpy(retrieved_trajectory).to(DEVICE).float()
            
            next_latent = self.vjepa_predictor(current_tensor, action_tensor)
        
        return next_latent.cpu().numpy()
    
    def generate_actions(
        self,
        condition: np.ndarray,
    ) -> np.ndarray:
        """
        Generate action sequence using diffusion policy.
        
        Args:
            condition: V-JEPA latent state for conditioning
        
        Returns:
            Generated action sequence
        """
        with torch.no_grad():
            actions = self.diffusion_policy.predict_action(
                condition,
                num_inference_steps=self.config.model.diffusion_num_inference_steps,
            )
        return actions
    
    def execute_trajectory(
        self,
        actions: np.ndarray,
        verbose: bool = False,
    ) -> Tuple[List[Dict], List[np.ndarray], List[float]]:
        """
        Execute action trajectory in the environment.
        
        Args:
            actions: Action sequence to execute
            verbose: If True, print execution info
        
        Returns:
            Tuple of (observations, actions, rewards)
        """
        observations = []
        rewards = []
        
        # Reset environment
        obs = self.env.reset()
        
        # Execute each action
        for i, action in enumerate(actions):
            # Clip action to valid range
            action = np.clip(action, -1.0, 1.0)
            
            # Execute step
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Store data
            observations.append(obs)
            rewards.append(reward)
            
            if verbose:
                print(f"Step {i}: reward={reward:.4f}, terminated={terminated}")
            
            if terminated or truncated:
                if verbose:
                    print(f"Episode terminated at step {i}")
                break
            
            # Control loop timing
            time.sleep(1.0 / self.config.env.control_freq)
        
        return observations, rewards
    
    def run_single_rollout(
        self,
        task_description: str,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run a single autonomous rollout.
        
        Args:
            task_description: User-provided task description
            verbose: If True, print progress
        
        Returns:
            Dictionary with rollout results
        """
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Running rollout: '{task_description}'")
            print(f"{'=' * 60}")
        
        # Step 1: Extract semantic target
        if verbose:
            print("Step 1: Extracting semantic target...")
        semantic_target = self.extract_semantic_target(task_description)
        
        # Step 2: Reset environment and get initial state
        if verbose:
            print("Step 2: Resetting environment...")
        initial_obs = self.env.reset()
        visual_state = self._extract_visual_state(initial_obs)
        
        # Step 3: Query memory
        if verbose:
            print("Step 3: Querying memory...")
        memory_results = self.query_memory(visual_state, semantic_target)
        
        if not memory_results:
            if verbose:
                print("No similar trajectories found in memory!")
            return {"success": False, "error": "No memory matches"}
        
        # Use best matching trajectory
        best_mem_id, best_score, best_trajectory = memory_results[0]
        if verbose:
            print(f"  Retrieved trajectory {best_mem_id} (score: {best_score:.4f})")
        
        # Step 4: Predict next state
        if verbose:
            print("Step 4: Predicting next state with V-JEPA...")
        current_latent = self.vjepa_encoder(
            torch.from_numpy(visual_state).to(DEVICE).float().unsqueeze(0)
        ).squeeze(0).cpu().numpy()
        
        next_latent = self.predict_next_state(current_latent, best_trajectory)
        
        # Step 5: Generate actions
        if verbose:
            print("Step 5: Generating actions with Diffusion Policy...")
        actions = self.generate_actions(next_latent)
        
        if verbose:
            print(f"  Generated {len(actions)} action steps")
        
        # Step 6: Execute trajectory
        if verbose:
            print("Step 6: Executing trajectory...")
        observations, rewards = self.execute_trajectory(actions, verbose=False)
        
        # Calculate statistics
        total_reward = sum(rewards)
        episode_length = len(observations)
        
        if verbose:
            print(f"\nResults:")
            print(f"  Total reward: {total_reward:.4f}")
            print(f"  Episode length: {episode_length} steps")
        
        return {
            "success": True,
            "task_description": task_description,
            "semantic_target": semantic_target,
            "memory_match_id": best_mem_id,
            "memory_score": best_score,
            "observations": observations,
            "rewards": rewards,
            "total_reward": total_reward,
            "episode_length": episode_length,
        }
    
    def _extract_visual_state(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Extract visual state from observations.
        
        Args:
            obs: Environment observation dictionary
        
        Returns:
            Visual state representation
        """
        # Get camera image
        image = obs.get("agentview_image", obs.get("robot0_agentview_image"))
        if image is None:
            raise ValueError("No camera observation found")
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(image).to(DEVICE).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        
        # Extract latent via V-JEPA encoder
        with torch.no_grad():
            latent = self.vjepa_encoder(image_tensor).squeeze(0).cpu().numpy()
        
        return latent
    
    def run_multiple_rollouts(
        self,
        task_descriptions: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Run multiple autonomous rollouts.
        
        Args:
            task_descriptions: List of task descriptions
        
        Returns:
            List of rollout results
        """
        results = []
        for task in task_descriptions:
            result = self.run_single_rollout(task)
            results.append(result)
        return results
    
    def close(self) -> None:
        """Clean up resources."""
        self.env.close()


# Main entry point
def main():
    """Main function to run autonomous rollout."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Autonomous rollout executor")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task description for the rollout",
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=1,
        help="Number of rollouts to run",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    args = parser.parse_args()
    
    # Initialize rollout pipeline
    rollout = AutonomousRollout()
    
    # Run rollouts
    try:
        if args.num_rollouts == 1:
            result = rollout.run_single_rollout(args.task, verbose=args.verbose)
            if result["success"]:
                print(f"\nRollout completed successfully!")
            else:
                print(f"\nRollout failed: {result.get('error', 'Unknown error')}")
        else:
            for i in range(args.num_rollouts):
                print(f"\n--- Rollout {i+1}/{args.num_rollouts} ---")
                result = rollout.run_single_rollout(args.task, verbose=args.verbose)
                print(f"Reward: {result.get('total_reward', 0):.4f}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        rollout.close()
        print("Done!")


if __name__ == "__main__":
    main()
