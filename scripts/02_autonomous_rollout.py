"""
Autonomous Rollout Script for Object Theater VLA

Main inference loop with Receding Horizon Control for dense V-JEPA 2.1
conditioning and Cross-Attention Diffusion Policy.

Implements:
1. Semantic targeting via SigLIP
2. Dense visual state extraction (3D feature maps)
3. Receding horizon execution with periodic re-planning
4. Memory querying with pooled (1D) features
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
from models.v_jepa_encoder import VJepaEncoder
from models.diffusion_policy import DiffusionPolicy


class AutonomousRollout:
    """
    Autonomous execution pipeline with dense VLA architecture.
    
    Implements Receding Horizon Control:
    1. Extract dense visual state [batch, patches, dim]
    2. Periodically (every N steps) re-plan using diffusion policy
    3. Execute actions one step at a time, re-planning periodically
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
        
        # V-JEPA 2.1 dense encoder (Gigantic model)
        self.vjepa_encoder = VJepaEncoder().to(DEVICE)
        
        # Diffusion policy with Cross-Attention conditioning
        self.diffusion_policy = DiffusionPolicy(
            latent_dim=self.config.model.vjepa_latent_dim,
            action_dim=self.config.model.diffusion_action_dim,
            action_horizon=self.config.model.diffusion_action_horizon,
            device=DEVICE,
        )
        
        # Ensure models are in eval mode with torch.no_grad()
        self.siglip.model.eval()
        self.vjepa_encoder.eval()
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
    
    def _extract_visual_state(self, obs: Dict[str, np.ndarray]) -> torch.Tensor:
        """
        Extract dense visual state from environment observation.
        
        Formats camera image to [batch_size, channels, frames, height, width]
        and returns dense V-JEPA feature map [batch, num_patches, latent_dim].
        
        Args:
            obs: Environment observation dictionary
        
        Returns:
            Dense visual state tensor
            Shape: [1, num_patches, 1024] for single frame
        """
        # Get camera image
        image = obs.get("agentview_image", obs.get("robot0_agentview_image"))
        if image is None:
            raise ValueError("No camera observation found")
        
        # Convert to tensor: [H, W, C] -> [C, H, W]
        image_tensor = torch.from_numpy(image).to(DEVICE).float()
        image_tensor = image_tensor.permute(2, 0, 1)  # (C, H, W)
        
        # Add frame dimension for video format: [C, H, W] -> [1, C, 1, H, W]
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(2)  # (1, C, 1, H, W)
        
        # Extract dense features via V-JEPA 2.1 encoder
        with torch.no_grad():
            dense_features = self.vjepa_encoder.extract_features(image_tensor)
        
        return dense_features  # Shape: [1, num_patches, 1024]
    
    def pool_visual_state(self, visual_state: torch.Tensor) -> torch.Tensor:
        """
        Pool dense visual state to 1D vector for memory querying.
        
        Args:
            visual_state: Dense feature map from V-JEPA
                Shape: [batch_size, num_patches, latent_dim]
        
        Returns:
            Pooled feature vector
            Shape: [batch_size, latent_dim]
        """
        # Mean-pool across spatial (patch) dimension
        return visual_state.mean(dim=1)  # (B, latent_dim)
    
    def query_memory(
        self,
        visual_state: torch.Tensor,
        semantic_target: np.ndarray,
    ) -> List[Tuple[int, float, np.ndarray]]:
        """
        Query LEMB for similar historical trajectories.
        
        The dense visual_state is pooled to 1D before memory query.
        
        Args:
            visual_state: Dense visual state from environment
                Shape: [1, num_patches, 1024]
            semantic_target: Target semantic embedding
        
        Returns:
            List of (memory_id, score, action_trajectory) tuples
        """
        # Pool dense features to 1D for FAISS compatibility
        pooled_state = self.pool_visual_state(visual_state).cpu().numpy()
        
        results = self.memory.retrieve_closest_trajectory(
            query_state=pooled_state,
            target_semantic_vector=semantic_target,
            k=self.config.memory.retrieval_k,
            alpha=self.config.memory.retrieval_alpha,
        )
        return results
    
    def generate_actions(
        self,
        condition: torch.Tensor,
        num_inference_steps: int = 20,
        memory_trajectory: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate action sequence using diffusion policy with optional trajectory priming.
        
        Passes the FULL dense condition tensor to the diffusion policy
        for Cross-Attention conditioning.
        
        Args:
            condition: V-JEPA dense feature map
                Shape: [batch_size, num_patches, latent_dim]
            num_inference_steps: Number of denoising steps
            memory_trajectory: Historical trajectory to prime the diffusion process
                Shape: [action_horizon, action_dim] or [batch_size, action_horizon, action_dim]
        
        Returns:
            Generated action sequence
            Shape: [batch_size, horizon, action_dim]
        """
        with torch.no_grad():
            actions = self.diffusion_policy.predict_action(
                condition.cpu().numpy(),
                num_inference_steps=num_inference_steps,
                memory_trajectory=memory_trajectory,
            )
        return actions
    
    def run_single_rollout(
        self,
        task_description: str,
        max_steps: int = 200,
        replan_interval: int = 8,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run a single autonomous rollout with Receding Horizon Control.
        
        Implements closed-loop execution:
        1. Extract dense visual state
        2. Every N steps, re-plan with diffusion policy
        3. Execute actions one step at a time
        
        Args:
            task_description: User-provided task description
            max_steps: Maximum steps per rollout
            replan_interval: Steps between re-planning
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
        
        # Step 2: Reset environment
        if verbose:
            print("Step 2: Resetting environment...")
        obs = self.env.reset()
        
        # Step 3: Extract initial dense visual state
        visual_state = self._extract_visual_state(obs)
        if verbose:
            print(f"  Visual state shape: {visual_state.shape}")
        
        # Initialize rollout tracking
        observations = []
        actions_executed = []
        rewards = []
        generated_sequences = []
        action_sequence_index = 0  # Track position in current action sequence
        
        # Generate initial action sequence
        if verbose:
            print("Step 3: Initial action planning...")
        
        # Check if we have a memory trajectory to prime with
        memory_results = self.query_memory(visual_state, semantic_target)
        memory_trajectory = None
        if memory_results:
            best_mem_id, best_score, best_trajectory, _ = memory_results[0]
            memory_trajectory = best_trajectory
            if verbose:
                print(f"  Memory match: ID={best_mem_id}, score={best_score:.4f}")
        
        current_actions = self.generate_actions(visual_state, memory_trajectory=memory_trajectory)
        generated_sequences.append(current_actions)
        if verbose:
            print(f"  Generated action sequence shape: {current_actions.shape}")
        
        # Step 4: Receding Horizon Execution Loop
        if verbose:
            print("Step 4: Executing with Receding Horizon Control...")
        
        for step in range(max_steps):
            # Validate action sequence index
            if action_sequence_index >= self.config.model.diffusion_action_horizon:
                # Need to re-plan - extract new visual state and generate new sequence
                visual_state = self._extract_visual_state(obs)
                memory_results = self.query_memory(visual_state, semantic_target)
                current_actions = self.generate_actions(visual_state)
                generated_sequences.append(current_actions)
                action_sequence_index = 0
            
            # Pop next action from generated sequence using current index
            next_action = current_actions[0, action_sequence_index]
            
            # Clip action to valid range
            next_action = np.clip(next_action, -1.0, 1.0)
            
            # Execute step in environment
            obs, reward, terminated, truncated, info = self.env.step(next_action)
            
            # Store data
            observations.append(obs)
            actions_executed.append(next_action)
            rewards.append(reward)
            
            if verbose and step % 10 == 0:
                print(f"  Step {step}: reward={reward:.4f}")
            
            # Check if task completed
            if terminated or truncated:
                if verbose:
                    print(f"  Episode terminated at step {step}")
                break
            
            # Re-plan every replan_interval steps (regardless of sequence index)
            if (step + 1) % replan_interval == 0:
                # Extract new dense visual state
                visual_state = self._extract_visual_state(obs)
                
                # Query memory (optional - for trajectory priming)
                memory_results = self.query_memory(visual_state, semantic_target)
                
                # Extract memory trajectory if available
                memory_trajectory = None
                if memory_results:
                    best_mem_id, best_score, best_trajectory = memory_results[0]
                    memory_trajectory = best_trajectory
                    if verbose:
                        print(f"  Memory match: ID={best_mem_id}, score={best_score:.4f}")
                
                # Generate new action sequence with trajectory priming
                current_actions = self.generate_actions(visual_state, memory_trajectory=memory_trajectory)
                generated_sequences.append(current_actions)
                action_sequence_index = 0
            else:
                # Step through current sequence
                action_sequence_index += 1
            
            # Control loop timing
            time.sleep(1.0 / self.config.env.control_freq)
        
        # Calculate statistics
        total_reward = sum(rewards)
        episode_length = len(observations)
        num_replans = len(generated_sequences) - 1
        
        if verbose:
            print(f"\nResults:")
            print(f"  Total reward: {total_reward:.4f}")
            print(f"  Episode length: {episode_length} steps")
            print(f"  Number of replans: {num_replans}")
        
        return {
            "success": True,
            "task_description": task_description,
            "semantic_target": semantic_target,
            "observations": observations,
            "actions_executed": actions_executed,
            "rewards": rewards,
            "total_reward": total_reward,
            "episode_length": episode_length,
            "num_replans": num_replans,
        }
    
    def run_multiple_rollouts(
        self,
        task_descriptions: List[str],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Run multiple autonomous rollouts.
        
        Args:
            task_descriptions: List of task descriptions
            **kwargs: Arguments passed to run_single_rollout
        
        Returns:
            List of rollout results
        """
        results = []
        for task in task_descriptions:
            result = self.run_single_rollout(task, **kwargs)
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
        "--max-steps",
        type=int,
        default=200,
        help="Maximum steps per rollout",
    )
    parser.add_argument(
        "--replan-interval",
        type=int,
        default=8,
        help="Steps between re-planning",
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
            result = rollout.run_single_rollout(
                args.task,
                max_steps=args.max_steps,
                replan_interval=args.replan_interval,
                verbose=args.verbose,
            )
            if result["success"]:
                print(f"\nRollout completed successfully!")
            else:
                print(f"\nRollout failed: {result.get('error', 'Unknown error')}")
        else:
            for i in range(args.num_rollouts):
                print(f"\n--- Rollout {i+1}/{args.num_rollouts} ---")
                result = rollout.run_single_rollout(
                    args.task,
                    max_steps=args.max_steps,
                    replan_interval=args.replan_interval,
                    verbose=args.verbose,
                )
                print(f"Reward: {result.get('total_reward', 0):.4f}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        rollout.close()
        print("Done!")


if __name__ == "__main__":
    main()
