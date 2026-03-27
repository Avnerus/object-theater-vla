"""
Visualization utilities for Object Theater VLA.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import torch


def plot_action_sequence(
    actions: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Action Sequence",
) -> None:
    """
    Plot action sequence as subplots.
    
    Args:
        actions: Action sequence array (horizon, action_dim)
        labels: Optional list of action dimension names
        title: Plot title
    """
    horizon, action_dim = actions.shape
    
    if labels is None:
        labels = [f"Action {i}" for i in range(action_dim)]
    
    fig, axes = plt.subplots(action_dim, 1, figsize=(12, 3 * action_dim))
    
    if action_dim == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        ax.plot(actions[:, i], marker='o', linestyle='-', linewidth=2)
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, horizon - 1)
    
    axes[-1].set_xlabel("Timestep")
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def animate_trajectory(
    observations: List[Dict[str, np.ndarray]],
    actions: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    fps: int = 20,
) -> None:
    """
    Create animation of trajectory.
    
    Args:
        observations: List of observation dictionaries with 'agentview_image'
        actions: Optional action sequence for overlay
        output_path: If provided, save as GIF/PNG
        fps: Frames per second
    """
    if not observations:
        print("No observations to animate")
        return
    
    # Get image from first observation
    first_obs = observations[0]
    image = first_obs.get("agentview_image", None)
    
    if image is None:
        # Try other camera keys
        for key in ["robot0_agentview_image", "image"]:
            if key in first_obs:
                image = first_obs[key]
                break
    
    if image is None:
        print("No image found in observations")
        return
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Display first frame
    img_artist = ax.imshow(image)
    ax.axis("off")
    
    # Add text for action overlay
    action_text = ax.text(
        0.02, 0.98, "", transform=ax.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    def update(frame):
        obs = observations[frame]
        
        # Update image
        image = obs.get("agentview_image", None)
        if image is None:
            for key in ["robot0_agentview_image", "image"]:
                if key in obs:
                    image = obs[key]
                    break
        
        if image is not None:
            img_artist.set_array(image)
        
        # Update action text
        if actions is not None and frame < len(actions):
            action = actions[frame]
            action_str = f"Action: [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}, ...]"
            action_text.set_text(action_str)
        
        return [img_artist, action_text]
    
    # Create animation
    anim = FuncAnimation(
        fig, update, frames=min(len(observations), 100),
        interval=1000/fps, blit=True
    )
    
    if output_path:
        anim.save(output_path, writer=PillowWriter(fps=fps))
        print(f"Saved animation to {output_path}")
    else:
        plt.show()


def plot_loss_curve(
    losses: List[float],
    rolling_window: int = 10,
    title: str = "Training Loss Curve",
) -> None:
    """
    Plot training loss curve with rolling average.
    
    Args:
        losses: List of loss values
        rolling_window: Window size for rolling average
        title: Plot title
    """
    if not losses:
        print("No losses to plot")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot raw losses
    ax.plot(losses, alpha=0.5, label="Raw Loss")
    
    # Plot rolling average
    if len(losses) >= rolling_window:
        rolling_avg = np.convolve(losses, np.ones(rolling_window)/rolling_window, mode='valid')
        ax.plot(range(rolling_window-1, len(losses)), rolling_avg, label=f"Rolling Avg (n={rolling_window})", linewidth=2)
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.show()


def visualize_memory_retrieval(
    query_vector: np.ndarray,
    retrieved_vectors: List[Tuple[int, float, np.ndarray]],
    metric: str = "cosine",
) -> None:
    """
    Visualize memory retrieval results.
    
    Args:
        query_vector: Query embedding vector
        retrieved_vectors: List of (memory_id, score, vector) tuples
        metric: Similarity metric used
    """
    # Create bar plot of retrieval scores
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar plot of scores
    memory_ids = [item[0] for item in retrieved_vectors]
    scores = [item[1] for item in retrieved_vectors]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(scores)))
    bars = axes[0].bar(memory_ids, scores, color=colors)
    axes[0].set_xlabel("Memory ID")
    axes[0].set_ylabel(f"Similarity ({metric})")
    axes[0].set_title("Retrieval Scores")
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot comparison of vectors
    if query_vector is not None:
        query_normalized = query_vector / (np.linalg.norm(query_vector) + 1e-8)
        
        for i, (mem_id, score, vector) in enumerate(retrieved_vectors):
            vec_normalized = vector / (np.linalg.norm(vector) + 1e-8)
            similarity = np.dot(query_normalized, vec_normalized)
            
            axes[1].plot(vec_normalized[:100], label=f'Memory {mem_id} (sim={similarity:.3f})', alpha=0.7)
        
        axes[1].set_xlabel("Dimension")
        axes[1].set_ylabel("Normalized Value")
        axes[1].set_title("Vector Comparison (first 100 dims)")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
