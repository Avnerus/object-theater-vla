"""
Dataset utilities for Object Theater VLA.

Provides utilities for loading and preprocessing demonstration data.
"""

from typing import Any, Callable, Dict, List, Tuple, Optional
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader

from configs.device import DEVICE


class DemonstrationDataset(Dataset):
    """
    PyTorch Dataset for demonstration data.
    
    Loads observations and actions from HDF5 files.
    """
    
    def __init__(
        self,
        hdf5_path: str,
        transform: Optional[Callable] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the demonstration dataset.
        
        Args:
            hdf5_path: Path to HDF5 file containing demonstrations
            transform: Optional transform function for observations
            device: Device to load tensors to (None for DEVICE global variable)
        """
        self.hdf5_path = hdf5_path
        self.transform = transform
        self.device = device or DEVICE
        
        # Open HDF5 file  # type: ignore[misc]
        self.file = h5py.File(hdf5_path, "r")  # type: ignore[misc]
        
        # Get metadata
        self.num_demonstrations = self.file.attrs.get("num_demonstrations", 0)
        self.image_size = self.file.attrs.get("image_size", [224, 224])
        self.action_dim = self.file.attrs.get("action_dim", 7)
        
        # Pre-load all trajectories for efficient access
        self._load_all_trajectories()
    
    def _load_all_trajectories(self) -> None:
        """Pre-load all trajectories into memory."""
        self.trajectories = []
        
        for i in range(self.num_demonstrations):
            demo_group = self.file[f"demonstration_{i}"]  # type: ignore[index]
            
            # Load observations
            obs_group = demo_group["observations"]  # type: ignore[index]
            observations = []
            for key in sorted(obs_group.keys(), key=lambda x: int(x.split("_")[1])):  # type: ignore[misc]
                step_group = obs_group[key]  # type: ignore[index]
                obs = {k: torch.from_numpy(v[:]) for k, v in step_group.items()}  # type: ignore[misc]
                observations.append(obs)
            
            # Load actions
            actions = torch.from_numpy(demo_group["actions"][:])  # type: ignore[index]
            
            # Get metadata
            text_label = demo_group.attrs.get("text_label", "")  # type: ignore[misc]
            
            self.trajectories.append({
                "observations": observations,
                "actions": actions,
                "text_label": text_label,
            })
    
    def __len__(self) -> int:
        """Return number of demonstrations."""
        return self.num_demonstrations
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a demonstration.
        
        Args:
            idx: Index of demonstration
        
        Returns:
            Dictionary with observations, actions, and metadata
        """
        demo = self.trajectories[idx]
        
        # Apply transform if provided
        if self.transform:
            demo = self.transform(demo)
        
        # Move tensors to device
        demo["actions"] = demo["actions"].to(self.device)
        
        # HER: Randomly select a future index within the same trajectory to act as the goal
        # Extract visual states from observations
        observations = demo["observations"]
        trajectory_length = len(observations)
        
        # Sample current and future indices for HER
        current_idx = np.random.randint(0, trajectory_length - 16)
        future_idx = np.random.randint(current_idx + 1, trajectory_length)
        
        # Extract current visual state (V-JEPA patches from observations)
        current_obs = observations[current_idx]
        # Assuming visual_state key exists in observations
        if "vjepa_patches" in current_obs:
            current_vjepa_patches = current_obs["vjepa_patches"]
        elif "visual_state" in current_obs:
            current_vjepa_patches = current_obs["visual_state"]
        else:
            # Fallback: try to find any tensor that looks like visual features
            for key, value in current_obs.items():
                if isinstance(value, torch.Tensor) and value.dim() >= 2:
                    current_vjepa_patches = value
                    break
            else:
                # Create dummy if no visual features found
                current_vjepa_patches = torch.zeros(196, 1024)  # Default 14x14 patches
        
        # Extract goal visual state (future pooled V-JEPA)
        future_obs = observations[future_idx]
        if "vjepa_pooled" in future_obs:
            future_vjepa_pooled = future_obs["vjepa_pooled"]
        elif "visual_state" in future_obs:
            future_vjepa_pooled = future_obs["visual_state"]
        else:
            # Fallback: use first available tensor
            for key, value in future_obs.items():
                if isinstance(value, torch.Tensor) and value.dim() == 1:
                    future_vjepa_pooled = value
                    break
            else:
                # Create dummy if no visual features found
                future_vjepa_pooled = torch.zeros(1024)
        
        # Add goal state to demo
        demo["visual_state"] = current_vjepa_patches  # [num_patches, latent_dim]
        demo["goal_state"] = future_vjepa_pooled      # [latent_dim]
        
        return demo
    
    def get_text_labels(self) -> List[str]:
        """Get all text labels."""
        return [d["text_label"] for d in self.trajectories]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        num_steps = sum(len(d["actions"]) for d in self.trajectories)
        action_means = []
        action_stds = []
        
        for demo in self.trajectories:
            actions = demo["actions"].numpy()
            action_means.append(actions.mean(axis=0))
            action_stds.append(actions.std(axis=0))
        
        return {
            "num_demonstrations": self.num_demonstrations,
            "total_steps": num_steps,
            "avg_steps_per_demo": num_steps / self.num_demonstrations if self.num_demonstrations > 0 else 0,
            "action_dim": self.action_dim,
            "image_size": self.image_size,
        }
    
    def close(self) -> None:
        """Close the HDF5 file."""
        self.file.close()  # type: ignore[misc]
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def create_data_loader(
    dataset: DemonstrationDataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a PyTorch DataLoader for demonstration data.
    
    Args:
        dataset: DemonstrationDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker threads
    
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda x: x[0],  # Simple collate for single-item batches
    )


def load_demonstrations(
    hdf5_path: str,
    device: Optional[torch.device] = None,
) -> List[Dict[str, Any]]:
    """
    Load all demonstrations from HDF5 file.
    
    Args:
        hdf5_path: Path to HDF5 file
        device: Device to load tensors to
    
    Returns:
        List of demonstration dictionaries
    """
    dataset = DemonstrationDataset(hdf5_path, device=device)
    demonstrations = []
    
    for i in range(len(dataset)):
        demonstrations.append(dataset[i])
    
    dataset.close()
    return demonstrations


def split_demonstrations(
    demonstrations: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split demonstrations into train/val/test sets.
    
    Args:
        demonstrations: List of demonstration dictionaries
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
    
    Returns:
        Tuple of (train, val, test) demonstration lists
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    n = len(demonstrations)
    indices = list(range(n))
    
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train = [demonstrations[i] for i in indices[:train_end]]
    val = [demonstrations[i] for i in indices[train_end:val_end]]
    test = [demonstrations[i] for i in indices[val_end:]]
    
    return train, val, test
