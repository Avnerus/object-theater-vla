"""
Diffusion Policy Training Script for Object Theater VLA

Trains the diffusion policy on collected demonstration data.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os

from configs.device import DEVICE
from configs.config import Config, default_config

from utils.dataset import DemonstrationDataset, load_demonstrations, split_demonstrations
from models.diffusion_policy import DiffusionPolicy


class DiffusionPolicyTrainer:
    """
    Trainer for the diffusion policy.
    
    Implements training loop with:
    - Forward diffusion process
    - Noise prediction loss
    - Gradient clipping
    - Checkpointing
    """
    
    def __init__(
        self,
        config: Config = None,
        model: DiffusionPolicy = None,
        device: torch.device = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration instance
            model: DiffusionPolicy instance
            device: Device to train on
        """
        self.config = config or default_config
        self.model = model or DiffusionPolicy(
            latent_dim=self.config.model.vjepa_latent_dim,
            action_dim=self.config.model.diffusion_action_dim,
            action_horizon=self.config.model.vjepa_action_horizon,
            device=device,
        )
        
        self.device = device or DEVICE
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.loss_history = []
        self.best_loss = float("inf")
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=1e-4,
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.training.num_epochs,
        )
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
    
    def prepare_data(
        self,
        dataset_path: str,
        batch_size: int = 1,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Load and prepare data for training.
        
        Args:
            dataset_path: Path to HDF5 file with demonstrations
            batch_size: Batch size
        
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Load demonstrations
        demonstrations = load_demonstrations(dataset_path, device=self.device)
        
        # Split into train/val
        train_data, val_data, _ = split_demonstrations(demonstrations, train_ratio=0.8)
        
        # Create datasets (with batch_size=1 for simplicity)
        train_dataset = DemonstrationDataset(
            dataset_path,
            device=self.device,
        )
        val_dataset = DemonstrationDataset(
            dataset_path,
            device=self.device,
        )
        
        # For now, use simple dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        
        return train_loader, val_loader
    
    def forward_diffusion(
        self,
        x_start: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply forward diffusion process.
        
        Args:
            x_start: Starting action sequence
        
        Returns:
            Tuple of (noisy_x, noise, timesteps)
        """
        batch_size = x_start.shape[0]
        
        # Sample timesteps
        t = torch.randint(
            0, self.model.num_timesteps, (batch_size,), device=self.device
        ).long()
        
        # Sample noise
        noise = torch.randn_like(x_start)
        
        # Forward diffusion
        noisy_x = self.model.q_sample(x_start, t, noise)
        
        return noisy_x, noise, t
    
    def compute_loss(
        self,
        noisy_x: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute training loss.
        
        Args:
            noisy_x: Noisy action sequence
            noise: Ground truth noise
            t: Timesteps
            condition: Conditioning vector
        
        Returns:
            Loss value
        """
        # Predict noise
        noise_pred = self.model.model(noisy_x, t, condition)
        
        # MSE loss
        loss = nn.functional.mse_loss(noise_pred, noise)
        
        return loss
    
    def train_epoch(
        self,
        train_loader: DataLoader,
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
        
        Returns:
            Average epoch loss
        """
        self.model.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            # Get data
            actions = batch["actions"]
            text_label = batch["text_label"]
            
            # Encode text to get condition
            condition = self.model.siglip.encode_text(text_label, normalize=True)
            condition = torch.from_numpy(condition).to(self.device).float()
            
            # Forward diffusion
            noisy_x, noise, t = self.forward_diffusion(actions)
            
            # Compute loss
            loss = self.compute_loss(noisy_x, noise, t, condition)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.model.parameters(), max_norm=1.0
            )
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Log
            if num_batches % 10 == 0:
                print(f"  Step {self.global_step}: loss={loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
    ) -> float:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Average validation loss
        """
        self.model.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in val_loader:
            actions = batch["actions"]
            text_label = batch["text_label"]
            
            condition = self.model.siglip.encode_text(text_label, normalize=True)
            condition = torch.from_numpy(condition).to(self.device).float()
            
            noisy_x, noise, t = self.forward_diffusion(actions)
            loss = self.compute_loss(noisy_x, noise, t, condition)
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def fit(
        self,
        dataset_path: str,
        num_epochs: int = None,
        batch_size: int = 1,
        val_interval: int = 5,
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            dataset_path: Path to HDF5 file
            num_epochs: Number of epochs
            batch_size: Batch size
            val_interval: Validation interval (epochs)
        
        Returns:
            Dictionary with training history
        """
        num_epochs = num_epochs or self.config.training.num_epochs
        
        # Prepare data
        train_loader, val_loader = self.prepare_data(dataset_path, batch_size)
        
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"  Training samples: {len(train_loader.dataset)}")
        print(f"  Validation samples: {len(val_loader.dataset)}")
        print(f"  Device: {self.device}")
        print()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            print(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            print(f"  Train loss: {train_loss:.4f}")
            
            # Validate
            if (epoch + 1) % val_interval == 0:
                val_loss = self.validate(val_loader)
                print(f"  Val loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint(
                        f"checkpoints/best_model_epoch_{epoch + 1}.pt"
                    )
                    print(f"  Saved best model (loss: {self.best_loss:.4f})")
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            print(f"  Learning rate: {current_lr:.6f}")
            print()
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }
    
    def save_checkpoint(self, filepath: str) -> None:
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "model_state_dict": self.model.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }
        
        torch.save(checkpoint, filepath)
        print(f"Saved checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint["best_loss"]
        self.model.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        
        print(f"Loaded checkpoint from {filepath} (epoch: {self.epoch})")
    
    def plot_training_history(self) -> None:
        """Plot training history."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        ax.plot(self.train_losses, label="Train", linewidth=2)
        if self.val_losses:
            ax.plot(self.val_losses, label="Validation", linewidth=2)
        
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Diffusion Policy Training History")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.show()


def main():
    """Main function to train diffusion policy."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train diffusion policy")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to HDF5 dataset file",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = DiffusionPolicyTrainer(device=DEVICE)
    
    # Resume from checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Train
    history = trainer.fit(
        dataset_path=args.dataset,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
    )
    
    # Save final model
    final_path = os.path.join(args.output_dir, "diffusion_policy_final.pt")
    trainer.save_checkpoint(final_path)
    
    # Plot training history
    trainer.plot_training_history()
    
    print(f"\nTraining complete! Final model saved to {final_path}")


if __name__ == "__main__":
    main()
