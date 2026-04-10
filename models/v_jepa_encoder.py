"""
V-JEPA 2.1 Encoder Wrapper for Object Theater VLA

Provides a clean interface for vision representation learning using
the official V-JEPA 2.1 ViT-G (Gigantic) dense feature extractor.

This implementation loads the pre-trained 2-Billion parameter model
from the facebookresearch/vjepa2 repository via torch.hub and uses
it as a frozen physical perception engine for downstream diffusion policy conditioning.
"""

import numpy as np
import torch
import torch.nn as nn


class VJepaEncoder(nn.Module):
    """
    V-JEPA 2.1 ViT dense feature extractor for downstream diffusion policy conditioning.
    
    This class wraps the official V-JEPA 2.1 models loaded via torch.hub.
    The model is frozen during initialization and used exclusively for extracting rich spatial
    latent representations from video frames. These dense feature maps serve as conditioning
    vectors for the downstream Diffusion Policy network.
    
    Device allocation automatically detects and uses CUDA (ROCm) for AMD GPUs or falls back to CPU.
    
    Args:
        model_name: V-JEPA model identifier (e.g., 'vjepa2_1_vit_base_384', 'vjepa2_1_vit_gigantic_384')
    """
    
    def __init__(self, model_name: str = "vjepa2_1_vit_gigantic_384") -> None:
        """
        Initialize the V-JEPA 2.1 encoder.
        
        Loads the official V-JEPA 2.1 preprocessor and the specified model
        from facebookresearch/vjepa2 via torch.hub.
        
        The encoder is set to eval() mode and all parameters are frozen to enforce
        the Bias Firewall - this model acts solely as a frozen physical perception engine.
        
        Args:
            model_name: V-JEPA model identifier (default: 'vjepa2_1_vit_gigantic_384')
        """
        super().__init__()
        
        # Determine device: cuda for AMD ROCm (PyTorch uses 'cuda' for ROCm), fallback to cpu
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the V-JEPA 2.1 preprocessor from official repository
        self.processor = torch.hub.load('Avnerus/vjepa2', 'vjepa2_preprocessor')  # type: ignore[misc]
        
        # Load the V-JEPA 2.1 model at 384 resolution
        # Models return (encoder, predictor) tuple for 2.1 versions, extract encoder
        encoder_or_tuple = torch.hub.load('Avnerus/vjepa2', model_name)  # type: ignore[misc]
        if isinstance(encoder_or_tuple, tuple):
            self.encoder = encoder_or_tuple[0]  # type: ignore[assignment]
        else:
            self.encoder = encoder_or_tuple  # type: ignore[assignment]
        
        # Freeze the encoder - enforce Bias Firewall
        self.encoder.eval()  # type: ignore[union-attr]
        for param in self.encoder.parameters():  # type: ignore[union-attr]
            param.requires_grad = False
        
        # Move to appropriate device
        self.encoder.to(self.device)  # type: ignore[union-attr]
    
    @torch.no_grad()
    def extract_features(
        self,
        video_frames: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract dense latent representations from a sequence of RGB video frames.
        
        Passes the input video through the V-JEPA 2.1 preprocessor and encoder to
        obtain the rich spatial feature map for downstream diffusion policy conditioning.
        
        Args:
            video_frames: RGB video frames tensor
                Shape: [batch_size, channels, frames, height, width]
                Expected format: B, C, T, H, W with values in [0, 255]
        
        Returns:
            Dense latent representation (feature map)
            Shape: [batch_size, num_patches, latent_dim] where num_patches = (H/16) * (W/16)
        
        Note:
            The V-JEPA 2.1 Gigantic model processes video frames and outputs dense tokens
            representing spatial features across the entire frame. These tokens serve as
            high-fidelity visual conditioning for the Diffusion Policy network.
        """
        # Process video frames through the preprocessor
        # The preprocessor expects a list of frames in [T, H, W, C] format
        # For batched input, process each video in the batch separately
        if video_frames.ndim == 5:  # [B, C, T, H, W] format
            processed_batch = []
            for b in range(video_frames.shape[0]):
                # Convert [C, T, H, W] to [T, H, W, C] for the preprocessor
                video_np = video_frames[b].permute(1, 2, 3, 0).cpu().numpy()
                # Preprocessor expects list of frames (T H W C)
                buffer = [video_np[t] for t in range(video_np.shape[0])]
                processed = self.processor(buffer)  # type: ignore[call-arg]
                # processed is a list with one item [C, T, H, W]
                processed_batch.append(processed[0])
            processed_video = torch.stack(processed_batch)
        else:
            # Single video case
            video_np = video_frames.permute(1, 2, 3, 0).cpu().numpy()
            buffer = [video_np[t] for t in range(video_np.shape[0])]
            processed = self.processor(buffer)  # type: ignore[call-arg]
            processed_video = processed[0].unsqueeze(0)  # Add batch dimension
        
        # Move processed video to the same device as encoder
        processed_video = processed_video.to(self.device)
        
        # Pass through encoder - returns dense feature map (not just CLS token)
        # The ViT-Gigantic model outputs all patch tokens for dense conditioning
        feature_map = self.encoder(processed_video)  # type: ignore[call-arg]
        
        return feature_map
    
    @torch.no_grad()
    def extract_features_batched(
        self,
        video_frames: torch.Tensor,
        chunk_size: int = 4,
    ) -> torch.Tensor:
        """
        Extract features with batched processing for memory efficiency.
        
        Args:
            video_frames: RGB video frames tensor
                Shape: [batch_size, channels, frames, height, width]
            chunk_size: Number of frames to process at once
        
        Returns:
            Dense latent representation
            Shape: [batch_size, num_patches, latent_dim]
        """
        if video_frames.shape[0] <= chunk_size:
            return self.extract_features(video_frames)
        
        features_list = []
        for i in range(0, video_frames.shape[0], chunk_size):
            chunk = video_frames[i:i + chunk_size]
            features = self.extract_features(chunk)
            features_list.append(features)
        
        return torch.cat(features_list, dim=0)


if __name__ == "__main__":
    # Initialize encoder with base model for testing
    encoder = VJepaEncoder(model_name="vjepa2_1_vit_base_384")
    
    # Test with single video frame (simulated as 3 frames for temporal context)
    batch_size = 2
    channels = 3
    frames = 3
    height = 224
    width = 224
    
    # Create dummy video frames (normalized to [0, 255])
    video_frames = torch.randint(0, 256, (batch_size, channels, frames, height, width), dtype=torch.uint8)
    
    print(f"Input video frames shape: {video_frames.shape}")
    
    # Extract features
    with torch.no_grad():
        features = encoder.extract_features(video_frames)
    
    print(f"Extracted features shape: {features.shape}")
    print(f"Device: {encoder.device}")
    
    # Verify feature extraction
    assert len(features.shape) == 3, f"Expected 3D output [batch, num_patches, latent_dim], got {features.shape}"
    
    # Check that features are on correct device
    assert features.device.type == encoder.device, f"Features device {features.device} != encoder device {encoder.device}"
    
    # Verify no NaN values
    assert not torch.isnan(features).any(), "Features contain NaN values"
    
    # Print shape details
    batch_size_out, num_patches, latent_dim = features.shape
    print(f"Batch size: {batch_size_out}")
    print(f"Number of patches: {num_patches}")
    print(f"Latent dimension: {latent_dim}")
    
    print("V-JEPA encoder test passed successfully!")
