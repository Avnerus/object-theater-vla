"""
V-JEPA 2.1 Encoder Wrapper for Object Theater VLA

Provides a clean interface for vision representation learning using
the official V-JEPA 2.1 ViT-G (Gigantic) dense feature extractor.

This implementation loads the pre-trained 2-Billion parameter model
from the facebookresearch/vjepa2 repository via torch.hub and uses
it as a frozen physical perception engine for downstream diffusion policy conditioning.
"""

import torch
import torch.nn as nn


class VJepaEncoder(nn.Module):
    """
    V-JEPA 2.1 ViT-G (Gigantic) dense feature extractor for downstream diffusion policy conditioning.
    
    This class wraps the official V-JEPA 2.1 Gigantic model (2B parameters) loaded via torch.hub.
    The model is frozen during initialization and used exclusively for extracting rich spatial
    latent representations from video frames. These dense feature maps serve as conditioning
    vectors for the downstream Diffusion Policy network.
    
    Device allocation automatically detects and uses CUDA (ROCm) for AMD GPUs or falls back to CPU.
    """
    
    def __init__(self) -> None:
        """
        Initialize the V-JEPA 2.1 encoder with pre-trained Gigantic weights.
        
        Loads the official V-JEPA 2.1 preprocessor and the ViT-Gigantic-384 model
        (2-Billion parameters) from facebookresearch/vjepa2 via torch.hub.
        
        The encoder is set to eval() mode and all parameters are frozen to enforce
        the Bias Firewall - this model acts solely as a frozen physical perception engine.
        """
        super().__init__()
        
        # Determine device: cuda for AMD ROCm (PyTorch uses 'cuda' for ROCm), fallback to cpu
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the V-JEPA 2.1 preprocessor from official repository
        self.processor = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_preprocessor')
        
        # Load the 2-Billion parameter Gigantic model at 384 resolution
        self.encoder = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_1_vit_gigantic_384')
        
        # Freeze the encoder - enforce Bias Firewall
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Move to appropriate device
        self.encoder.to(self.device)
    
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
        # The preprocessor handles normalization, resizing, and tensor conversion
        processed_video = self.processor(video_frames)
        
        # Move processed video to the same device as encoder
        processed_video = processed_video.to(self.device)
        
        # Pass through encoder - returns dense feature map (not just CLS token)
        # The ViT-Gigantic model outputs all patch tokens for dense conditioning
        feature_map = self.encoder(processed_video)
        
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
