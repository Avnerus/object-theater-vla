"""
Diffusion Policy for Object Theater VLA

Implements a 1D Conditional UNet for generating action sequences
conditioned on V-JEPA latent states.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet1DModel


class ConditionalUnet1D(nn.Module):
    """
    1D Conditional UNet for diffusion policy.
    
    Takes V-JEPA latent states as conditioning and outputs
    a sequence of action commands.
    """
    
    def __init__(
        self,
        input_dim: int = 1024,  # V-JEPA latent dimension
        output_dim: int = 7,    # Action dimension (OSC_POSE)
        horizon: int = 16,      # Action horizon
        dim: int = 256,
        dim_mults: Tuple[int, ...] = (1, 2, 4, 8),
        condition_dim: int = 1024,
        condition_type: str = "concat",
    ):
        """
        Initialize the conditional UNet.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output action dimension
            horizon: Length of action sequence
            dim: Base dimension for the network
            dim_mults: Dimension multipliers for each level
            condition_dim: Conditioning vector dimension
            condition_type: How to incorporate conditioning ('concat' or 'add')
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.condition_dim = condition_dim
        self.condition_type = condition_type
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, dim)
        
        # Condition projection
        self.condition_proj = nn.Linear(condition_dim, dim)
        
        # Time embedding (for diffusion)
        self.time_emb = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        
        # Encoder blocks
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        dims = [dim] + [dim * m for m in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        # Encoder
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= len(in_out) - 1
            self.encoders.append(
                nn.ModuleList([
                    ResnetBlock(dim_in, dim_out, time_embed_dim=dim),
                    ResnetBlock(dim_out, dim_out, time_embed_dim=dim),
                    Residual(Attention(dim_out)),
                    Downsample(dim_out) if not is_last else nn.Identity(),
                ])
            )
        
        # Middle
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_embed_dim=dim)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_embed_dim=dim)
        self.mid_attn = Residual(Attention(mid_dim))
        
        # Decoder
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == len(in_out) - 1
            self.decoders.append(
                nn.ModuleList([
                    ResnetBlock(dim_out * 2, dim_in, time_embed_dim=dim),
                    ResnetBlock(dim_in, dim_in, time_embed_dim=dim),
                    Residual(Attention(dim_in)),
                    Upsample(dim_in) if not is_last else nn.Identity(),
                ])
            )
        
        # Output head
        self.final_conv = nn.Sequential(
            ResnetBlock(dim, dim, time_embed_dim=dim),
            nn.Conv1d(dim, output_dim, 1),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input noisy action sequence
                Shape: (batch_size, horizon, input_dim)
            time: Time step for diffusion
                Shape: (batch_size,)
            condition: V-JEPA latent condition
                Shape: (batch_size, condition_dim)
        
        Returns:
            Predicted action sequence
            Shape: (batch_size, horizon, output_dim)
        """
        batch_size = x.shape[0]
        
        # Project input
        x = self.input_proj(x)  # (B, horizon, dim)
        x = x.transpose(1, 2)   # (B, dim, horizon)
        
        # Project and expand condition
        condition = self.condition_proj(condition)  # (B, dim)
        condition = condition.unsqueeze(-1)          # (B, dim, 1)
        condition = condition.expand(-1, -1, x.shape[-1])  # (B, dim, horizon)
        
        # Combine condition with input
        if self.condition_type == "concat":
            x = torch.cat([x, condition], dim=1)  # (B, dim*2, horizon)
        
        # Time embedding
        time_emb = self.time_emb(time)
        
        # Encoder
        skips = []
        for block1, block2, attn, downsample in self.encoders:
            x = block1(x, time_emb)
            x = block2(x, time_emb)
            x = attn(x)
            skips.append(x)
            x = downsample(x)
        
        # Middle
        x = self.mid_block1(x, time_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, time_emb)
        
        # Decoder
        for block1, block2, attn, upsample in self.decoders:
            x = torch.cat([x, skips.pop()], dim=1)
            x = block1(x, time_emb)
            x = block2(x, time_emb)
            x = attn(x)
            x = upsample(x)
        
        # Final output
        x = self.final_conv(x)
        x = x.transpose(1, 2)  # (B, horizon, output_dim)
        
        return x


class ResnetBlock(nn.Module):
    """ResNet block with time embedding."""
    
    def __init__(self, dim_in: int, dim_out: int, time_embed_dim: Optional[int] = None):
        super().__init__()
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, dim_in),
            nn.SiLU(),
            nn.Conv1d(dim_in, dim_out, 3, padding=1),
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, dim_out),
            nn.SiLU(),
            nn.Conv1d(dim_out, dim_out, 3, padding=1),
        )
        
        self.res_conv = nn.Conv1d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()
        
        self.time_emb = None
        if time_embed_dim is not None:
            self.time_emb = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_embed_dim, dim_out),
            )
    
    def forward(self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.block1(x)
        
        if time_emb is not None and self.time_emb is not None:
            time_scale = self.time_emb(time_emb).unsqueeze(-1)
            h = h + time_scale
        
        h = self.block2(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    """Multi-head self-attention."""
    
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2) 
                   for t in qkv]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = attn @ v
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        out = self.to_out(out)
        
        return out


class Residual(nn.Module):
    """Residual connection wrapper."""
    
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x) + x


class Upsample(nn.Module):
    """Upsampling layer."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Downsample(nn.Module):
    """Downsampling layer."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DiffusionPolicy:
    """
    Diffusion Policy wrapper for action generation.
    
    Uses a Conditional UNet to generate action sequences
    conditioned on V-JEPA latent states.
    """
    
    def __init__(
        self,
        latent_dim: int = 1024,
        action_dim: int = 7,
        action_horizon: int = 16,
        device: Optional[str] = None,
    ):
        """
        Initialize the diffusion policy.
        
        Args:
            latent_dim: Dimension of V-JEPA latent states
            action_dim: Dimension of action vectors
            action_horizon: Number of actions to generate
            device: Device to run on
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        
        # Initialize UNet
        self.model = ConditionalUnet1D(
            input_dim=action_dim,
            output_dim=action_dim,
            horizon=action_horizon,
            condition_dim=latent_dim,
        )
        self.model.to(device)
        
        # Diffusion parameters
        self.num_timesteps = 1000
        self.beta_schedule = "linear"
        self.beta_start = 1e-4
        self.beta_end = 0.02
        
        # Initialize betas
        self._init_beta_schedule()
    
    def _init_beta_schedule(self) -> None:
        """Initialize beta schedule for diffusion."""
        if self.beta_schedule == "linear":
            self.betas = torch.linspace(
                self.beta_start, self.beta_end, self.num_timesteps, device=self.device
            )
        elif self.beta_schedule == "quadratic":
            self.betas = torch.linspace(
                self.beta_start ** 0.5, self.beta_end ** 0.5, self.num_timesteps, device=self.device
            ) ** 2
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=self.device), self.alphas_cumprod[:-1]]
        )
    
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward diffusion process."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[t])
        
        return (
            sqrt_alphas_cumprod[:, None, None] * x_start
            + sqrt_one_minus_alphas_cumprod[:, None, None] * noise
        )
    
    def p_sample(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Single denoising step."""
        # Predict noise
        noise_pred = self.model(x, t, condition)
        
        # Compute x_0 prediction
        alphas_cumprod_t = self.alphas_cumprod[t]
        sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        sqrt_recipm1_alphas = torch.sqrt(1.0 / self.alphas - 1.0)
        
        x_0_pred = (
            sqrt_recip_alphas[t][:, None, None] * x
            - sqrt_recipm1_alphas[t][:, None, None] * noise_pred
        )
        
        # Compute mean
        mean = (
            self.alphas_cumprod_prev[t][:, None, None] * x_0_pred
            + torch.sqrt(self.betas[t][:, None, None]) * noise_pred
        )
        
        # Sample noise
        noise = torch.randn_like(x)
        noise[t == 0] = 0  # No noise at t=0
        
        return mean + torch.sqrt(self.betas[t][:, None, None]) * noise
    
    def predict_action(
        self,
        condition: np.ndarray,
        num_inference_steps: int = 50,
    ) -> np.ndarray:
        """
        Generate action sequence from condition.
        
        Args:
            condition: V-JEPA latent state
                Shape: (batch_size, latent_dim) or (latent_dim,)
            num_inference_steps: Number of denoising steps
        
        Returns:
            Generated action sequence
            Shape: (batch_size, action_horizon, action_dim)
        """
        self.model.eval()
        
        condition = torch.from_numpy(condition).to(self.device).float()
        if condition.dim() == 1:
            condition = condition.unsqueeze(0)
        
        batch_size = condition.shape[0]
        
        # Initialize with noise
        x = torch.randn(
            batch_size, self.action_horizon, self.action_dim, device=self.device
        )
        
        # Sample timesteps
        timesteps = np.linspace(0, self.num_timesteps - 1, num_inference_steps)
        timesteps = timesteps[::-1]  # Reverse for denoising
        
        with torch.no_grad():
            for t in timesteps:
                t_tensor = torch.full((batch_size,), int(t), device=self.device, dtype=torch.long)
                x = self.p_sample(x, t_tensor, condition)
        
        return x.cpu().numpy()
    
    def save(self, filepath: str) -> None:
        """Save model weights."""
        torch.save(self.model.state_dict(), filepath)
    
    def load(self, filepath: str) -> None:
        """Load model weights."""
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))


# Test the diffusion policy
if __name__ == "__main__":
    policy = DiffusionPolicy(device="cpu")
    
    # Test action prediction
    condition = np.random.randn(2, 1024).astype(np.float32)
    actions = policy.predict_action(condition, num_inference_steps=10)
    print(f"Generated action sequence shape: {actions.shape}")
