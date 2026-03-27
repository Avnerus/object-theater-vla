"""
Object Theater VLA Training Utilities Package
"""

from .train_diffusion_policy import DiffusionPolicyTrainer, main as train_diffusion_main

__all__ = [
    'DiffusionPolicyTrainer',
    'train_diffusion_main',
]
