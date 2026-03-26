"""
Configuration dataclasses for Object Theater VLA hyperparameters.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import yaml


@dataclass
class EnvironmentConfig:
    """Environment configuration parameters."""
    render_mode: str = "rgb_array"
    camera_name: str = "agentview"
    image_size: Tuple[int, int] = (224, 224)
    control_freq: int = 20
    horizon: int = 1000
    object_types: Tuple[str, ...] = ("box", "cylinder", "sphere")
    diffusion_action_dim: int = 7


@dataclass
class MemoryConfig:
    """Episodic memory configuration parameters."""
    embedding_dim: int = 768
    use_cosine_similarity: bool = True
    max_memory_chunks: int = 10000
    retrieval_k: int = 3
    retrieval_alpha: float = 0.5


@dataclass
class ModelConfig:
    """Model architecture configuration parameters."""
    # SigLIP
    siglip_model_name: str = "google/siglip-base-patch16-224"
    
    # V-JEPA
    vjepa_latent_dim: int = 1024
    vjepa_action_dim: int = 7
    vjepa_action_horizon: int = 16
    vjepa_num_layers: int = 4
    vjepa_num_heads: int = 16
    
    # Diffusion Policy
    diffusion_dim: int = 256
    diffusion_dim_mults: Tuple[int, ...] = (1, 2, 4, 8)
    diffusion_num_timesteps: int = 1000
    diffusion_num_inference_steps: int = 50
    diffusion_action_dim: int = 7  # OSC_POSE action dimension


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    gradient_accumulation_steps: int = 4


@dataclass
class Config:
    """Main configuration container."""
    env: EnvironmentConfig = None
    memory: MemoryConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None
    
    def __post_init__(self):
        if self.env is None:
            self.env = EnvironmentConfig()
        if self.memory is None:
            self.memory = MemoryConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "env": vars(self.env),
            "memory": vars(self.memory),
            "model": vars(self.model),
            "training": vars(self.training),
        }
    
    def to_yaml(self) -> str:
        """Convert config to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load config from YAML file."""
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        
        config = cls()
        if "env" in data:
            config.env = EnvironmentConfig(**data["env"])
        if "memory" in data:
            config.memory = MemoryConfig(**data["memory"])
        if "model" in data:
            config.model = ModelConfig(**data["model"])
        if "training" in data:
            config.training = TrainingConfig(**data["training"])
        
        return config
    
    def save(self, yaml_path: str) -> None:
        """Save config to YAML file."""
        with open(yaml_path, "w") as f:
            f.write(self.to_yaml())
    
    @classmethod
    def load(cls, yaml_path: str) -> "Config":
        """Load config from YAML file (alias for from_yaml)."""
        return cls.from_yaml(yaml_path)


# Default configuration instance
default_config = Config()
