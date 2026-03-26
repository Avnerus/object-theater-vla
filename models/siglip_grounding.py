"""
SigLIP Text Encoder Wrapper for Object Theater VLA

Provides a clean interface for generating semantic embeddings from text.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


class SigLIPTextEncoder:
    """
    Wrapper for SigLIP text encoder model.
    
    Takes text strings and returns normalized dense embeddings.
    Uses the SigLIP model for multimodal contrastive learning.
    """
    
    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize the SigLIP text encoder.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ('cpu', 'cuda', or None for auto)
            dtype: Data type for computations
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.dtype = dtype
        self.model_name = model_name
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        
        # Get embedding dimension from config
        self.embedding_dim = self.model.config.projection_dim
        
        # Ensure model is in eval mode and frozen
        for param in self.model.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def encode_text(
        self,
        text: Union[str, List[str]],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode text to embedding vector(s).
        
        Args:
            text: Single text string or list of text strings
            normalize: If True, return L2-normalized embeddings
        
        Returns:
            Embedding array of shape (batch_size, embedding_dim)
        """
        # Tokenize input text
        if isinstance(text, str):
            text = [text]
        
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        outputs = self.model.get_text_features(**inputs)
        
        # Normalize if requested
        if normalize:
            outputs = torch.nn.functional.normalize(outputs, p=2, dim=-1)
        
        # Convert to numpy
        embeddings = outputs.cpu().numpy().astype(np.float32)
        return embeddings
    
    @torch.no_grad()
    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode a batch of texts efficiently.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            normalize: If True, return L2-normalized embeddings
        
        Returns:
            Embedding array of shape (len(texts), embedding_dim)
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_embeddings = self.encode_text(batch_texts, normalize=normalize)
            all_embeddings.append(batch_embeddings)
        
        return np.concatenate(all_embeddings, axis=0)
    
    def __call__(self, text: Union[str, List[str]]) -> np.ndarray:
        """Callable alias for encode_text."""
        return self.encode_text(text)
    
    def get_embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self.embedding_dim


# Test the text encoder
if __name__ == "__main__":
    # Initialize encoder
    encoder = SigLIPTextEncoder(device="cpu")
    
    # Test encoding
    texts = [
        "a red box on the table",
        "a blue cylinder in the workspace",
        "a green sphere near the robot",
    ]
    
    embeddings = encoder.encode_text(texts)
    print(f"Encoded {len(texts)} texts")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dimension: {encoder.get_embedding_dim()}")
    
    # Verify normalization
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"Embedding norms (should be ~1.0): {norms}")
