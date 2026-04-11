"""
Object Theater VLA Memory Package

Provides FAISS-based episodic memory storage and retrieval:

- `EpisodicMemoryBuffer`: Core memory buffer with semantic and visual similarity search
- `retrieve_closest_trajectory`: Returns 4-tuple (memory_id, score, action_trajectory, visual_state)
"""

from .lemb_core import EpisodicMemoryBuffer

__all__ = ["EpisodicMemoryBuffer"]
