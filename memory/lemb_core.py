"""
Localized Episodic Memory Buffer (LEMB) for Object Theater VLA

Implements a FAISS-based vector database for storing and retrieving
episodic memories using semantic and visual similarity.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import faiss
import torch

from configs.device import DEVICE


class EpisodicMemoryBuffer:
    """
    Localized Episodic Memory Buffer using FAISS for efficient similarity search.
    
    Stores memory chunks containing:
    - semantic_vector: Text embedding (e.g., from SigLIP, dim: 768)
    - visual_state: V-JEPA latent state
    - action_trajectory: Sequence of OSC_POSE actions
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        use_cosine_similarity: bool = True,
        max_memory_chunks: int = 10000,
    ):
        """
        Initialize the episodic memory buffer.
        
        Args:
            embedding_dim: Dimension of semantic embeddings (default: 768 for SigLIP)
            use_cosine_similarity: If True, use IndexFlatIP (inner product = cosine)
                                   If False, use IndexFlatL2 (Euclidean distance)
            max_memory_chunks: Maximum number of memory chunks to store
        """
        self.embedding_dim = embedding_dim
        self.use_cosine_similarity = use_cosine_similarity
        self.max_memory_chunks = max_memory_chunks
        
        # Initialize FAISS index
        if use_cosine_similarity:
            # IndexFlatIP uses inner product, which is equivalent to cosine similarity
            # when vectors are normalized
            self.index = faiss.IndexFlatIP(embedding_dim)
        else:
            # IndexFlatL2 uses Euclidean distance
            self.index = faiss.IndexFlatL2(embedding_dim)
        
        # Memory storage: each entry contains semantic_vector, visual_state, action_trajectory
        self.memory_chunks: List[Dict[str, np.ndarray]] = []
        self.id_to_idx: Dict[int, int] = {}  # Maps memory_id to internal index
        
        # Tracking metrics
        self.total_additions = 0
        self.total_retrievals = 0
    
    def add_memory(
        self,
        memory_id: int,
        semantic_vector: np.ndarray,
        visual_state: np.ndarray,
        action_trajectory: np.ndarray,
        task_label: str = None,
    ) -> None:
        """
        Add a new memory chunk to the buffer.
        
        Args:
            memory_id: Unique identifier for this memory
            semantic_vector: Text embedding vector (normalized for cosine similarity)
            visual_state: V-JEPA latent representation
            action_trajectory: Sequence of actions [action_horizon, action_dim]
            task_label: Optional label describing the task (stored for SLM chat)
        """
        if len(self.memory_chunks) >= self.max_memory_chunks:
            # Simple eviction: remove oldest memory
            oldest_id = min(self.id_to_idx.keys())
            oldest_idx = self.id_to_idx[oldest_id]
            del self.id_to_idx[oldest_id]
            # Remove from list and update indices
            removed = self.memory_chunks.pop(oldest_idx)
            for i, _ in enumerate(self.memory_chunks):
                for k, v in list(self.id_to_idx.items()):
                    if v > oldest_idx:
                        self.id_to_idx[k] = v - 1
        
        # Validate dimensions
        if semantic_vector.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Semantic vector dimension {semantic_vector.shape[0]} "
                f"doesn't match expected {self.embedding_dim}"
            )
        
        # Store memory chunk
        memory_entry = {
            "memory_id": memory_id,
            "semantic_vector": semantic_vector.astype(np.float32),
            "visual_state": visual_state.astype(np.float32),
            "action_trajectory": action_trajectory.astype(np.float32),
            "task_label": task_label,  # Add task label for SLM chat
        }
        
        self.memory_chunks.append(memory_entry)
        self.id_to_idx[memory_id] = len(self.memory_chunks) - 1
        
        # Add vector to FAISS index (normalize for cosine similarity)
        vector = semantic_vector / (np.linalg.norm(semantic_vector) + 1e-8)
        self.index.add(vector.reshape(1, -1))
        
        self.total_additions += 1
    
    def retrieve_closest_trajectory(
        self,
        query_state: np.ndarray,
        target_semantic_vector: np.ndarray,
        k: int = 3,
        alpha: float = 0.5,
    ) -> List[Tuple[int, float, np.ndarray]]:
        """
        Retrieve the most similar action trajectories.
        
        Args:
            query_state: Current visual state for hybrid retrieval
            target_semantic_vector: Target semantic embedding
            k: Number of nearest neighbors to retrieve
            alpha: Weight for semantic vs visual similarity (0=visual only, 1=semantic only)
        
        Returns:
            List of (memory_id, score, action_trajectory) tuples
        """
        self.total_retrievals += 1
        
        # Normalize query vectors
        semantic_vec = target_semantic_vector / (np.linalg.norm(target_semantic_vector) + 1e-8)
        visual_vec = query_state / (np.linalg.norm(query_state) + 1e-8)
        
        # Query FAISS index for semantic similarity
        D_sem, I_sem = self.index.search(semantic_vec.reshape(1, -1), k)
        
        # For visual similarity, we'd need a separate index or compute distances
        # Here we use semantic-based retrieval as the primary method
        results = []
        for i in range(min(k, len(I_sem[0]))):
            idx = I_sem[0][i]
            if idx < 0 or idx >= len(self.memory_chunks):
                continue
            
            memory = self.memory_chunks[idx]
            memory_id = memory["memory_id"]
            
            # Compute combined score
            semantic_score = D_sem[0][i] if self.use_cosine_similarity else -D_sem[0][i]
            
            # Visual similarity (using cosine)
            visual_sim = np.dot(visual_vec, memory["visual_state"]) / (
                np.linalg.norm(visual_vec) * np.linalg.norm(memory["visual_state"]) + 1e-8
            )
            visual_score = visual_sim
            
            # Combined score
            combined_score = alpha * semantic_score + (1 - alpha) * visual_score
            
            results.append((
                memory_id,
                combined_score,
                memory["action_trajectory"],
            ))
        
        # Sort by score (higher is better)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return 4-tuple: (memory_id, score, action_trajectory, visual_state)
        results_with_visual = []
        for memory_id, score, traj in results:
            # Find the memory entry to get visual_state
            for mem in self.memory_chunks:
                if mem["memory_id"] == memory_id:
                    results_with_visual.append((
                        memory_id,
                        score,
                        traj,
                        mem["visual_state"]
                    ))
                    break
        return results_with_visual
    
    def get_memory_by_id(self, memory_id: int) -> Optional[Dict[str, np.ndarray]]:
        """Retrieve a specific memory chunk by ID."""
        idx = self.id_to_idx.get(memory_id)
        if idx is not None and idx < len(self.memory_chunks):
            return self.memory_chunks[idx].copy()
        return None
    
    def get_all_memories(self) -> List[Dict[str, np.ndarray]]:
        """Get all memory chunks."""
        return [mem.copy() for mem in self.memory_chunks]

    def get_all_task_labels(self) -> List[str]:
        """Get all task labels from stored memories."""
        labels = []
        for mem in self.memory_chunks:
            label = mem.get("task_label")
            if label is not None:
                labels.append(label)
        return labels
    
    def clear(self) -> None:
        """Clear all memories from the buffer."""
        self.memory_chunks.clear()
        self.id_to_idx.clear()
        if self.index.ntotal > 0:
            self.index.reset()
        self.total_additions = 0
        self.total_retrievals = 0
    
    @property
    def size(self) -> int:
        """Current number of stored memories."""
        return len(self.memory_chunks)
    
    def save(self, filepath: str) -> None:
        """Save the index and memory data to disk."""
        faiss.write_index(self.index, filepath + ".index")
        np.savez(
            filepath + ".data",
            memory_chunks=self.memory_chunks,
            id_to_idx=self.id_to_idx,
        )
    
    def load(self, filepath: str) -> None:
        """Load the index and memory data from disk."""
        self.index = faiss.read_index(filepath + ".index")
        data = np.load(filepath + ".data.npz", allow_pickle=True)
        self.memory_chunks = data["memory_chunks"].tolist()
        self.id_to_idx = data["id_to_idx"].tolist()


# Test the memory buffer
if __name__ == "__main__":
    # Create a memory buffer
    memory = EpisodicMemoryBuffer(embedding_dim=768, use_cosine_similarity=True)
    
    # Add some dummy memories
    for i in range(10):
        semantic_vec = np.random.randn(768).astype(np.float32)
        semantic_vec /= np.linalg.norm(semantic_vec)
        visual_state = np.random.randn(512).astype(np.float32)
        action_trajectory = np.random.randn(16, 7).astype(np.float32)
        
        memory.add_memory(
            memory_id=i,
            semantic_vector=semantic_vec,
            visual_state=visual_state,
            action_trajectory=action_trajectory,
        )
    
    print(f"Added {memory.size} memories")
    
    # Retrieve
    query_semantic = np.random.randn(768).astype(np.float32)
    query_semantic /= np.linalg.norm(query_semantic)
    query_visual = np.random.randn(512).astype(np.float32)
    
    results = memory.retrieve_closest_trajectory(query_visual, query_semantic, k=3)
    print(f"Retrieved {len(results)} similar trajectories:")
    for mem_id, score, traj in results:
        print(f"  Memory {mem_id}: score={score:.4f}, trajectory shape={traj.shape}")
