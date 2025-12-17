"""
Memory module for Vision-Language-Action (VLA) systems
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import uuid


@dataclass
class MemoryEntry:
    """Represents an entry in the VLA memory system"""
    id: str
    content: Dict[str, Any]
    memory_type: str  # "episodic", "semantic", "working"
    timestamp: datetime
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    importance: float = 0.5
    lifetime: Optional[timedelta] = None  # How long to keep this memory


class MemoryModule:
    """Memory module for VLA systems with episodic, semantic, and working memory"""

    def __init__(self, max_capacity: int = 1000):
        self.max_capacity = max_capacity
        self.logger = logging.getLogger(__name__)

        # Memory banks for different types
        self.episodic_memory = []  # Personal experiences and events
        self.semantic_memory = []  # General knowledge and facts
        self.working_memory = []   # Temporary information for current tasks
        self.long_term_memory = [] # Important memories that persist

        # Initialize memory components
        self._initialize_memory()

    def _initialize_memory(self):
        """Initialize memory structures and parameters"""
        self.feature_dim = 512  # Dimension for memory embeddings
        self.similarity_threshold = 0.7  # Threshold for memory retrieval
        self.importance_decay_rate = 0.95  # Rate at which importance decays

    def write_to_memory(self, content: Dict[str, Any], memory_type: str = "episodic",
                       metadata: Optional[Dict[str, Any]] = None,
                       tags: Optional[List[str]] = None,
                       importance: float = 0.5,
                       lifetime: Optional[timedelta] = None) -> str:
        """Write information to memory"""
        try:
            # Generate embedding for the content (simulated)
            embedding = self._generate_embedding(content)

            # Create memory entry
            memory_id = str(uuid.uuid4())
            memory_entry = MemoryEntry(
                id=memory_id,
                content=content,
                memory_type=memory_type,
                timestamp=datetime.now(),
                embedding=embedding,
                metadata=metadata or {},
                tags=tags or [],
                importance=importance,
                lifetime=lifetime
            )

            # Add to appropriate memory bank
            if memory_type == "episodic":
                self.episodic_memory.append(memory_entry)
            elif memory_type == "semantic":
                self.semantic_memory.append(memory_entry)
            elif memory_type == "working":
                self.working_memory.append(memory_entry)
            else:
                self.long_term_memory.append(memory_entry)

            # Manage memory capacity
            self._manage_memory_capacity()

            self.logger.info(f"Memory entry added: {memory_id} to {memory_type} memory")

            return memory_id
        except Exception as e:
            self.logger.error(f"Error writing to memory: {str(e)}")
            raise

    def read_from_memory(self, query: str, memory_type: str = "episodic",
                        max_results: int = 5,
                        similarity_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Read information from memory based on query"""
        try:
            # Generate embedding for query (simulated)
            query_embedding = self._generate_embedding({"text": query})

            # Select memory bank
            memory_bank = self._get_memory_bank(memory_type)

            # Find similar memories
            similar_memories = []
            for entry in memory_bank:
                if entry.embedding is not None:
                    similarity = self._calculate_similarity(query_embedding, entry.embedding)
                    if similarity >= similarity_threshold:
                        similar_memories.append((entry, similarity))

            # Sort by similarity (descending)
            similar_memories.sort(key=lambda x: x[1], reverse=True)

            # Return top results
            results = []
            for entry, similarity in similar_memories[:max_results]:
                results.append({
                    "id": entry.id,
                    "content": entry.content,
                    "similarity": float(similarity),
                    "timestamp": entry.timestamp.isoformat(),
                    "metadata": entry.metadata,
                    "tags": entry.tags
                })

            self.logger.info(f"Retrieved {len(results)} memories from {memory_type} memory")

            return results
        except Exception as e:
            self.logger.error(f"Error reading from memory: {str(e)}")
            raise

    def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing memory entry"""
        try:
            # Find memory entry across all banks
            entry, bank = self._find_memory_entry(memory_id)
            if entry is None:
                return False

            # Update the entry
            for key, value in updates.items():
                if hasattr(entry, key):
                    setattr(entry, key, value)
                else:
                    entry.content[key] = value

            # Update timestamp
            entry.timestamp = datetime.now()

            self.logger.info(f"Memory entry updated: {memory_id}")

            return True
        except Exception as e:
            self.logger.error(f"Error updating memory: {str(e)}")
            return False

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory entry"""
        try:
            entry, bank = self._find_memory_entry(memory_id)
            if entry is None:
                return False

            if bank == "episodic":
                self.episodic_memory.remove(entry)
            elif bank == "semantic":
                self.semantic_memory.remove(entry)
            elif bank == "working":
                self.working_memory.remove(entry)
            else:
                self.long_term_memory.remove(entry)

            self.logger.info(f"Memory entry deleted: {memory_id}")

            return True
        except Exception as e:
            self.logger.error(f"Error deleting memory: {str(e)}")
            return False

    def _get_memory_bank(self, memory_type: str) -> List[MemoryEntry]:
        """Get the appropriate memory bank"""
        if memory_type == "episodic":
            return self.episodic_memory
        elif memory_type == "semantic":
            return self.semantic_memory
        elif memory_type == "working":
            return self.working_memory
        else:
            return self.long_term_memory

    def _find_memory_entry(self, memory_id: str) -> Tuple[Optional[MemoryEntry], Optional[str]]:
        """Find a memory entry across all banks"""
        for entry in self.episodic_memory:
            if entry.id == memory_id:
                return entry, "episodic"
        for entry in self.semantic_memory:
            if entry.id == memory_id:
                return entry, "semantic"
        for entry in self.working_memory:
            if entry.id == memory_id:
                return entry, "working"
        for entry in self.long_term_memory:
            if entry.id == memory_id:
                return entry, "long_term"

        return None, None

    def _generate_embedding(self, content: Dict[str, Any]) -> np.ndarray:
        """Generate embedding for content (simulated)"""
        # In a real implementation, this would use a neural network to generate embeddings
        # For simulation, we'll generate random embeddings based on content
        content_str = str(content)
        hash_val = hash(content_str) % (2**32)

        # Generate reproducible random embedding based on hash
        rng = np.random.RandomState(hash_val)
        embedding = rng.random(self.feature_dim).astype(np.float32)

        return embedding

    def _calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate similarity between two embeddings"""
        # Use cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        return float(similarity)

    def _manage_memory_capacity(self):
        """Manage memory capacity by removing old or less important entries"""
        # Merge all memories for capacity management
        all_memories = (self.episodic_memory + self.semantic_memory +
                       self.working_memory + self.long_term_memory)

        if len(all_memories) > self.max_capacity:
            # Sort by importance and timestamp (keep important and recent memories)
            all_memories.sort(key=lambda x: (x.importance,
                                           x.timestamp.timestamp()),
                            reverse=True)

            # Calculate how many to remove
            excess = len(all_memories) - self.max_capacity
            memories_to_remove = all_memories[-excess:]

            # Remove from respective banks
            for memory in memories_to_remove:
                if memory in self.episodic_memory:
                    self.episodic_memory.remove(memory)
                elif memory in self.semantic_memory:
                    self.semantic_memory.remove(memory)
                elif memory in self.working_memory:
                    self.working_memory.remove(memory)
                elif memory in self.long_term_memory:
                    self.long_term_memory.remove(memory)

    def store_episode(self, episode_data: Dict[str, Any],
                     task_description: str,
                     outcome: str) -> str:
        """Store an entire episode (sequence of states, actions, rewards)"""
        try:
            episode_content = {
                "task_description": task_description,
                "episode_data": episode_data,
                "outcome": outcome,
                "timestamp": datetime.now().isoformat()
            }

            return self.write_to_memory(
                content=episode_content,
                memory_type="episodic",
                tags=["episode", task_description.split()[0] if task_description else "task"],
                importance=0.8 if outcome == "success" else 0.5
            )
        except Exception as e:
            self.logger.error(f"Error storing episode: {str(e)}")
            raise

    def retrieve_relevant_episodes(self, current_state: Dict[str, Any],
                                 task_description: str,
                                 max_episodes: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant episodes for current task"""
        try:
            # Create query based on current state and task
            query = f"task: {task_description}, state: {str(current_state)[:100]}"

            # Retrieve relevant episodic memories
            episodes = self.read_from_memory(
                query=query,
                memory_type="episodic",
                max_results=max_episodes,
                similarity_threshold=0.4
            )

            return episodes
        except Exception as e:
            self.logger.error(f"Error retrieving relevant episodes: {str(e)}")
            return []

    def store_knowledge(self, concept: str, definition: str,
                       examples: Optional[List[str]] = None,
                       relations: Optional[Dict[str, str]] = None) -> str:
        """Store semantic knowledge"""
        try:
            knowledge_content = {
                "concept": concept,
                "definition": definition,
                "examples": examples or [],
                "relations": relations or {},
                "timestamp": datetime.now().isoformat()
            }

            return self.write_to_memory(
                content=knowledge_content,
                memory_type="semantic",
                tags=["knowledge", "concept", concept.split()[0] if concept else "unknown"],
                importance=0.7
            )
        except Exception as e:
            self.logger.error(f"Error storing knowledge: {str(e)}")
            raise

    def retrieve_knowledge(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieve semantic knowledge"""
        try:
            return self.read_from_memory(
                query=query,
                memory_type="semantic",
                max_results=max_results,
                similarity_threshold=0.5
            )
        except Exception as e:
            self.logger.error(f"Error retrieving knowledge: {str(e)}")
            return []

    def store_working_memory(self, key: str, value: Any,
                           task_context: Optional[str] = None) -> str:
        """Store temporary information in working memory"""
        try:
            content = {
                "key": key,
                "value": value,
                "task_context": task_context,
                "timestamp": datetime.now().isoformat()
            }

            # Set short lifetime for working memory (e.g., 1 hour)
            lifetime = timedelta(hours=1)

            return self.write_to_memory(
                content=content,
                memory_type="working",
                tags=["working", "temporary"] + ([task_context] if task_context else []),
                importance=0.9,  # High importance for current task
                lifetime=lifetime
            )
        except Exception as e:
            self.logger.error(f"Error storing working memory: {str(e)}")
            raise

    def retrieve_working_memory(self, key: str = None,
                              task_context: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve from working memory"""
        try:
            if key:
                query = key
            elif task_context:
                query = f"task_context: {task_context}"
            else:
                query = "working memory"

            results = self.read_from_memory(
                query=query,
                memory_type="working",
                max_results=10,
                similarity_threshold=0.3
            )

            # Filter by task context if specified
            if task_context:
                results = [r for r in results
                          if task_context in str(r.get('content', {}).get('task_context', ''))]

            return results
        except Exception as e:
            self.logger.error(f"Error retrieving working memory: {str(e)}")
            return []

    def consolidate_memory(self):
        """Consolidate important memories from working/episodic to long-term"""
        try:
            # Find high-importance working memories
            for entry in self.working_memory[:]:  # Copy to avoid modification during iteration
                if entry.importance > 0.7:
                    # Move to long-term memory
                    self.long_term_memory.append(entry)
                    self.working_memory.remove(entry)
                    self.logger.info(f"Consolidated memory {entry.id} to long-term")

            # Find important episodic memories
            for entry in self.episodic_memory[:]:
                if (entry.importance > 0.8 or
                    "success" in str(entry.content.get('outcome', '')).lower()):
                    # Move to long-term memory
                    self.long_term_memory.append(entry)
                    self.episodic_memory.remove(entry)
                    self.logger.info(f"Consolidated episode {entry.id} to long-term")

        except Exception as e:
            self.logger.error(f"Error consolidating memory: {str(e)}")

    def decay_memory_importance(self):
        """Decay the importance of memories over time"""
        try:
            current_time = datetime.now()

            for memory_bank in [self.episodic_memory, self.semantic_memory,
                              self.working_memory, self.long_term_memory]:
                for entry in memory_bank:
                    # Calculate time since entry
                    time_diff = current_time - entry.timestamp
                    hours_passed = time_diff.total_seconds() / 3600

                    # Apply decay (faster for working memory, slower for long-term)
                    if entry.memory_type == "working":
                        decay_factor = self.importance_decay_rate ** (hours_passed * 2)
                    else:
                        decay_factor = self.importance_decay_rate ** hours_passed

                    entry.importance *= decay_factor

                    # Remove if importance too low and has lifetime
                    if (entry.importance < 0.1 and entry.lifetime and
                        current_time > entry.timestamp + entry.lifetime):
                        memory_bank.remove(entry)

        except Exception as e:
            self.logger.error(f"Error decaying memory importance: {str(e)}")

    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get statistics about memory usage"""
        return {
            "episodic_count": len(self.episodic_memory),
            "semantic_count": len(self.semantic_memory),
            "working_count": len(self.working_memory),
            "long_term_count": len(self.long_term_memory),
            "total_count": (len(self.episodic_memory) + len(self.semantic_memory) +
                           len(self.working_memory) + len(self.long_term_memory)),
            "capacity_utilization": min(1.0, (len(self.episodic_memory) + len(self.semantic_memory) +
                                            len(self.working_memory) + len(self.long_term_memory)) / self.max_capacity),
            "average_importance": {
                "episodic": np.mean([e.importance for e in self.episodic_memory]) if self.episodic_memory else 0,
                "semantic": np.mean([e.importance for e in self.semantic_memory]) if self.semantic_memory else 0,
                "working": np.mean([e.importance for e in self.working_memory]) if self.working_memory else 0,
                "long_term": np.mean([e.importance for e in self.long_term_memory]) if self.long_term_memory else 0
            }
        }

    def clear_working_memory(self):
        """Clear working memory (for task completion)"""
        self.working_memory.clear()
        self.logger.info("Working memory cleared")


# Singleton instance
memory_module = MemoryModule()