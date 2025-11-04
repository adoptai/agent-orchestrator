import time
import faiss
import numpy as np
from typing import List, Optional, Tuple
from config import settings
from src.models import Action, SearchResult
from src.search.base import SearchEngine


class FAISSSearch(SearchEngine):
    
    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.actions: Optional[List[Action]] = None
        self.dimension = settings.EMBEDDING_DIMENSION
    
    def get_name(self) -> str:
        return "FAISS"
    
    def index_actions(self, actions: List[Action], embeddings: np.ndarray) -> None:
        # Store the actions so we can map indices back to Action objects later
        self.actions = actions
        
        # Create a FAISS index using L2 distance
        # L2 distance is also called Euclidean distance
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # FAISS requires embeddings to be float32
        # If they're already float32, this is a no-op
        embeddings = embeddings.astype('float32')
        
        # Add all embeddings to the index
        # After this, the index contains all our action embeddings
        self.index.add(embeddings)
        
        print(f"✅ Indexed {len(actions)} actions in FAISS")
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        k: int = None
    ) -> Tuple[SearchResult, List[dict]]:
        """
        Search for the k most similar actions to the query.
        
        Args:
            query_embedding: The embedding to search for (can be 1D or 2D)
            k: Number of results to return (default: from settings)
            
        Returns:
            Tuple of (primary SearchResult, list of all results)
            
        Raises:
            Exception: If the index hasn't been created yet
        """
        # Validate that we have an index
        if self.index is None or self.actions is None:
            raise Exception("Index not initialized. Call index_actions() first.")
        
        # Use default k from settings if not provided
        if k is None:
            k = settings.TOP_K_RESULTS
        
        # Prepare the query embedding
        # FAISS expects 2D arrays of shape (n_queries, dimension)
        if query_embedding.ndim == 1:
            # Reshape from (dimension,) to (1, dimension)
            query_embedding = query_embedding.reshape(1, -1)
        
        # Convert to float32 (FAISS requirement)
        query_embedding = query_embedding.astype('float32')
        
        # Time the search operation (useful for performance monitoring)
        start_time = time.time()
        
        # Perform the search
        # Returns:
        # - distances: Array of shape (n_queries, k) with L2 distances
        # - indices: Array of shape (n_queries, k) with indices of nearest neighbors
        distances, indices = self.index.search(query_embedding, k)
        
        # Calculate elapsed time
        search_time = time.time() - start_time
        
        # Extract results from the first (and only) query
        # distances[0] and indices[0] are arrays of length k
        result_distances = distances[0]
        result_indices = indices[0]
        
        # Build the results list
        all_results = []
        for i, (idx, distance) in enumerate(zip(result_indices, result_distances)):
            # idx is the position in our original actions list
            # Make sure idx is valid (FAISS can return -1 if fewer than k results)
            if idx >= 0 and idx < len(self.actions):
                action = self.actions[idx]
                result_dict = {
                    "action_id": action.id,
                    "action": action,
                    "distance": float(distance)
                }
                all_results.append(result_dict)
        
        # Create the primary SearchResult from the best (first) result
        if all_results:
            best = all_results[0]
            primary_result = SearchResult(
                action_id=best["action_id"],
                action=best["action"],
                distance=best["distance"],
                search_method=self.get_name(),
                time_taken=search_time
            )
        else:
            raise Exception("No search results found")
        
        return primary_result, all_results

