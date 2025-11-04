from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
from src.models import Action, SearchResult
from config import settings


class SearchEngine(ABC):
    
    @abstractmethod
    def index_actions(self, actions: List[Action], embeddings: np.ndarray) -> None:
        """
        Store actions and their embeddings in the search engine.
        
        This is the "setup" phase - we're building an index that will
        allow fast similarity searches later.
        
        Args:
            actions: List of Action objects to index
            embeddings: Numpy array of shape (n_actions, embedding_dimension)
                       where embeddings[i] is the embedding for actions[i]
                       
        Returns:
            None
        """
        pass
    
    @abstractmethod
    def search(
        self, 
        query_embedding: np.ndarray, 
        k: int = None
    ) -> Tuple[SearchResult, List[dict]]:
        """
        Search for actions similar to the query embedding.
        
        This is the "query" phase - we're finding the k most similar
        actions to a query.
        
        Args:
            query_embedding: The embedding vector to search for (1D array)
            k: Number of results to return. If None, uses settings.TOP_K_RESULTS
            
        Returns:
            A tuple of:
            - SearchResult: The primary (best) result
            - List[dict]: All results with their distances and metadata
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of this search engine.
        
        Returns:
            A string like "FAISS", "ChromaDB", etc.
        """
        pass

