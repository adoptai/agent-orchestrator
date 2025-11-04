import numpy as np
from typing import Union, List
from sentence_transformers import SentenceTransformer
from config import settings


class EmbeddingGenerator:
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        
        print(f"📥 Loading embedding model: {self.model_name}")
        print(f"    Output dimension: {settings.EMBEDDING_DIMENSION}")
        
        # Load the pre-trained model
        # This downloads the model if it's not already cached locally
        self.model = SentenceTransformer(self.model_name)
        
        print("✅ Model loaded successfully")
    
    def generate(self, texts: Union[str, List[str]]) -> np.ndarray:
        # Convert single string to list for consistent processing
        if isinstance(texts, str):
            texts = [texts]
        
        # Generate embeddings using the model
        # convert_to_numpy=True ensures we get numpy arrays (required by FAISS)
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        return embeddings
    
    def generate_single(self, text: str) -> np.ndarray:
        # Generate embeddings (will return shape (1, dimension))
        embeddings = self.generate(text)
        
        # Return just the first (and only) embedding
        # This converts from 2D (1, dimension) to 1D (dimension,)
        return embeddings[0]