import numpy as np
from typing import Union, List
from config import settings


class EmbeddingGenerator:
    def __init__(self, model_name: str = None, provider: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.provider = provider or settings.EMBEDDING_PROVIDER
        
        print(f"📥 Loading embedding model: {self.model_name}")
        print(f"    Provider: {self.provider}")
        print(f"    Output dimension: {self.get_dimension()}")
        
        # Initialize the appropriate embedding client based on provider
        self._initialize_provider()
        
        print("✅ Model loaded successfully")
    
    def _initialize_provider(self):
        """
        Initialize the embedding provider based on configuration.
        This method sets up the client for the chosen provider.
        """
        if self.provider == "local":
            # Local provider uses sentence-transformers
            # This runs on your computer and doesn't need an API key
            from sentence_transformers import SentenceTransformer
            # Disable multiprocessing to avoid crashes on macOS
            import os
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            self.model = SentenceTransformer(self.model_name)
            
        elif self.provider == "openai":
            # OpenAI provider uses LangChain's OpenAI integration
            from langchain_openai import OpenAIEmbeddings
            
            # Check if API key is provided
            if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY == "your_openai_key_here":
                raise ValueError(
                    "OpenAI API key not found! Please set OPENAI_API_KEY in your .env file."
                )
            
            # Initialize OpenAI embeddings client
            self.model = OpenAIEmbeddings(
                model=self.model_name,
                openai_api_key=settings.OPENAI_API_KEY
            )
            
        elif self.provider == "voyage":
            # Voyage AI provider uses LangChain's Voyage integration
            from langchain_voyageai import VoyageAIEmbeddings
            
            # Check if API key is provided
            if not settings.VOYAGE_API_KEY or settings.VOYAGE_API_KEY == "your_voyage_key_here":
                raise ValueError(
                    "Voyage AI API key not found! Please set VOYAGE_API_KEY in your .env file."
                )
            
            # Initialize Voyage AI embeddings client
            self.model = VoyageAIEmbeddings(
                model=self.model_name,
                voyage_api_key=settings.VOYAGE_API_KEY
            )
            
        elif self.provider == "titan":
            # AWS Bedrock Titan provider uses LangChain's AWS integration
            from langchain_aws import BedrockEmbeddings
            import boto3
            
            # Check if AWS credentials are provided
            if not settings.AWS_ACCESS_KEY_ID or settings.AWS_ACCESS_KEY_ID == "your_aws_access_key":
                raise ValueError(
                    "AWS credentials not found! Please set AWS_ACCESS_KEY_ID and "
                    "AWS_SECRET_ACCESS_KEY in your .env file."
                )
            
            # Create a boto3 client for Bedrock
            # This is needed to authenticate with AWS
            bedrock_client = boto3.client(
                service_name='bedrock-runtime',
                region_name=settings.AWS_REGION,
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
            )
            
            # Initialize Bedrock embeddings client
            self.model = BedrockEmbeddings(
                model_id=self.model_name,
                client=bedrock_client
            )
            
        else:
            raise ValueError(
                f"Unknown embedding provider: {self.provider}. "
                f"Supported providers: local, openai, voyage, titan"
            )
    
    def get_dimension(self) -> int:
        """
        Get the embedding dimension for the current model.
        
        Returns:
            int: The number of dimensions in the embedding vector
        """
        # Look up the dimension from our mapping in settings
        if self.model_name not in settings.EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Unknown embedding model: {self.model_name}. "
                f"Supported models: {list(settings.EMBEDDING_DIMENSIONS.keys())}"
            )
        return settings.EMBEDDING_DIMENSIONS[self.model_name]
    
    def generate(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for one or more texts.
        
        Args:
            texts: A single string or list of strings to embed
            
        Returns:
            numpy array of shape (n_texts, dimension) containing the embeddings
        """
        # Convert single string to list for consistent processing
        if isinstance(texts, str):
            texts = [texts]
        
        # Generate embeddings based on provider
        if self.provider == "local":
            # Local provider: use sentence-transformers directly
            # convert_to_numpy=True ensures we get numpy arrays (required by FAISS)
            # show_progress_bar=False avoids multiprocessing issues
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            
        else:
            # LangChain providers (openai, voyage, titan)
            # Use the embed_documents method which is standard across LangChain
            embeddings_list = self.model.embed_documents(texts)
            
            # Convert from list of lists to numpy array
            # This ensures compatibility with FAISS which needs numpy arrays
            embeddings = np.array(embeddings_list, dtype=np.float32)
        
        return embeddings
    
    def generate_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        This is optimized for single queries and returns a 1D array.
        
        Args:
            text: The text to embed
            
        Returns:
            numpy array of shape (dimension,) containing the embedding
        """
        if self.provider == "local":
            # Local provider: use the batch method and extract first result
            embeddings = self.generate(text)
            # Return just the first (and only) embedding
            # This converts from 2D (1, dimension) to 1D (dimension,)
            return embeddings[0]
            
        else:
            # LangChain providers (openai, voyage, titan)
            # Use embed_query which is optimized for single text
            embedding_list = self.model.embed_query(text)
            
            # Convert from list to numpy array (1D)
            embedding = np.array(embedding_list, dtype=np.float32)
            
            return embedding