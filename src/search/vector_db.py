import time
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Tuple
from config import settings
from src.models import Action, SearchResult
from src.search.base import SearchEngine


class VectorDBSearch(SearchEngine):
    def __init__(self, embedding_generator=None):
        self.client = chromadb.Client()
        
        # ChromaDB uses its own embedding function based on the provider
        # We need to use the appropriate embedding function for each provider
        if settings.EMBEDDING_PROVIDER == "local":
            # For local models, use SentenceTransformer
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=settings.EMBEDDING_MODEL
            )
        elif settings.EMBEDDING_PROVIDER == "openai":
            # For OpenAI, use OpenAI embedding function
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=settings.OPENAI_API_KEY,
                model_name=settings.EMBEDDING_MODEL
            )
        else:
            raise ValueError(
                f"ChromaDB does not support provider: {settings.EMBEDDING_PROVIDER}. "
                f"Supported providers: local, openai"
            )
        
        self.collection = None
        self.actions = None
        self.embedding_generator = embedding_generator
        
        print(f"📦 ChromaDB initialized with {settings.EMBEDDING_PROVIDER}/{settings.EMBEDDING_MODEL}")
    
    def get_name(self) -> str:
        return "ChromaDB"
    
    def index_actions(self, actions: List[Action], embeddings=None) -> None:
        self.actions = actions
        
        try:
            self.client.delete_collection("actions")
        except:
            pass  # Collection doesn't exist yet, that's fine
        
        self.collection = self.client.create_collection(
            name="actions",
            embedding_function=self.embedding_function
        )
        
        documents = []  # The text to embed
        ids = []        # Unique identifiers
        metadatas = []  # Additional information about each action
        
        for action in actions:
            # Convert action to text (same method FAISS uses)
            documents.append(action.to_embedding_text())
            
            # Use action ID as the document ID
            ids.append(action.id)
            
            metadatas.append({
                "title": action.title,
                "description": action.description,
                "endpoint": action.endpoint,
                "method": action.method,
                # Convert list to string for metadata storage
                "required_inputs": str(action.required_inputs)
            })
        
        # Add everything to the collection
        # ChromaDB will automatically generate embeddings for each document
        self.collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )
        
        print(f"✅ Indexed {len(actions)} actions in ChromaDB")
    
    def search(
        self, 
        query_embedding=None, 
        k: int = None,
        query_text: str = None
    ) -> Tuple[SearchResult, List[dict]]:
        # Validate that we have a collection
        if self.collection is None or self.actions is None:
            raise Exception("Collection not initialized. Call index_actions() first.")
        
        if k is None:
            k = settings.TOP_K_RESULTS
        
        # We need query_text for ChromaDB
        if query_text is None:
            raise Exception("ChromaDB requires query_text parameter")
        
        # Time the search operation for comparison with FAISS
        start_time = time.time()
        
        # Query the collection
        # ChromaDB will automatically:
        # 1. Convert query_text to an embedding
        # 2. Find the k most similar documents
        # 3. Return results with distances and metadata
        results = self.collection.query(
            query_texts=[query_text],  # Must be a list even for single query
            n_results=k
        )
        
        # Calculate elapsed time
        search_time = time.time() - start_time
        
        # Parse ChromaDB results
        # Results structure: {'ids': [[...]], 'distances': [[...]], 'metadatas': [[...]]}
        ids = results['ids'][0]              # IDs of matching documents
        distances = results['distances'][0]  # Distances (lower = more similar)
        metadatas = results['metadatas'][0]  # Metadata we stored
        
        # Build our results list
        all_results = []
        for i, (doc_id, distance, metadata) in enumerate(zip(ids, distances, metadatas)):
            # Find the full Action object by ID
            action = None
            for a in self.actions:
                if a.id == doc_id:
                    action = a
                    break
            
            if action:
                result_dict = {
                    "action_id": action.id,
                    "action": action,
                    "distance": float(distance)
                }
                all_results.append(result_dict)
        
        # Create the primary SearchResult from the best result
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