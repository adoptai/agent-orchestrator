# ============================================================================
# CRITICAL: OpenMP Initialization Control
# ============================================================================
# This MUST be the first import to properly configure OpenMP before any
# libraries (numpy, faiss, chromadb) initialize their OpenMP runtimes.
# This prevents the "multiple copies of OpenMP runtime" error.
import openmp_init

# Now we can safely import packages that use OpenMP
from config import settings
from src.auth.client import AuthClient
from src.api.actions import ActionsAPI
from src.api.conversations import ConversationsAPI
from src.embeddings.generator import EmbeddingGenerator
from src.search.faiss_search import FAISSSearch
from src.search.vector_db import VectorDBSearch
from src.matching.matcher import IntentMatcher
from src.utils.logger import logger


def print_header(text: str, char: str = "="):
    print("\n" + char * 80)
    print(text.center(80))
    print(char * 80 + "\n")


def main():
    try:
        print_header("🎯 INTENT MATCHER", "=")
        print("Vector-Based Action Matching System")
        print("Analyzing user messages to find the best matching organizational actions\n")
        
        # Authentication
        print("🔐 Step 1: Authenticating with API...")
        auth_client = AuthClient()
        token = auth_client.get_token()
        print(f"✅ Authentication successful")
        logger.info("Successfully authenticated with API")
        
        # Fetching List of Actions
        print("\n📥 Step 2: Fetching available actions...")
        actions_api = ActionsAPI(auth_client)
        actions = actions_api.fetch_actions()
        print(f"✅ Fetched {len(actions)} actions from the organization")
        logger.info(f"Retrieved {len(actions)} actions")
        
        # Validate that we have actions to work with
        if not actions:
            print("⚠️  Warning: No actions found. Cannot proceed.")
            return
        
        # Display a few example actions for context
        print("\n   Example actions:")
        for action in actions[:3]:
            print(f"   • {action.title}")
        if len(actions) > 3:
            print(f"   ... and {len(actions) - 3} more")
        
        # Fetch Conversation
        print("\n💬 Step 3: Fetching conversation...")
        conversations_api = ConversationsAPI(auth_client)
        user_message = conversations_api.get_last_user_message(settings.CONVERSATION_ID)
        
        # Display a preview of the message (truncate if too long)
        preview = user_message if len(user_message) <= 100 else user_message[:100] + "..."
        print(f"✅ Found user message: \"{preview}\"")
        logger.info(f"Analyzing message: {user_message}")
        
        # Generate Embedding for Actions
        print("\n🧠 Step 4: Generating embeddings for actions...")
        embedding_generator = EmbeddingGenerator()
        
        # Convert each action to text suitable for embedding
        action_texts = [action.to_embedding_text() for action in actions]
        
        # Generate embeddings for all actions at once (batching is efficient)
        print(f"   Converting {len(action_texts)} actions to {settings.EMBEDDING_DIMENSION}-dimensional vectors...")
        action_embeddings = embedding_generator.generate(action_texts)
        print(f"✅ Generated embeddings for {len(actions)} actions")
        logger.info(f"Generated {len(actions)} action embeddings")
        
        # Generate Embedding for User Message
        print("\n🧠 Step 5: Generating embedding for user message...")
        user_embedding = embedding_generator.generate_single(user_message)
        print(f"✅ Generated user message embedding (dimension: {len(user_embedding)})")
        logger.info("Generated user message embedding")
        
        # =====================================================================
        # COMPARISON: FAISS vs ChromaDB
        # =====================================================================
        print("\n" + "=" * 80)
        print("🔬 SEARCH ENGINE COMPARISON: FAISS vs ChromaDB".center(80))
        print("=" * 80)
        
        # ---------------------------------------------------------------------
        # METHOD 1: FAISS (Manual Embedding Management)
        # ---------------------------------------------------------------------
        print("\n🔍 Method 1: FAISS Search")
        print("   Approach: We generate embeddings manually and pass them to FAISS")
        
        # Index in FAISS
        print("   • Indexing actions in FAISS...")
        faiss_engine = FAISSSearch(embedding_generator)
        faiss_engine.index_actions(actions, action_embeddings)
        
        # Search with FAISS
        print("   • Searching for matching actions...")
        faiss_result, faiss_all_results = faiss_engine.search(
            user_embedding, 
            k=settings.TOP_K_RESULTS
        )
        print(f"   ✅ FAISS search completed in {faiss_result.time_taken:.4f}s")
        logger.info(f"FAISS search completed in {faiss_result.time_taken:.4f}s")
        
        # Apply Matching Logic to FAISS results
        matcher = IntentMatcher()
        faiss_match = matcher.match(faiss_result, faiss_all_results)
        
        # ---------------------------------------------------------------------
        # METHOD 2: ChromaDB (Automatic Embedding Management)
        # ---------------------------------------------------------------------
        print("\n📦 Method 2: ChromaDB Search")
        print("   Approach: ChromaDB generates embeddings automatically from text")
        
        # Index in ChromaDB
        print("   • Indexing actions in ChromaDB...")
        vectordb_engine = VectorDBSearch(embedding_generator)
        vectordb_engine.index_actions(actions)  # No embeddings parameter!
        
        # Search with ChromaDB 
        print("   • Searching for matching actions...")
        vectordb_result, vectordb_all_results = vectordb_engine.search(
            query_text=user_message,  # Pass text, not embedding!
            k=settings.TOP_K_RESULTS
        )
        print(f"   ✅ ChromaDB search completed in {vectordb_result.time_taken:.4f}s")
        logger.info(f"ChromaDB search completed in {vectordb_result.time_taken:.4f}s")
        
        # Apply Matching Logic to ChromaDB results
        vectordb_match = matcher.match(vectordb_result, vectordb_all_results)
        
        # =====================================================================
        # COMPARISON RESULTS
        # =====================================================================
        print("\n" + "=" * 80)
        print("📊 COMPARISON RESULTS".center(80))
        print("=" * 80)
        
        # Display FAISS results
        print("\n" + "-" * 80)
        print("FAISS Results:")
        print("-" * 80)
        formatted_faiss = matcher.format_output(faiss_match, faiss_all_results)
        print(formatted_faiss)
        
        # Display ChromaDB results
        print("\n" + "-" * 80)
        print("ChromaDB Results:")
        print("-" * 80)
        formatted_vectordb = matcher.format_output(vectordb_match, vectordb_all_results)
        print(formatted_vectordb)
        
        # Performance comparison
        print("\n" + "=" * 80)
        print("⚡ PERFORMANCE COMPARISON".center(80))
        print("=" * 80)
        print(f"\n⏱️  FAISS Search Time:    {faiss_result.time_taken:.4f}s")
        print(f"⏱️  ChromaDB Search Time: {vectordb_result.time_taken:.4f}s")
        
        if faiss_result.time_taken < vectordb_result.time_taken:
            speedup = vectordb_result.time_taken / faiss_result.time_taken
            print(f"\n🏆 FAISS is {speedup:.2f}x faster!")
        else:
            speedup = faiss_result.time_taken / vectordb_result.time_taken
            print(f"\n🏆 ChromaDB is {speedup:.2f}x faster!")
        
        # Check if both methods found the same action
        faiss_best = faiss_all_results[0]["action"].title if faiss_all_results else "None"
        vectordb_best = vectordb_all_results[0]["action"].title if vectordb_all_results else "None"
        
        print(f"\n🎯 Best Match Comparison:")
        print(f"   FAISS:    {faiss_best}")
        print(f"   ChromaDB: {vectordb_best}")
        
        if faiss_best == vectordb_best:
            print(f"   ✅ Both methods found the same best action!")
        else:
            print(f"   ⚠️  Methods found different best actions")
        
        # Completion
        print("=" * 80)
        print("✅ Analysis complete!\n")
        logger.info("Intent matching workflow completed successfully")
        
    except Exception as e:
        # Catch any errors that occurred during execution
        print(f"\n❌ Error: {str(e)}")
        logger.error(f"Error in main workflow: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()