from config import settings
from src.auth.client import AuthClient
from src.api.actions import ActionsAPI
from src.api.conversations import ConversationsAPI
from src.embeddings.generator import EmbeddingGenerator
from src.search.faiss_search import FAISSSearch
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
        
        # Index in FAISS
        print("\n🔍 Step 6: Indexing actions in FAISS...")
        search_engine = FAISSSearch()
        search_engine.index_actions(actions, action_embeddings)
        logger.info("Actions indexed in FAISS")
        
        # Search for Matching Actions
        print("\n🎯 Step 7: Searching for matching actions...")
        search_result, all_results = search_engine.search(
            user_embedding, 
            k=settings.TOP_K_RESULTS
        )
        print(f"✅ Found {len(all_results)} potential matches")
        logger.info(f"Search completed in {search_result.time_taken:.4f}s")
        
        # Apply Matching Logic
        print("\n🤔 Step 8: Applying matching logic...")
        matcher = IntentMatcher()
        match_result = matcher.match(search_result, all_results)
        logger.info(f"Match type: {match_result.match_type}")
        
        # Display Results
        print("\n📊 Step 9: Results")
        formatted_output = matcher.format_output(match_result, all_results)
        print(formatted_output)
        
        # Completion
        print("✅ Analysis complete!\n")
        logger.info("Intent matching workflow completed successfully")
        
    except Exception as e:
        # Catch any errors that occurred during execution
        print(f"\n❌ Error: {str(e)}")
        logger.error(f"Error in main workflow: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()