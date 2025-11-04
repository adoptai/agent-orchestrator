# 🎯 Intent Matcher - Vector-Based Action Matching

An intelligent orchestration agent that identifies which AdoptAI agent/action/tool to invoke based on chat history and user messages. This system uses semantic embeddings and vector similarity search to match user intents with organizational capabilities.

## 📖 Overview

The Intent Matcher analyzes user messages from conversations and automatically determines which organizational action best matches the user's intent. It uses:

- **Sentence Transformers** to convert text into semantic embeddings (768-dimensional vectors)
- **FAISS** for fast similarity search across action embeddings
- **Configurable thresholds** to determine match quality and handle ambiguous cases

### Key Features

✅ **Semantic Understanding**: Uses AI embeddings to understand meaning, not just keywords  
✅ **Fast Search**: FAISS enables millisecond searches even with thousands of actions  
✅ **Intelligent Matching**: Handles clear matches, no matches, and confusing cases  
✅ **Highly Configurable**: Adjust thresholds to tune behavior without code changes  
✅ **Modular Architecture**: Clean separation of concerns, easy to extend  
✅ **Type-Safe**: Full type hints for better IDE support and fewer bugs

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- API credentials from AdoptAI (CLIENT_ID, SECRET_KEY, CONVERSATION_ID)

### Installation

1. **Clone the repository** (or ensure you're in the project directory):
```bash
cd agent-orchestrator
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

This will install:
- `sentence-transformers` - For generating semantic embeddings
- `faiss-cpu` - For fast vector similarity search
- `requests` - For HTTP API calls
- `python-dotenv` - For environment variable management
- `numpy` - For numerical operations
- `chromadb` - For potential future comparisons

3. **Configure environment variables**:

Create a `.env` file in the project root with your credentials:
```bash
# Create the .env file
touch .env
```

Add the following content to `.env`:
```
CLIENT_ID=your_actual_client_id_here
SECRET_KEY=your_actual_secret_key_here
CONVERSATION_ID=your_actual_conversation_id_here
```

⚠️ **Important**: Replace the placeholder values with your actual API credentials!

### Running the Application

Simply run:
```bash
python main.py
```

No command-line arguments needed - all configuration is in `config/settings.py` and `.env`.

## 🎛️ Configuration & Experimentation

### Adjusting Thresholds

The matching behavior is controlled by thresholds in `config/settings.py`:

```python
# NO_MATCH_THRESHOLD: Distance above which we reject matches
# Higher = more lenient (more matches accepted)
# Lower = more strict (fewer matches accepted)
# Typical range: 0.5 to 1.0
NO_MATCH_THRESHOLD = 0.7

# CONFUSION_MARGIN: Difference below which we consider results "too close"
# Higher = less confusion detection
# Lower = more confusion detection  
# Typical range: 0.05 to 0.2
CONFUSION_MARGIN = 0.1

# TOP_K_RESULTS: How many results to retrieve from search
TOP_K_RESULTS = 3
```

### Experimentation Workflow

1. **Run with default settings**:
   ```bash
   python main.py
   ```
   
2. **Observe the results**:
   - Check the match type (none/single/multiple)
   - Look at the distances in the output
   - Read the reasoning

3. **Adjust thresholds** in `config/settings.py`:
   - If too many "no match" → increase `NO_MATCH_THRESHOLD`
   - If too many false positives → decrease `NO_MATCH_THRESHOLD`
   - If too much confusion → increase `CONFUSION_MARGIN`
   - If missing legitimate confusion → decrease `CONFUSION_MARGIN`

4. **Run again** and compare results

5. **Repeat** until you find optimal values for your use case

## 📊 Understanding the Output

When you run the application, you'll see output like this:

```
================================================================================
SEARCH RESULTS
================================================================================

⏱️  Search Method: FAISS
⏱️  Search Time: 0.0023s

🎯 Top Matches:

  1. Create Project
     Distance: 0.3521
     Description: Creates a new project in the system

  2. Update Project
     Distance: 0.4102
     Description: Updates an existing project's information

  3. Delete Project
     Distance: 0.5834
     Description: Removes a project from the system

--------------------------------------------------------------------------------
MATCH RESULT
--------------------------------------------------------------------------------

📊 Match Type: ✅ SINGLE MATCH

🎯 Recommended Action(s):
   • Create Project

💡 Reasoning:
   Clear match found with distance 0.3521 (threshold: 0.7)

================================================================================
```

## 🚧 Future Enhancements

Potential improvements to explore:

- [ ] Add Vector DB support for comparison with FAISS
- [ ] Implement caching of action embeddings to disk
- [ ] Add support for conversation context (not just last message)
- [ ] Add unit tests for all modules
- [ ] Support batch processing of multiple conversations
- [ ] Add metrics tracking (accuracy, precision, recall)
- [ ] Implement more sophisticated matching algorithms

## 📚 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| requests | ≥2.31.0 | HTTP API communication |
| python-dotenv | ≥1.0.0 | Environment variable management |
| sentence-transformers | ≥2.2.2 | Generate semantic embeddings |
| faiss-cpu | ≥1.7.4 | Fast similarity search |
| numpy | ≥1.24.0 | Numerical operations |
| chromadb | ≥0.4.0 | Future vector database comparison |