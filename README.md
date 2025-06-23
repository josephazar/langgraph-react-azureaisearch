# HR Assistant Agent with Azure AI Search

A ReAct (Reasoning and Action) agent built with LangGraph that provides HR assistance using Azure AI Search for RAG (Retrieval Augmented Generation) capabilities.

## Features

- **Agentic Reasoning**: The agent intelligently decides when to search the knowledge base vs. respond directly
- **Hybrid Search**: Combines vector similarity and keyword matching for better search results
- **Azure Integration**: Uses Azure OpenAI and Azure Cognitive Search
- **Conversation Memory**: Maintains context across multiple turns
- **Source Citations**: Automatically includes sources when information comes from the knowledge base

## Architecture

The agent uses the ReAct pattern with LangGraph:
1. **Reasoning**: Analyzes user queries to determine the appropriate action
2. **Action**: Searches the knowledge base when needed using Azure AI Search
3. **Observation**: Processes search results and formulates responses
4. **Response**: Provides formatted answers with proper citations

## Setup

### Prerequisites

- Python 3.8+
- Azure OpenAI resource
- Azure Cognitive Search resource with an indexed knowledge base
- Required Python packages (see requirements.txt)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd react-agent-main
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your Azure credentials:
```env
OPENAI_API_KEY=your-azure-openai-key
OPENAI_API_VERSION=2024-05-01-preview
OPENAI_API_BASE=https://your-resource.openai.azure.com/
OPENAI_CHAT_API_KEY=your-azure-openai-key
OPENAI_CHAT_API_VERSION=2024-05-01-preview
OPENAI_CHAT_API_BASE=https://your-resource.openai.azure.com/
COG_SEARCH_ENDPOINT=https://your-search.search.windows.net
COG_SEARCH_KEY=your-search-key
COG_SEARCH_INDEX_NAME=your-index-name
GPT4OMINI=gpt-4o-mini
emb_model=text-embedding-ada-002
```

## Usage

Run the interactive chat interface:

```bash
python main.py
```

The agent will:
- Greet you and respond to small talk without searching
- Search the knowledge base for policy, benefits, and procedure questions
- Provide general HR knowledge when organization-specific info isn't found
- Politely decline non-HR related questions

### Example Interactions

```
You: Hello!
Assistant: Hello! How can I help you with HR-related questions today?

You: What are the travel policies?
Assistant: [Searches knowledge base and provides specific travel policy information with sources]

You: What's the weather like?
Assistant: I apologize, but I can only assist with HR-related questions...
```

## Project Structure

```
react-agent-main/
├── main.py                 # Entry point for the application
├── requirements.txt        # Python dependencies
├── .env                   # Environment variables (not in git)
├── .gitignore            # Git ignore file
└── react_agent/          # Main agent package
    ├── __init__.py       # Package initialization
    ├── configuration.py  # Agent configuration
    ├── graph.py         # LangGraph agent definition
    ├── prompts.py       # System prompts
    ├── state.py         # State management
    ├── tools.py         # Azure AI Search tool
    └── utils.py         # Utility functions
```

## Key Components

### Tools (tools.py)
- `azure_ai_search`: Performs hybrid vector + keyword search on Azure Cognitive Search

### Graph (graph.py)
- Defines the ReAct agent flow with reasoning and tool execution nodes
- Handles source citation formatting

### State (state.py)
- Tracks conversation messages, search results, and sources used

### Configuration (configuration.py)
- Manages Azure settings and model configuration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

[Your License Here]