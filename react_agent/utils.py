"""Utility & helper functions."""

import os
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_openai import AzureChatOpenAI
from openai import AzureOpenAI
from typing import List, Dict, Any


def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
    """
    provider, model = fully_specified_name.split("/", maxsplit=1)
    
    if provider == "azure_openai":
        # Load Azure OpenAI model using environment variables
        return AzureChatOpenAI(
            azure_endpoint=os.environ.get("OPENAI_API_BASE"),
            api_key=os.environ.get("OPENAI_API_KEY"),
            api_version=os.environ.get("OPENAI_API_VERSION"),
            azure_deployment=os.environ.get("GPT4OMINI", model),
            temperature=0.1,
            max_tokens=1000
        )
    else:
        return init_chat_model(model, model_provider=provider)


def get_azure_openai_client() -> AzureOpenAI:
    """Get Azure OpenAI client for embeddings and other operations."""
    return AzureOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        api_version=os.environ.get("OPENAI_API_VERSION"),
        azure_endpoint=os.environ.get("OPENAI_API_BASE")
    )


def format_search_results_for_context(search_results: List[Dict[str, Any]]) -> str:
    """Format search results into a context string for the model."""
    if not search_results:
        return "No search results found."
    
    context_parts = []
    for i, result in enumerate(search_results, 1):
        title = result.get("title", "Untitled")
        content = result.get("content", "")
        context_parts.append(f"Document {i} - {title}:\n{content}\n")
    
    return "\n".join(context_parts)


def extract_sources_from_results(search_results: List[Dict[str, Any]]) -> List[str]:
    """Extract unique source filenames from search results."""
    sources = []
    for result in search_results:
        title = result.get("title", "")
        if title and title not in sources:
            sources.append(title)
    return sources
