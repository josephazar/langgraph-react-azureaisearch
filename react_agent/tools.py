"""This module provides tools for Azure AI Search functionality.

It includes an Azure Cognitive Search function that performs hybrid vector + keyword search
on the configured knowledge base index.
"""

import os
from typing import Any, Callable, List, Optional, Dict
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery
from azure.core.credentials import AzureKeyCredential

from react_agent.configuration import Configuration


async def azure_ai_search(query: str) -> Optional[dict[str, Any]]:
    """Search the HR knowledge base using Azure AI Search with hybrid vector + keyword search.

    This function performs a hybrid search combining vector similarity and keyword matching
    on the configured Azure Cognitive Search index. It returns relevant documents with their
    content and metadata.
    
    Args:
        query: The search query string
    """
    configuration = Configuration.from_context()
    
    # Get search configuration
    search_endpoint = os.environ.get("COG_SEARCH_ENDPOINT")
    search_key = os.environ.get("COG_SEARCH_KEY")
    index_name = os.environ.get("COG_SEARCH_INDEX_NAME")
    
    if not all([search_endpoint, search_key, index_name]):
        missing = []
        if not search_endpoint:
            missing.append("COG_SEARCH_ENDPOINT")
        if not search_key:
            missing.append("COG_SEARCH_KEY")
        if not index_name:
            missing.append("COG_SEARCH_INDEX_NAME")
        return {"error": f"Azure Search configuration missing: {', '.join(missing)}"}
    
    try:
        # Create search client
        credential = AzureKeyCredential(search_key)
        search_client = SearchClient(
            endpoint=search_endpoint,
            index_name=index_name,
            credential=credential
        )
        
        # Create vector query for hybrid search
        vector_query = VectorizableTextQuery(
            text=query,
            k_nearest_neighbors=50,
            fields="content_vector"
        )
        
        # Perform hybrid search
        results = search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            select=["title", "content", "keyPhrases", "category", "document_type"],
            top=configuration.max_search_results
        )
        
        # Format results for the agent
        formatted_results = []
        for doc in results:
            formatted_results.append({
                "title": doc.get("title", ""),
                "content": doc.get("content", ""),
                "keyPhrases": doc.get("keyPhrases", []),
                "category": doc.get("category", ""),
                "document_type": doc.get("document_type", ""),
                "score": doc.get("@search.score", 0)
            })
        
        if not formatted_results:
            return {
                "results": [],
                "total_count": 0,
                "query": query,
                "message": "No documents found matching your query"
            }
        
        return {
            "results": formatted_results,
            "total_count": len(formatted_results),
            "query": query
        }
        
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}


TOOLS: List[Callable[..., Any]] = [azure_ai_search]
