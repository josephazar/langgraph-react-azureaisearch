"""Main entry point for testing the HR Assistant agent."""

import asyncio
import os
from typing import Optional
from langchain_core.messages import HumanMessage, AIMessage
from react_agent.graph import graph
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


async def chat_with_agent(question: str, conversation_history: Optional[list] = None):
    """
    Chat with the HR Assistant agent.
    
    Args:
        question: The user's question
        conversation_history: Optional list of previous messages
        
    Returns:
        The agent's response
    """
    # Initialize messages with conversation history if provided
    messages = conversation_history or []
    messages.append(HumanMessage(content=question))
    
    # Create the initial state
    initial_state = {
        "messages": messages
    }
    
    # Run the agent
    try:
        # Stream the response
        async for event in graph.astream(initial_state):
            # Get the last message from the event
            if "call_model" in event and event["call_model"].get("messages"):
                last_message = event["call_model"]["messages"][-1]
                if isinstance(last_message, AIMessage) and not last_message.tool_calls:
                    return last_message.content
                    
    except Exception as e:
        return f"Error: {str(e)}"
    
    return "No response generated."


async def main():
    """Main function to demonstrate the agent."""
    print("HR Assistant Agent - ReAct with Azure AI Search")
    print("=" * 50)
    print("\nThis agent can help with HR-related questions.")
    print("It will search the knowledge base when needed.")
    print("\nType 'quit' to exit.\n")
    
    conversation_history = []
    
    while True:
        # Get user input
        question = input("\nYou: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
            
        if not question:
            continue
        
        # Get agent response
        print("\nAssistant: ", end="", flush=True)
        response = await chat_with_agent(question, conversation_history)
        print(response)
        
        # Update conversation history
        conversation_history.append(HumanMessage(content=question))
        conversation_history.append(AIMessage(content=response))


if __name__ == "__main__":
    # Example usage for testing specific questions
    test_questions = [
        "Hello!",
        "What are the company's vacation policies?",
        "Tell me about employee benefits",
        "Tell me about travel policies",
        "What's the weather like?",
        "How do I request time off?"
    ]
    
    print("Running test questions...\n")
    
    # Run a single test question
    async def test_single_question():
        question = "What are the health insurance options?"
        print(f"Question: {question}")
        response = await chat_with_agent(question)
        print(f"Response: {response}\n")
    
    # Uncomment to run the test
    # asyncio.run(test_single_question())
    
    # Run interactive chat
    asyncio.run(main())