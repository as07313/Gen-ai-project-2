import os
import json
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage # Import SystemMessage

def generate_response(query, documents):
    """
    Generate a response based on the query and retrieved documents, including citations.

    Args:
        query (str): The user's query.
        documents (list): List of relevant Document objects.

    Returns:
        tuple: (Generated response string, List of documents used for context)
               or (Error message string, Empty list) on error.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        error_msg = "Error: OpenAI API key not found. Please check your .env file."
        print(f"ERROR: {error_msg}")
        return error_msg, [] # Return tuple on error

    try:
        print(f"Generating response for query: {query}")
        print(f"Using {len(documents)} documents for context")

        llm = ChatOpenAI(api_key=api_key)

        # Format the context from retrieved documents, including source information
        context = "\n\n".join([
            f"Document {i+1} (Source: {doc.metadata.get('filename', 'unknown')}, Page: {doc.metadata.get('page_number', doc.metadata.get('page', 'unknown'))}): {doc.page_content}"
            for i, doc in enumerate(documents)
        ])

        # Define the system role/instructions
        system_prompt = """You are a helpful assistant specializing in maternal health information.
        Answer the user's question based *only* on the provided context documents.
        Be concise and informative.
        Cite your sources clearly using the format [Source: Document Number, Page Number].
        If the context doesn't contain the answer, state that clearly.
        If the user's question is in Roman Urdu, please provide the answer in standard Urdu script.""" # Added instruction for Roman Urdu input

        # Create the user prompt (contains context and the actual question)
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

        # Save prompt components for debugging
        debug_prompt_path = os.path.join(os.path.dirname(__file__), "..", "debug_prompt.json") # Save in app/
        debug_info = {
            "query": query,
            "num_documents": len(documents),
            "document_sources": [doc.metadata.get('filename', 'unknown') for doc in documents],
            "system_prompt": system_prompt, # Save system prompt
            "user_prompt": user_prompt      # Save user prompt
        }
        try:
            with open(debug_prompt_path, "w") as f:
                json.dump(debug_info, f, indent=2)
        except Exception as debug_e:
             print(f"Warning: Could not write debug_prompt.json: {debug_e}")


        print(f"System Prompt: {system_prompt}")
        print(f"User Prompt length: {len(user_prompt)} characters")

        # Use SystemMessage for role and HumanMessage for query/context
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        response = llm(messages).content

        print(f"Generated response length: {len(response)} characters")

        # Save response for debugging (ensure path is correct if needed)
        debug_response_path = os.path.join(os.path.dirname(__file__), "..", "debug_response.txt") # Save in app/
        try:
            with open(debug_response_path, "w") as f:
                f.write(f"QUERY:\n{query}\n\nSYSTEM PROMPT:\n{system_prompt}\n\nUSER PROMPT (excerpt):\n{user_prompt[:500]}...\n\nRESPONSE:\n{response}")
        except Exception as debug_e:
             print(f"Warning: Could not write debug_response.txt: {debug_e}")

        # Return both the response and the documents used
        return response, documents # Return tuple

    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        print(f"ERROR: {error_msg}")
        return error_msg, [] # Return tuple on error