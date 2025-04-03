import os
import json
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

def generate_response(query, documents):
    """
    Generate a response based on the query and retrieved documents.

    Args:
        query (str): The user's query.
        documents (list): List of relevant Document objects.

    Returns:
        str: Generated response.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Error: OpenAI API key not found. Please check your .env file."
    
    try:
        print(f"Generating response for query: {query}")
        print(f"Using {len(documents)} documents for context")
        
        llm = ChatOpenAI(api_key=api_key)
        
        # Format the context from retrieved documents
        context = "\n\n".join([f"Document {i+1} (Source: {doc.metadata.get('source', 'unknown')}): {doc.page_content}" 
                               for i, doc in enumerate(documents)])
        
        # Create the prompt
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        
        # Save prompt for debugging
        debug_info = {
            "query": query,
            "num_documents": len(documents),
            "document_sources": [doc.metadata.get('source', 'unknown') for doc in documents],
            "prompt": prompt
        }
        
        with open("debug_prompt.json", "w") as f:
            json.dump(debug_info, f, indent=2)
        
        print(f"Prompt length: {len(prompt)} characters")
        
        messages = [HumanMessage(content=prompt)]
        response = llm(messages).content
        
        print(f"Generated response length: {len(response)} characters")
        
        # Save response for debugging
        with open("debug_response.txt", "w") as f:
            f.write(f"QUERY:\n{query}\n\nRESPONSE:\n{response}")
        
        return response
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        print(f"ERROR: {error_msg}")
        return error_msg