import os
import cohere
import json
from typing import List
from langchain_core.documents import Document

def rerank_documents(query: str, documents: List[Document], top_n: int = 5) -> List[Document]:
    """
    Reranks retrieved documents using Cohere's reranking API for more accurate results.
    
    Args:
        query (str): The user's query.
        documents (List[Document]): List of retrieved Document objects.
        top_n (int): Number of top documents to keep after reranking.
        
    Returns:
        List[Document]: Reranked list of Document objects.
    """
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        print("WARNING: Cohere API key not found. Skipping reranking.")
        return documents[:top_n]
    
    try:
        print(f"Reranking {len(documents)} documents for query: {query}")
        
        # Initialize Cohere client
        co = cohere.Client(api_key)
        
        # Extract document texts and their indices
        doc_texts = [doc.page_content for doc in documents]
        
        # Call Cohere's rerank API
        rerank_results = co.rerank(
            query=query,
            documents=doc_texts,
            top_n=top_n,
            model="rerank-english-v2.0"
        )
        
        # Log reranking results
        reranking_debug = {
            "query": query,
            "original_docs_count": len(documents),
            "reranked_docs": []
        }
        
        # Get reranked documents 
        reranked_documents = []
        for result in rerank_results.results:
            idx = result.index
            relevance_score = result.relevance_score
            
            reranking_debug["reranked_docs"].append({
                "original_index": idx,
                "relevance_score": relevance_score,
                "content_preview": doc_texts[idx][:100] + "..." if len(doc_texts[idx]) > 100 else doc_texts[idx]
            })
            
            reranked_documents.append(documents[idx])
        
        # Save reranking debug info
        with open("debug_reranking.json", "w") as f:
            json.dump(reranking_debug, f, indent=2)
        
        print(f"Reranking complete. Kept top {len(reranked_documents)} documents.")
        return reranked_documents
    
    except Exception as e:
        print(f"ERROR in reranking: {str(e)}")
        # Fallback to original documents if reranking fails
        return documents[:top_n]