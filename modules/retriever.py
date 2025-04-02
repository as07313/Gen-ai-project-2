import os
import json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def create_vector_store(documents, index_path="embeddings/faiss_index"):
    """
    Create a FAISS vector store from documents.

    Args:
        documents (list): List of LangChain Document objects.
        index_path (str): Path to save the FAISS index.

    Returns:
        FAISS: FAISS vector store object.
    """
    print(f"Creating vector store at {index_path} with {len(documents)} documents")
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    # Get a sample embedding to see dimensions
    sample_text = documents[0].page_content
    sample_embedding = embeddings.embed_query(sample_text)
    print(f"Sample embedding dimension: {len(sample_embedding)}")
    print(f"Sample embedding first 5 values: {sample_embedding[:5]}")
    
    vector_store = FAISS.from_documents(documents, embeddings)
    print(f"Vector store created. Doc count: {len(vector_store.docstore._dict)}")
    
    vector_store.save_local(index_path)
    print(f"Vector store saved to {index_path}")
    return vector_store

def load_vector_store(index_path="embeddings/faiss_index"):
    """
    Load a FAISS vector store from disk.

    Args:
        index_path (str): Path to the saved FAISS index.

    Returns:
        FAISS: Loaded FAISS vector store object.
    """
    print(f"Loading vector store from {index_path}")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    print(f"Vector store loaded. Doc count: {len(vector_store.docstore._dict)}")
    return vector_store

def retrieve_documents(query, vector_store, top_k=5):
    """
    Retrieve relevant documents based on a query.

    Args:
        query (str): The user's query.
        vector_store (FAISS): FAISS vector store object.
        top_k (int): Number of top documents to retrieve.

    Returns:
        list: List of relevant Document objects.
    """
    print(f"Retrieving documents for query: {query}")
    
    # Get embeddings for debug purposes
    embeddings = vector_store.embeddings
    query_embedding = embeddings.embed_query(query)
    print(f"Query embedding dimension: {len(query_embedding)}")
    
    docs = vector_store.similarity_search(query, k=top_k)
    
    print(f"Retrieved {len(docs)} documents")
    
    # Log retrieval results to a file
    retrieval_results = []
    for i, doc in enumerate(docs):
        retrieval_results.append({
            "rank": i+1,
            "metadata": doc.metadata,
            "content": doc.page_content,
            "content_length": len(doc.page_content)
        })
    
    with open("debug_retrieval.json", "w") as f:
        json.dump(retrieval_results, f, indent=2)
    
    return docs