import os
import json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from typing import List
from langchain_core.documents import Document
from collections import defaultdict


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



def retrieve_documents(query: str, vector_store: FAISS, documents: List[Document], top_k: int = 5, rrf_k: int = 60, semantic_weight: float = 0.5) -> List[Document]:
    """
    Retrieve relevant documents based on a query, combining vector store and BM25 retrieval using RRF.

    Args:
        query (str): The user's query.
        vector_store (FAISS): FAISS vector store object.
        documents (List[Document]): List of all documents corresponding to the vector store for BM25.
        top_k (int): Number of top documents to retrieve from each method before fusion.
        rrf_k (int): RRF constant to dampen lower-ranked documents.
        semantic_weight (float): Weight assigned to semantic search (0.0 to 1.0). Keyword weight is (1 - semantic_weight).

    Returns:
        list: List of relevant Document objects after RRF.
    """
    print(f"Retrieving documents for query: {query} (Top K={top_k}, Semantic Weight={semantic_weight})")

    # Ensure weight is within bounds
    semantic_weight = max(0.0, min(1.0, semantic_weight))
    keyword_weight = 1.0 - semantic_weight

    # --- Vector store retrieval ---
    vector_store_docs = []
    if semantic_weight > 0: # Only run if weight is > 0
        vector_store_docs = vector_store.similarity_search(query, k=top_k)
        print(f"Retrieved {len(vector_store_docs)} documents from vector store (semantic search)")
        # ... (optional logging) ...

    # --- BM25 retrieval ---
    bm25_docs = []
    if keyword_weight > 0: # Only run if weight is > 0
        print("Performing BM25 retrieval (keyword search)")
        corpus = [doc.page_content for doc in documents]
        if not corpus:
             print("Warning: BM25 corpus is empty. Skipping keyword search.")
        else:
            try:
                bm25 = BM25Okapi(corpus)
                tokenized_query = query.split(" ")
                bm25_docs = bm25.get_top_n(tokenized_query, documents, n=top_k)
                print(f"Retrieved {len(bm25_docs)} documents from BM25")

            except Exception as e:
                print(f"Error during BM25 retrieval: {e}. Skipping keyword search.")

    # --- Assign ranks and prepare for RRF ---
    doc_id_to_doc = {}
    vs_ranking = {}
    for rank, doc in enumerate(vector_store_docs):
        # Use persistent_chunk_id if available, otherwise fallback to id()
        doc_id = doc.metadata.get("persistent_chunk_id", id(doc))
        vs_ranking[doc_id] = rank + 1
        doc_id_to_doc[doc_id] = doc

    bm25_ranking = {}
    for rank, doc in enumerate(bm25_docs):
        doc_id = doc.metadata.get("persistent_chunk_id", id(doc))
        bm25_ranking[doc_id] = rank + 1
        doc_id_to_doc[doc_id] = doc # Add BM25 docs that might not be in VS results

    # --- Apply RRF ---
    print(f"Applying Reciprocal Rank Fusion (Semantic Weight={semantic_weight}, Keyword Weight={keyword_weight})")
    rrf_scores = defaultdict(float)
    all_doc_ids = set(vs_ranking.keys()) | set(bm25_ranking.keys()) # Combine all unique doc IDs

    if not all_doc_ids:
        print("No documents found by either retrieval method.")
        return []

    for doc_id in all_doc_ids:
        # Penalize if not found, use a rank worse than max possible (e.g., top_k * 2 or just a large number)
        vs_rank = vs_ranking.get(doc_id, top_k * 2)
        bm25_rank = bm25_ranking.get(doc_id, top_k * 2)

        # Calculate weighted RRF score
        score = 0.0
        if semantic_weight > 0:
            score += semantic_weight * (1 / (rrf_k + vs_rank))
        if keyword_weight > 0:
            score += keyword_weight * (1 / (rrf_k + bm25_rank))

        rrf_scores[doc_id] = score
        # print(f"  Doc id={doc_id} | Semantic rank={vs_rank} | BM25 rank={bm25_rank} | RRF score={rrf_scores[doc_id]:.6f}") # Optional detailed log

    # Sort documents by RRF score
    # Ensure doc_id exists in doc_id_to_doc before accessing
    sorted_doc_ids = sorted(
        [(doc_id, score) for doc_id, score in rrf_scores.items() if doc_id in doc_id_to_doc],
        key=lambda x: x[1],
        reverse=True
    )

    # Return the top_k documents overall after fusion
    combined_docs = [doc_id_to_doc[doc_id] for doc_id, _ in sorted_doc_ids[:top_k]]

    print("Top documents after fusion:")
    # Use the sorted list with scores for logging
    for i, (doc_id, score) in enumerate(sorted_doc_ids[:top_k]):
        doc = doc_id_to_doc[doc_id]
        # Check for 'page_number' first, then 'page'
        page_info = doc.metadata.get('page_number', doc.metadata.get('page', 'unknown'))
        print(f"  [Final Rank {i+1}] Doc id={doc_id} | RRF score={score:.6f} | Page={page_info} | Source={doc.metadata.get('source', 'unknown')[:40]}...")

    # Log retrieval results to a file (ensure path is correct)
    debug_retrieval_path = os.path.join(os.path.dirname(__file__), "..", "debug_retrieval.json") # Save in app/
    retrieval_results = []
    for i, doc in enumerate(combined_docs):
        retrieval_results.append({
            "rank": i+1,
            "metadata": doc.metadata,
            "content": doc.page_content,
            "content_length": len(doc.page_content)
        })

    try:
        with open(debug_retrieval_path, "w") as f:
            json.dump(retrieval_results, f, indent=2)
    except Exception as debug_e:
        print(f"Warning: Could not write debug_retrieval.json: {debug_e}")

    return combined_docs