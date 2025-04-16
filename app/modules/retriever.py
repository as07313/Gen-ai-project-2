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



def retrieve_documents(query, vector_store, documents: List[Document], top_k=5, rrf_k=60):
    """
    Retrieve relevant documents based on a query, combining vector store and BM25 retrieval using RRF.

    Args:
        query (str): The user's query.
        vector_store (FAISS): FAISS vector store object.
        documents (List[Document]): List of all documents.
        top_k (int): Number of top documents to retrieve.
        rrf_k (int): RRF constant to dampen lower-ranked documents.

    Returns:
        list: List of relevant Document objects.
    """
    print(f"Retrieving documents for query: {query}")
    
    # Vector store retrieval
    embeddings = vector_store.embeddings
    vector_store_docs = vector_store.similarity_search(query, k=top_k)
    print(f"Retrieved {len(vector_store_docs)} documents from vector store (semantic search)")
    for idx, doc in enumerate(vector_store_docs):
        print(f"  [Semantic Rank {idx+1}] Doc id={id(doc)} | Page={doc.metadata.get('page', 'unknown')} | Source={doc.metadata.get('source', 'unknown')[:40]}...")

    # BM25 retrieval
    print("Performing BM25 retrieval (keyword search)")
    corpus = [doc.page_content for doc in documents]
    bm25 = BM25Okapi(corpus)
    tokenized_query = query.split(" ")
    bm25_docs = bm25.get_top_n(tokenized_query, documents, n=top_k)
    print(f"Retrieved {len(bm25_docs)} documents from BM25")
    for idx, doc in enumerate(bm25_docs):
        print(f"  [BM25 Rank {idx+1}] Doc id={id(doc)} | Page={doc.metadata.get('page', 'unknown')} | Source={doc.metadata.get('source', 'unknown')[:40]}...")

    # Assign ranks
    doc_id_to_doc = {}
    rankings = []

    # Vector store rankings
    vs_ranking = {}
    for rank, doc in enumerate(vector_store_docs):
        doc_id = id(doc)
        vs_ranking[doc_id] = rank + 1  # ranks start from 1
        doc_id_to_doc[doc_id] = doc
    rankings.append(vs_ranking)



    # BM25 rankings
    bm25_ranking = {}
    for rank, doc in enumerate(bm25_docs):
        doc_id = id(doc)
        bm25_ranking[doc_id] = rank + 1
        doc_id_to_doc[doc_id] = doc
    rankings.append(bm25_ranking)

    # Apply RRF (50% semantic, 50% keyword)
    print("Applying Reciprocal Rank Fusion (50% semantic, 50% keyword)")
    rrf_scores = defaultdict(float)
    for doc_id in doc_id_to_doc:
        vs_rank = vs_ranking.get(doc_id, top_k + 1)
        bm25_rank = bm25_ranking.get(doc_id, top_k + 1)
        # 50% weight to each
        rrf_scores[doc_id] = 0.5 * (1 / (rrf_k + vs_rank)) + 0.5 * (1 / (rrf_k + bm25_rank))
        print(f"  Doc id={doc_id} | Semantic rank={vs_rank} | BM25 rank={bm25_rank} | RRF score={rrf_scores[doc_id]:.6f}")

    # Sort documents by RRF score
    sorted_doc_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    combined_docs = [doc_id_to_doc[doc_id] for doc_id, _ in sorted_doc_ids[:top_k]]

    print("Top documents after fusion:")
    for i, (doc_id, score) in enumerate(sorted_doc_ids[:top_k]):
        doc = doc_id_to_doc[doc_id]
        print(f"  [Final Rank {i+1}] Doc id={doc_id} | RRF score={score:.6f} | Page={doc.metadata.get('page_number', 'unknown')} | Source={doc.metadata.get('source', 'unknown')[:40]}...")

    # Log retrieval results to a file
    retrieval_results = []
    for i, doc in enumerate(combined_docs):
        retrieval_results.append({
            "rank": i+1,
            "metadata": doc.metadata,
            "content": doc.page_content,
            "content_length": len(doc.page_content)
        })
    
    with open("debug_retrieval.json", "w") as f:
        json.dump(retrieval_results, f, indent=2)
    
    return combined_docs
