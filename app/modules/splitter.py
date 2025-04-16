from langchain_text_splitters import RecursiveCharacterTextSplitter
import json

def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    """
    Split documents into smaller chunks.

    Args:
        documents (list): List of LangChain Document objects.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        list: List of split Document objects.
    """
    print(f"Original document count: {len(documents)}")
    print(f"Sample document metadata: {documents[0].metadata}")
    print(f"Sample document content first 100 chars: {documents[0].page_content[:100]}...")
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    
    print(f"Total chunks after splitting: {len(chunks)}")
    print(f"Sample chunk metadata: {chunks[0].metadata}")
    print(f"Sample chunk content: {chunks[0].page_content}")
    
    # Save some sample chunks to a file for inspection
    sample_chunks = []
    for i, chunk in enumerate(chunks[:5]):  # Take first 5 chunks
        sample_chunks.append({
            "chunk_index": i,
            "metadata": chunk.metadata,
            "content": chunk.page_content,
            "content_length": len(chunk.page_content)
        })
    
    with open("debug_chunks.json", "w") as f:
        json.dump(sample_chunks, f, indent=2)
    
    return chunks