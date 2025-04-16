from langchain_community.document_loaders import UnstructuredPDFLoader, PyPDFLoader # Keep PyPDFLoader import if needed elsewhere, or remove if not

def load_documents(file_path):
    """
    Load documents from a PDF file using UnstructuredPDFLoader.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        list: List of LangChain Document objects.
    """
    # Use UnstructuredPDFLoader instead of PyPDFLoader
    # Default mode="single" loads the document content as one item
    loader = UnstructuredPDFLoader(file_path, mode="elements")
    print(f"Loading document: {file_path} using UnstructuredPDFLoader")
    try:
        documents = loader.load()
        print(f"Successfully loaded {len(documents)} document(s) from {file_path}")
        # UnstructuredPDFLoader in "single" mode returns one document for the whole PDF
        # If it returns more than expected, it might indicate an issue or different behavior
        if documents:
             print(f"Sample loaded content (first 200 chars): {documents[0].page_content[:200]}...")
        return documents
    except Exception as e:
        print(f"Error loading {file_path} with UnstructuredPDFLoader: {e}")
        # Optionally, fallback or raise the error
        # Fallback example (uncomment if needed):
        print("Falling back to PyPDFLoader...")
        loader = PyPDFLoader(file_path)
        return loader.load()
    
