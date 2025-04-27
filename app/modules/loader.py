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
        return documents
    except Exception as e:
        print(f"Error loading {file_path} with UnstructuredPDFLoader: {e}")
        print("Falling back to PyPDFLoader...")
        loader = PyPDFLoader(file_path)
        return loader.load()
    
