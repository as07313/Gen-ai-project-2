from langchain_community.document_loaders import UnstructuredPDFLoader, PyPDFLoader

def load_documents(file_path):
    """
    Load documents from a PDF file.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        list: List of LangChain Document objects.
    """
    loader = PyPDFLoader(file_path)
    return loader.load()
