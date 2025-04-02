import os
import streamlit as st
from modules.loader import load_documents
from modules.splitter import split_documents
from modules.retriever import create_vector_store, load_vector_store, retrieve_documents
from modules.generator import generate_response
from dotenv import load_dotenv

os.environ["STREAMLIT_SERVER_WATCH_MODULES"] = "false"

load_dotenv()

# Ensure the data and embeddings directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("embeddings", exist_ok=True)

# Initialize session state
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "vector_stores" not in st.session_state:
    st.session_state.vector_stores = {}
if "current_file" not in st.session_state:
    st.session_state.current_file = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "query" not in st.session_state:
    st.session_state.query = ""

# Function to handle query submission
def handle_query_submit():
    if st.session_state.query_input.strip():  # Only if non-empty query
        st.session_state.query = st.session_state.query_input
        st.session_state.query_input = ""  # This is allowed during a callback

# App title and description
st.title("RAG-based PDF Q&A System")
st.markdown("Upload PDF documents and ask questions about them.")

# Sidebar for document management
with st.sidebar:
    st.header("Document Management")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    
    if uploaded_file and uploaded_file.name not in [f.name for f in st.session_state.uploaded_files]:
        with st.spinner("Processing document..."):
            file_path = os.path.join("data", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Load and process the document
            documents = load_documents(file_path)
            chunks = split_documents(documents)

            # Create or load the vector store
            index_path = f"embeddings/{os.path.splitext(uploaded_file.name)[0]}"
            if not os.path.exists(index_path):
                vector_store = create_vector_store(chunks, index_path)
            else:
                vector_store = load_vector_store(index_path)
            
            # Store in session state
            st.session_state.uploaded_files.append(uploaded_file)
            st.session_state.vector_stores[uploaded_file.name] = vector_store
            st.session_state.current_file = uploaded_file.name
            
            st.success(f"Document '{uploaded_file.name}' processed successfully!")
    
    # Document selector (only show if we have documents)
    if st.session_state.uploaded_files:
        st.subheader("Select Document")
        file_names = [f.name for f in st.session_state.uploaded_files]
        selected_file = st.selectbox("Choose a document to query", file_names, 
                                    index=file_names.index(st.session_state.current_file) if st.session_state.current_file else 0)
        
        if selected_file != st.session_state.current_file:
            st.session_state.current_file = selected_file
            st.info(f"Now querying document: {selected_file}")
    
    # Clear chat history button
    if st.session_state.chat_history:
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []

# Main chat area
col1, col2 = st.columns([3, 1])

with col1:
    # Display chat messages
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        st.markdown(f"**Question:**")
        st.markdown(f"{question}")
        st.markdown(f"**Answer:**")
        st.markdown(f"{answer}")
        
        # Add a separator between messages
        if i < len(st.session_state.chat_history) - 1:
            st.markdown("---")

with col2:
    # Show current document info if any
    if st.session_state.current_file:
        st.info(f"Currently querying: {st.session_state.current_file}")
        
        # Show document stats if available
        vector_store = st.session_state.vector_stores[st.session_state.current_file]
        st.metric("Document Chunks", len(vector_store.docstore._dict))

# User query input - always available
query_input = st.text_input("Enter your question:", 
                     placeholder="Ask a question about your documents...",
                     key="query_input")

button_col1, button_col2 = st.columns([1, 9])
with button_col1:
    submit_button = st.button("Submit")

# Process the query when the button is clicked or Enter is pressed
if submit_button or (query_input and query_input != st.session_state.query):
    if query_input:  # Ensure there's an actual query
        st.session_state.query = query_input
        query = query_input
    
        if not st.session_state.uploaded_files:
            st.warning("Please upload a document first!")
        else:
            with st.spinner("Generating response..."):
                # Get the current vector store
                vector_store = st.session_state.vector_stores[st.session_state.current_file]
                
                # Retrieve relevant documents and generate response
                relevant_docs = retrieve_documents(query, vector_store)
                response = generate_response(query, relevant_docs)
                
                # Add to chat history
                st.session_state.chat_history.append((query, response))
            
            # Force a rerun to clear the input
            st.rerun()