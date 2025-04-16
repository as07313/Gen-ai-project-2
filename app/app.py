import os
import streamlit as st
from modules.loader import load_documents
from modules.splitter import split_documents
from modules.retriever import create_vector_store, load_vector_store, retrieve_documents
from modules.generator import generate_response
# Remove reranker import as it's not used directly in the app anymore
# from modules.reranker import rerank_documents
from dotenv import load_dotenv
import glob
from langchain_core.documents import Document # Import Document type

os.environ["STREAMLIT_SERVER_WATCH_MODULES"] = "false"

load_dotenv()

# Ensure the data and embeddings directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("embeddings", exist_ok=True)

# Initialize session state
if "vector_stores" not in st.session_state:
    st.session_state.vector_stores = {} # Stores {filename: vector_store}
if "all_documents" not in st.session_state:
    st.session_state.all_documents = {} # Stores {filename: List[Document]}
if "corpus_loaded" not in st.session_state:
    st.session_state.corpus_loaded = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "query" not in st.session_state:
    st.session_state.query = ""
# Remove unused session state
# if "use_reranking" not in st.session_state:
#     st.session_state.use_reranking = True
if "query_processed" not in st.session_state:
    st.session_state.query_processed = False

# App title and description
st.title("Maternal Health Assistant")
st.markdown("Ask questions about maternal health based on our comprehensive resource collection.")

# Function to load all PDFs from the data directory
def load_corpus():
    with st.spinner("Loading maternal health corpus... This may take a few minutes."):
        corpus_files = glob.glob("data/*.pdf")

        if not corpus_files:
            st.error("No PDF files found in the data directory. Please add maternal health PDFs to the 'data' folder.")
            return False

        # Display the corpus files being processed
        st.write(f"Processing {len(corpus_files)} documents:")
        progress_bar = st.progress(0)
        st.session_state.vector_stores = {} # Reset stores on reload
        st.session_state.all_documents = {} # Reset documents on reload

        for i, file_path in enumerate(corpus_files):
            filename = os.path.basename(file_path)
            st.write(f"Processing: {filename}")

            # Extract filename without extension for the index path
            base_name = os.path.splitext(filename)[0]
            index_path = f"embeddings/{base_name}"

            # Load or create vector store and store documents
            if os.path.exists(index_path):
                # Load existing vector store
                vector_store = load_vector_store(index_path)
                # Need to load original documents/chunks for BM25
                # Assuming chunks are implicitly loaded or accessible via vector_store.docstore
                # If not, we need a way to load/store the chunks separately
                # For simplicity, let's assume docstore holds them. If not, adjust loading.
                # A robust way would be to save/load chunks alongside the index.
                # Let's load the source PDF again to get the chunks for BM25
                raw_documents = load_documents(file_path)
                chunks = split_documents(raw_documents)
                st.session_state.all_documents[filename] = chunks

            else:
                # Create new vector store
                raw_documents = load_documents(file_path)
                chunks = split_documents(raw_documents)
                vector_store = create_vector_store(chunks, index_path)
                st.session_state.all_documents[filename] = chunks # Store chunks

            # Store vector store in session state
            st.session_state.vector_stores[filename] = vector_store

            # Update progress
            progress_bar.progress((i + 1) / len(corpus_files))

        st.success(f"Successfully processed {len(corpus_files)} documents in the maternal health corpus!")
        return True

# Sidebar for settings and controls
with st.sidebar:
    # Load corpus button
    if not st.session_state.corpus_loaded:
        if st.button("Load Maternal Health Corpus"):
            success = load_corpus()
            if success:
                st.session_state.corpus_loaded = True
                st.rerun() # Rerun to update sidebar stats
    else:
        st.success("Maternal health corpus loaded!")

        # Show corpus stats
        st.subheader("Corpus Statistics")
        st.write(f"Documents in corpus: {len(st.session_state.vector_stores)}")
        # Calculate total chunks from the stored documents list
        total_chunks = sum(len(docs) for docs in st.session_state.all_documents.values())
        st.write(f"Total knowledge chunks: {total_chunks}")

    # Simplified Search Settings
    st.subheader("Search Settings")
    # Remove reranking toggle and rerank_top_n slider
    # st.session_state.use_reranking = st.toggle("Use Enhanced Relevance Ranking", value=st.session_state.use_reranking)
    top_k = st.slider("Search depth (documents to retrieve)", min_value=5, max_value=20, value=10)
    # rerank_top_n = st.slider("Final relevance depth", min_value=3, max_value=10, value=5) # Removed

    # Clear chat history
    if st.session_state.chat_history and st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.query = "" # Clear query input as well
        st.session_state.query_processed = False
        st.success("Chat history cleared")
        st.rerun()

# User query area
st.subheader("Ask about maternal health:")

# Use text_area for potentially longer questions and better display
query = st.text_area("Enter your question:",
                     placeholder="Example: What are the signs of preeclampsia?",
                     key="query_input",
                     value=st.session_state.get("query", ""))

# Process the query when the user submits (e.g., presses Enter with text_input, or we add a button)
# Using a button for explicit submission with text_area
submit_button = st.button("Ask")

# Process the query
if submit_button and query and not st.session_state.query_processed:
    st.session_state.query_processed = True  # Set the flag to True
    st.session_state.query = query # Store the submitted query
    if not st.session_state.corpus_loaded:
        st.warning("Please load the maternal health corpus first by clicking the button in the sidebar.")
        st.session_state.query_processed = False # Reset flag if corpus not loaded
    else:
        with st.spinner("Searching for the most relevant information using Reciprocal Rank Fusion..."):
            # Collect relevant documents using RRF from all vector stores
            final_docs = []
            # Consolidate all chunks from all loaded documents for BM25 context if needed by retriever
            # Or pass documents per file if retriever handles it internally
            # Assuming retrieve_documents needs the specific vector_store and its corresponding documents
            all_retrieved_docs = []
            for filename, vector_store in st.session_state.vector_stores.items():
                # Get the documents/chunks associated with this specific vector store
                doc_chunks = st.session_state.all_documents.get(filename, [])
                if not doc_chunks:
                    st.error(f"Could not find document chunks for {filename}. Skipping.")
                    continue

                # Retrieve using RRF (semantic + BM25)
                # Pass the specific chunks for this file to retriever for BM25 calculation
                retrieved = retrieve_documents(query, vector_store, documents=doc_chunks, top_k=top_k)
                all_retrieved_docs.extend(retrieved) # Collect all retrieved docs

            # Optional: Apply a final global ranking or limit if too many docs collected
            # For simplicity, let's just use the collected docs up to a reasonable limit
            final_docs = all_retrieved_docs[:top_k] # Limit the final list

            ranking_method = "Reciprocal Rank Fusion (Semantic + Keyword)"

            # Generate response based on the RRF results
            response = generate_response(query, final_docs)

            # Add to chat history
            st.session_state.chat_history.append((st.session_state.query, response, ranking_method))

            # Clear the query state variables for the next input
            st.session_state.query = ""
            # query_input key will be cleared on rerun if value is ""

            # Rerun the app to refresh the interface and clear input
            st.rerun()

# Reset the processing flag if the query is cleared or processed
elif not query:
     st.session_state.query_processed = False


st.subheader("Conversation History")

if not st.session_state.chat_history:
    st.info("Ask a question about maternal health to get started.")
else:
    # Display chat history in reverse order (newest first)
    for i, (question, answer, method) in enumerate(reversed(st.session_state.chat_history)):
        # Use expander for better organization of Q&A pairs
        with st.expander(f"**Q: {question}**", expanded=(i == 0)): # Expand the latest question
            # Display Answer with method
            st.markdown(f"**Answer:** *(Retrieved using {method})*")
            st.markdown(answer)

        # Add a separator between messages, except for the last one
        if i < len(st.session_state.chat_history) - 1:
            st.divider()
