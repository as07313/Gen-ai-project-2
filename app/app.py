import os
import streamlit as st
from modules.loader import load_documents
from modules.splitter import split_documents
# Make sure retrieve_documents handles semantic_weight=1.0 correctly for Naive RAG
from modules.retriever import create_vector_store, load_vector_store, retrieve_documents
from modules.generator import generate_response
# Import the reranker function
from modules.reranker import rerank_documents
from dotenv import load_dotenv
import glob
from langchain_core.documents import Document # Import Document type
import collections # Import collections for download button logic
import pickle # For persisted chunks

os.environ["STREAMLIT_SERVER_WATCH_MODULES"] = "false"

load_dotenv()

# Ensure the data and embeddings directories exist relative to app.py
APP_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(APP_DIR, "data")
EMBEDDINGS_DIR = os.path.join(APP_DIR, "embeddings")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Initialize session state
if "vector_stores" not in st.session_state:
    st.session_state.vector_stores = {}
if "all_documents" not in st.session_state:
    st.session_state.all_documents = {}
if "corpus_loaded" not in st.session_state:
    st.session_state.corpus_loaded = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "query" not in st.session_state:
    st.session_state.query = ""
if "query_processed" not in st.session_state:
    st.session_state.query_processed = False
# --- New Session State Variables ---
if "selected_rag_mode" not in st.session_state:
    # Define modes clearly
    st.session_state.rag_modes = ["Naive RAG (Semantic Only)", "Hybrid RAG (RRF)", "Naive RAG + Rerank (Cohere)"]
    st.session_state.selected_rag_mode = st.session_state.rag_modes[1] # Default to Hybrid
if "semantic_weight" not in st.session_state:
    st.session_state.semantic_weight = 0.5 # Default RRF weight for Hybrid mode
if "rerank_top_n" not in st.session_state:
    st.session_state.rerank_top_n = 5 # Default for reranking

# App title and description
st.title("Maternal Health Assistant")
st.markdown("Ask questions about maternal health based on our comprehensive resource collection.")

# Function to load all PDFs from the data directory
def load_corpus():
    with st.spinner("Loading maternal health corpus... This may take a few minutes."):
        corpus_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))

        if not corpus_files:
            st.error(f"No PDF files found in the {DATA_DIR} directory. Please add maternal health PDFs.")
            return False

        progress_bar = st.progress(0.0)
        st.session_state.vector_stores = {}
        st.session_state.all_documents = {}

        for i, file_path in enumerate(corpus_files):
            filename = os.path.basename(file_path)
            st.write(f"Processing: {filename}")
            base_name = os.path.splitext(filename)[0]
            index_path = os.path.join(EMBEDDINGS_DIR, base_name)
            chunks_pickle_path = os.path.join(index_path, "chunks.pkl")
            chunks = None

            # --- Load/Create Vector Store & Chunks ---
# ... inside load_corpus loop ...
            if os.path.exists(index_path):
                vector_store = load_vector_store(index_path)
                # Try loading persisted chunks first
                if os.path.exists(chunks_pickle_path):
                    try:
                        print(f"Loading persisted chunks from {chunks_pickle_path}")
                        with open(chunks_pickle_path, 'rb') as f:
                            chunks = pickle.load(f)
                    except Exception as e:
                        st.warning(f"Could not load persisted chunks for {filename}: {e}. Will re-process.")
                        chunks = None # Ensure chunks is None if loading fails
                # If chunks weren't loaded, re-process
                if chunks is None:
                    print(f"Re-loading/splitting {filename} for context...")
                    raw_documents = load_documents(file_path)
                    chunks = split_documents(raw_documents)
                st.session_state.all_documents[filename] = chunks
            else:
                # Create new vector store and chunks
                # Load/split only if not loaded from pickle (shouldn't happen here, but safe check)
                if chunks is None:
                        raw_documents = load_documents(file_path)
                        chunks = split_documents(raw_documents)
                vector_store = create_vector_store(chunks, index_path)
                st.session_state.all_documents[filename] = chunks # Store newly created chunks

                # --- Persist Chunks (Save) ---
                try:
                    # Ensure the directory exists before saving
                    os.makedirs(index_path, exist_ok=True)
                    with open(chunks_pickle_path, 'wb') as f:
                        pickle.dump(chunks, f)
                    print(f"Saved chunks to {chunks_pickle_path}")
                except Exception as e:
                    st.warning(f"Could not save chunks for {filename}: {e}")
                # --- End Persist Chunks ---

            st.session_state.vector_stores[filename] = vector_store
            progress_bar.progress((i + 1) / len(corpus_files))
# ... rest of load_corpus ...
        # --- End Loop ---

        st.success("Corpus loading complete!")
        return True

# Sidebar for settings and controls
with st.sidebar:
    st.header("Setup & Settings")
    # Load corpus button
    if not st.session_state.corpus_loaded:
        if st.button("Load Maternal Health Corpus"):
            success = load_corpus()
            if success:
                st.session_state.corpus_loaded = True
                st.rerun() # Rerun to update sidebar stats and enable query
    elif st.session_state.corpus_loaded:
        st.success("Maternal health corpus loaded!")
        st.subheader("Corpus Statistics")
        st.write(f"Documents: {len(st.session_state.vector_stores)}")
        total_chunks = sum(len(docs) for docs in st.session_state.all_documents.values())
        st.write(f"Knowledge chunks: {total_chunks}")

    # --- RAG Mode Selection ---
    st.subheader("Retrieval Mode")
    # Use the list from session state
    st.session_state.selected_rag_mode = st.radio(
        "Select RAG Strategy:",
        options=st.session_state.rag_modes,
        index=st.session_state.rag_modes.index(st.session_state.selected_rag_mode), # Maintain selection
        key="rag_mode_selector"
    )

    # --- Search Settings ---
    st.subheader("Search Settings")
    top_k = st.slider(
        "Initial retrieval depth (k)",
        min_value=5, max_value=25, value=10, key="top_k_slider",
        help="Number of documents initially retrieved by semantic/hybrid search."
    )

    # Conditional settings based on mode
    if st.session_state.selected_rag_mode == "Hybrid RAG (RRF)":
        st.session_state.semantic_weight = st.slider(
            "Semantic Search Weight (RRF)",
            min_value=0.0, max_value=1.0, value=st.session_state.semantic_weight, step=0.1,
            key="semantic_weight_slider",
            help="Controls the balance between semantic (vector) and keyword (BM25) search in Hybrid mode. 1.0 = Pure Semantic, 0.0 = Pure Keyword."
        )
        st.caption(f"Keyword Weight: {1.0 - st.session_state.semantic_weight:.1f}")
    elif st.session_state.selected_rag_mode == "Naive RAG + Rerank (Cohere)":
        st.session_state.rerank_top_n = st.slider(
            "Final relevance depth (n)",
            min_value=3, max_value=10, value=st.session_state.rerank_top_n,
            key="rerank_top_n_slider",
            help="Number of documents to keep after Cohere reranking."
        )

    # Clear chat history
    if st.session_state.chat_history and st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.query = ""
        st.session_state.query_processed = False
        st.success("Chat history cleared")
        st.rerun()

# User query area
st.subheader("Ask about maternal health:")
query = st.text_area("Enter your question:",
                     placeholder="Example: What are the signs of preeclampsia?",
                     key="query_input",
                     value=st.session_state.get("query", ""))
submit_button = st.button("Ask")

# Process the query
if submit_button and query and not st.session_state.query_processed:
    st.session_state.query_processed = True
    st.session_state.query = query
    if not st.session_state.corpus_loaded:
        st.warning("Please load the maternal health corpus first...")
        st.session_state.query_processed = False
    else:
        retrieved_docs = []
        ranking_method = st.session_state.selected_rag_mode # Use selected mode as method name

        with st.spinner(f"Running {st.session_state.selected_rag_mode}..."):
            # --- Retrieval Logic based on Mode ---
            initial_retrieved_docs = [] # Docs before potential reranking

            if st.session_state.selected_rag_mode == "Naive RAG (Semantic Only)" or st.session_state.selected_rag_mode == "Naive RAG + Rerank (Cohere)":
                # Both start with semantic search
                temp_retrieved = []
                for filename, vector_store in st.session_state.vector_stores.items():
                    vs_docs = vector_store.similarity_search(query, k=top_k)
                    temp_retrieved.extend(vs_docs)
                # Simple deduplication based on content
                seen_content = set()
                for doc in temp_retrieved:
                    if doc.page_content not in seen_content:
                        initial_retrieved_docs.append(doc)
                        seen_content.add(doc.page_content)
                # Limit initial semantic results if needed, though reranker might benefit from more
                initial_retrieved_docs = initial_retrieved_docs[:top_k * len(st.session_state.vector_stores)] # Allow more initially for reranker

            elif st.session_state.selected_rag_mode == "Hybrid RAG (RRF)":
                temp_retrieved = []
                for filename, vector_store in st.session_state.vector_stores.items():
                    doc_chunks = st.session_state.all_documents.get(filename, [])
                    if not doc_chunks: continue
                    # Call retrieve_documents with the selected semantic weight
                    # Ensure retrieve_documents returns top_k results after fusion
                    rrf_docs = retrieve_documents(
                        query,
                        vector_store,
                        documents=doc_chunks,
                        top_k=top_k, # Retrieve top_k after fusion
                        semantic_weight=st.session_state.semantic_weight
                    )
                    temp_retrieved.extend(rrf_docs)
                # Deduplicate results from RRF across files (using persistent ID if available)
                seen_ids = set()
                for doc in temp_retrieved:
                    doc_id = doc.metadata.get("persistent_chunk_id", doc.page_content)
                    if doc_id not in seen_ids:
                        initial_retrieved_docs.append(doc)
                        seen_ids.add(doc_id)
                # RRF retriever already limits to top_k, deduplication is secondary here
                initial_retrieved_docs = initial_retrieved_docs[:top_k] # Ensure final limit


            # --- Optional Reranking Step ---
            if st.session_state.selected_rag_mode == "Naive RAG + Rerank (Cohere)":
                if not initial_retrieved_docs:
                     st.warning("No documents found by initial semantic search to rerank.")
                     retrieved_docs = []
                else:
                     with st.spinner("Reranking retrieved documents with Cohere..."):
                          # Use the rerank_top_n value from the slider
                          retrieved_docs = rerank_documents(query, initial_retrieved_docs, top_n=st.session_state.rerank_top_n)
            else:
                 # For Naive RAG and Hybrid RAG, the initially retrieved docs are the final ones
                 retrieved_docs = initial_retrieved_docs[:top_k] # Apply final limit if not reranking


            # --- Generation ---
            if not retrieved_docs:
                 st.warning("Could not retrieve relevant documents for your query.")
                 response_text = "Sorry, I couldn't find relevant information for your query."
                 source_docs_used = []
            else:
                 # Ensure generate_response returns tuple: (text, docs_used)
                 response_text, source_docs_used = generate_response(query, retrieved_docs)

            # Add to chat history
            st.session_state.chat_history.append((st.session_state.query, response_text, ranking_method, source_docs_used))

            # Clear query state and rerun
            st.session_state.query = ""
            st.rerun()

# Reset processing flag if query is cleared
elif not query:
     st.session_state.query_processed = False

# Display chat history
st.subheader("Conversation History")
# ... (Rest of the chat history display logic remains the same as previous correct version) ...
if not st.session_state.chat_history:
    st.info("Ask a question about maternal health to get started.")
else:
    history_item_length = len(st.session_state.chat_history[0]) if st.session_state.chat_history else 0
    for i, history_item in enumerate(reversed(st.session_state.chat_history)):
        if history_item_length == 4:
            question, answer, method, source_docs = history_item
        else:
            question, answer, method = history_item
            source_docs = []

        with st.expander(f"**Q: {question}**", expanded=(i == 0)):
            st.markdown(f"**Answer:** *(Retrieved using {method})*")
            st.markdown(answer)

            if source_docs:
                st.markdown("---")
                st.markdown("**Sources Used:**")
                sources_by_file = collections.defaultdict(list)
                unique_files_data = {}

                for idx, doc in enumerate(source_docs):
                    source_path_rel = doc.metadata.get('source', 'unknown')
                    source_path_abs = os.path.join(DATA_DIR, os.path.basename(source_path_rel))
                    page_num = doc.metadata.get('page_number', doc.metadata.get('page', 'unknown'))
                    doc_num = idx + 1
                    display_name = os.path.basename(source_path_rel)
                    sources_by_file[display_name].append(f"Doc {doc_num}, Page {page_num}")

                    if display_name not in unique_files_data:
                         try:
                             if os.path.exists(source_path_abs):
                                 with open(source_path_abs, "rb") as f:
                                     unique_files_data[display_name] = {"path": source_path_abs, "content": f.read()}
                             else:
                                  unique_files_data[display_name] = {"path": source_path_abs, "content": None}
                         except Exception as e:
                             unique_files_data[display_name] = {"path": source_path_abs, "content": None}

                for display_name, details in sources_by_file.items():
                    file_info = unique_files_data.get(display_name)
                    if file_info and file_info["content"]:
                        st.download_button(label=f"Download {display_name}", data=file_info["content"], file_name=display_name, mime="application/pdf")
                        st.caption(f"Context from: {'; '.join(details)}")
                    else:
                        st.caption(f"{display_name} (File not accessible)")
                        st.caption(f"Context from: {'; '.join(details)}")

        if i < len(st.session_state.chat_history) - 1:
            st.divider()