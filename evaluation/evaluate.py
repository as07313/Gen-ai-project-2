import json
import os
import glob
import nltk
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import collections # Import collections
import pickle # For loading persisted chunks (if implemented)

# --- Adapt these imports based on your potential refactoring ---
# Assume functions exist to load resources and run pipeline steps
from app.modules.loader import load_documents
from app.modules.splitter import split_documents
from app.modules.retriever import load_vector_store, retrieve_documents
from app.modules.generator import generate_response # Assuming it returns only the text response string
# --- End adaptation section ---

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except (LookupError, nltk.downloader.DownloadError):
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)

def initialize_pipeline_resources(embeddings_dir="app/embeddings", data_dir="app/data"):
    """
    Loads vector stores and document chunks required for the RAG pipeline.
    Prioritizes loading persisted chunks if available.
    """
    print("Initializing pipeline resources...")
    vector_stores = {}
    all_documents_chunks = {} # Maps filename to list of Document chunks

    corpus_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    if not corpus_files:
        raise FileNotFoundError(f"No PDF files found in {data_dir}")

    for file_path in corpus_files:
        filename = os.path.basename(file_path)
        base_name = os.path.splitext(filename)[0]
        index_path = os.path.join(embeddings_dir, base_name)
        chunks_pickle_path = os.path.join(index_path, "chunks.pkl") # Path for persisted chunks

        if os.path.exists(index_path):
            print(f"Loading vector store from {index_path}...")
            vector_store = load_vector_store(index_path)
            vector_stores[filename] = vector_store

            # --- Load corresponding chunks ---
            if os.path.exists(chunks_pickle_path):
                print(f"Loading persisted chunks from {chunks_pickle_path}...")
                with open(chunks_pickle_path, 'rb') as f:
                    chunks = pickle.load(f)
                all_documents_chunks[filename] = chunks
            else:
                # Fallback: Re-load and split (less efficient)
                print(f"Persisted chunks not found. Loading/splitting {filename} for BM25...")
                raw_docs = load_documents(file_path) # Uses your loader.py logic
                chunks = split_documents(raw_docs) # Uses your splitter.py logic
                all_documents_chunks[filename] = chunks
                # Optionally save chunks here if they weren't persisted before
                # os.makedirs(index_path, exist_ok=True)
                # with open(chunks_pickle_path, 'wb') as f:
                #     pickle.dump(chunks, f)
            # --- End chunk loading ---
        else:
            print(f"Warning: Index not found at {index_path}. Cannot evaluate this file.")

    if not vector_stores:
         raise RuntimeError("No vector stores could be loaded. Cannot proceed with evaluation.")

    print(f"Loaded {len(vector_stores)} vector stores and corresponding chunks.")
    return vector_stores, all_documents_chunks

def run_rag_for_query(query, vector_stores, all_documents_chunks, top_k=10):
    """
    Executes the retrieval and generation steps for a given query.
    """
    all_retrieved_docs = []
    # Mimic retrieval loop from app.py
    for filename, vector_store in vector_stores.items():
        doc_chunks = all_documents_chunks.get(filename, [])
        if not doc_chunks: continue
        # Use retrieve_documents from your retriever module
        retrieved = retrieve_documents(query, vector_store, documents=doc_chunks, top_k=top_k)
        all_retrieved_docs.extend(retrieved)

    # Simple de-duplication and limiting (as in app.py)
    # Consider a more sophisticated global ranking/deduplication if needed
    unique_docs = []
    seen_content = set()
    for doc in all_retrieved_docs:
        # Use persistent_chunk_id if available for more reliable deduplication
        doc_id = doc.metadata.get("persistent_chunk_id", doc.page_content)
        if doc_id not in seen_content:
            unique_docs.append(doc)
            seen_content.add(doc_id)

    final_docs = unique_docs[:top_k] # Limit context for generator

    # Use generate_response from your generator module
    # Ensure it returns only the response string for evaluation
    generated_text = generate_response(query, final_docs)
    if not isinstance(generated_text, str):
         # Handle cases where generate_response might return a tuple (text, docs)
         # or an error message. Adjust based on your generator's actual return type.
         if isinstance(generated_text, tuple) and len(generated_text) > 0:
             generated_text = generated_text[0]
         elif "Error" in str(generated_text): # Basic error check
             raise RuntimeError(f"Generation failed: {generated_text}")
         else:
             raise TypeError(f"Unexpected return type from generate_response: {type(generated_text)}")

    return generated_text


def calculate_metrics(generated, references):
    """Calculates ROUGE (1, 2, L F1) and BLEU scores."""
    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    # Calculate against all references and take the best score for each metric
    best_scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    for ref in references:
        scores = scorer.score(ref, generated)
        best_scores['rouge1'] = max(best_scores['rouge1'], scores['rouge1'].fmeasure)
        best_scores['rouge2'] = max(best_scores['rouge2'], scores['rouge2'].fmeasure)
        best_scores['rougeL'] = max(best_scores['rougeL'], scores['rougeL'].fmeasure)

    # BLEU
    tokenized_refs = [word_tokenize(ref.lower()) for ref in references]
    tokenized_gen = word_tokenize(generated.lower())
    smoothie = SmoothingFunction().method1 # Using method1 smoothing
    # Calculate BLEU score (typically up to 4-grams)
    bleu_score = sentence_bleu(tokenized_refs, tokenized_gen, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

    return best_scores, bleu_score


# --- Main Evaluation Script ---
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct paths relative to the script directory
    EVAL_DATASET_PATH = os.path.join(script_dir, "evaluation_dataset.json")
    RESULTS_OUTPUT_PATH = os.path.join(script_dir, "evaluation_results.json")


    # 1. Load Evaluation Dataset
    if not os.path.exists(EVAL_DATASET_PATH):
        print(f"Error: Evaluation dataset not found at {EVAL_DATASET_PATH}")
        exit(1)
    with open(EVAL_DATASET_PATH, 'r', encoding='utf-8') as f:
        evaluation_data = json.load(f)
    print(f"Loaded {len(evaluation_data)} evaluation items.")

    # 2. Initialize Pipeline Resources
    try:
        vector_stores, all_documents_chunks = initialize_pipeline_resources()
    except Exception as e:
        print(f"Fatal Error initializing pipeline: {e}")
        exit(1)

    # 3. Run Evaluation Loop
    evaluation_results = []
    total_scores = collections.defaultdict(float)
    successful_evals = 0

    for i, item in enumerate(evaluation_data):
        query = item["question"]
        references = item["answer"]
        print(f"\n[{i+1}/{len(evaluation_data)}] Evaluating query: {query}")

        if not references:
            print("  Skipping item due to missing reference answers.")
            evaluation_results.append({"query": query, "error": "Missing reference answers"})
            continue

        try:
            # Run the RAG pipeline
            generated_answer = run_rag_for_query(query, vector_stores, all_documents_chunks)
            print(f"  Generated Answer (preview): {generated_answer[:150]}...")

            # Calculate metrics
            rouge_f1_scores, bleu_score = calculate_metrics(generated_answer, references)
            print(f"  Metrics - ROUGE-1: {rouge_f1_scores['rouge1']:.4f}, ROUGE-2: {rouge_f1_scores['rouge2']:.4f}, ROUGE-L: {rouge_f1_scores['rougeL']:.4f}, BLEU: {bleu_score:.4f}")

            # Store results
            result_item = {
                "query": query,
                "generated_answer": generated_answer,
                "reference_answers": references,
                "rouge1_f1": rouge_f1_scores['rouge1'],
                "rouge2_f1": rouge_f1_scores['rouge2'],
                "rougeL_f1": rouge_f1_scores['rougeL'],
                "bleu": bleu_score
            }
            evaluation_results.append(result_item)

            # Accumulate scores for averaging
            total_scores['rouge1'] += rouge_f1_scores['rouge1']
            total_scores['rouge2'] += rouge_f1_scores['rouge2']
            total_scores['rougeL'] += rouge_f1_scores['rougeL']
            total_scores['bleu'] += bleu_score
            successful_evals += 1

        except Exception as e:
            print(f"  ERROR during evaluation for this query: {e}")
            evaluation_results.append({"query": query, "error": str(e)})

    # 4. Calculate and Print Average Scores
    if successful_evals > 0:
        print("\n--- Average Scores ---")
        print(f"Average ROUGE-1 F1: {total_scores['rouge1'] / successful_evals:.4f}")
        print(f"Average ROUGE-2 F1: {total_scores['rouge2'] / successful_evals:.4f}")
        print(f"Average ROUGE-L F1: {total_scores['rougeL'] / successful_evals:.4f}")
        print(f"Average BLEU:       {total_scores['bleu'] / successful_evals:.4f}")
    else:
        print("\nNo queries were successfully evaluated.")

    # 5. Save Detailed Results
    with open(RESULTS_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed evaluation results saved to {os.path.abspath(RESULTS_OUTPUT_PATH)}")
