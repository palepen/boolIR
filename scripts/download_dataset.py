import ir_datasets
import os
import re

def sanitize_filename(doc_id):
    """
    Sanitize document ID to create a valid filename.
    Replaces any non-alphanumeric characters with underscores.
    """
    return re.sub(r'[^a-zA-Z0-9_-]', '_', doc_id)

def download_and_format_trec_covid():
    """
    Downloads the TREC-COVID dataset and saves each document as an individual file.
    The filename is the document ID, and the content is plain text (title + abstract).
    """
    dataset_id = "cord19/trec-covid"
    sanitized_dataset_name = dataset_id.replace('/', '-')
    
    # Define output directories and file paths
    base_output_dir = "data"
    corpus_output_dir = os.path.join(base_output_dir, f"{sanitized_dataset_name}_corpus")
    topics_path = os.path.join(base_output_dir, f"topics.{sanitized_dataset_name}.txt")
    qrels_path = os.path.join(base_output_dir, f"qrels.{sanitized_dataset_name}.txt")

    # Ensure output directories exist
    print(f"Creating output directory: {corpus_output_dir}")
    os.makedirs(corpus_output_dir, exist_ok=True)
    os.makedirs(base_output_dir, exist_ok=True)

    # Load the dataset
    print(f"Loading ir_datasets '{dataset_id}'...")
    dataset = ir_datasets.load(dataset_id)
    print("Dataset loaded successfully.")

    # 1. Process and save each document as an individual file
    print(f"\nProcessing {dataset.docs_count()} documents...")
    total_docs_processed = 0
    skipped_docs = 0

    for doc in dataset.docs_iter():
        # Sanitize the document ID for use as a filename
        safe_doc_id = sanitize_filename(doc.doc_id)
        filename = f"{safe_doc_id}.txt"
        filepath = os.path.join(corpus_output_dir, filename)
        
        # Combine title and abstract, handling None values
        title = doc.title or ""
        abstract = doc.abstract or ""
        full_text = f"{title} {abstract}".strip()
        
        # Skip documents with no content
        if not full_text:
            skipped_docs += 1
            continue
        
        # Write the document content to its own file
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(full_text)
            total_docs_processed += 1
            
            # Progress indicator every 1000 documents
            if total_docs_processed % 1000 == 0:
                print(f"  Processed {total_docs_processed}/{dataset.docs_count()} documents...")
        except Exception as e:
            print(f"  Error writing document {doc.doc_id}: {e}")
            skipped_docs += 1

    print(f"\nCorpus complete:")
    print(f"  Total documents saved: {total_docs_processed}")
    print(f"  Skipped documents: {skipped_docs}")
    print(f"  Output directory: {corpus_output_dir}")

    # 2. Process and save the topics (queries) - UNCHANGED
    print(f"\nProcessing {dataset.queries_count()} topics...")
    with open(topics_path, "w", encoding="utf-8") as f_topics:
        for query in dataset.queries_iter():
            f_topics.write("<top>\n")
            f_topics.write(f"<num>Number: {query.query_id}</num>\n")
            f_topics.write(f"<title>{query.title}</title>\n")
            f_topics.write("</top>\n")
    print(f"Topics saved to '{topics_path}'")

    # 3. Process and save the qrels (relevance judgments) - UNCHANGED
    print(f"\nProcessing {dataset.qrels_count()} qrels...")
    qrels_processed = 0
    with open(qrels_path, "w", encoding="utf-8") as f_qrels:
        for qrel in dataset.qrels_iter():
            f_qrels.write(f"{qrel.query_id} 0 {qrel.doc_id} {qrel.relevance}\n")
            qrels_processed += 1
    print(f"Processed {qrels_processed} qrels and saved to '{qrels_path}'")

    print("\n" + "="*80)
    print("DATASET DOWNLOAD AND FORMATTING COMPLETE")
    print("="*80)
    print(f"Documents: {corpus_output_dir}")
    print(f"Topics:    {topics_path}")
    print(f"Qrels:     {qrels_path}")
    print("="*80)

if __name__ == "__main__":
    download_and_format_trec_covid()