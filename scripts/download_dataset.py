import ir_datasets
import os
import re
from tqdm import tqdm

def sanitize_filename(doc_id):
    """
    Sanitize document ID to create a valid filename.
    Replaces any non-alphanumeric characters with underscores.
    """
    return re.sub(r'[^a-zA-Z0-9_-]', '_', doc_id)

def download_and_format_trec_covid():
    """
    Downloads the TREC-COVID dataset and saves each document as an individual file.
    Creates a qrels file that ONLY references documents that were successfully downloaded.
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

    # Step 1: Download and save documents, tracking which ones we actually saved
    print(f"\nProcessing {dataset.docs_count()} documents...")
    total_docs_processed = 0
    skipped_docs = 0
    saved_doc_ids = set()  # Track successfully saved document IDs

    for doc in tqdm(dataset.docs_iter(), total=dataset.docs_count(), desc="Downloading documents"):
        # Use original doc_id as filename (CORD-19 IDs are already clean)
        doc_id = doc.doc_id
        filename = f"{doc_id}.txt"
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
            saved_doc_ids.add(doc_id)  # Track that we saved this document
            
        except Exception as e:
            print(f"\n  Error writing document {doc.doc_id}: {e}")
            skipped_docs += 1

    print(f"\nCorpus download complete:")
    print(f"  Total documents saved: {total_docs_processed}")
    print(f"  Skipped documents (empty): {skipped_docs}")
    print(f"  Output directory: {corpus_output_dir}")

    # Step 2: Process and save the topics (queries)
    print(f"\nProcessing {dataset.queries_count()} topics...")
    with open(topics_path, "w", encoding="utf-8") as f_topics:
        for query in dataset.queries_iter():
            f_topics.write("<top>\n")
            f_topics.write(f"<num>Number: {query.query_id}</num>\n")
            f_topics.write(f"<title>{query.title}</title>\n")
            f_topics.write("</top>\n")
    print(f"Topics saved to '{topics_path}'")

    # Step 3: Process qrels and FILTER to only include documents we actually saved
    print(f"\nProcessing and filtering {dataset.qrels_count()} qrels...")
    qrels_total = 0
    qrels_saved = 0
    qrels_skipped = 0
    queries_with_relevant = set()
    
    with open(qrels_path, "w", encoding="utf-8") as f_qrels:
        for qrel in dataset.qrels_iter():
            qrels_total += 1
            
            # Only include qrels for documents that exist in our corpus
            if qrel.doc_id in saved_doc_ids:
                f_qrels.write(f"{qrel.query_id} 0 {qrel.doc_id} {qrel.relevance}\n")
                qrels_saved += 1
                if qrel.relevance > 0:
                    queries_with_relevant.add(qrel.query_id)
            else:
                qrels_skipped += 1
    
    print(f"Qrels processing complete:")
    print(f"  Total qrels from dataset: {qrels_total}")
    print(f"  Qrels saved (docs exist): {qrels_saved}")
    print(f"  Qrels skipped (docs missing): {qrels_skipped}")
    print(f"  Queries with relevant docs: {len(queries_with_relevant)}")
    print(f"  Saved to: {qrels_path}")

    # Step 4: Verification
    print(f"\n" + "="*80)
    print("VERIFICATION")
    print("="*80)
    
    # Count actual files in corpus
    actual_files = len([f for f in os.listdir(corpus_output_dir) if f.endswith('.txt')])
    print(f"Documents in corpus directory: {actual_files}")
    print(f"Documents tracked in saved_doc_ids: {len(saved_doc_ids)}")
    
    # Verify qrels references valid docs
    print(f"\nVerifying qrels file...")
    with open(qrels_path, 'r') as f:
        qrels_relevant_count = 0
        qrels_doc_ids = set()
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                query_id, _, doc_id, relevance = parts
                qrels_doc_ids.add(doc_id)
                if int(relevance) > 0:
                    qrels_relevant_count += 1
    
    print(f"Unique doc IDs in qrels: {len(qrels_doc_ids)}")
    print(f"Relevant judgments in qrels: {qrels_relevant_count}")
    
    # Check for any mismatches
    mismatches = qrels_doc_ids - saved_doc_ids
    if mismatches:
        print(f"\nWARNING: {len(mismatches)} doc IDs in qrels not in corpus!")
        print(f"  This should not happen. Sample: {list(mismatches)[:5]}")
    else:
        print(f"\nSUCCESS: All qrels reference documents that exist in corpus!")

    print("\n" + "="*80)
    print("DATASET DOWNLOAD AND FORMATTING COMPLETE")
    print("="*80)
    print(f"Documents: {corpus_output_dir}")
    print(f"Topics:    {topics_path}")
    print(f"Qrels:     {qrels_path}")
    print("\nNext steps:")
    print("  1. Build index: make build-index")
    print("  2. Run benchmark: make benchmark")
    print("="*80)

if __name__ == "__main__":
    download_and_format_trec_covid()