import ir_datasets
import os
import itertools

def download_and_format_trec_covid():
    """
    Downloads the TREC-COVID dataset, a permission-free dataset with under 500k documents.
    Batches corpus documents into larger files for compatibility with C++ applications.
    """
    # Use the full dataset ID required by ir_datasets
    dataset_id = "cord19/trec-covid"
    
    # Sanitize dataset ID to create valid file/directory names
    sanitized_dataset_name = dataset_id.replace('/', '-')
    
    # Define output directories and file paths
    base_output_dir = "data"
    corpus_output_dir = os.path.join(base_output_dir, f"{sanitized_dataset_name}_corpus_batched")
    topics_path = os.path.join(base_output_dir, f"topics.{sanitized_dataset_name}.txt")
    qrels_path = os.path.join(base_output_dir, f"qrels.{sanitized_dataset_name}.txt")

    # Ensure output directories exist
    print(f"Ensuring output directories '{base_output_dir}' and '{corpus_output_dir}' exist...")
    os.makedirs(corpus_output_dir, exist_ok=True)

    # Load the dataset
    print(f"Loading ir_datasets '{dataset_id}'...")
    dataset = ir_datasets.load(dataset_id)
    print("Dataset loaded successfully.")

    # 1. Process and save the corpus (documents) in batches
    print(f"\nProcessing {dataset.docs_count()} documents...")
    batch_size = 10000
    doc_iterator = dataset.docs_iter()
    total_docs_processed = 0

    for batch_num in itertools.count():
        batch_docs = list(itertools.islice(doc_iterator, batch_size))
        if not batch_docs:
            break

        batch_filename = f"batch_{batch_num:05d}.txt"
        filepath = os.path.join(corpus_output_dir, batch_filename)
        
        with open(filepath, "w", encoding="utf-8") as f_doc:
            for doc in batch_docs:
                f_doc.write("<DOC>\n")
                f_doc.write(f"<DOCNO>{doc.doc_id}</DOCNO>\n")
                f_doc.write("<TEXT>\n")
                # Combine title and abstract, handling None values
                title = doc.title or ""
                abstract = doc.abstract or ""
                full_text = f"{title} {abstract}".strip().replace('\n', ' ')
                f_doc.write(f"{full_text}\n")
                f_doc.write("</TEXT>\n")
                f_doc.write("</DOC>\n")
        
        total_docs_processed += len(batch_docs)
        print(f"Wrote batch {batch_num} to {filepath} ({total_docs_processed}/{dataset.docs_count()} docs)")

    print(f"Corpus of {total_docs_processed} documents saved to '{corpus_output_dir}'")

    # 2. Process and save the topics (queries)
    print(f"\nProcessing {dataset.queries_count()} topics...")
    with open(topics_path, "w", encoding="utf-8") as f_topics:
        for query in dataset.queries_iter():
            f_topics.write("<top>\n")
            f_topics.write(f"<num>Number: {query.query_id}</num>\n")
            # FIXED: Use query.title instead of query.query
            f_topics.write(f"<title>{query.title}</title>\n")
            f_topics.write("</top>\n")
    print(f"Topics saved to '{topics_path}'")

    # 3. Process and save the qrels (relevance judgments)
    print(f"\nProcessing {dataset.qrels_count()} qrels...")
    qrels_processed = 0
    with open(qrels_path, "w", encoding="utf-8") as f_qrels:
        for qrel in dataset.qrels_iter():
            f_qrels.write(f"{qrel.query_id} 0 {qrel.doc_id} {qrel.relevance}\n")
            qrels_processed += 1
    print(f"Processed {qrels_processed} qrels and saved to '{qrels_path}'")

    print("\nAll files downloaded and formatted successfully.")

if __name__ == "__main__":
    download_and_format_trec_covid()