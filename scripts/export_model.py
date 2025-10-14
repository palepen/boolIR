import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import argparse

def export_to_onnx(model_name: str, output_path: str):
    """
    Exports a cross-encoder model to ONNX format for reranking.
    """
    print(f"Loading pre-trained cross-encoder model: {model_name}...")
    # Use CPU for model conversion to avoid putting the conversion process on the GPU
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load cross-encoder model (outputs relevance scores)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    # Create dummy inputs for a query-document pair to trace the model
    query = "What is the incubation period of COVID-19?"
    document = "The incubation period for COVID-19 is typically 5-6 days but can be up to 14 days."

    inputs = tokenizer(
        query,
        document,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Exporting model to ONNX at: {output_path}")

    # Test model output shape before export
    with torch.no_grad():
        test_output = model(input_ids, attention_mask)
        print(f"Model output shape: {test_output.logits.shape}")
        print(f"Sample score: {test_output.logits[0].tolist()}")

    # Export with a single 'logits' output
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        output_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size'},
            'attention_mask': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        },
        opset_version=14,
        export_params=True
    )
    print("Model exported successfully.")

    # Also save tokenizer's vocabulary file, which is needed by the C++ WordPiece tokenizer
    tokenizer.save_vocabulary(os.path.dirname(output_path))
    print(f"Vocabulary saved to: {os.path.join(os.path.dirname(output_path), 'vocab.txt')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export a cross-encoder model to ONNX for reranking."
    )
    parser.add_argument(
        "--model",
        type=str,
        # --- MODIFICATION: Changed to a different reranker model ---
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="The name of the cross-encoder model to export from Hugging Face."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/bert_model.onnx",
        help="The path to save the exported ONNX model."
    )
    args = parser.parse_args()

    export_to_onnx(args.model, args.output)