import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import argparse

def export_to_onnx(model_name: str, output_path: str):
    """
    Exports a cross-encoder model to ONNX format for reranking.
    """
    print(f"Loading pre-trained cross-encoder model: {model_name}...")
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Use AutoModelForSequenceClassification for cross-encoders
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    # Create dummy inputs for a query-document pair
    query = "What is the incubation period of COVID-19?"
    document = "The incubation period for COVID-19 is typically 5-6 days but can be up to 14 days."
    
    inputs = tokenizer(
        query,
        document,
        padding='max_length',
        truncation=True,
        max_length=512,  # Cross-encoders can handle longer sequences
        return_tensors="pt"
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Exporting model to ONNX at: {output_path}")

    # The output is now 'logits', not 'last_hidden_state'
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        output_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'], # CRITICAL CHANGE
        dynamic_axes={
            'input_ids': {0: 'batch_size'},
            'attention_mask': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        },
        opset_version=14,
        export_params=True
    )
    print("Model exported successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export a cross-encoder model to ONNX for reranking."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="allenai/scibert_scivocab_uncased", # Use a proper cross-encoder
        help="The name of the cross-encoder model to export."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/bert_model.onnx",
        help="The path to save the exported ONNX model."
    )
    args = parser.parse_args()
    
    export_to_onnx(args.model, args.output)