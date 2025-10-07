import torch
from sentence_transformers import SentenceTransformer
import os
import argparse

def export_to_onnx(model_name: str, output_path: str):
    """
    Exports a SentenceTransformer model to ONNX format.
    """
    print(f"Loading pre-trained model: {model_name}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SentenceTransformer(model_name).to(device)
    core_model = model[0].auto_model

    dummy_text = "This is a sample sentence."
    inputs = model.tokenizer(
        dummy_text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Exporting model to ONNX at: {output_path}")

    torch.onnx.export(
        core_model,
        (input_ids, attention_mask),
        output_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['last_hidden_state'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'sentence_embedding': {0: 'batch_size'}
        },
        opset_version=14,  # <-- FIX: Changed from 13 to 14
        export_params=True
    )
    print("Model exported successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export a SentenceTransformer model to ONNX."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="The name of the SentenceTransformer model to export."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/bert_model.onnx",
        help="The path to save the exported ONNX model."
    )
    args = parser.parse_args()
    
    export_to_onnx(args.model, args.output)