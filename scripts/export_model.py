# scripts/export_model.py
import argparse
from sentence_transformers import SentenceTransformer
import torch

def export_to_onnx(model_name: str, output_path: str):
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Example input (max 128 tokens for efficiency on GTX 1650)
    example = ["This is a sample query for ONNX export."]
    features = model.tokenizer(
        example,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    print("Exporting to ONNX...")
    torch.onnx.export(
        model._first_module().auto_model,
        (features['input_ids'], features['attention_mask']),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["output"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "output": {0: "batch"}
        },
        opset_version=13,
        export_params=True
    )
    print(f"✅ ONNX model saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    export_to_onnx(args.model, args.output)