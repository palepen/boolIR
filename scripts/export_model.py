import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import argparse

DEFAULT_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_OUTPUT_PATH = "models/bert_model.pt"
DEFAULT_MAX_LENGTH = 512


class RerankerModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        # Directly return the logits tensor from the model's output
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

def export_to_torchscript(model_name: str, output_path: str):
    """
    Exports a cross-encoder model to TorchScript format for reranking.
    """
    print(f"Loading pre-trained cross-encoder model: {model_name}...")
    device = torch.device("cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    
    model_to_export = RerankerModelWrapper(hf_model)
    model_to_export.eval()

    query = "What is the incubation period of COVID-19?"
    document = "The incubation period for COVID-19 is typically 5-6 days but can be up to 14 days."
    # 1. USE CONFIG
    inputs = tokenizer(query, document, padding='max_length', truncation=True, max_length=DEFAULT_MAX_LENGTH, return_tensors="pt")
    dummy_inputs = (inputs['input_ids'], inputs['attention_mask'])

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Exporting model to TorchScript at: {output_path}")

    try:
        traced_model = torch.jit.trace(model_to_export, dummy_inputs, strict=False)
        traced_model.save(output_path)
        print("Model exported successfully to TorchScript.")
    except Exception as e:
        print(f"An error occurred during tracing: {e}")
        return

    tokenizer.save_vocabulary(output_dir)
    print(f"Vocabulary saved to: {os.path.join(output_dir, 'vocab.txt')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a cross-encoder model to TorchScript for C++.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    export_to_torchscript(args.model, args.output)