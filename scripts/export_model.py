import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class SimpleBertWrapper(nn.Module):
    """Simplified BERT wrapper that avoids problematic operations"""
    
    def __init__(self, model_name):
        super(SimpleBertWrapper, self).__init__()
        # Load just the transformer model without sentence-transformers wrapper
        from transformers import AutoModel, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Freeze all parameters to avoid training-specific operations
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.model.eval()
        
    def forward(self, input_ids, attention_mask):
        # Get token embeddings from BERT
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            token_embeddings = outputs.last_hidden_state
            
            # Simple mean pooling (manually implemented to avoid complex operations)
            # Expand attention mask for broadcasting
            input_mask_expanded = attention_mask.unsqueeze(-1).float()
            
            # Apply mask and compute mean
            masked_embeddings = token_embeddings * input_mask_expanded
            sum_embeddings = torch.sum(masked_embeddings, dim=1)
            sum_mask = torch.sum(input_mask_expanded, dim=1)
            
            # Avoid division by zero
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            
            return mean_embeddings

def export_simple_model():
    """Export a simplified version that's more compatible with ONNX"""
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    output_path = "models/bert_model.onnx"
    
    print(f"Loading simplified model: {model_name}...")
    
    try:
        # Create simplified wrapper
        model = SimpleBertWrapper(model_name)
        model.eval()
        
        # Create dummy inputs
        batch_size = 1
        seq_length = 64  # Reduced sequence length for compatibility
        
        dummy_input_ids = torch.randint(1, 1000, (batch_size, seq_length), dtype=torch.long)
        dummy_attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
        
        print(f"Input shapes: input_ids={dummy_input_ids.shape}, attention_mask={dummy_attention_mask.shape}")
        
        # Test the model first
        with torch.no_grad():
            test_output = model(dummy_input_ids, dummy_attention_mask)
            print(f"Test output shape: {test_output.shape}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"Exporting to ONNX: {output_path}")
        
        # Export with higher opset version and additional settings
        torch.onnx.export(
            model,
            args=(dummy_input_ids, dummy_attention_mask),
            f=output_path,
            input_names=['input_ids', 'attention_mask'],
            output_names=['sentence_embedding'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_len'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_len'},
                'sentence_embedding': {0: 'batch_size'}
            },
            opset_version=14,  # Updated to version 14
            do_constant_folding=True,
            verbose=False,
            export_params=True,
            training=torch.onnx.TrainingMode.EVAL
        )
        
        print("\n✅ ONNX export successful!")
        print(f"Model saved at: {output_path}")
        
        # Verify the model
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("✅ ONNX model validation passed!")
            
            # Print model info
            print(f"Model inputs: {[input.name for input in onnx_model.graph.input]}")
            print(f"Model outputs: {[output.name for output in onnx_model.graph.output]}")
            
        except ImportError:
            print("⚠️  ONNX package not available for validation")
        except Exception as e:
            print(f"⚠️  ONNX validation failed: {e}")
            
        return True
        
    except Exception as e:
        print(f"❌ Simplified export failed: {e}")
        return False

def export_fallback_model():
    """Ultra-simple fallback that just saves model weights"""
    try:
        print("🔄 Trying fallback approach...")
        model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        
        # Just download and save the sentence transformer model
        model = SentenceTransformer(model_name)
        
        # Save as PyTorch model instead
        fallback_path = "models/bert_model.pt"
        torch.save(model.state_dict(), fallback_path)
        
        print(f"✅ Fallback: Saved PyTorch model to {fallback_path}")
        print("Note: You'll need to load this with SentenceTransformer in your C++ code via Python bindings")
        
        # Also save a simple embedding example
        sample_text = "This is a sample document for indexing and retrieval."
        embeddings = model.encode([sample_text])
        
        embedding_path = "models/sample_embedding.txt"
        with open(embedding_path, 'w') as f:
            f.write(f"Sample text: {sample_text}\n")
            f.write(f"Embedding dimensions: {embeddings.shape}\n")
            f.write("Embedding values:\n")
            f.write(" ".join(map(str, embeddings[0])))
        
        print(f"✅ Sample embedding saved to {embedding_path}")
        return True
        
    except Exception as e:
        print(f"❌ Fallback also failed: {e}")
        return False

def main():
    """Main export function with multiple fallback strategies"""
    print("Starting BERT model export...")
    
    # Try simplified ONNX export first
    if export_simple_model():
        return
    
    print("\n" + "="*50)
    print("ONNX export failed, trying fallback...")
    
    # Try fallback approach
    if export_fallback_model():
        return
    
    print("\n❌ All export methods failed!")
    print("Recommendations:")
    print("1. Check your PyTorch version: pip install torch==1.13.1")
    print("2. Update transformers: pip install transformers>=4.20.0")
    print("3. Install ONNX: pip install onnx>=1.12.0")
    print("4. Try with a different model or skip neural features for now")

if __name__ == "__main__":
    if not os.path.exists("models"):
        os.makedirs("models")
    main()