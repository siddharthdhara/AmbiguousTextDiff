import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class TextEncoder(nn.Module):
    """Text encoder for medical reports using pretrained language models."""
    
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT", max_length=128, output_dim=768):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_safetensors=True
    )
        self.max_length = max_length
        self.output_dim = output_dim
        
        # Optional projection layer if needed to match dimensions with image features
        self.projection = nn.Linear(self.encoder.config.hidden_size, output_dim)
        
    def forward(self, text):
        """
        Args:
            text: List of text descriptions
        Returns:
            Text embeddings
        """
        # Tokenize the input texts
        inputs = self.tokenizer(text, padding=True, truncation=True, 
                               max_length=self.max_length, return_tensors="pt")
        
        # Move to the same device as the model
        inputs = {k: v.to(self.encoder.device) for k, v in inputs.items()}
        
        # Get the BERT embeddings
        outputs = self.encoder(**inputs)
        
        # Use the [CLS] token embedding as the text representation
        text_features = outputs.last_hidden_state[:, 0]  # [batch_size, hidden_size]
        
        # Project to desired dimension if needed
        text_features = self.projection(text_features)  # [batch_size, output_dim]
        
        return text_features
    
    def freeze(self):
        """Freeze the text encoder for efficiency."""
        for param in self.encoder.parameters():
            param.requires_grad = False