import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from typing import Optional

class CrossAttention(nn.Module):
    """Cross-attention mechanism to fuse text and image features."""
    
    def __init__(self, query_dim, key_dim, value_dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.inner_dim = inner_dim
        self.context_dim = key_dim  # Store context dimension for later use
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dropout = dropout
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(key_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(value_dim, inner_dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, context):
        """
        Args:
            x: Image features (queries) [batch, h*w, c]
            context: Text features (keys and values) [batch, seq_len, d]
        Returns:
            Attended features
        """
        h = self.heads
        current_dim = x.shape[-1]
    
    # If the input dimension has changed, recreate the projection layers
        if hasattr(self.to_q, 'in_features') and self.to_q.in_features != current_dim:
            device = self.to_q.weight.device
        # Recreate the projection layers with the new dimensions
            self.to_q = nn.Linear(current_dim, self.inner_dim, bias=False).to(device)
            self.to_k = nn.Linear(self.context_dim, self.inner_dim, bias=False).to(device)
            self.to_v = nn.Linear(self.context_dim, self.inner_dim, bias=False).to(device)
            self.to_out = nn.Sequential(
                nn.Linear(self.inner_dim, current_dim),
                nn.Dropout(self.dropout)
            ).to(device)
        
        
        q = self.to_q(x)
        context = x if context is None else context
        k = self.to_k(context)
        v = self.to_v(context)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        
        # Compute attention scores
        sim = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = F.softmax(sim, dim=-1)
        
        # Compute weighted sum
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(-3, -2).reshape(*x.shape[:-1], -1)
        
        return self.to_out(out)

class CrossModalFusion(nn.Module):
    """Fuses text and image features using cross-attention."""
    
    def __init__(self, image_dim, text_dim, hidden_dim=512, heads=8, dropout=0.1):
        super().__init__()
        # Add a projection layer to handle dimension mismatch
        self.target_image_dim = image_dim  # Internal working dimension
        self.text_dim = text_dim
        self.hidden_dim_ff = hidden_dim
        self.heads = heads
        self.dropout_rate = dropout

        # Projections will be nn.Identity or Linear, defined in the first forward pass
        self._input_projection = None
        self._output_projection = None
        
        self.norm_img = nn.LayerNorm(self.target_image_dim) # Expects target_image_dim
        self.norm_text = nn.LayerNorm(self.text_dim)
        
        # Cross-attention: Image attends to Text
        self.cross_attn = CrossAttention(
            query_dim=self.target_image_dim,  # Operates on target_image_dim
            key_dim=self.text_dim,
            value_dim=self.text_dim,
            heads=self.heads,
            dropout=self.dropout_rate
        )
        
        
        # Feed-forward network, operates on target_image_dim
        self.ff_norm = nn.LayerNorm(self.target_image_dim)
        self.ff_linear1 = nn.Linear(self.target_image_dim, self.hidden_dim_ff)
        self.ff_gelu = nn.GELU()
        self.ff_dropout1 = nn.Dropout(self.dropout_rate)
        self.ff_linear2 = nn.Linear(self.hidden_dim_ff, self.target_image_dim)
        self.ff_dropout2 = nn.Dropout(self.dropout_rate)
        
    def forward(self, img_features, text_features):
        """
        Args:
            img_features: Image features [batch, c, h, w]
            text_features: Text features [batch, d]
        Returns:
            Fused features with same shape as img_features
        """
        original_shape = img_features.shape
        actual_channel_dim = original_shape[1]
        batch_size = original_shape[0]
        spatial_dims = original_shape[2:]

        # Reshape image features to [batch, seq_len, channels]
        img_features_flat = img_features.reshape(batch_size, actual_channel_dim, -1).permute(0, 2, 1)
        
        # Initialize or update projection layers if needed
        if self._input_projection is None or \
           (not isinstance(self._input_projection, nn.Identity) and self._input_projection.in_features != actual_channel_dim) or \
           (isinstance(self._input_projection, nn.Identity) and actual_channel_dim != self.target_image_dim): # handles switch from Identity to Linear
            if actual_channel_dim == self.target_image_dim:
                self._input_projection = nn.Identity().to(img_features.device)
                self._output_projection = nn.Identity().to(img_features.device)
            else:
                self._input_projection = nn.Linear(actual_channel_dim, self.target_image_dim).to(img_features.device)
                self._output_projection = nn.Linear(self.target_image_dim, actual_channel_dim).to(img_features.device)

        # Project to target_image_dim if necessary
        current_img_features_flat = self._input_projection(img_features_flat) # Now shape [B, H*W, target_image_dim]
        
        # Expand text features
        text_features_expanded = text_features.unsqueeze(1)  # [b, 1, d]
        
        # Apply layer normalization
        norm_img = self.norm_img(current_img_features_flat) # Expects target_image_dim
        norm_text = self.norm_text(text_features_expanded)
        
        # Cross-attention
        # CrossAttention query_dim is target_image_dim, input norm_img is target_image_dim
        attended = self.cross_attn(norm_img, norm_text) # Output is target_image_dim
        
        # Residual connection for image features
        x_for_ff = attended + current_img_features_flat # Both are target_image_dim
        
        # Feed-forward
        # All ff components expect target_image_dim
        ff_out = self.ff_norm(x_for_ff)
        ff_out = self.ff_linear1(ff_out)
        ff_out = self.ff_gelu(ff_out)
        ff_out = self.ff_dropout1(ff_out)
        ff_out = self.ff_linear2(ff_out)
        ff_out = self.ff_dropout2(ff_out)
        
        output_after_ff = ff_out + x_for_ff # Residual for ff, dim is target_image_dim
        
        # Project back to actual_channel_dim if necessary
        final_flat_output = self._output_projection(output_after_ff) # Shape [B, H*W, actual_channel_dim]
    
        # Reshape back to original image format
        output = final_flat_output.permute(0, 2, 1).reshape(batch_size, actual_channel_dim, *spatial_dims)
        
        return output