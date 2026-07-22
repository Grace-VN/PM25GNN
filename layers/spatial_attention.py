# ============================================================================
# NEW FILE: src/layers/spatial_attention.py
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveSectorMSA(nn.Module):
    """
    Adaptive Dartboard Sector Multi-Head Self-Attention
    
    Replaces the fixed DS_MSA with learnable sector assignment
    Sectors are assigned based on feature similarity, not geography
    """
    
    def __init__(self, dim, heads=4, qkv_bias=False, qk_scale=None,
                 dropout=0.0, num_sectors=17, feature_dim=32, 
                 temperature=0.5):
        super().__init__()
        
        assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}"
        
        self.dim = dim
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5
        self.num_sectors = num_sectors
        self.temperature = temperature
        
        # Learnable sector centroids
        self.sector_centroids = nn.Parameter(
            torch.randn(num_sectors, feature_dim)
        )
        nn.init.xavier_uniform_(self.sector_centroids)
        
        # Feature projection for assignment
        self.feature_projection = nn.Conv2d(dim, feature_dim, 1)
        
        # Standard attention layers
        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_linear = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.relative_bias = nn.Parameter(torch.randn(heads, 1, num_sectors))
        
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
    
    def compute_soft_assignment(self, features):
        """
        Compute soft assignment based on feature similarity
        
        Args:
            features: [batch, feature_dim, nodes] averaged over time
        Returns:
            assignment: [batch, nodes, num_sectors]
        """
        batch_size = features.shape[0]
        num_nodes = features.shape[-1]
        
        features_t = features.permute(0, 2, 1)  # [batch, nodes, feature_dim]
        
        # Euclidean distance to sector centroids
        distances = torch.cdist(features_t, self.sector_centroids)  # [batch, nodes, sectors]
        
        # Soft assignment via softmax with temperature
        assignment_soft = torch.softmax(-distances / self.temperature, dim=-1)
        
        return assignment_soft  # [batch, nodes, num_sectors]
    
    def forward(self, x):
        """
        Args:
            x: [b, c, n, t]
        Returns:
            out: [b, c, n, t]
        """
        b, c, n, t = x.shape

        # Permute to [b, t, n, c] for feature extraction
        x_permuted = x.permute(0, 3, 2, 1)  # [b, t, n, c]

        # Extract features for sector assignment by averaging over time
        x_features = x_permuted.mean(dim=1)  # [b, n, c]

        # Project node features to sector-assignment features
        features = x_features.permute(0, 2, 1).unsqueeze(-1)  # [b, c, n, 1]
        features = self.feature_projection(features).squeeze(-1)  # [b, feature_dim, n]
        features = features.permute(0, 2, 1)  # [b, n, feature_dim]

        # Compute soft sector assignment for each node
        assignment = self.compute_soft_assignment(features.permute(0, 2, 1))  # [b, n, num_sectors]
        assignment = assignment.reshape(b, n, self.num_sectors)

        # Reshape x for attention: [b*t, n, c]
        x_att = x_permuted.reshape(-1, n, c)  # [b*t, n, c]

        # Aggregate node features into sector representations
        x_sectors = torch.einsum('bnc,bns->bsc', x_att, assignment.repeat_interleave(t, dim=0))
        x_sectors = x_sectors.reshape(-1, self.num_sectors, c)  # [b*t, num_sectors, c]

        # Multi-head attention over sectors for each node query
        x_query = x_att  # [b*t, n, c]
        q = self.q_linear(x_query).reshape(-1, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv_linear(x_sectors).reshape(-1, self.num_sectors, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self.relative_bias.view(1, self.num_heads, 1, self.num_sectors)
        attn = attn.softmax(dim=-1)

        # Apply attention to values
        x_out = (attn @ v).transpose(1, 2).reshape(-1, n, c)
        x_out = x_out.reshape(b, t, n, c).permute(0, 3, 2, 1)  # Back to [b, c, n, t]

        x_out = self.proj(x_out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_out = self.proj_drop(x_out)

        return x_out