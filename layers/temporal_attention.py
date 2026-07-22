import torch
import torch.nn as nn
class AdaptiveTemporalAttention(nn.Module):
    """
    Learnable temporal attention with adaptive window sizes
    - Learns effective window size for each block
    - Learns temperature for soft windowing
    """
    
    def __init__(self, dim, heads=2, window_size=12, 
                 qkv_bias=False, qk_scale=None, dropout=0.,
                 causal=True, device=None, learnable_window=True):
        super().__init__()
        
        assert dim % heads == 0
        
        self.dim = dim
        self.num_heads = heads
        self.causal = causal
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5
        self.window_size_init = window_size
        self.learnable_window = learnable_window
        self.device = device
        
        # NEW: Learn window size multiplier
        if learnable_window:
            # Initialize to 1.0 (means keep original window size)
            self.window_multiplier = nn.Parameter(torch.ones(1) * 1.0)
        else:
            self.register_buffer('window_multiplier', torch.ones(1))
        
        # NEW: Learn temperature for soft masking
        self.mask_temperature = nn.Parameter(torch.ones(1) * 5.0)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        
        # Create soft mask (learnable form)
        self._create_soft_mask(window_size, device)
    
    def _create_soft_mask(self, window_size, device):
        """
        Create soft causal mask with learnable temperature
        
        Instead of hard mask [0, 0, 0, 1, 1, 1]
        Use soft mask [0.01, 0.05, 0.2, 0.8, 0.95, 0.99]
        Controlled by temperature parameter
        """
        # Create triangular distance matrix
        positions = torch.arange(window_size, device=device)
        distance = (positions[:, None] - positions[None, :]).float()
        
        # Create soft mask: higher temperature → softer transition
        self.register_buffer('distance_matrix', distance)
        self.window_size = window_size
    
    def forward(self, x):
        """
        Args:
            x: [B, T, C] or [B, C, N, T]
        Returns:
            x: same shape as input
        """
        is_4d = x.dim() == 4
        if is_4d:
            b, c, n, t = x.shape
            x = x.permute(0, 2, 3, 1).reshape(-1, t, c)
        else:
            b = None
            c = None
            n = None
            t = None

        B_merged, T, C = x.shape

        # Compute effective window size
        if self.learnable_window:
            effective_window = int(self.window_size_init * torch.sigmoid(self.window_multiplier))
            # Clamp to reasonable range
            effective_window = max(1, min(effective_window, T))
        else:
            effective_window = self.window_size_init

# Reshape into windows. If the sequence length is not divisible by the
        # effective window size, pad it so the reshape is valid.
        if effective_window > 0:
            if T % effective_window != 0:
                pad_len = effective_window - (T % effective_window)
                x = torch.cat([x, x.new_zeros(x.shape[0], pad_len, C)], dim=1)
                T_padded = T + pad_len
            else:
                T_padded = T
            x = x.reshape(-1, effective_window, C)
        else:
            T_padded = T

        B_windowed, window_sz, C = x.shape

        # Compute QKV
        qkv = self.qkv(x).reshape(B_windowed, -1, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply soft causal mask with temperature
        if self.causal:
            # Create soft mask for this window
            distance = self.distance_matrix[:window_sz, :window_sz]
            soft_mask = torch.sigmoid(self.mask_temperature * distance)
            # soft_mask: [T, T], ranges from 0.5 (future) to ~1.0 (past)

            attn = attn * soft_mask.unsqueeze(0).unsqueeze(0)  # Broadcast

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_windowed, window_sz, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if effective_window > 0:
            x = x.reshape(B_merged, T_padded, C)
            if T_padded != T:
                x = x[:, :T, :]

        if is_4d:
            x = x.reshape(b, n, t, C).permute(0, 3, 1, 2)

        return x