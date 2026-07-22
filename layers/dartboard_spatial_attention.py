import torch
import torch.nn as nn


class PreNorm(nn.Module):
    """Standard pre-norm wrapper: LayerNorm before fn, fn's output returned
    as-is (residual is added by the caller, matching DS_MSA's usage below)."""
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """Standard 2-layer GELU MLP."""
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class SpatialAttention(nn.Module):
    # dartboard project + MSA
    def __init__(self,
                 dim,
                 heads=4,
                 qkv_bias=False,
                 qk_scale=None,
                 dropout=0.,
                 num_sectors=17,
                 assignment=None,
                 mask=None):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}."

        self.dim = dim
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5
        self.num_sector = num_sectors
        # Registered as buffers (not plain attributes) so they correctly
        # follow the module across .to(device)/.cuda() calls and appear in
        # state_dict, rather than silently staying on whatever device they
        # were constructed on.
        self.register_buffer('assignment', assignment)  # [n, n, num_sector]
        self.register_buffer('mask', mask)  # [n, num_sector]

        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_linear = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.relative_bias = nn.Parameter(torch.randn(heads, 1, num_sectors))
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: [b, n, c]

        B, N, C = x.shape

        # query: [bn, 1, c]
        # key/value target: [bn, num_sector, c]
        # [b, n, num_sector, c]
        pre_kv = torch.einsum('bnc,mnr->bmrc', x, self.assignment)

        pre_kv = pre_kv.reshape(-1, self.num_sector, C)  # [bn, num_sector, c]
        pre_q = x.reshape(-1, 1, C)  # [bn, 1, c]

        q = self.q_linear(pre_q).reshape(B*N, -1, self.num_heads, C //
                                         self.num_heads).permute(0, 2, 1, 3)  # [bn, num_heads, 1, c//num_heads]
        kv = self.kv_linear(pre_kv).reshape(B*N, -1, 2, self.num_heads,
                                            C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # [bn, num_heads, num_sector, c//num_heads]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.reshape(B, N, self.num_heads, 1,
                            self.num_sector) + self.relative_bias # you can fuse external factors here as well
        mask = self.mask.reshape(1, N, 1, 1, self.num_sector)

        # masking
        attn = attn.masked_fill_(mask, float(
            "-inf")).reshape(B * N, self.num_heads, 1, self.num_sector).softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DS_MSA(nn.Module):
    # Dartboard Spatial MSA
    def __init__(self,
                 dim,  # hidden dimension
                 depth,  # number of MSA in DS-MSA
                 heads,  # number of heads
                 mlp_dim,  # mlp dimension
                 assignment,  # dartboard assignment matrix
                 mask,  # mask
                 dropout=0.):  # dropout rate
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                SpatialAttention(dim, heads=heads, dropout=dropout,
                                 assignment=assignment, mask=mask,
                                 num_sectors=assignment.shape[-1]),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        # x: [b, c, n, t]
        b, c, n, t = x.shape
        x = x.permute(0, 3, 2, 1).reshape(b*t, n, c)  # [b*t, n, c]
        # x = x + self.pos_embedding  # [b*t, n, c]  we use relative PE instead
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        x = x.reshape(b, t, n, c).permute(0, 3, 2, 1)
        return x