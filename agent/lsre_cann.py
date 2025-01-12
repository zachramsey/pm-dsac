'''
Derived from: github.com/jiahaoli57/LSRE-CAAN/blob/main/LSRE_CAAN.py
'''

import torch
import torch.nn as nn
from einops import rearrange, repeat
import numpy as np


# class PositionalEncoding(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         position = torch.arange(cfg["window_size"]).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, cfg["feat_dim"], 2) * -(np.log(10000.0) / cfg["feat_dim"]))
#         pe = torch.zeros(cfg["window_size"], cfg["feat_dim"])
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.pe = nn.Buffer(pe)

#     def forward(self, x):
#         ''' ### Forward pass of PositionalEncoding
#         Args:
#             x (torch.Tensor): Input tensor of shape (batch_dim*asset_dim, window_size, feat_dim)
#         Returns:
#             x (torch.Tensor): Output tensor of shape (batch_dim*asset_dim, window_size, feat_dim)
#         '''
#         x = rearrange(x, 'b w f -> w b f')
#         x = x + self.pe[:x.size(0)]
#         x = rearrange(x, 'w b f -> b w f')
#         return x


class QuickGELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)

class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, q_dim, kv_dim):
        super().__init__()
        inner_dim = num_heads * head_dim
        self.scale = head_dim ** -0.5
        self.num_heads = num_heads

        self.q_linear = nn.Linear(q_dim, inner_dim, bias=False)
        self.k_linear = nn.Linear(kv_dim, inner_dim, bias=False)
        self.v_linear = nn.Linear(kv_dim, inner_dim, bias=False)
        self.out_linear = nn.Linear(inner_dim, q_dim)

    def forward(self, q, kv, is_causal = False):
        ''' ### Forward pass of Attention
        Args:
            q (torch.Tensor): Query tensor of shape (batch_dim*asset_dim, num_latents, latent_dim)
            kv (torch.Tensor): Key-Value tensor of shape (batch_dim*asset_dim, window_size, feat_dim)
            is_causal (bool): Whether to apply causal masking
        Returns:
            out (torch.Tensor): Output tensor of shape (batch_dim*asset_dim, num_latents, latent_dim)
        '''
        query = self.q_linear(q)    # (b*a, n, l) -> (b*a, n, h*d)
        key = self.k_linear(kv)     # (b*a, w, f) -> (b*a, w, h*d)
        value = self.v_linear(kv)   # (b*a, w, f) -> (b*a, w, h*d)
        
        query = rearrange(query, 'b w (h d) -> (b h) w d', h=self.num_heads)    # (b*a, n, h*d) -> (b*a*h, n, d)
        key = rearrange(key, 'b w (h d) -> (b h) w d', h=self.num_heads)        # (b*a, w, h*d) -> (b*a*h, w, d)
        value = rearrange(value, 'b w (h d) -> (b h) w d', h=self.num_heads)    # (b*a, w, h*d) -> (b*a*h, w, d)

        # (b*a*h, n, d) x (b*a*h, w, d) -> (b*a*h, n, w)
        scores = torch.einsum('b i d, b j d -> b i j', query, key) * self.scale

        # Derived from: pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        if is_causal:
            mask = torch.ones(q.shape[-2], kv.shape[-2], dtype=torch.bool).tril(diagonal=0)
            scores.masked_fill_(mask.logical_not(), float('-inf'))
            scores.to(query.dtype)

        # (b*a*h, n, w) -> (b*a*h, n, w)
        attn = torch.softmax(scores, dim=-1)

        # (b*a*h, n, w) x (b*a*h, w, d) -> (a*h, n, d)
        out = torch.einsum('b i j, b j d -> b i d', attn, value)
        
        # (b*a*h, n, d) -> (b*a, n, h*d)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self.num_heads)

        # (b*a, n, h*d) -> (b*a, n, l)
        out = self.out_linear(out)
        
        return out
    

class AttentionBlock(nn.Module):
    def __init__(self, num_heads, head_dim, q_dim, kv_dim=None):
        super().__init__()
        inner_dim = np.power(2, np.ceil(np.log2(q_dim))).astype(int)    # Next power of 2

        self.norm_q = nn.LayerNorm(q_dim)
        if kv_dim is None: kv_dim = q_dim
        else: self.norm_kv = nn.LayerNorm(kv_dim)

        self.attn = Attention(num_heads, head_dim, q_dim, kv_dim)
        self.norm_z = nn.LayerNorm(q_dim)
        self.ff = nn.Sequential(
            nn.Linear(q_dim, inner_dim),
            QuickGELU(),
            nn.Linear(inner_dim, q_dim)
        )

    def forward(self, q, kv = None):
        ''' ### Forward pass of AttentionBlock
        Args:
            q (torch.Tensor): Query tensor of shape (batch_dim*asset_dim, num_latents, latent_dim)
            kv (torch.Tensor): Key-Value tensor of shape (batch_dim*asset_dim, window_size, feat_dim)
        Returns:
            z (torch.Tensor): Output tensor of shape (batch_dim*asset_dim, num_latents, latent_dim)
        '''
        q_norm = self.norm_q(q)
        kv_norm = q_norm if kv is None else self.norm_kv(kv)
        z = self.attn(q_norm, kv_norm)
        z = z + q
        z = z + self.ff(self.norm_z(z))
        return z


class LSRE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        depth = cfg['depth']                        # Number of self attention layers
        f = cfg['feat_dim']                         # Number of features

        asset_dim = cfg['asset_dim']                # Number of assets

        num_latents = cfg['num_latents']            # Number of latents
        latent_dim = cfg['latent_dim']              # Dimension of latents

        num_cross_heads = cfg['num_cross_heads']    # Number of cross attention heads
        cross_head_dim = cfg['cross_head_dim']      # Dimension of cross attention heads

        num_latent_heads = cfg['num_latent_heads']  # Number of self attention heads
        latent_head_dim = cfg['latent_head_dim']    # Dimension of self attention heads

        self.z = nn.Parameter(torch.randn(asset_dim, num_latents, latent_dim))
        self.cross_attn = AttentionBlock(num_cross_heads, cross_head_dim, latent_dim, f)
        self.self_attns = nn.ModuleList([AttentionBlock(num_latent_heads, latent_head_dim, latent_dim) for _ in range(depth)])

        self.out = nn.Linear(num_latents, 1)

    def forward(self, x):
        ''' ### Forward pass of LSRE
        Args:
            x (torch.Tensor): Input tensor of shape (batch_dim, asset_dim, window_size, feat_dim)
        Returns:
            z (torch.Tensor): Latent tensor of shape (batch_dim, asset_dim, latent_dim)
        '''
        batch_dim = x.shape[0]
        z = repeat(self.z, 'a n d -> b a n d', b=batch_dim)
        z = rearrange(z, 'b a n d -> (b a) n d')
        x = rearrange(x, 'b a w f -> (b a) w f')

        # (b*a, n, d), (b*a, w, f) -> (b*a, n, d)
        z = self.cross_attn(z, x)

        for self_attn in self.self_attns:
            z = self_attn(z)

        # (b*a, n, d) -> (b*a, d, n)
        z = z.transpose(1, 2)
        
        # (b*a, d, n) -> (b*a, d, 1)
        z = self.out(z)

        # (b*a, d, 1) -> (b, a, d)
        z = rearrange(z, '(b a) d 1 -> b a d', b=batch_dim)

        return z


class CANN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        latent_dim = cfg['latent_dim']
        self.scale = latent_dim ** -0.5
        
        self.q_linear = nn.Linear(latent_dim, latent_dim)
        self.k_linear = nn.Linear(latent_dim, latent_dim)
        self.v_linear = nn.Linear(latent_dim, latent_dim)

    def forward(self, z):
        ''' ### Forward pass of CANN
        Args:
            z (torch.Tensor): Latent tensor of shape (batch_dim, asset_dim, latent_dim)
        Returns:
            h (torch.Tensor): Hidden state of shape (batch_dim, asset_dim, latent_dim)
        '''
        q = self.q_linear(z)    # (batch_dim, asset_dim, latent_dim)
        k = self.k_linear(z)    # (batch_dim, asset_dim, latent_dim)
        v = self.v_linear(z)    # (batch_dim, asset_dim, latent_dim)

        # (batch_dim, asset_dim, latent_dim) x (batch_dim, latent_dim, asset_dim) 
        beta = torch.matmul(q, k.transpose(1, 2)) * self.scale  # (batch_dim, asset_dim, asset_dim)
        beta = torch.softmax(beta, dim=-1)

        # (batch_dim, asset_dim, asset_dim) x (batch_dim, asset_dim, latent_dim)
        h = torch.matmul(beta, v)   # (batch_dim, asset_dim, latent_dim)
        
        return h

class LSRE_CANN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # self.pos_enc = PositionalEncoding(cfg)
        # self.pos_emb = nn.Embedding(cfg["window_size"], cfg["feat_dim"])  #NOTE Original implementation
        self.lsre = LSRE(cfg)
        self.dropout = nn.Dropout(cfg["dropout"])
        self.cann = CANN(cfg)

    def forward(self, x):
        ''' ### Forward pass of LSRE_CANN
        Args:
            x (torch.Tensor): Input tensor of shape (batch_dim, asset_dim, window_size, feat_dim)
            reset (bool): Whether to reset the latent tensor
        Returns:
            h (torch.Tensor): Hidden state of shape (batch_dim, asset_dim, latent_dim)
        '''
        # (asset_dim, window_size, feat_dim) -> (1, asset_dim, window_size, feat_dim)
        if x.ndim == 3: x = x.unsqueeze(0)

        # x = self.pos_enc(x)
        # pos_emb = self.pos_emb(torch.arange(self.window_size))    #NOTE Original implementation
        # x = x + rearrange(pos_emb, "n d -> () n d")

        # (batch_dim, asset_dim, window_size, feat_dim) -> (batch_dim, asset_dim, latent_dim)
        z = self.lsre(x)
        z = self.dropout(z)

        # (batch_dim, asset_dim, latent_dim) -> (batch_dim, asset_dim, latent_dim)
        h = self.cann(z)

        return h
        