'''
Derived from: github.com/jiahaoli57/LSRE-CAAN/blob/main/LSRE_CAAN.py
'''

import torch
import torch.nn as nn
from einops import rearrange, repeat
import numpy as np

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
            q (torch.Tensor): Query tensor of shape (asset_dim, window_size, feat_dim)
            kv (torch.Tensor): Key-Value tensor of shape (asset_dim, window_size, feat_dim)
            is_causal (bool): Whether to apply causal masking
        Returns:
            out (torch.Tensor): Output tensor of shape (asset_dim, window_size, feat_dim)
        '''
        # (asset_dim, window_size, feat_dim) -> (asset_dim, window_size, (num_heads * head_dim))
        query = self.q_linear(q)
        key = self.k_linear(kv)
        value = self.v_linear(kv)

        # (asset_dim, window_size, (num_heads * head_dim)) -> ((asset_dim * num_heads), window_size, head_dim)
        query = rearrange(query, 'b s (h d) -> (b h) s d', h=self.num_heads)    
        key = rearrange(key, 'b s (h d) -> (b h) s d', h=self.num_heads)
        value = rearrange(value, 'b s (h d) -> (b h) s d', h=self.num_heads)

        # ((asset_dim * num_heads), window_size, head_dim) x ((asset_dim * num_heads), window_size, head_dim) -> ((asset_dim * num_heads), window_size, window_size)
        scores = torch.einsum('b i d, b j d -> b i j', query, key) * self.scale

        # Derived from: pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        if is_causal:
            mask = torch.ones(q.shape[-2], kv.shape[-2], dtype=torch.bool).tril(diagonal=0)
            scores.masked_fill_(mask.logical_not(), float('-inf'))
            scores.to(query.dtype)

        attn = torch.softmax(scores, dim=-1)

        # ((asset_dim * num_heads), window_size, window_size) x ((asset_dim * num_heads), window_size, head_dim) -> ((asset_dim * num_heads), window_size, head_dim)
        out = torch.einsum('b i j, b j d -> b i d', attn, value)
        
        # ((asset_dim * num_heads), window_size, head_dim) -> (asset_dim, window_size, (num_heads * head_dim))
        out = rearrange(out, '(b h) s d -> b s (h d)', h=self.num_heads)

        # (asset_dim, window_size, (num_heads * head_dim)) -> (asset_dim, window_size, feat_dim)
        out = self.out_linear(out)
        
        return out
    

class AttentionBlock(nn.Module):
    def __init__(self, num_heads, head_dim, q_dim, kv_dim=None):
        super().__init__()
        if kv_dim is None: kv_dim = q_dim
        inner_dim = np.power(2, np.ceil(np.log2(q_dim))).astype(int)    # Next power of 2

        # TODO: Does the "context" part (kv) need to be normed seperately?
        self.attn = Attention(num_heads, head_dim, q_dim, kv_dim)
        self.attn_norm = nn.LayerNorm(q_dim)
        self.ff = nn.Sequential(
            nn.Linear(q_dim, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, q_dim)
        )
        self.ff_norm = nn.LayerNorm(q_dim)

    def forward(self, q, kv = None):
        ''' ### Forward pass of AttentionBlock
        Args:
            q (torch.Tensor): Query tensor of shape (asset_dim, window_size, feat_dim)
            kv (torch.Tensor): Key-Value tensor of shape (asset_dim, window_size, feat_dim)
        Returns:
            z (torch.Tensor): Output tensor of shape (asset_dim, window_size, feat_dim)
        '''
        if kv is None: kv = q
        z = self.attn_norm(q + self.attn(q, kv))
        z = self.ff_norm(z + self.ff(z))
        return z


class LSRE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        depth = cfg['depth']                        # Number of self attention layers
        f = cfg['feat_dim']                         # Number of features

        num_latents = cfg['num_latents']            # Number of latents
        latent_dim = cfg['latent_dim']              # Dimension of latents

        num_cross_heads = cfg['num_cross_heads']    # Number of cross attention heads
        cross_head_dim = cfg['cross_head_dim']      # Dimension of cross attention heads

        num_latent_heads = cfg['num_latent_heads']  # Number of self attention heads
        latent_head_dim = cfg['latent_head_dim']    # Dimension of self attention heads

        self.z = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.cross_attn = AttentionBlock(num_cross_heads, cross_head_dim, latent_dim, f)
        self.self_attns = nn.ModuleList([AttentionBlock(num_latent_heads, latent_head_dim, latent_dim) for _ in range(depth)])

    def forward(self, x):
        ''' ### Forward pass of LSRE
        Args:
            x (torch.Tensor): Input tensor of shape (asset_dim, window_size, feat_dim)
        Returns:
            z (torch.Tensor): Latent tensor of shape (asset_dim, latent_dim)
        '''
        z = repeat(self.z, 'n d -> b n d', b=x.shape[0])
        x = self.cross_attn(z, x)
        for self_attn in self.self_attns:
            z = self_attn(z)
        return z.squeeze(1)


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
            z (torch.Tensor): Latent tensor of shape (asset_dim, latent_dim)
        Returns:
            h (torch.Tensor): Contextual tensor of shape (asset_dim, latent_dim)
        '''
        q = self.q_linear(z)
        k = self.k_linear(z)
        v = self.v_linear(z)

        # (asset_dim, latent_dim) x (latent_dim, asset_dim) -> (asset_dim, asset_dim)
        scores = torch.matmul(q, k.transpose(0, 1)) * self.scale

        # (asset_dim, asset_dim) -> (asset_dim, asset_dim)
        attn = torch.softmax(scores, dim=-1)
        
        # (asset_dim, latent_dim) -> (1, asset_dim, latent_dim)
        v = v.unsqueeze(0)

        # (asset_dim, asset_dim) -> (asset_dim, asset_dim, 1)
        attn = attn.unsqueeze(-1)

        # (asset_dim, asset_dim, 1) x (1, asset_dim, latent_dim) -> (asset_dim, asset_dim, latent_dim)
        h = attn * v

        # (asset_dim, asset_dim, latent_dim) -> (asset_dim, latent_dim)
        h = torch.sum(h, dim=1)

        return h


class LSRE_CANN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        latent_dim = cfg['latent_dim']              # Dimension of latents
        self.min_log_std = cfg["min_log_std"]
        self.max_log_std = cfg["max_log_std"]

        # TODO: Add positional encoding (?)

        self.lsre = LSRE(cfg)
        self.cann = CANN(cfg)
        self.out = nn.Linear(latent_dim, 2)

    def forward(self, x):
        ''' ### Forward pass of LSRE_CANN
        Args:
            x (torch.Tensor): Input tensor of shape (asset_dim, window_size, feat_dim)
        Returns:
            mu (torch.Tensor): Mean tensor of shape (asset_dim, 1)
            std (torch.Tensor): Standard deviation tensor of shape (asset_dim, 1)
        '''
        z = self.lsre(x)
        h = self.cann(z)
        logits = self.out(h)
        mu, log_std = torch.chunk(logits, chunks=2, dim=-1) 
        std = torch.exp(torch.clamp(log_std, self.min_log_std, self.max_log_std))
        return mu, std