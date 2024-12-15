'''
Heavily inspired by https://github.com/jiahaoli57/LSRE-CAAN/blob/main/LSRE_CAAN.py#L12
'''

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' Forward pass of the Feed Forward module
        params:
            x: input tensor of shape (num_assets, num_latents, in_dim)
        returns:
            out: output tensor of shape (num_assets, num_latents, out_dim)
        '''
        return self.fc2(torch.relu(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, hidden_dim, q_dim, kv_dim):
        super().__init__()
        self.num_heads = num_heads
        self.scale = head_dim ** -0.5

        self.q_linear = nn.Linear(q_dim, hidden_dim, bias=False)
        self.kv_linear = nn.Linear(kv_dim, hidden_dim * 2, bias=False)
        self.out_linear = nn.Linear(hidden_dim, q_dim)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        ''' Forward pass of the Attention module
        params:
            q: query tensor of shape (num_assets, num_latents, q_dim)
            kv: key-value tensor of shape (num_assets, window_size, kv_dim)
        returns:
            out: output tensor of shape (num_assets, num_latents, q_dim)
        '''
        query = self.q_linear(q)
        key, value = torch.chunk(self.kv_linear(kv), chunks=2, dim=-1)

        query = query.reshape(*query.shape[:-1], self.num_heads, -1).permute(0, 2, 1, 3)
        key = key.reshape(*key.shape[:-1], self.num_heads, -1).permute(0, 2, 3, 1)
        value = value.reshape(*value.shape[:-1], self.num_heads, -1).permute(0, 2, 1, 3)

        scores = torch.einsum('bhid,bhjd->bhij', query, key) * self.scale

        # from https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        if is_causal:
            mask = torch.ones(q.shape[-2], kv.shape[-2], dtype=torch.bool).tril(diagonal=0)
            scores.masked_fill_(mask.logical_not(), float('-inf'))
            scores.to(query.dtype)

        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, value).reshape(*query.shape[:-2], -1)
        return self.out_linear(out)
    

class AttentionBlock(nn.Module):
    def __init__(self, num_heads, head_dim, q_dim, kv_dim=None):
        super().__init__()
        if kv_dim is None: kv_dim = q_dim
        hidden_dim = num_heads * head_dim

        self.attn = Attention(num_heads, head_dim, hidden_dim, q_dim, kv_dim)
        self.attn_norm = nn.LayerNorm(q_dim)
        self.ff = FeedForward(q_dim, hidden_dim, q_dim)
        self.ff_norm = nn.LayerNorm(q_dim)

    def forward(self, q: torch.Tensor, kv: torch.Tensor = None) -> torch.Tensor:
        ''' Forward pass of the AttentionBlock
        params:
            q: query tensor of shape (num_assets, num_latents, latent_dim)
            kv (optional): key-value tensor of shape (num_assets, window_size, num_features)
        returns:
            z: latent representation tensor of shape (num_assets, num_latents, latent_dim)
        '''
        if kv is None: kv = q
        z = self.attn_norm(q + self.attn(q, kv))
        z = self.ff_norm(z + self.ff(z))
        return z


class LSRE(nn.Module):
    def __init__(self, depth, feature_dim, 
                 num_latents, latent_dim, 
                 num_cross_heads, cross_head_dim, 
                 num_self_heads, self_head_dim):
        super().__init__()
        self.z = nn.Parameter(torch.randn(num_latents, latent_dim))   # (M, D) = (1, 32)
        self.cross_attn = AttentionBlock(num_cross_heads, cross_head_dim, latent_dim, feature_dim)
        self.self_attns = nn.ModuleList([AttentionBlock(num_self_heads, self_head_dim, latent_dim) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' Forward pass of the LSREBlock
        params:
            x: input tensor of shape (num_assets, window_size, num_features)
        returns:
            z: latent representation tensor of shape (num_assets, latent_dim)
        '''
        # NOTE: may need causal mask somewhere here
        self.z = self.cross_attn(self.z, x)
        for self_attn in self.self_attns:
            z = self_attn(z)
        return z.squeeze(1) # (m, M, D) -> (m, D) since M=1 in this case


class CANN(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.scale = latent_dim ** -0.5
        
        self.q_linear = nn.Linear(latent_dim, latent_dim)
        self.k_linear = nn.Linear(latent_dim, latent_dim)
        self.v_linear = nn.Linear(latent_dim, latent_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        ''' Forward pass of the CANNBlock
        params:
            z: latent representation tensor of shape (num_assets, latent_dim)
        returns:
            h: representation tensor of shape (num_assets, latent_dim)
        '''
        q = self.q_linear(z)
        k = self.k_linear(z)
        v = self.v_linear(z)

        scores = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = torch.softmax(scores, dim=-1)
        h = torch.einsum('bhij,bhjd->bhid', attn, v).reshape(*z.shape[:-2], -1)
        return h


class LSRE_CANN(nn.Module):
    def __init__(self, **cfg):
        super().__init__()
        depth = cfg.get('depth', 1)
        feature_dim = cfg['num_features']
        window_size = cfg['window_size']

        num_latents = cfg.get('num_latents', 1)
        latent_dim = cfg.get('latent_dim', 32)

        n_cross_heads = cfg.get('n_cross_heads', 1)
        cross_dim = cfg.get('cross_head_dim', 64)

        n_self_heads = cfg.get('n_self_heads', 1)
        self_dim = cfg.get('self_head_dim', 32)

        self.pos_emb = nn.Embedding(window_size, feature_dim)

        self.lsre = LSRE(depth, feature_dim, num_latents, latent_dim, n_cross_heads, cross_dim, n_self_heads, self_dim)
        self.cann = CANN(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' Forward pass of the LSRE-CANN model
        params:
            x: input tensor of shape (num_assets, window_size, num_features)
        returns:
            h: representation tensor of shape (num_assets, latent_dim)
        '''
        pos_emb = self.pos_emb(torch.arange(x.shape[1], device=x.device)).unsqueeze(0)
        x = x + pos_emb
        z = self.lsre(x)
        h = self.cann(z)
        return h
    