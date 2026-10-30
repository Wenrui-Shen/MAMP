"""AMR first step: cross-attention decoder with the existing NAT target/loss."""
import torch
from torch import nn

from .drop import DropPath
from .transformer import MLP, trunc_normal_
from .transformer_nat import Transformer as NATTransformer


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0., drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, query, memory):
        batch, queries, dim = query.shape
        tokens = memory.shape[1]
        q = self.q(query).reshape(
            batch, queries, self.num_heads, dim // self.num_heads
        ).permute(0, 2, 1, 3)
        kv = self.kv(memory).reshape(
            batch, tokens, 2, self.num_heads, dim // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attention = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        attention = self.attn_drop(attention)
        output = (attention @ v).transpose(1, 2).reshape(batch, queries, dim)
        return self.proj_drop(self.proj(output))


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.memory_norm = norm_layer(dim)
        self.attn = CrossAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, query, memory):
        query = query + self.drop_path(
            self.attn(self.norm1(query), self.memory_norm(memory)))
        return query + self.drop_path(self.mlp(self.norm2(query)))


class Transformer(NATTransformer):
    def __init__(self, contrast_queue_size=8192, **kwargs):
        super().__init__(contrast_queue_size=contrast_queue_size, **kwargs)
        dim = self.decoder_embed.out_features
        drop_rates = torch.linspace(
            0, kwargs.get('drop_path_rate', 0.), len(self.blocks)).tolist()
        self.decoder_blocks = nn.ModuleList([
            CrossAttentionBlock(
                dim, num_heads=kwargs.get('num_heads', 8),
                mlp_ratio=kwargs.get('mlp_ratio', 4.),
                qkv_bias=kwargs.get('qkv_bias', True),
                drop=kwargs.get('drop_rate', 0.),
                attn_drop=kwargs.get('attn_drop_rate', 0.),
                drop_path=drop_rates[index],
                norm_layer=kwargs.get('norm_layer', nn.LayerNorm))
            for index in range(len(self.decoder_blocks))
        ])
        self.decoder_blocks.apply(self._init_weights)
        self.decoder_query = nn.Parameter(
            torch.zeros(1, self.joints_embed.num_patches, dim))
        trunc_normal_(self.decoder_query, std=.02)
        del self.mask_token

    def forward_decoder(self, latent, ids_restore):
        # Memory remains connected to the encoder for reconstruction gradients.
        memory = self.decoder_embed(latent)
        position = (self.decoder_temp_embed + self.decoder_pos_embed).flatten(1, 2)
        # Queries already follow the target's temporal/joint order.
        query = (self.decoder_query + position).expand(memory.shape[0], -1, -1)
        for block in self.decoder_blocks:
            query = block(query, memory)
        return self.decoder_pred(self.decoder_norm(query))
