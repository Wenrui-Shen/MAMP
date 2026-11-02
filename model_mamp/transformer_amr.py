"""Cross-attention decoding with independently sized NAT reconstruction targets."""
import torch
from torch import nn

from .drop import DropPath
from .transformer import MLP, trunc_normal_
from .transformer_nat import Transformer as NATTransformer, normalize_reconstruction_target


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
    def __init__(self, contrast_queue_size=8192, decoder_t_patch_size=None,
                 loss_on_all_patches=False, focal_reconstruction=False,
                 focal_window_size=4, focal_k=.01, cross_person_decoder=False, **kwargs):
        super().__init__(contrast_queue_size=contrast_queue_size, **kwargs)
        self.decoder_t_patch_size = (
            self.t_patch_size if decoder_t_patch_size is None else decoder_t_patch_size)
        self.loss_on_all_patches = loss_on_all_patches
        self.focal_reconstruction = focal_reconstruction
        self.focal_window_size = focal_window_size
        self.focal_k = focal_k
        self.cross_person_decoder = cross_person_decoder
        if cross_person_decoder and len(self.decoder_blocks) != 3:
            raise ValueError('Cross-person decoding requires three decoder layers')
        if focal_reconstruction and not loss_on_all_patches:
            raise ValueError('Focal reconstruction requires loss_on_all_patches=True')
        if focal_reconstruction and self.num_frames % focal_window_size:
            raise ValueError('num_frames must be divisible by focal_window_size')
        if self.num_frames % self.decoder_t_patch_size:
            raise ValueError('num_frames must be divisible by decoder_t_patch_size')
        dim = self.decoder_embed.out_features
        target_temporal = self.num_frames // self.decoder_t_patch_size
        if self.decoder_t_patch_size != self.t_patch_size:
            self.decoder_temp_embed = nn.Parameter(torch.zeros(1, target_temporal, 1, dim))
            trunc_normal_(self.decoder_temp_embed, std=.02)
            self.decoder_pred = nn.Linear(
                dim, self.decoder_t_patch_size * self.patch_size * self.joints_embed.proj.in_channels)
            self.decoder_pred.apply(self._init_weights)
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
            torch.zeros(1, target_temporal * self.joints_embed.grid_size, dim))
        trunc_normal_(self.decoder_query, std=.02)
        del self.mask_token

    def patchify_target(self, x):
        batch, frames, joints, channels = x.shape
        temporal = self.decoder_t_patch_size
        spatial = self.patch_size
        target = x.reshape(
            batch, frames // temporal, temporal, joints // spatial, spatial, channels)
        return target.permute(0, 1, 3, 2, 4, 5).reshape(
            batch, (frames // temporal) * (joints // spatial), temporal * spatial * channels)

    def make_reconstruction_target(self, x, motion_stride=1):
        if self.reconstruction_target == 'motion':
            x = self.extract_motion(x, motion_stride)
        return normalize_reconstruction_target(
            self.patchify_target(x), self.decoder_t_patch_size, self.patch_size,
            x.shape[-1], self.target_normalization)

    @torch.no_grad()
    def focal_weights(self, x):
        # AMR Eq. (3): population variance summed over xyz, before NAT.
        batch, frames, joints, channels = x.shape
        window_size = self.focal_window_size
        windows = x.float().reshape(batch, frames // window_size, window_size, joints, channels)
        centered = windows - windows.mean(dim=2, keepdim=True)
        energy = centered.square().sum(dim=-1).mean(dim=2)
        threshold = energy.mean(dim=(1, 2), keepdim=True)
        weight = (energy + 1.e-6) / (energy + 1.e-6 + self.focal_k * threshold)
        weight = weight.repeat_interleave(window_size, dim=1).unsqueeze(-1).expand_as(x)
        return self.patchify_target(weight)

    def reconstruction_loss(self, x, prediction, target, mask):
        if self.loss_on_all_patches:
            # Large targets mix visible and masked encoder patches.
            error = (prediction - target) ** 2
            if self.focal_reconstruction:
                error = error * self.focal_weights(x)
            return error.mean()
        return super().reconstruction_loss(x, prediction, target, mask)

    def decode_reconstruction(self, latent, ids_restore, valid_people):
        partner_index = None
        if self.cross_person_decoder:
            if valid_people.shape[1] != 2:
                raise ValueError('Cross-person decoding requires two person slots per video')
            # Map original slots to packed rows before looking up the other person.
            packed = torch.full(
                valid_people.shape, -1, dtype=torch.long, device=latent.device)
            own_index = torch.arange(latent.shape[0], device=latent.device)
            packed[valid_people] = own_index
            partner_index = packed.flip([1])[valid_people]
            # A single valid person keeps its own memory in the middle layer.
            partner_index = torch.where(partner_index >= 0, partner_index, own_index)
        return self.forward_decoder(latent, ids_restore, partner_index)

    def forward_decoder(self, latent, ids_restore, partner_index=None):
        # Memory remains connected to the encoder for reconstruction gradients.
        memory = self.decoder_embed(latent)
        middle_memory = memory if partner_index is None else memory.index_select(0, partner_index)
        position = (self.decoder_temp_embed + self.decoder_pos_embed).flatten(1, 2)
        # Queries already follow the target's temporal/joint order.
        query = (self.decoder_query + position).expand(memory.shape[0], -1, -1)
        for index, block in enumerate(self.decoder_blocks):
            query = block(query, middle_memory if index == 1 else memory)
        return self.decoder_pred(self.decoder_norm(query))
