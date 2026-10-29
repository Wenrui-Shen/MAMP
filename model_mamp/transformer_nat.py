"""Controlled joint/motion reconstruction and NAT + InfoNCE experiments.

NAT follows Gong et al., AAAI 2025, Eq. (5): normalize each coordinate
of each joint over the timestamps inside one temporal patch. The input,
masking, two-person handling, and downstream encoder keys stay native.
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist

from .transformer import Transformer as MAMPTransformer


def normalize_reconstruction_target(patches, temporal_size, spatial_size,
                                    channels, mode, eps=1.e-6):
    """patches: [people_batch, tokens, temporal_size * spatial_size * channels]."""
    if mode == 'none':
        return patches
    if mode == 'patch':
        # Preserve MAMP's reduction, sample variance, and epsilon exactly.
        mean = patches.mean(dim=-1, keepdim=True)
        var = patches.var(dim=-1, keepdim=True)
        return (patches - mean) / (var + eps).sqrt()
    if mode != 'nat':
        raise ValueError('Unknown target normalization: ' + str(mode))
    if temporal_size < 2:
        raise ValueError('NAT requires at least two timestamps per patch')
    batch, tokens, width = patches.shape
    if width != temporal_size * spatial_size * channels:
        raise ValueError('Target width does not match the patch geometry')
    target = patches.reshape(batch, tokens, temporal_size, spatial_size, channels)
    mean = target.mean(dim=2, keepdim=True)
    var = target.var(dim=2, keepdim=True)
    return ((target - mean) / (var + eps).sqrt()).reshape_as(patches)


class FullContextEncoder(nn.Module):
    """Frozen EMA copy of only the online embedding and encoder."""
    def __init__(self, online):
        super().__init__()
        self.joints_embed = copy.deepcopy(online.joints_embed)
        self.blocks = copy.deepcopy(online.blocks)
        self.norm = copy.deepcopy(online.norm)
        self.pos_embed = nn.Parameter(online.pos_embed.detach().clone())
        self.temp_embed = nn.Parameter(online.temp_embed.detach().clone())
        for parameter in self.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        x = self.joints_embed(x)
        batch, temporal, joints, dim = x.shape
        x = x + self.pos_embed[:, :, :joints] + self.temp_embed[:, :temporal]
        x = x.reshape(batch, temporal * joints, dim)
        # Do not call random_masking: the teacher uses every token.
        for block in self.blocks:
            x = block(x)
        return self.norm(x).float().mean(dim=1)

    @torch.no_grad()
    def update_from(self, online, momentum):
        sources = dict(online.named_parameters())
        for name, parameter in self.named_parameters():
            parameter.mul_(momentum).add_(sources[name].detach(), alpha=1. - momentum)


class Transformer(MAMPTransformer):
    def __init__(self, reconstruction_target='joint', target_normalization='nat',
                 contrastive_weight=0., ema_momentum=.999,
                 contrast_temperature=.07, contrast_queue_size=65536,
                 teacher_batch_size=8, **kwargs):
        if reconstruction_target not in ('joint', 'motion'):
            raise ValueError('reconstruction_target must be joint or motion')
        if target_normalization not in ('patch', 'nat', 'none'):
            raise ValueError('target_normalization must be patch, nat, or none')
        for name, value in (('contrastive_weight', contrastive_weight),
                            ('ema_momentum', ema_momentum),
                            ('contrast_temperature', contrast_temperature)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(name + ' must be a finite number')
        if contrastive_weight < 0 or not 0 <= ema_momentum < 1 or contrast_temperature <= 0:
            raise ValueError('Invalid contrastive weight, EMA momentum, or temperature')
        for name, value in (('contrast_queue_size', contrast_queue_size),
                            ('teacher_batch_size', teacher_batch_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(name + ' must be a positive integer')
        if target_normalization == 'nat' and kwargs.get('t_patch_size', 4) < 2:
            raise ValueError('NAT requires t_patch_size >= 2')
        kwargs['norm_skes_loss'] = target_normalization != 'none'
        super().__init__(**kwargs)
        self.reconstruction_target = reconstruction_target
        self.target_normalization = target_normalization
        self.contrastive_weight = contrastive_weight
        self.ema_momentum = ema_momentum
        self.contrast_temperature = contrast_temperature
        self.teacher_batch_size = teacher_batch_size
        self._pending_keys = []
        self._metrics = {}
        if contrastive_weight > 0:
            self.ema_teacher = FullContextEncoder(self)
            self.ema_teacher.eval()
            # Paper does not specify queue initialization: start empty and fill
            # only with real full-context features, without random negatives.
            self.register_buffer('contrast_queue', torch.zeros(contrast_queue_size, self.dim_feat))
            self.register_buffer('contrast_queue_ptr', torch.zeros((), dtype=torch.long))
            self.register_buffer('contrast_queue_count', torch.zeros((), dtype=torch.long))
            self.register_buffer('teacher_updates', torch.zeros((), dtype=torch.long))

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, 'ema_teacher'):
            self.ema_teacher.eval()
        return self

    def begin_pretrain_epoch(self):
        # Native MAMP discards incomplete accumulation windows at epoch end.
        self._pending_keys.clear()

    def pretrain_metrics(self):
        return self._metrics

    def make_reconstruction_target(self, x, motion_stride=1):
        if self.reconstruction_target == 'motion':
            x = self.extract_motion(x, motion_stride)
        return normalize_reconstruction_target(
            self.patchify(x), self.t_patch_size, self.patch_size,
            self.joints_embed.proj.in_channels, self.target_normalization)

    @torch.no_grad()
    def teacher_features(self, x, videos, people):
        # Process full contexts before the online forward. With LayerNorm and
        # an eval-mode teacher, chunking preserves the per-video features.
        features = torch.cat([
            self.ema_teacher(chunk) for chunk in x.split(self.teacher_batch_size)
        ], dim=0)
        # One key per video. Pool both people just as the native LP head does;
        # do not turn empty-person padding into separate contrastive instances.
        features = features.reshape(videos, people, -1).mean(dim=1)
        return F.normalize(features, dim=-1, eps=1.e-6)

    def contrastive_loss(self, latent, keys, videos, people):
        with torch.cuda.amp.autocast(enabled=False):
            queries = latent.float().reshape(videos, people, latent.shape[1], -1)
            queries = F.normalize(queries.mean(dim=(1, 2)), dim=-1, eps=1.e-6)
            positive = (queries * keys.detach()).sum(dim=-1, keepdim=True)
            count = int(self.contrast_queue_count.item())
            negatives = queries @ self.contrast_queue[:count].detach().T
            logits = torch.cat((positive, negatives), dim=1) / self.contrast_temperature
            labels = torch.zeros(videos, device=queries.device, dtype=torch.long)
            return F.cross_entropy(logits, labels)

    @torch.no_grad()
    def _enqueue(self, keys):
        capacity = self.contrast_queue.shape[0]
        if keys.shape[0] >= capacity:
            self.contrast_queue.copy_(keys[-capacity:])
            self.contrast_queue_ptr.zero_()
            self.contrast_queue_count.fill_(capacity)
            return
        pointer = int(self.contrast_queue_ptr.item())
        first = min(capacity - pointer, keys.shape[0])
        self.contrast_queue[pointer:pointer + first].copy_(keys[:first])
        if first < keys.shape[0]:
            self.contrast_queue[:keys.shape[0] - first].copy_(keys[first:])
        self.contrast_queue_ptr.fill_((pointer + keys.shape[0]) % capacity)
        self.contrast_queue_count.fill_(min(
            capacity, int(self.contrast_queue_count.item()) + keys.shape[0]))

    @torch.no_grad()
    def after_optimizer_step(self, successful):
        if not self.contrastive_weight or not self._pending_keys:
            return
        keys = torch.cat(self._pending_keys, dim=0)
        self._pending_keys.clear()
        if not successful:
            return
        if dist.is_available() and dist.is_initialized():
            gathered = [torch.empty_like(keys) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, keys)
            keys = torch.cat(gathered, dim=0)
        # Mutate the queue after backward, avoiding autograd version errors.
        self._enqueue(keys)
        self.ema_teacher.update_from(self, self.ema_momentum)
        self.teacher_updates.add_(1)

    def validate_resume_args(self, saved_args, current_args):
        """Changing an ablation definition requires a new run, not --resume."""
        if saved_args is None:
            raise ValueError('NAT resume requires checkpoint experiment arguments')
        saved = vars(saved_args) if not isinstance(saved_args, dict) else saved_args
        current = vars(current_args) if not isinstance(current_args, dict) else current_args
        fields = ('model', 'model_args', 'train_feeder_args', 'mask_ratio',
                  'motion_stride', 'motion_aware_tau', 'batch_size', 'accum_iter',
                  'world_size', 'epochs', 'warmup_epochs', 'lr', 'min_lr',
                  'weight_decay', 'enable_amp', 'seed')
        differences = [field for field in fields if saved.get(field) != current.get(field)]
        if differences:
            raise ValueError('NAT resume definition changed: ' + ', '.join(differences))

    def forward(self, x, mask_ratio=.9, motion_stride=1, motion_aware_tau=.8, **kwargs):
        if not 0 < mask_ratio < 1:
            raise ValueError('Reconstruction requires 0 < mask_ratio < 1')
        videos, channels, frames, joints, people = x.shape
        x = x.permute(0, 4, 2, 3, 1).contiguous().view(videos * people, frames, joints, channels)
        keys = self.teacher_features(x, videos, people) if self.contrastive_weight else None
        target = self.make_reconstruction_target(x, motion_stride)
        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio, motion_aware_tau)
        prediction = self.forward_decoder(latent, ids_restore)
        reconstruction = ((prediction - target) ** 2).mean(dim=-1)
        reconstruction = (reconstruction * mask).sum() / mask.sum()
        contrastive = reconstruction.new_zeros(())
        if self.contrastive_weight:
            contrastive = self.contrastive_loss(latent, keys, videos, people)
            if self.training:
                self._pending_keys.append(keys.detach())
        loss = reconstruction + self.contrastive_weight * contrastive
        self._metrics = dict(loss_reconstruction=reconstruction.detach(),
                             loss_contrastive=contrastive.detach())
        if self.contrastive_weight:
            self._metrics['contrast_queue_count'] = self.contrast_queue_count.detach().clone()
        return loss, prediction, mask
