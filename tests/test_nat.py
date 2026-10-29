"""CPU tests; run with: python -m unittest discover -s tests -p 'test_nat*.py'."""
import copy
import ast
import csv
import importlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import subprocess
import types
import unittest
from unittest import mock

import torch
import yaml
import torch.distributed as dist
from torch import nn

from model_mamp.transformer import Transformer as MAMP
from model_mae.transformer import Transformer as MAE
from model_mamp.transformer_nat import Transformer, normalize_reconstruction_target
from model.transformer import Transformer as Downstream


torch.set_num_threads(1)


def options():
    return dict(dim_feat=16, decoder_dim_feat=16, depth=2, decoder_depth=1,
                num_heads=4, num_frames=8, num_joints=4, t_patch_size=4,
                norm_skes_loss=True)


def inputs():
    return torch.randn(2, 3, 8, 4, 2)


def cl_model(**kwargs):
    settings = dict(contrastive_weight=1.e-4, contrast_queue_size=8,
                    teacher_batch_size=1)
    settings.update(kwargs)
    return Transformer(**settings, **options())


def _ddp_worker(rank, store_path):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file:///' + store_path.replace('\\', '/'),
                            rank=rank, world_size=2)
    try:
        torch.manual_seed(10 + rank)
        model = cl_model()
        wrapped = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        optimizer = torch.optim.SGD(wrapped.parameters(), lr=.01)
        for step in range(2):
            optimizer.zero_grad()
            loss, _, _ = wrapped(inputs(), mask_ratio=.5, motion_aware_tau=0.)
            loss.backward()
            optimizer.step()
            model.after_optimizer_step(True)
            assert model.contrast_queue_count.item() == (step + 1) * 4
            assert model.teacher_updates.item() == step + 1
            if step:
                assert model.pretrain_metrics()['loss_contrastive'].item() > 0
        for value in (model.contrast_queue, model.contrast_queue_ptr,
                      model.ema_teacher.pos_embed):
            gathered = [torch.empty_like(value) for _ in range(2)]
            dist.all_gather(gathered, value)
            assert torch.equal(gathered[0], gathered[1])
    finally:
        dist.destroy_process_group()


class TargetTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_nat_matches_each_joint_coordinate_time_series(self):
        raw = torch.randn(2, 3, 4, 2, 3)
        patches = raw.reshape(2, 3, -1)
        actual = normalize_reconstruction_target(patches, 4, 2, 3, 'nat')
        expected = torch.empty_like(raw)
        for batch in range(2):
            for token in range(3):
                for joint in range(2):
                    for channel in range(3):
                        series = raw[batch, token, :, joint, channel]
                        expected[batch, token, :, joint, channel] = (
                            series - series.mean()) / (series.var() + 1.e-6).sqrt()
        self.assertTrue(torch.allclose(actual, expected.reshape_as(actual)))

    def test_nat_removes_independent_coordinate_offsets(self):
        patches = torch.randn(2, 3, 4, 1, 3)
        shifted = patches + torch.tensor([20., -100., 300.])
        before = normalize_reconstruction_target(patches.reshape(2, 3, 12), 4, 1, 3, 'nat')
        after = normalize_reconstruction_target(shifted.reshape(2, 3, 12), 4, 1, 3, 'nat')
        self.assertTrue(torch.allclose(before, after, atol=5.e-5, rtol=5.e-5))
        old_before = normalize_reconstruction_target(patches.reshape(2, 3, 12), 4, 1, 3, 'patch')
        old_after = normalize_reconstruction_target(shifted.reshape(2, 3, 12), 4, 1, 3, 'patch')
        self.assertFalse(torch.allclose(old_before, old_after))

    def test_constant_and_empty_targets_are_finite_zero(self):
        for value in (torch.ones(2, 3, 12), torch.zeros(2, 3, 12)):
            actual = normalize_reconstruction_target(value, 4, 1, 3, 'nat')
            self.assertTrue(torch.isfinite(actual).all())
            self.assertEqual(actual.abs().max().item(), 0.)

    def test_invalid_nat_geometry_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_reconstruction_target(torch.zeros(1, 1, 3), 1, 1, 3, 'nat')
        with self.assertRaises(ValueError):
            normalize_reconstruction_target(torch.zeros(1, 1, 9), 4, 1, 3, 'nat')

    def test_target_processing_does_not_modify_input(self):
        model = Transformer(**options())
        raw = torch.randn(4, 8, 4, 3)
        saved = raw.clone()
        model.make_reconstruction_target(raw)
        self.assertTrue(torch.equal(raw, saved))

    def test_motion_patch_matches_original_mamp_forward_and_gradients(self):
        original = MAMP(**options())
        candidate = Transformer(reconstruction_target='motion',
                                target_normalization='patch', **options())
        candidate.load_state_dict(original.state_dict(), strict=True)
        data = inputs()
        torch.manual_seed(1)
        reference = original(data, mask_ratio=.5, motion_aware_tau=.8)
        torch.manual_seed(1)
        result = candidate(data, mask_ratio=.5, motion_aware_tau=.8)
        for actual, expected in zip(result, reference):
            self.assertTrue(torch.allclose(actual, expected, atol=1.e-7, rtol=1.e-6))
        reference[0].backward()
        result[0].backward()
        for (name, parameter), (other_name, other_parameter) in zip(
                original.named_parameters(), candidate.named_parameters()):
            self.assertEqual(name, other_name)
            if parameter.grad is not None:
                self.assertTrue(torch.allclose(parameter.grad, other_parameter.grad,
                                               atol=1.e-6, rtol=1.e-5), name)

    def test_joint_patch_matches_original_mae_with_random_masking(self):
        original = MAE(**options())
        candidate = Transformer(target_normalization='patch', **options())
        candidate.load_state_dict(original.state_dict(), strict=True)
        data = inputs()
        torch.manual_seed(2)
        expected = original(data, mask_ratio=.5)
        torch.manual_seed(2)
        actual = candidate(data, mask_ratio=.5, motion_aware_tau=0.)
        for value, reference in zip(actual, expected):
            self.assertTrue(torch.allclose(value, reference, atol=1.e-7, rtol=1.e-6))

    def test_zero_cl_has_no_teacher_or_queue_parameters(self):
        model = Transformer(**options())
        self.assertFalse(hasattr(model, 'ema_teacher'))
        self.assertEqual(set(model.state_dict()), set(MAMP(**options()).state_dict()))


class ContrastiveTests(unittest.TestCase):
    def test_full_teacher_and_person_pooling_with_chunks(self):
        model = cl_model()
        model.train()
        self.assertFalse(model.ema_teacher.training)
        data = inputs()
        flat = data.permute(0, 4, 2, 3, 1).contiguous().reshape(4, 8, 4, 3)
        token_counts = []
        handle = model.ema_teacher.blocks[0].register_forward_pre_hook(
            lambda _, args: token_counts.append(args[0].shape[1]))
        chunked = model.teacher_features(flat, 2, 2)
        handle.remove()
        self.assertEqual(token_counts, [8, 8, 8, 8])
        model.teacher_batch_size = 16
        whole = model.teacher_features(flat, 2, 2)
        self.assertEqual(tuple(chunked.shape), (2, 16))
        self.assertTrue(torch.allclose(chunked, whole, atol=1.e-6, rtol=1.e-5))
        self.assertFalse(chunked.requires_grad)

    def test_cl_uses_cosine_infonce_and_only_encoder_gradients(self):
        model = cl_model(contrast_temperature=.2)
        latent = torch.randn(4, 3, 16, requires_grad=True)
        keys = torch.nn.functional.normalize(torch.randn(2, 16), dim=-1)
        negatives = torch.nn.functional.normalize(torch.randn(5, 16), dim=-1)
        model._enqueue(negatives)
        queries = torch.nn.functional.normalize(latent.reshape(2, 2, 3, 16).mean(dim=(1, 2)), dim=-1)
        scores = torch.cat(((queries * keys).sum(-1, keepdim=True), queries @ negatives.T), 1) / .2
        expected = -(scores[:, 0] - torch.logsumexp(scores, dim=1)).mean()
        actual = model.contrastive_loss(latent, keys, 2, 2)
        self.assertTrue(torch.allclose(actual, expected))
        actual.backward()
        self.assertGreater(latent.grad.abs().sum().item(), 0.)

    def test_forward_backward_freezes_teacher_and_preserves_queue_until_commit(self):
        model = cl_model()
        model._enqueue(torch.nn.functional.normalize(torch.randn(3, 16), dim=-1))
        queue = model.contrast_queue.clone()
        loss, _, _ = model(inputs(), mask_ratio=.5, motion_aware_tau=0.)
        loss.backward()
        self.assertTrue(torch.equal(queue, model.contrast_queue))
        self.assertGreater(model.blocks[0].attn.qkv.weight.grad.abs().sum().item(), 0.)
        self.assertGreater(model.decoder_pred.weight.grad.abs().sum().item(), 0.)
        self.assertTrue(all(p.grad is None for p in model.ema_teacher.parameters()))
        model.after_optimizer_step(True)
        self.assertEqual(model.contrast_queue_count.item(), 5)
        self.assertEqual(model.teacher_updates.item(), 1)

    def test_ema_update_happens_after_successful_optimizer_step(self):
        model = cl_model(ema_momentum=.9)
        before = model.ema_teacher.pos_embed.clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        loss, _, _ = model(inputs(), mask_ratio=.5, motion_aware_tau=0.)
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.equal(before, model.ema_teacher.pos_embed))
        expected = before * .9 + model.pos_embed.detach() * .1
        model.after_optimizer_step(True)
        self.assertTrue(torch.allclose(expected, model.ema_teacher.pos_embed))

    def test_skipped_step_does_not_update_teacher_or_queue(self):
        model = cl_model()
        before = copy.deepcopy(model.state_dict())
        model(inputs(), mask_ratio=.5, motion_aware_tau=0.)
        model.after_optimizer_step(False)
        self.assertFalse(model._pending_keys)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(before[name], value), name)

    def test_accumulation_commits_all_keys_once(self):
        model = cl_model()
        for _ in range(2):
            loss, _, _ = model(inputs(), mask_ratio=.5, motion_aware_tau=0.)
            (loss / 2).backward()
        self.assertEqual(model.contrast_queue_count.item(), 0)
        model.after_optimizer_step(True)
        self.assertEqual(model.contrast_queue_count.item(), 4)
        self.assertEqual(model.teacher_updates.item(), 1)

    def test_epoch_start_discards_incomplete_accumulation_keys(self):
        model = cl_model()
        model(inputs(), mask_ratio=.5, motion_aware_tau=0.)
        self.assertTrue(model._pending_keys)
        model.begin_pretrain_epoch()
        model.after_optimizer_step(True)
        self.assertEqual(model.contrast_queue_count.item(), 0)

    def test_queue_wraparound_and_over_capacity(self):
        model = cl_model(contrast_queue_size=5)
        keys = torch.arange(7 * 16).float().reshape(7, 16)
        model._enqueue(keys[:3])
        model._enqueue(keys[3:6])
        self.assertEqual(model.contrast_queue_ptr.item(), 1)
        self.assertTrue(torch.equal(model.contrast_queue, keys[[5, 1, 2, 3, 4]]))
        model._enqueue(keys)
        self.assertEqual(model.contrast_queue_ptr.item(), 0)
        self.assertTrue(torch.equal(model.contrast_queue, keys[-5:]))

    def test_checkpoint_restores_teacher_queue_and_next_loss(self):
        model = cl_model()
        model(inputs(), mask_ratio=.5, motion_aware_tau=0.)
        model.after_optimizer_step(True)
        restored = cl_model()
        restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
        data = inputs()
        torch.manual_seed(3)
        expected = model(data, mask_ratio=.5, motion_aware_tau=0.)
        torch.manual_seed(3)
        actual = restored(data, mask_ratio=.5, motion_aware_tau=0.)
        self.assertTrue(torch.equal(expected[0], actual[0]))
        self.assertEqual(restored.teacher_updates.item(), 1)

    def test_resume_refuses_task_normalization_or_data_change(self):
        model = cl_model()
        saved = dict(model='model_mamp.transformer_nat.Transformer',
                     model_args=dict(reconstruction_target='joint', target_normalization='nat'),
                     train_feeder_args=dict(data_path='dataset-a.npz'), seed=0)
        model.validate_resume_args(saved, dict(saved))
        for field, replacement in (
                ('model_args', dict(reconstruction_target='motion', target_normalization='patch')),
                ('train_feeder_args', dict(data_path='dataset-b.npz')),
                ('seed', 1)):
            changed = dict(saved)
            changed[field] = replacement
            with self.assertRaises(ValueError):
                model.validate_resume_args(saved, changed)

    def test_native_lp_loads_online_encoder_and_ignores_teacher(self):
        source = cl_model()
        downstream_options = options()
        for name in ('decoder_dim_feat', 'decoder_depth', 'norm_skes_loss'):
            downstream_options.pop(name)
        downstream = Downstream(num_classes=60, **downstream_options)
        result = downstream.load_state_dict(source.state_dict(), strict=False)
        self.assertEqual(set(result.missing_keys), {'head.fc.weight', 'head.fc.bias'})
        self.assertTrue(torch.equal(downstream.pos_embed, source.pos_embed))
        self.assertTrue(any(key.startswith('ema_teacher.') for key in result.unexpected_keys))
        downstream.eval()
        self.assertEqual(tuple(downstream(inputs()).shape), (2, 60))


class _TestScaler:
    def __init__(self, skip=False):
        self._scaler = self
        self.scale = 1.
        self.skip = skip

    def get_scale(self):
        return self.scale

    def __call__(self, loss, optimizer, parameters, update_grad):
        loss.backward()
        if update_grad:
            if self.skip:
                self.scale /= 2
            else:
                optimizer.step()


class EngineTests(unittest.TestCase):
    def run_engine(self, skip, accumulation):
        # The repo targets torch 1.8. Use a test-only shim for torch._six,
        # removed in the newer CPU torch used by this local test environment.
        if not hasattr(torch, '_six'):
            sys.modules.setdefault('torch._six', types.SimpleNamespace(inf=math.inf))
        engine = importlib.import_module('engine_pretrain')
        model = cl_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        args = types.SimpleNamespace(
            accum_iter=accumulation, warmup_epochs=0, lr=.01, min_lr=0.,
            epochs=1, enable_amp=False, mask_ratio=.5, motion_stride=1,
            motion_aware_tau=0.)
        batches = [(inputs(), None, None) for _ in range(accumulation)]
        with mock.patch('torch.cuda.synchronize'):
            stats = engine.train_one_epoch(
                model, batches, optimizer, torch.device('cpu'), 0,
                _TestScaler(skip), args=args)
        self.assertIn('loss_reconstruction', stats)
        self.assertIn('loss_contrastive', stats)
        return model

    def test_real_engine_commits_at_accumulation_boundary(self):
        model = self.run_engine(False, 2)
        self.assertEqual(model.contrast_queue_count.item(), 4)
        self.assertEqual(model.teacher_updates.item(), 1)

    def test_real_engine_does_not_commit_on_scaler_skip(self):
        model = self.run_engine(True, 1)
        self.assertEqual(model.contrast_queue_count.item(), 0)
        self.assertEqual(model.teacher_updates.item(), 0)


class ConfigurationTests(unittest.TestCase):
    def test_all_groups_hold_native_data_backbone_mask_and_schedule_fixed(self):
        root = Path(__file__).resolve().parents[1]
        native = yaml.safe_load((root / 'config/ntu60_xsub_joint/pretrain_mamp_t120_layer8+3_mask90.yaml').read_text())
        parser_tree = ast.parse((root / 'main_pretrain.py').read_text())
        function = next(node for node in parser_tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == 'get_args_parser')
        namespace = dict(argparse=importlib.import_module('argparse'))
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<native-parser>', 'exec'), namespace)
        for name in ('motion_patch', 'joint_patch', 'joint_nat', 'joint_nat_cl'):
            config = yaml.safe_load((root / ('config/ntu60_xsub_joint/nat/pretrain_' + name + '.yaml')).read_text())
            self.assertEqual(config['train_feeder_args'], native['train_feeder_args'])
            for field in ('epochs', 'warmup_epochs', 'batch_size', 'lr', 'min_lr',
                          'weight_decay', 'mask_ratio', 'motion_aware_tau'):
                self.assertEqual(config[field], native[field], (name, field))
            for field, value in native['model_args'].items():
                self.assertEqual(config['model_args'][field], value, (name, field))
            parser = namespace['get_args_parser']()
            self.assertFalse(set(config) - set(vars(parser.parse_args([]))))
            parser.set_defaults(**config)
            args = parser.parse_args([])
            self.assertEqual(args.lr, .001)
            self.assertEqual(args.min_lr, .0005)
            self.assertIs(args.enable_amp, False)
            if name == 'joint_nat_cl':
                self.assertEqual(args.model_args['contrastive_weight'], .0001)
                self.assertEqual(args.model_args['contrast_queue_size'], 65536)

    def test_summary_keeps_supplied_baseline_and_best_and_last_separate(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='mamp_nat_summary_') as directory:
            run = Path(directory)
            log = run / 'joint_nat/lp/log.txt'
            log.parent.mkdir(parents=True)
            log.write_text('\n'.join(json.dumps(record) for record in [
                dict(epoch=0, test_acc1=85.2), dict(epoch=1, test_acc1=84.8)]))
            subprocess.run([sys.executable, str(root / 'tools/summarize_nat_runs.py'),
                            '--root', str(run)], check=True, capture_output=True, text=True)
            with (run / 'summary.csv').open(newline='') as stream:
                rows = {row['experiment']: row for row in csv.DictReader(stream)}
            self.assertEqual(rows['motion_patch']['source'], 'supplied_baseline_logs')
            self.assertAlmostEqual(float(rows['motion_patch']['best_lp_acc1']), 84.940562850234)
            self.assertEqual(rows['joint_nat']['best_lp_acc1'], '85.2')
            self.assertEqual(rows['joint_nat']['last_lp_acc1'], '84.8')
            self.assertEqual(rows['joint_nat_cl']['source'], 'not_run')

    def test_summary_rejects_duplicate_or_missing_epochs(self):
        from tools.summarize_nat_runs import read_log
        with tempfile.TemporaryDirectory(prefix='mamp_nat_logs_') as directory:
            path = Path(directory) / 'log.txt'
            for epochs in ([0, 0], [0, 2]):
                path.write_text('\n'.join(json.dumps(dict(epoch=epoch)) for epoch in epochs))
                with self.assertRaises(ValueError):
                    read_log(path)


@unittest.skipUnless(os.environ.get('NAT_RUN_DDP_TESTS') == '1',
                     'Set NAT_RUN_DDP_TESTS=1 to run the real two-process Gloo test')
class DistributedTests(unittest.TestCase):
    def test_two_rank_queue_and_ema_stay_identical(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory(prefix='mamp_nat_gloo_') as directory:
            store = str(Path(directory) / 'store')
            mp.spawn(_ddp_worker, args=(store,), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
