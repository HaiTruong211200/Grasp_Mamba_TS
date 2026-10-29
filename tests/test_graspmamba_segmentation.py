"""CPU contract tests with tiny encoders; no downloads or CUDA extensions."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch
from torch import nn

from inference.models.graspmamba import Grasp_Mamba
from inference.models.segmentation import SegHead, segmentation_loss
from utils.data.grasp_anything_language_data import GraspAnythingLanguageDataset, language_collate


def tiny_model():
    # Load the actual backbone implementation without requiring transformers.
    module_name = 'inference.models.mamba.mamba_L_Siglip'
    spec = importlib.util.spec_from_file_location(
        module_name, Path(__file__).resolve().parents[1] / 'inference/models/mamba/mamba_L_Siglip.py')
    module = importlib.util.module_from_spec(spec)
    transformers = types.ModuleType('transformers')
    transformers.AutoModel = transformers.AutoProcessor = object
    with patch.dict(sys.modules, {'transformers': transformers}):
        spec.loader.exec_module(module)

    class Text(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1), requires_grad=False)

        def get_text_features(self, input_ids):
            return input_ids.float().repeat(1, 3) * self.weight

    class Vision(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 1)

        def forward(self, x):
            x = self.conv(x)
            return None, [torch.nn.functional.adaptive_avg_pool2d(x, n) for n in (8, 4, 2, 1)]

    class GraspHead(nn.Conv2d):
        def forward(self, x):
            maps = torch.nn.functional.interpolate(super().forward(x), (32, 32))
            return maps.split(1, dim=1)

    backbone = module.GraspMamba.__new__(module.GraspMamba)
    nn.Module.__init__(backbone)
    backbone.siglip_model = Text()
    backbone.siglip_processor = lambda text, **kwargs: {
        'input_ids': torch.tensor([[len(t)] for t in text])}
    backbone.model = Vision()
    for stage in range(1, 5):
        setattr(backbone, f'fusion_stage{stage}', module.TextImageFusionBlock(4, 3, 4))
    for stage in range(1, 4):
        setattr(backbone, f'upscale{stage}', module.UpscaleBlock(4, 4))
    backbone.conv = nn.Conv2d(4, 4, 1)
    backbone.output = GraspHead(4, 4, 1)
    backbone.seg_head = SegHead(4)
    model = Grasp_Mamba.__new__(Grasp_Mamba)
    nn.Module.__init__(model)
    model.grasp_mamba = backbone
    model.segmentation = True
    model.lambda_seg = 0.5
    model.lambda_dice = 1.0
    model.grasp_loss = 'mse'
    return model


class SegmentationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        torch.set_num_threads(1)

    def test_dice_is_per_sample_and_bce_uses_logits(self):
        logits = torch.zeros(2, 1, 2, 2, requires_grad=True)
        target = torch.zeros_like(logits)
        target[0] = 1
        total, bce, dice = segmentation_loss(logits, target)
        expected = ((1 - (4 + 1e-6) / (6 + 1e-6)) +
                    (1 - 1e-6 / (2 + 1e-6))) / 2
        self.assertAlmostEqual(bce.item(), np.log(2), places=6)
        self.assertAlmostEqual(dice.item(), expected, places=6)
        total.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        with self.assertRaises(ValueError):
            segmentation_loss(logits, target + 2)

    def test_segmentation_gradients_and_inference_skip(self):
        model = tiny_model().train()
        self.assertFalse(model.grasp_mamba.siglip_model.training)
        img, text = torch.randn(2, 3, 32, 32), ['cup', 'handle']
        out = model(img, text, return_seg=True)
        self.assertEqual(out[4].shape, (2, 1, 32, 32))
        seg, _, _ = segmentation_loss(out[4], torch.ones_like(out[4]))
        seg.backward()
        self.assertGreater(model.grasp_mamba.model.conv.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.grasp_mamba.fusion_stage1.conv_fusion.weight.grad.abs().sum().item(), 0)
        self.assertIsNone(model.grasp_mamba.siglip_model.weight.grad)
        self.assertIsNone(model.grasp_mamba.output.weight.grad)
        calls = []
        model.eval()
        hook = model.grasp_mamba.seg_head.register_forward_hook(lambda *a: calls.append(1))
        grasp = model(img, text)
        self.assertEqual(calls, [])
        with_seg = model(img, text, return_seg=True)
        hook.remove()
        self.assertEqual(len(calls), 1)
        for a, b in zip(grasp, with_seg[:4]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_total_loss_and_missing_mask(self):
        model = tiny_model()
        batch = {'image': torch.randn(2, 3, 32, 32), 'instruction': ['a', 'b'],
                 'grasp_targets': {k: torch.zeros(2, 1, 32, 32)
                                   for k in ('pos', 'cos', 'sin', 'width')},
                 'target_mask': torch.ones(2, 1, 32, 32)}
        result = model.compute_loss(batch)
        torch.testing.assert_close(result['loss'], result['losses']['grasp_loss'] +
                                   0.5 * result['losses']['seg_loss'])
        del batch['target_mask']
        with self.assertRaises(ValueError):
            model.compute_loss(batch)
        model.segmentation = False
        model.grasp_mamba.seg_head = None
        baseline = model.compute_loss(batch['image'],
                                      list(batch['grasp_targets'].values()), batch['instruction'])
        self.assertEqual(set(baseline['pred']), {'pos', 'cos', 'sin', 'width'})
        self.assertNotIn('seg_loss', baseline['losses'])
        with self.assertRaises(ValueError):
            model(batch['image'], batch['instruction'], return_seg=True)

    def test_head_can_overfit_one_mask(self):
        head = SegHead(4)
        features = torch.randn(1, 4, 8, 8)
        target = (features[:, :1] > 0).float()
        optimizer = torch.optim.Adam(head.parameters(), lr=0.03)
        initial = segmentation_loss(head(features, (8, 8)), target)[0].item()
        for _ in range(80):
            optimizer.zero_grad()
            loss = segmentation_loss(head(features, (8, 8)), target)[0]
            loss.backward()
            optimizer.step()
        self.assertLess(loss.item(), initial * 0.2)

    def test_manifest_geometry_instances_and_missing_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cv2.imwrite(str(root / 'scene_12.jpg'), np.zeros((40, 80, 3), np.uint8))
            torch.save(torch.tensor([[0, 40, 20, 20, 10, 0.0]]), root / 'grasp.pt')
            mask = np.zeros((40, 80), np.int32)
            mask[10:30, 20:60] = 12
            np.save(root / 'mask.npy', mask)
            row = dict(sample_id='scene_12_object_12', scene_id='scene_12',
                       instruction='grasp object twelve', rgb_path='scene_12.jpg',
                       grasp_path='grasp.pt', annotation_height=40, annotation_width=80,
                       object_mask_path='mask.npy', mask_instance_id=12, split='train')
            manifest = root / 'manifest.jsonl'
            manifest.write_text(json.dumps(row), encoding='utf-8')
            dataset = GraspAnythingLanguageDataset(manifest, segmentation=True, output_size=32)
            sample = dataset[0]
            self.assertEqual(sample['target_mask'].sum().item(), 256)
            np.testing.assert_allclose(dataset.get_gtbb(0)[0].center, [16, 16])
            batch = language_collate([sample, sample])
            self.assertEqual(batch['target_mask'].shape, (2, 1, 32, 32))
            self.assertIsInstance(batch['metadata'], list)
            np.save(root / 'mask.npy', np.zeros_like(mask))
            with self.assertRaises(ValueError):
                dataset[0]
            row['object_mask_path'] = 'missing.npy'
            manifest.write_text(json.dumps(row), encoding='utf-8')
            with self.assertRaises(FileNotFoundError):
                GraspAnythingLanguageDataset(manifest, segmentation=True)
            GraspAnythingLanguageDataset(manifest, segmentation=False)
            row2 = dict(row, sample_id='val_copy', split='val')
            manifest.write_text(json.dumps(row) + '\n' + json.dumps(row2), encoding='utf-8')
            with self.assertRaises(ValueError):
                GraspAnythingLanguageDataset(manifest)

    def test_training_update_count_and_batched_validation(self):
        # Logging services are not required to exercise train/validation functions.
        with patch.dict(sys.modules, {'tensorboardX': types.ModuleType('tensorboardX'),
                                      'wandb': types.ModuleType('wandb')}):
            import train_network as trainer
        model = tiny_model()
        batch = {'image': torch.randn(2, 3, 32, 32), 'instruction': ['cup', 'handle'],
                 'grasp_targets': {k: torch.zeros(2, 1, 32, 32)
                                   for k in ('pos', 'cos', 'sin', 'width')},
                 'target_mask': torch.ones(2, 1, 32, 32),
                 'metadata': [{'index': 0}, {'index': 1}]}
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        with patch.object(optimizer, 'step', wraps=optimizer.step) as update:
            result = trainer.train(0, model, 'cpu', [batch], optimizer, 3,
                                   types.SimpleNamespace(lr=0.001), vis=False)
        self.assertEqual(update.call_count, 3)
        self.assertTrue(np.isfinite(result['loss']))
        loader = [batch]

        class Loader(list):
            dataset = types.SimpleNamespace(get_gtbb=lambda index: index)

        with patch.object(trainer.evaluation, 'calculate_iou_match', return_value=True) as match:
            results = trainer.validate(model, 'cpu', Loader(loader), 0.25)
        self.assertEqual(match.call_count, 2)
        self.assertEqual(results['correct'], 2)
        self.assertIn('mask_iou', results)
        with self.assertRaises(ValueError):
            trainer.train(0, model, 'cpu', [], optimizer, 3,
                          types.SimpleNamespace(lr=0.001), vis=False)

    def test_width_decode_round_trip(self):
        from inference.post_process import post_process_output
        for size in (224, 300):
            maps = torch.ones(1, 1, size, size)
            width = post_process_output(maps, maps, maps * 0, maps * 0.5,
                                        width_scale=size / 2)[2]
            np.testing.assert_allclose(width, size / 4)


if __name__ == '__main__':
    unittest.main()
