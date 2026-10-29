"""Pretrained location routing with fake encoders; no downloads or CUDA."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import MagicMock, call, patch
from torch import nn
from inference.models.graspmamba import Grasp_Mamba

class PretrainedLocationTests(unittest.TestCase):
    def test_backbone_loads_model_and_processor_from_selected_locations(self):
        spec = importlib.util.spec_from_file_location('location_test_backbone', Path(__file__).resolve().parents[1] / 'inference/models/mamba/mamba_L_Siglip.py')
        module = importlib.util.module_from_spec(spec)
        transformers = types.ModuleType('transformers')
        transformers.AutoModel = MagicMock()
        transformers.AutoProcessor = MagicMock()
        with patch.dict(sys.modules, {'transformers': transformers}):
            spec.loader.exec_module(module)
        for kwargs, vision, text in [({}, 'nvidia/MambaVision-L-1K', 'google/siglip-base-patch16-224'),
                                    ({'vision_model': '/models/vision snapshot', 'text_model': '/models/text snapshot'}, '/models/vision snapshot', '/models/text snapshot')]:
            with self.subTest(vision=vision):
                transformers.AutoModel.reset_mock()
                transformers.AutoProcessor.reset_mock()
                transformers.AutoModel.from_pretrained.side_effect = [nn.Linear(1,1), nn.Linear(1,1)]
                with patch.object(module,'Head', side_effect=lambda **k: nn.Identity()), patch.object(module,'TextImageFusionBlock', side_effect=lambda *a: nn.Identity()), patch.object(module,'UpscaleBlock', side_effect=lambda *a: nn.Identity()):
                    model = module.GraspMamba(**kwargs)
                self.assertEqual(transformers.AutoModel.from_pretrained.call_args_list, [call(text), call(vision, trust_remote_code=True)])
                transformers.AutoProcessor.from_pretrained.assert_called_once_with(text)
                self.assertTrue(all(not p.requires_grad for p in model.siglip_model.parameters()))

    def test_wrapper_forwards_locations_to_backbone(self):
        module = types.ModuleType('inference.models.mamba.mamba_L_Siglip')
        module.GraspMamba = MagicMock(return_value=nn.Identity())
        with patch.dict(sys.modules, {module.__name__:module}):
            Grasp_Mamba(vision_model='/models/vision snapshot', text_model='/models/text snapshot')
        module.GraspMamba.assert_called_once_with(resolution=224, segmentation=False, vision_model='/models/vision snapshot', text_model='/models/text snapshot')

if __name__ == '__main__':
    unittest.main()
