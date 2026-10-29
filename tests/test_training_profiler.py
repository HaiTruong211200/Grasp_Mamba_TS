"""CPU correctness checks for mask rotation fast path and profiling output."""
import importlib.util
from pathlib import Path
import types
import unittest
from unittest.mock import patch
import numpy as np
from skimage.transform import rotate
import torch
from utils.data.grasp_anywhere_data import _rotate_part_mask

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('training_profiler',ROOT/'tools/profile_training.py')
profiler=importlib.util.module_from_spec(spec)
spec.loader.exec_module(profiler)

class TrainingProfilerTests(unittest.TestCase):
    def test_quarter_turn_matches_previous_mask_pixels_exactly(self):
        generator=np.random.default_rng(123)
        for size in (31,32,415,416):
            target=(generator.random((size,size))>.6).astype(np.float32)
            for quarter in (0,1,2,3):
                angle=quarter*np.pi/2
                expected=rotate(target,angle/np.pi*180,order=0,mode='symmetric',preserve_range=True)
                np.testing.assert_array_equal(_rotate_part_mask(target,angle),expected)
    def test_other_angles_and_rectangles_keep_original_rotation(self):
        target=np.eye(31,32,dtype=np.float32)
        for angle in (.27,np.pi/2):
            expected=rotate(target,angle/np.pi*180,order=0,mode='symmetric',preserve_range=True)
            np.testing.assert_array_equal(_rotate_part_mask(target,angle),expected)
    def test_summary_reports_mean_and_median(self):
        result=profiler.summarize([{'total':1.,'data_wait':.2},{'total':3.,'data_wait':.4}])
        self.assertEqual(result['total'],{'mean_ms':2.,'median_ms':2.})
        self.assertAlmostEqual(result['data_wait']['mean_ms'],.3)
    def test_data_only_comparison_does_not_need_cuda_or_model(self):
        class Data(torch.utils.data.Dataset):
            segmentation=False
            visited=[]
            def __len__(self):
                return 6
            def __getitem__(self,index):
                self.visited.append(self.segmentation)
                base=(torch.zeros(3,4,4),[torch.zeros(1,4,4) for _ in range(4)],index,0.,1.,'','cup')
                return (*base,torch.ones(1,4,4)) if self.segmentation else base
        args=types.SimpleNamespace(mode='both',batch_size=2,num_workers=0,warmup=1,batches=2)
        dataset=Data()
        with patch('torch.cuda.synchronize',side_effect=AssertionError('Unexpected CUDA call')):
            result=profiler.measure(args,dataset,list(range(6)),None,torch.device('cpu'))
        self.assertEqual(set(result),{'baseline','segmentation'})
        self.assertEqual(dataset.visited,[False]*6+[True]*6)
        self.assertEqual(set(result['baseline']['timings']),{'data_wait','total'})

if __name__=='__main__':
    unittest.main()
