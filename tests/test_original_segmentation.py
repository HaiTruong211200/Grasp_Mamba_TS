"""CPU checks for optional segmentation on the original ++ loader, without CUDA/downloads."""
import ast
import copy
import io
import logging
import math
from pathlib import Path
import pickle
import types
import unittest
from unittest.mock import patch
import tempfile
import numpy as np
from PIL import Image
from skimage.transform import resize
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from inference.models.segmentation import SegHead, segmentation_loss
from utils.data.grasp_anywhere_data import GraspAnywhereDataset
from utils.data.language_grasp_data import LanguageGraspDatasetBase
from inference.post_process import post_process_output

ROOT=Path(__file__).resolve().parents[1]
REAL_CONV=nn.Conv2d

def model_classes():
    ns={'torch':torch,'nn':nn,'F':F,'math':math,'SegHead':lambda channels: SegHead(4)}
    text=(ROOT/'inference/models/mamba/mamba_L_Siglip.py').read_text()
    nodes=[n for n in ast.parse(text).body if isinstance(n,ast.ClassDef)]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<actual backbone classes>','exec'),ns)
    fusion,up=ns['TextImageFusionBlock'],ns['UpscaleBlock']
    ns['TextImageFusionBlock']=lambda *a: fusion(4,3,4)
    ns['UpscaleBlock']=lambda *a: up(4,4)
    class Text(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight=nn.Parameter(torch.ones(1))
            self.config=types.SimpleNamespace(model_type='siglip', text_config=types.SimpleNamespace(hidden_size=3))
        def get_text_features(self,input_ids):
            return input_ids.float().repeat(1,3)*self.weight
    class Vision(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv=REAL_CONV(3,4,1)
        def cuda(self):
            return self
        def forward(self,x):
            x=self.conv(x)
            return None,[F.adaptive_avg_pool2d(x,n) for n in (8,4,2,1)]
    class Head(nn.Module):
        def __init__(self,**kw):
            super().__init__()
            self.conv=REAL_CONV(4,4,1)
        def forward(self,x):
            return F.interpolate(self.conv(x),size=(32,32)).split(1,dim=1)
    ns['Head']=Head
    ns['AutoModel']=types.SimpleNamespace(from_pretrained=lambda name,**kw: Text() if 'siglip' in name else Vision())
    ns['AutoProcessor']=types.SimpleNamespace(from_pretrained=lambda *a,**kw: lambda text,**kwargs: {'input_ids':torch.tensor([[len(t)] for t in text])})
    wrapper_ns={'torch':torch,'nn':nn,'F':F,'math':math,'segmentation_loss':segmentation_loss,'GraspMamba':ns['GraspMamba']}
    nodes=[n for n in ast.parse((ROOT/'inference/models/graspmamba.py').read_text()).body if isinstance(n,ast.ClassDef)]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<actual wrapper>','exec'),wrapper_ns)
    return wrapper_ns['Grasp_Mamba']

def tiny_model(segmentation=False,lambda_dice=1):
    cls=model_classes()
    def conv(a,b,*args,**kwargs):
        if (a,b)==(96,1024):
            a,b=4,4
        return REAL_CONV(a,b,*args,**kwargs)
    with patch.object(nn,'Conv2d',side_effect=conv):
        return cls(input_size=32,segmentation=segmentation,lambda_dice=lambda_dice)

def function(name):
    text=(ROOT/'train_network.py').read_text()
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
    evaluation=types.SimpleNamespace(calculate_iou_match=lambda *a,**kw: kw['no_grasps']==15)
    ns={'torch':torch,'np':np,'logging':logging,'tqdm':tqdm,'sys':types.SimpleNamespace(stdout=io.StringIO()),'post_process_output':post_process_output,'evaluation':evaluation}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual trainer>','exec'),ns)
    return ns[name]

class OriginalSegmentationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(8)
    def test_optional_head_preserves_initial_grasp_weights_and_rng(self):
        torch.manual_seed(123)
        baseline=tiny_model(False)
        state=torch.get_rng_state()
        torch.manual_seed(123)
        seg=tiny_model(True)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        for key,value in baseline.state_dict().items():
            self.assertTrue(torch.equal(value,seg.state_dict()[key]),key)
        self.assertIsNone(baseline.grasp_mamba.seg_head)
        self.assertTrue(any('seg_head' in k for k in seg.state_dict()))
        x=torch.randn(2,3,32,32)
        for a,b in zip(baseline(x,['cup','cup']),seg(x,['cup','cup'])):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
    def test_segmentation_loss_gradients_and_default_inference_skip(self):
        model=tiny_model(True).train()
        self.assertTrue(model.grasp_mamba.siglip_model.training) # Preserve original mode behavior.
        x=torch.randn(2,3,32,32)
        targets=[torch.zeros(2,1,32,32) for _ in range(4)]
        mask=torch.ones(2,1,32,32)
        result=model.compute_loss(x,targets,['cup','handle'],target_mask=mask)
        torch.testing.assert_close(result['loss'],result['losses']['grasp_loss']+.5*result['losses']['seg_loss'])
        result['loss'].backward()
        self.assertGreater(model.grasp_mamba.seg_head.layers[0].weight.grad.abs().sum().item(),0)
        self.assertGreater(model.grasp_mamba.model.conv.weight.grad.abs().sum().item(),0)
        self.assertIsNone(model.grasp_mamba.siglip_model.weight.grad)
        called=[]
        hook=model.grasp_mamba.seg_head.register_forward_hook(lambda *a: called.append(1))
        self.assertEqual(len(model(x,['cup','handle'])),4)
        self.assertEqual(called,[])
        hook.remove()
        with self.assertRaises(ValueError):
            model.compute_loss(x,targets,['cup','handle'])
    def test_segmentation_gradient_does_not_use_grasp_head(self):
        model=tiny_model(True)
        x=torch.randn(1,3,32,32)
        output=model(x,['cup'],return_seg=True)
        seg,_,_=segmentation_loss(output[4],torch.ones_like(output[4]))
        seg.backward()
        self.assertIsNone(model.grasp_mamba.output.conv.weight.grad)
        self.assertGreater(model.grasp_mamba.fusion_stage1.conv_fusion.weight.grad.abs().sum().item(),0)
    def test_dice_zero_keeps_only_bce_in_segmentation_term(self):
        model=tiny_model(True,lambda_dice=0)
        result=model.compute_loss(torch.randn(1,3,32,32),[torch.zeros(1,1,32,32) for _ in range(4)],['cup'],torch.ones(1,1,32,32))
        torch.testing.assert_close(result['losses']['seg_loss'],result['losses']['seg_bce'])
    def dataset(self,folder,segmentation):
        scene='a'*64
        for name in ('image','grasp_label_positive','grasp_instructions','part_mask'):
            (folder/name).mkdir(exist_ok=True)
        target=np.zeros((32,32),dtype=np.uint8)
        target[7:18,18:27]=1
        Image.fromarray(np.repeat((target*255)[...,None],3,axis=2)).save(folder/'image'/f'{scene}.jpg',quality=100,subsampling=0)
        np.save(folder/'part_mask'/f'{scene}_0_0.npy',target)
        (folder/'grasp_instructions'/f'{scene}_0_0.pkl').write_bytes(pickle.dumps('Lift the cup by its handle.'))
        ds=GraspAnywhereDataset.__new__(GraspAnywhereDataset)
        LanguageGraspDatasetBase.__init__(ds,output_size=32,include_rgb=True,include_depth=False,random_rotate=True,random_zoom=True)
        ds.segmentation=segmentation
        ds.text_source='instruction'
        ds.rgb_dir=str(folder/'image')
        ds.part_mask_dir=str(folder/'part_mask')
        ds.instruction_dir=str(folder/'grasp_instructions')
        ds.prompt_dir=str(folder/'no_scene_description')
        ds.grasp_files=[(folder/'grasp_label_positive'/f'{scene}_0_0.pt').as_posix()]
        ds.get_gtbb=lambda *a,**kw: types.SimpleNamespace(draw=lambda shape:(np.ones(shape),np.zeros(shape),np.ones(shape)))
        return ds,target
    def test_mask_uses_same_rotation_zoom_and_original_sample_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            ds,target=self.dataset(Path(temp),True)
            with patch('random.choice',return_value=np.pi/2),patch('numpy.random.uniform',return_value=.75):
                sample=ds[0]
            ds.segmentation=False
            with patch.object(ds,'get_target_mask',side_effect=AssertionError('baseline must not read masks')),patch('random.choice',return_value=np.pi/2),patch('numpy.random.uniform',return_value=.75):
                baseline=ds[0]
            self.assertEqual(len(baseline),7)
            self.assertEqual(len(sample),8)
            self.assertTrue(torch.equal(sample[0],baseline[0]))
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(sample[1],baseline[1])))
            self.assertEqual(sample[2:7],baseline[2:7])
            expected=resize(np.rot90(target)[4:28,4:28],(32,32),order=0,anti_aliasing=False,preserve_range=True)
            np.testing.assert_array_equal(sample[7][0].numpy(),expected)
            rgb=ds.get_rgb(0,rot=np.pi/2,zoom=.75,normalise=False)[...,0]>127
            intersection=(rgb & (expected>0)).sum()
            self.assertGreater(intersection/(rgb | (expected>0)).sum(),.95)
    def test_mask_failures_and_empty_augmented_crop(self):
        with tempfile.TemporaryDirectory() as temp:
            ds,target=self.dataset(Path(temp),True)
            file=Path(ds.part_mask_dir)/(Path(ds.grasp_files[0]).stem+'.npy')
            for bad in (np.zeros((32,32)),np.full((32,32),2),np.ones((16,16)),np.full((32,32),np.nan)):
                np.save(file,bad)
                with self.assertRaises(ValueError):
                    ds.get_target_mask(0)
            target[:]=0
            target[:2,:2]=1
            np.save(file,target)
            self.assertEqual(ds.get_target_mask(0,zoom=.5).sum().item(),0)
            file.unlink()
            with self.assertRaises(FileNotFoundError):
                ds.get_target_mask(0)
    def test_original_update_count_and_segmentation_validation(self):
        model=tiny_model(True)
        x=torch.randn(2,3,32,32)
        y=[torch.zeros(2,1,32,32) for _ in range(4)]
        batch=(x,y,torch.tensor([0,1]),torch.zeros(2),torch.ones(2),['',''],['cup','handle'],torch.ones(2,1,32,32))
        optimizer=torch.optim.SGD(model.parameters(),lr=.001)
        steps=[]
        hook=optimizer.register_step_post_hook(lambda *a: steps.append(1))
        result=function('train')(0,model,torch.device('cpu'),[batch],optimizer,4,types.SimpleNamespace(lr=.001),vis=False)
        hook.remove()
        self.assertEqual(len(steps),3)
        self.assertTrue(math.isfinite(result['loss']))
        self.assertIn('seg_loss',result['losses'])
        # Original validation uses batch size 1 and 15 grasp candidates.
        class Data(list):
            dataset=types.SimpleNamespace(get_gtbb=lambda *a: object())
        single=tuple(v[:1] if isinstance(v,(torch.Tensor,list)) else v for v in batch)
        single=list(single)
        single[1]=[v[:1] for v in y]
        validation=function('validate')(model,torch.device('cpu'),Data([single]),.25)
        self.assertEqual(validation['correct'],1)
        self.assertEqual(validation['failed'],0)
        self.assertIn('seg_loss',validation['losses'])

if __name__=='__main__':
    unittest.main()
