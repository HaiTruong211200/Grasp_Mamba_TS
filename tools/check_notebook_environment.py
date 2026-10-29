"""Check notebook imports, compiled CUDA scan, and optionally pretrained outputs."""
import argparse
import importlib
from importlib.metadata import version, PackageNotFoundError
import json
import platform
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-only', action='store_true')
    parser.add_argument('--model-smoke', action='store_true', help='Loads/downloads pretrained models and runs forward')
    args = parser.parse_args()
    if args.data_only and args.model_smoke:
        parser.error('--data-only and --model-smoke are mutually exclusive')
    packages = ('torch','torchvision','numpy','scipy','matplotlib','Pillow','opencv-python-headless',
                'scikit-image','imageio','transformers','huggingface-hub','timm','einops',
                'tensorboardX','wandb','ipykernel','mamba-ssm','causal-conv1d','triton','sentencepiece','safetensors')
    info = {'python': sys.version, 'platform': platform.platform(), 'executable': sys.executable,
            'nvcc': shutil.which('nvcc'), 'versions': {}}
    for name in packages:
        try:
            info['versions'][name] = version(name)
        except PackageNotFoundError:
            info['versions'][name] = None
    import torch
    info['torch_cuda_version'] = torch.version.cuda
    info['cuda_available'] = torch.cuda.is_available()
    print(json.dumps(info, indent=2), flush=True)
    for name in ('cv2','numpy','scipy','skimage','matplotlib','PIL','imageio','tqdm'):
        importlib.import_module(name)
    from utils.data.grasp_anything_language_data import GraspAnythingLanguageDataset, language_collate
    if args.data_only:
        print('PASS: CPU/data imports; CUDA/model imports were not tested')
        return
    if sys.platform != 'linux':
        raise RuntimeError('This CUDA profile targets Linux GPU runtimes (Colab/Kaggle/WSL2)')
    for name in ('torchvision','transformers','timm','einops','sentencepiece','tensorboardX','wandb'):
        importlib.import_module(name)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable: enable the notebook GPU accelerator and check CUDA-enabled torch')
    import mamba_ssm
    import selective_scan_cuda
    import causal_conv1d
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    info['mamba_python_path'] = mamba_ssm.__file__
    info['scan_extension_path'] = selective_scan_cuda.__file__
    info['gpu'] = torch.cuda.get_device_name(0)
    info['compute_capability'] = torch.cuda.get_device_capability(0)
    print(json.dumps({k: info[k] for k in ('mamba_python_path','scan_extension_path','gpu','compute_capability')}, indent=2), flush=True)
    u = torch.randn(1, 8, 16, device='cuda', requires_grad=True)
    delta = torch.randn_like(u, requires_grad=True)
    a = -torch.ones(8, 4, device='cuda')
    b = torch.randn(1, 4, 16, device='cuda')
    c = torch.randn(1, 4, 16, device='cuda')
    result = selective_scan_fn(u, delta, a, b, c, delta_softplus=True)
    result.square().mean().backward()
    if not torch.isfinite(result).all() or u.grad is None or not torch.isfinite(u.grad).all():
        raise RuntimeError('Nonfinite CUDA selective-scan result/gradient')
    print('PASS: CUDA selective scan forward/backward', flush=True)
    if args.model_smoke:
        from inference.models.graspmamba import Grasp_Mamba
        model = Grasp_Mamba(input_size=224, segmentation=True).cuda().eval()
        with torch.no_grad():
            outputs = model(torch.randn(1, 3, 224, 224, device='cuda'), ['grasp the cup by its handle'], return_seg=True)
        shapes = [list(x.shape) for x in outputs]
        if len(outputs) != 5 or any(x.shape != (1,1,224,224) or not torch.isfinite(x).all() for x in outputs):
            raise RuntimeError(f'Unexpected model outputs: {shapes}')
        print('PASS: pretrained grasp/segmentation output shapes ' + json.dumps(shapes))


if __name__ == '__main__':
    main()
