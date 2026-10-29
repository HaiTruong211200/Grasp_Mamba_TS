"""Evaluate the prepared language/segmentation manifest with a GraspMamba state_dict."""
import argparse
import json
from pathlib import Path

import torch

from hardware.device import get_device
from inference.models.graspmamba import Grasp_Mamba
from train_network import validate
from utils.data.grasp_anything_language_data import GraspAnythingLanguageDataset, language_collate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--eval-split', choices=['all', 'val', 'test_seen', 'test_unseen'], default='all')
    parser.add_argument('--mask-target', choices=['part', 'object'], default='part')
    parser.add_argument('--vision-model', default='nvidia/MambaVision-L-1K',
                        help='MambaVision-L Hugging Face ID or local pretrained directory')
    parser.add_argument('--text-model', default='google/siglip-base-patch16-224',
                        help='SigLIP Hugging Face ID or local model and processor directory')
    parser.add_argument('--input-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--iou-threshold', type=float, default=0.25)
    parser.add_argument('--normalization', choices=['legacy', 'imagenet'], default='legacy')
    parser.add_argument('--grasp-loss', choices=['mse', 'smooth_l1'], default='smooth_l1')
    parser.add_argument('--lambda-seg', type=float, default=0.5)
    parser.add_argument('--lambda-dice', type=float, default=1.0)
    parser.add_argument('--cpu', dest='force_cpu', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    device = get_device(args.force_cpu)
    state = torch.load(args.model_path, map_location='cpu', weights_only=True)
    segmentation = any(k.startswith('grasp_mamba.seg_head.') for k in state)
    net = Grasp_Mamba(input_size=args.input_size, segmentation=segmentation,
                      lambda_seg=args.lambda_seg, lambda_dice=args.lambda_dice,
                      grasp_loss=args.grasp_loss,
                      vision_model=args.vision_model, text_model=args.text_model)
    net.load_state_dict(state, strict=True)
    net.to(device)
    results = {}
    splits = ['test_seen', 'test_unseen'] if args.eval_split == 'all' else [args.eval_split]
    for split in splits:
        dataset = GraspAnythingLanguageDataset(args.manifest, split=split,
            segmentation=segmentation, mask_target=args.mask_target,
            output_size=args.input_size, normalization=args.normalization)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers, collate_fn=language_collate)
        metrics = validate(net, device, loader, args.iou_threshold)
        metrics['success_rate'] = metrics['correct'] / (metrics['correct'] + metrics['failed'])
        metrics['samples'] = len(dataset)
        results[split] = metrics
    if args.eval_split == 'all':
        s, u = (results[k]['success_rate'] for k in ('test_seen', 'test_unseen'))
        results['S'], results['U'] = s, u
        results['H'] = 2*s*u/(s+u) if s+u else 0.0
    text = json.dumps(results, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
