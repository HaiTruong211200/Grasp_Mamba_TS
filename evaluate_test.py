import argparse
import torch
import torch.utils.data
import logging
from tqdm import tqdm
import time
from hardware.device import get_device
from inference.models import get_network
from inference.post_process import post_process_output
from utils.data import get_dataset
from utils.dataset_processing import evaluation
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='Validate trained network')

    # Network
    parser.add_argument('--network', type=str, default='graspmamba', help='Network name in inference/models')
    parser.add_argument('--vision-model', default='nvidia/MambaVision-L-1K')
    parser.add_argument('--text-model', default='google/siglip-base-patch16-224')
    parser.add_argument('--input-size', type=int, default=224, help='Input image size for the network')
    parser.add_argument('--use-depth', type=int, default=0, help='Use Depth image for validation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1, help='Use RGB image for validation (1/0)')
    parser.add_argument('--iou-threshold', type=float, default=0.25, help='Threshold for IOU matching')

    # Dataset
    parser.add_argument('--dataset', type=str, help='Dataset Name ("cornell" or "jacquard")')
    parser.add_argument('--dataset-path', type=str, help='Path to dataset')
    parser.add_argument('--seen', type=int, default=1, help='Flag for using seen classes, only work for Grasp-Anything dataset')
    parser.add_argument('--add-file-path', type=str, default='data/grasp-anywhere', help='Specific for Grasp-Anywhere')

    # Checkpoint
    parser.add_argument('--model-path', type=str, help='Path to the trained model file')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')
    parser.add_argument('--text-source', choices=['object', 'instruction'], default='object',
                        help='grasp-anywhere text: original object query or detailed part instruction')
    parser.add_argument('--segmentation', action='store_true', help='Add part-mask auxiliary loss on grasp-anywhere')
    parser.add_argument('--lambda-seg', type=float, default=0.5)
    parser.add_argument('--lambda-dice', type=float, default=1.0)
    parser.add_argument('--eval-split', choices=['test_seen', 'test_unseen'], default=None,
                        help='Explicit official ++ test split; overrides legacy --seen selection')
    args = parser.parse_args()
    if args.eval_split is not None and args.dataset != 'grasp-anywhere':
        parser.error('--eval-split requires --dataset grasp-anywhere')
    if args.segmentation and args.dataset != 'grasp-anywhere':
        parser.error('--segmentation requires --dataset grasp-anywhere and part_mask/')
    if args.text_source == 'instruction' and args.dataset != 'grasp-anywhere':
        parser.error('--text-source instruction requires --dataset grasp-anywhere')
    return args

def validate(net, device, val_data, iou_threshold, plot=False):
    net.eval()
    
    results = {
        'correct': 0,
        'failed': 0,
        'loss': 0,
        'losses': {}
    }
    if plot:
        fig = plt.figure()
        count = 0
    start_time = time.time()
    with torch.no_grad():
        for batch in tqdm(val_data):
            x, y, didx, rot, zoom_factor, prompt, query = batch[:7]
            xc = x.to(device)
            yc = [yy.to(device) for yy in y]
            mask_options = {'target_mask': batch[7].to(device)} if len(batch) == 8 else {}
            lossd = net.compute_loss(xc, yc, query, **mask_options)

            loss = lossd['loss']

            results['loss'] += loss.item()
            for ln, l in lossd['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item()

            q_out, ang_out, w_out = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                        lossd['pred']['sin'], lossd['pred']['width'])
            
            s = evaluation.calculate_iou_match(q_out,
                                               ang_out,
                                               val_data.dataset.get_gtbb(didx, rot, zoom_factor),
                                               no_grasps=5,
                                               grasp_width=w_out,
                                               threshold=iou_threshold
                                               )

            if plot:
                
                    if isinstance(rot, torch.Tensor):
                        rot = rot.item()
                    rgb_img = val_data.dataset.get_rgb(didx, rot, zoom_factor, normalise=False)
                    depth_img = x[0, 3, :, :].cpu().numpy() if x.shape[1] == 4 else None
                    # Convert angle from tensor to scalar if necessary
                    # evaluation.plot_output(fig, rgb_img, q_out, ang_out, val_data.dataset.get_gtbb(didx, rot, zoom_factor),
                    #                                     query, prompt, depth_img, no_grasps=1, grasp_width_img=w_out, save_folder='output_100grasps')
                    evaluation.save_output(rgb_img, q_out, ang_out, w_out, no_grasps=1, save_folder='results')

            if s:
                results['correct'] += 1
            else:
                results['failed'] += 1
    
    avg_time = (time.time() - start_time) / len(val_data)
    logging.info('Average evaluation time per image: {}ms'.format(avg_time * 1000))

    return results

def run():
    args = parse_args()

    # Get the compute device
    device = get_device(args.force_cpu)

    # Load Dataset
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)
    text_options = {'text_source': args.text_source, 'segmentation': args.segmentation} if args.dataset == 'grasp-anywhere' else {}
    if args.eval_split is not None:
        text_options['eval_split'] = args.eval_split
    logging.info('Evaluation split: %s', args.eval_split or ('train_seen' if args.seen else 'test_unseen'))
    logging.info('Text source: %s', args.text_source)
    dataset = Dataset(args.dataset_path,
                      output_size=args.input_size,
                      ds_rotate=0,
                      random_rotate=False,
                      random_zoom=False,
                      include_depth=args.use_depth,
                      include_rgb=args.use_rgb,
                      seen=args.seen,
                      add_file_path=args.add_file_path,
                      **text_options)
    
    val_data = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )
    logging.info('Validation size: {}'.format(dataset.length))

    # Load the network
    logging.info('Loading Network...')
    input_channels = args.use_depth + 3 * args.use_rgb
    network = get_network(args.network)
    net = network(input_size=args.input_size,
                  vision_model=args.vision_model, text_model=args.text_model,
                  segmentation=args.segmentation, lambda_seg=args.lambda_seg,
                  lambda_dice=args.lambda_dice)
    net.load_state_dict(torch.load(args.model_path))
    net = net.to(device)
    # Add this line to display the number of parameters
    logging.info(f'Number of model parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad):,}')
    logging.info('Done')

    # Run Validation
    logging.info('Validating...')
    test_results = validate(net, device, val_data, args.iou_threshold, plot=False)
    logging.info('IOU Results: %d/%d = %f' % (test_results['correct'],
                                              test_results['correct'] + test_results['failed'],
                                              test_results['correct'] / (test_results['correct'] + test_results['failed'])))
    logging.info('Loss: {:0.4f}'.format(test_results['loss']))


if __name__ == '__main__':
    run()