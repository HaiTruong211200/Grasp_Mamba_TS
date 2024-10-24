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
    parser.add_argument('--network', type=str, default='graspmambav7', help='Network name in inference/models')
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
    args = parser.parse_args()
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
        for x, y, didx, rot, zoom_factor, prompt, query in tqdm(val_data):
            xc = x.to(device)
            yc = [yy.to(device) for yy in y]
            lossd = net.compute_loss(xc, yc, query)

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
            # Get intermediate features
            features = net.get_intermediate_features()
            

            if plot:
                
                    if isinstance(rot, torch.Tensor):
                        rot = rot.item()
                    rgb_img = val_data.dataset.get_rgb(didx, rot, zoom_factor, normalise=False)
                    depth_img = x[0, 3, :, :].cpu().numpy() if x.shape[1] == 4 else None
                    # Convert angle from tensor to scalar if necessary
                    # evaluation.plot_output(fig, rgb_img, q_out, ang_out, val_data.dataset.get_gtbb(didx, rot, zoom_factor),
                    #                                     query, prompt, depth_img, no_grasps=1, grasp_width_img=w_out, save_folder='output_100grasps')
                    evaluation.save_output(rgb_img, q_out, ang_out, w_out, no_grasps=1, save_folder='cliport_exp1')
          
            
                    # rgb_img = np.uint8(rgb_img * 255)
                    # # Create comparison heatmaps for each fusion stage
                    # create_comparison_heatmap(
                    #     features['final_features'],
                    #     features['pos_output'],
                    #     rgb_img,
                    #     f"Image_{count}_final_vs_Final",
                    #       )

            if s:
                results['correct'] += 1
            else:
                results['failed'] += 1
    
    avg_time = (time.time() - start_time) / len(val_data)
    logging.info('Average evaluation time per image: {}ms'.format(avg_time * 1000))

    return results

def create_comparison_heatmap(pre_fusion, post_fusion, original_image, title, output_folder="test_features"):
    def process_feature_map(feature_map):
        if isinstance(feature_map, torch.Tensor):
            feature_map = feature_map.cpu().numpy()
        if len(feature_map.shape) == 4:
            feature_map = feature_map[0]
        return np.mean(feature_map, axis=0)

    pre_fusion_map = process_feature_map(pre_fusion)
    post_fusion_map = process_feature_map(post_fusion)

    # Normalize feature maps
    pre_fusion_map = (pre_fusion_map - pre_fusion_map.min()) / (pre_fusion_map.max() - pre_fusion_map.min())
    post_fusion_map = (post_fusion_map - post_fusion_map.min()) / (post_fusion_map.max() - post_fusion_map.min())

    # Resize heatmaps to match original image size
    pre_fusion_map = cv2.resize(pre_fusion_map, (original_image.shape[1], original_image.shape[0]))
    post_fusion_map = cv2.resize(post_fusion_map, (original_image.shape[1], original_image.shape[0]))

    # Apply color map
    pre_fusion_heatmap = cv2.applyColorMap(np.uint8(255 * pre_fusion_map), cv2.COLORMAP_JET)
    post_fusion_heatmap = cv2.applyColorMap(np.uint8(255 * post_fusion_map), cv2.COLORMAP_JET)

    # Overlay heatmaps on original image
    pre_fusion_output = cv2.addWeighted(original_image, 0.6, pre_fusion_heatmap, 0.4, 0)
    post_fusion_output = cv2.addWeighted(original_image, 0.6, post_fusion_heatmap, 0.4, 0)

    # Create side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(cv2.cvtColor(pre_fusion_output, cv2.COLOR_BGR2RGB))
    ax1.set_title("Pre-Fusion")
    ax1.axis('off')
    ax2.imshow(cv2.cvtColor(post_fusion_output, cv2.COLOR_BGR2RGB))
    ax2.set_title("Post-Fusion")
    ax2.axis('off')
    
    plt.suptitle(title)
    
    # Save the figure
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, f"{title.replace(' ', '_')}.png"))
    plt.close()

def run():
    args = parse_args()

    # Get the compute device
    device = get_device(args.force_cpu)

    # Load Dataset
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)
    dataset = Dataset(args.dataset_path,
                      output_size=args.input_size,
                      ds_rotate=0,
                      random_rotate=False,
                      random_zoom=False,
                      include_depth=args.use_depth,
                      include_rgb=args.use_rgb,
                      seen=args.seen,
                      add_file_path=args.add_file_path)
    
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
    net = network(
        input_channels=input_channels,
        dropout=False,
        prob=0,
        channel_size=32
    )
    net.load_state_dict(torch.load(args.model_path))
    net = net.to(device)
    # Add this line to display the number of parameters
    logging.info(f'Number of model parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad):,}')
    logging.info('Done')

    # Run Validation
    logging.info('Validating...')
    test_results = validate(net, device, val_data, args.iou_threshold, plot=True)
    logging.info('IOU Results: %d/%d = %f' % (test_results['correct'],
                                              test_results['correct'] + test_results['failed'],
                                              test_results['correct'] / (test_results['correct'] + test_results['failed'])))
    logging.info('Loss: {:0.4f}'.format(test_results['loss']))


if __name__ == '__main__':
    run()