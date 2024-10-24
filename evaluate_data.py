import torch
import cv2
import numpy as np
import clip
from inference.post_process import post_process_output
import argparse
import torch.utils.data
import logging
from hardware.device import get_device
from inference.models import get_network
from utils.dataset_processing import image, evaluation
import matplotlib.pyplot as plt
import os
from PIL import Image



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
    parser.add_argument('--image-path', type=str, default='/0000000.png')
    parser.add_argument('--query-text', type=str, default='pick the hammer', help='type the text')
    parser.add_argument('--add-file-path', type=str, default='data/grasp-anywhere', help='Specific for Grasp-Anywhere')

    # Checkpoint

    parser.add_argument('--model-path', type=str, help='Path to the trained model file')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,

    help='Force code to run in CPU mode')

    args = parser.parse_args()

    return args

def infer_single_image(net, image_path, query_text, device, input_size=224, use_rgb=True, use_depth=False):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The image file does not exist: {image_path}")

    # Open the image using PIL
    with Image.open(image_path) as img:
        # Resize the image
        img = img.resize((input_size, input_size))
        
        # Convert to RGB if it's not already
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(img)
    
    # print(f"Image shape after resizing: {img_array.shape}")
    
    # Normalize the image
    img_array = img_array.astype(np.float32) / 255.0
    
    # Convert to tensor and add batch dimension
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    # print(f"Tensor shape: {img_tensor.shape}")
    
    img_tensor = img_tensor.to(device)
    
    net.eval()
    with torch.no_grad():
        pos_output, cos_output, sin_output, width_output = net(img_tensor, query_text)
    
    # print(f"Network output shapes:")
    # print(f"pos_output: {pos_output.shape}, range: [{pos_output.min():.2f}, {pos_output.max():.2f}]")
    # print(f"cos_output: {cos_output.shape}, range: [{cos_output.min():.2f}, {cos_output.max():.2f}]")
    # print(f"sin_output: {sin_output.shape}, range: [{sin_output.min():.2f}, {sin_output.max():.2f}]")
    # print(f"width_output: {width_output.shape}, range: [{width_output.min():.2f}, {width_output.max():.2f}]")
    
    q_out, ang_out, w_out = post_process_output(pos_output, cos_output, sin_output, width_output)
    
    # print(f"Post-processed output shapes:")
    # print(f"q_out: {q_out.shape}, range: [{q_out.min():.2f}, {q_out.max():.2f}]")
    # print(f"ang_out: {ang_out.shape}, range: [{ang_out.min():.2f}, {ang_out.max():.2f}]")
    # print(f"w_out: {w_out.shape}, range: [{w_out.min():.2f}, {w_out.max():.2f}]")
    
    return img_array, q_out, ang_out, w_out


def run():
    args = parse_args()
    device = get_device(args.force_cpu)

    single_image_path = args.image_path
    query_text = args.query_text
    
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

    if single_image_path and query_text:
        img, q_out, ang_out, w_out = infer_single_image(
            net, single_image_path, query_text, device, 
            input_size=args.input_size, 
            use_rgb=args.use_rgb, 
            use_depth=args.use_depth
        )
        
        evaluation.save_output(img, q_out, ang_out, 
                    grasp_width_img=w_out, 
                    no_grasps=1, save_folder='test_real')

if __name__ == '__main__':
    run()