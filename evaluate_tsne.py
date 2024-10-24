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
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting
import os
from PIL import Image
from sklearn.manifold import TSNE

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
    parser.add_argument('--query-text', type=str, default='grasp the pen', help='type the text')
    parser.add_argument('--add-file-path', type=str, default='data/grasp-anywhere', help='Specific for Grasp-Anywhere')

    # Checkpoint

    parser.add_argument('--model-path', type=str, help='Path to the trained model file')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,help='Force code to run in CPU mode')
    parser.add_argument('--num-samples', type=int, default=1000, help='Number of samples to use for t-SNE')
    parser.add_argument('--feature-type', type=str, default='combined1', help='Type of feature to visualize (stage1, stage2, stage3, stage4, text_features, combined1, combined2, combined3)')
    return parser.parse_args()

def apply_tsne(features, n_components=3, perplexity=30, n_iter=1000, learning_rate='auto'):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, 
                learning_rate=learning_rate, random_state=42)
    return tsne.fit_transform(features)

def visualize_tsne_3d(tsne_results, labels, title):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize the results to a smaller range
    tsne_normalized = (tsne_results - tsne_results.min()) / (tsne_results.max() - tsne_results.min())
    tsne_normalized = (tsne_normalized - 0.5) * 16  # Scale to roughly -8 to 8
    
    scatter = ax.scatter(tsne_normalized[:, 0], tsne_normalized[:, 1], tsne_normalized[:, 2], 
                         c=labels, cmap='tab20', s=5, alpha=0.6)
    
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_zlabel('t-SNE 3')
    ax.set_title(title)
    
    # Set consistent axis limits
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_zlim(-8, 8)
    
    plt.colorbar(scatter)
    plt.savefig(f'{title.lower().replace(" ", "_")}_tsne_3d.png', dpi=300, bbox_inches='tight')
    plt.close()

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
    
    print(f"Image shape after resizing: {img_array.shape}")
    
    # Normalize the image
    img_array = img_array.astype(np.float32) / 255.0
    
    # Convert to tensor and add batch dimension
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    print(f"Tensor shape: {img_tensor.shape}")
    
    img_tensor = img_tensor.to(device)
    
    net.eval()
    with torch.no_grad():
        pos_output, cos_output, sin_output, width_output = net(img_tensor, query_text)
    
    q_out, ang_out, w_out = post_process_output(pos_output, cos_output, sin_output, width_output)
    
    return img_array, q_out, ang_out, w_out, net.intermediate_features

def run():
    args = parse_args()
    device = get_device(args.force_cpu)
    
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

    # Prepare data for t-SNE
    features = []
    labels = []

    for i in range(args.num_samples):
        # You might need to modify this part to load different images and queries
        img, q_out, ang_out, w_out, intermediate_features = infer_single_image(
            net, args.image_path, args.query_text, device, 
            input_size=args.input_size, 
            use_rgb=args.use_rgb, 
            use_depth=args.use_depth
        )
        
        # Extract relevant features based on the specified feature type
        if args.feature_type in ['stage1', 'stage2', 'stage3', 'stage4']:
            feature = intermediate_features[args.feature_type].mean(dim=[2, 3]).cpu().numpy()
        elif args.feature_type == 'text_features':
            feature = intermediate_features[args.feature_type].cpu().numpy()
        elif args.feature_type in ['combined1', 'combined2', 'combined3']:
            feature = intermediate_features[args.feature_type].mean(dim=[2, 3]).cpu().numpy()
        else:
            raise ValueError(f"Unknown feature type: {args.feature_type}")
        
        features.append(feature)
        labels.append(i % 10)  # Just for demonstration, you might want to use actual labels

    # Convert to numpy arrays
    features = np.vstack(features)
    labels = np.array(labels)

    # Apply t-SNE with adjusted parameters
    tsne_results = apply_tsne(features, n_components=3, perplexity=50, n_iter=5000, learning_rate='auto')

    # Visualize t-SNE results in 3D
    visualize_tsne_3d(tsne_results, labels, f"{args.feature_type.capitalize()} Features t-SNE 3D")

    print(f"3D t-SNE visualization for {args.feature_type} features has been saved.")


if __name__ == '__main__':
    run()