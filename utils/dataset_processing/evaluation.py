import warnings
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
warnings.filterwarnings("ignore")
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw
import logging
from .grasp import GraspRectangles, detect_grasps, Grasp
import cv2


def plot_output(fig, rgb_img, grasp_q_img, grasp_angle_img, ground_truth_grasps, query, prompt, depth_img=None, no_grasps=1, grasp_width_img=None, save_folder='output_test'):
    """
    Plot the output of a network, draw detected grasps, and save the result
    :param fig: Figure to plot the output
    :param rgb_img: RGB Image
    :param depth_img: Depth Image
    :param grasp_q_img: Q output of network
    :param grasp_angle_img: Angle output of network
    :param ground_truth_grasps: Ground truth grasp positions
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of network
    :param save_folder: Folder to save the output images
    :return: Number of grasps detected
    """
    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps)
    
    plt.ion()
    plt.clf()
    fig.set_size_inches(15, 10)
    
    # Normalize RGB image if needed
    if rgb_img.dtype == np.float32 or rgb_img.dtype == np.float64:
        rgb_img = np.clip(rgb_img, 0, 1)
    else:
        rgb_img = np.clip(rgb_img, 0, 255).astype(np.uint8)
    
    rgb_with_grasps = rgb_img.copy()
    rgb_with_ground_truth = rgb_img.copy()
    
    # Plot RGB image with detected grasps
    ax = fig.add_subplot(2, 3, 1)
    ax.imshow(rgb_with_grasps)
    for g in gs:
        g.plot(ax)
    ax.set_title('RGB with Detected Grasps')
    ax.axis('off')
    
    # Plot RGB image with ground truth grasps
    ax = fig.add_subplot(2, 3, 2)
    ax.imshow(rgb_with_ground_truth)
    for g in ground_truth_grasps:
        g.plot(ax, color='green')
    ax.set_title('RGB with Ground Truth Grasps')
    ax.axis('off')
    
    # Plot the query output
    ax = fig.add_subplot(2, 3, 3)
    ax.text(0.5, 0.5, query, ha='center', va='center', wrap=True)
    ax.set_title('Query')
    ax.axis('off')
    
    # Plot Q heatmap overlaid on RGB image
    ax = fig.add_subplot(2, 3, 4)
    ax.imshow(rgb_img)
    q_plot = ax.imshow(grasp_q_img, cmap='jet', vmin=0, vmax=1, alpha=0.7)
    ax.set_title('Q')
    ax.axis('off')
    plt.colorbar(q_plot, ax=ax)
    
    # Plot Angle heatmap overlaid on RGB image
    ax = fig.add_subplot(2, 3, 5)
    ax.imshow(rgb_img)
    angle_plot = ax.imshow(grasp_angle_img, cmap='hsv', vmin=-np.pi/2, vmax=np.pi/2, alpha=0.7)
    ax.set_title('Angle')
    ax.axis('off')
    plt.colorbar(angle_plot, ax=ax)
    
   # Plot the RGB image instead of the prompt
    ax = fig.add_subplot(2, 3, 6)
    ax.imshow(rgb_img)
    ax.set_title('Original RGB Image')
    ax.axis('off')
    
    plt.tight_layout()
    
    # Create the save folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)
    
    # Save the figure
    save_path = os.path.join(save_folder, f'grasp_detection_{len(os.listdir(save_folder))}.png')
    plt.savefig(save_path)
    plt.pause(0.1)
    fig.canvas.draw()

def draw_grasp_rectangle(draw, center, angle, length, width):
    """
    Draw a rectangle with longer sides in red and shorter sides in green
    """
    cos_t, sin_t = np.cos(angle), np.sin(angle)
    half_length, half_width = length / 2, width / 2
    corners = np.array([
        [-half_length, -half_width],
        [half_length, -half_width],
        [half_length, half_width],
        [-half_length, half_width]
    ])
    
    rotated_corners = np.dot(corners, np.array([[cos_t, -sin_t], [sin_t, cos_t]]))
    corners_xy = (rotated_corners + np.array(center)).astype(int)
    
    # Draw longer sides (red)
    draw.line(tuple(corners_xy[0]) + tuple(corners_xy[1]), fill=(255, 0, 0), width=2)
    draw.line(tuple(corners_xy[2]) + tuple(corners_xy[3]), fill=(255, 0, 0), width=2)
    
    # Draw shorter sides (green)
    draw.line(tuple(corners_xy[1]) + tuple(corners_xy[2]), fill=(0, 255, 0), width=2)
    draw.line(tuple(corners_xy[3]) + tuple(corners_xy[0]), fill=(0, 255, 0), width=2)

def create_heatmap_overlay(q_img, rgb_img):
    """
    Create a heatmap overlay on the RGB image
    """
    # Normalize Q image
    q_norm = (q_img - q_img.min()) / (q_img.max() - q_img.min())
    
    # Create a color map
    cmap = plt.get_cmap('jet')
    heatmap = (cmap(q_norm) * 255).astype(np.uint8)
    
    # Create PIL images
    heatmap_img = Image.fromarray(heatmap[:,:,:3])
    rgb_img_pil = Image.fromarray(rgb_img)
    
    # Overlay heatmap on RGB image
    overlay = Image.blend(rgb_img_pil, heatmap_img, 0.5)
    
    return overlay

def save_output(rgb_img, grasp_q_img, grasp_angle_img, grasp_width_img=None, no_grasps=1, save_folder='experiments'):
    """
    Save RGB image with two-color grasp rectangles and Q heatmap as 224x224 images
    :param rgb_img: RGB Image (assumed to be 224x224)
    :param grasp_q_img: Q output of network
    :param grasp_angle_img: Angle output of network
    :param grasp_width_img: (optional) Width output of network
    :param no_grasps: Maximum number of grasps to plot
    :param save_folder: Folder to save the output images
    :return: None
    """
    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps)
    
    # Ensure RGB image is in the correct format
    if rgb_img.dtype == np.float32 or rgb_img.dtype == np.float64:
        rgb_img = (np.clip(rgb_img, 0, 1) * 255).astype(np.uint8)
    else:
        rgb_img = np.clip(rgb_img, 0, 255).astype(np.uint8)
    
    # Create the save folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)
    
    # Generate a unique identifier for this set of images
    unique_id = len(os.listdir(save_folder))
    
    # Create and save RGB image with grasp rectangles
    img = Image.fromarray(rgb_img)
    draw = ImageDraw.Draw(img)
    
    for g in gs:
        draw_grasp_rectangle(draw, (g.center[1], g.center[0]), g.angle, g.length, g.width)

    img.save(os.path.join(save_folder, f'rgb_grasps_{unique_id}.png'))
    
    # Create and save Q heatmap
    heatmap_overlay = create_heatmap_overlay(grasp_q_img, rgb_img)
    heatmap_overlay.save(os.path.join(save_folder, f'q_heatmap_{unique_id}.png'))

    # Create and save angle heatmap
    angle_heatmap_overlay = create_heatmap_overlay(grasp_angle_img, rgb_img)
    angle_heatmap_overlay.save(os.path.join(save_folder, f'angle_heatmap_{unique_id}.png'))
    
    print(f"Saved RGB with detected grasps and Q heatmap in folder: {save_folder}")


def calculate_iou_match(grasp_q, grasp_angle, ground_truth_bbs, no_grasps=1, grasp_width=None, threshold=0.25):
    """
    Calculate grasp success using the IoU (Jacquard) metric (e.g. in https://arxiv.org/abs/1301.3592)
    A success is counted if grasp rectangle has a 25% IoU with a ground truth, and is withing 30 degrees.
    :param grasp_q: Q outputs of network (Nx300x300x3)
    :param grasp_angle: Angle outputs of network
    :param ground_truth_bbs: Corresponding ground-truth BoundingBoxes
    :param no_grasps: Maximum number of grasps to consider per image.
    :param grasp_width: (optional) Width output from network
    :param threshold: Threshold for IOU matching. Detect with IOU ≥ threshold
    :return: success
    """

    if not isinstance(ground_truth_bbs, GraspRectangles):
        gt_bbs = GraspRectangles.load_from_array(ground_truth_bbs)
    else:
        gt_bbs = ground_truth_bbs
    gs = detect_grasps(grasp_q, grasp_angle, width_img=grasp_width, no_grasps=no_grasps)
    for g in gs:
        if g.max_iou(gt_bbs) > threshold:
            return True
    else:
        return False
    