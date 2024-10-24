import os
import glob
import shutil
import random
import re

# Define the source and destination directories
src_dir = '/home/cds-robotlab-10/master_thesis/grasp-anything/dataset_10percent'
dst_dir_exp = '/home/cds-robotlab-10/master_thesis/grasp-anything/dataset_exp'

# Create destination directory if it doesn't exist
os.makedirs(dst_dir_exp, exist_ok=True)

# Get all positive grasp label files (we'll use these as the basis for splitting)
grasp_files = glob.glob(os.path.join(src_dir, 'grasp_label_positive', '*.pt'))

# Get unique identifiers
identifiers = [os.path.splitext(os.path.basename(f))[0] for f in grasp_files]

# Randomly shuffle the identifiers
random.shuffle(identifiers)

# Calculate the split point for approximately 100 images
split_point = min(100, len(identifiers))
print('split_point', split_point)

# Split the identifiers
identifiers_exp = identifiers[:split_point]

def get_corresponding_files(identifier):
    # Grasp label positive
    grasp_pos = os.path.join(src_dir, 'grasp_label_positive', f"{identifier}.pt")
    # Grasp label negative
    grasp_neg = os.path.join(src_dir, 'grasp_label_negative', f"{identifier}.pt")
    # Image
    image = os.path.join(src_dir, 'image', f"{identifier.rsplit('_', 1)[0]}.jpg")
    # Mask
    mask = os.path.join(src_dir, 'mask', f"{identifier}.npy")
    # Scene description (prompt)
    scene_desc = os.path.join(src_dir, 'scene_description', f"{identifier.rsplit('_', 1)[0]}.pkl")
    
    return grasp_pos, grasp_neg, image, mask, scene_desc

# Function to copy files based on identifiers
def copy_files(id_list, dst):
    for identifier in id_list:
        grasp_pos, grasp_neg, image, mask, scene_desc = get_corresponding_files(identifier)
        
        # Copy files
        for src_file, folder in zip([grasp_pos, grasp_neg, image, mask, scene_desc],
                                    ['grasp_label_positive', 'grasp_label_negative', 'image', 'mask', 'scene_description']):
            dst_folder = os.path.join(dst, folder)
            os.makedirs(dst_folder, exist_ok=True)
            if os.path.exists(src_file):
                shutil.copy2(src_file, os.path.join(dst_folder, os.path.basename(src_file)))
            else:
                print(f"Warning: File not found: {src_file}")

# Copy files to the experimental dataset
copy_files(identifiers_exp, dst_dir_exp)
print(f"Experimental dataset created in {dst_dir_exp} with {len(identifiers_exp)} samples")