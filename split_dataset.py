import os
import glob
import shutil
import random
import re

# Define the source and destination directories
src_dir = '/home/cds-robotlab-10/master_thesis/grasp-anything/data_exp1'  # Replace with your actual source directory
dst_dir_exp = '/home/cds-robotlab-10/master_thesis/grasp-anything/data_exp2'
# dst_dir_10 = '/home/robot-learning/master_thesis/grasp-anything/dataset_10percent'
# dst_dir_90 = '/home/robot-learning/master_thesis/grasp-anything/dataset_90percent'

# Create destination directories if they don't exist
os.makedirs(dst_dir_exp, exist_ok=True)

# os.makedirs(dst_dir_10, exist_ok=True)
# os.makedirs(dst_dir_90, exist_ok=True)

# Get all positive grasp label files (we'll use these as the basis for splitting)
grasp_files = glob.glob(os.path.join(src_dir, 'grasp_label_positive', '*.pt'))

# Get unique identifiers
identifiers = [os.path.splitext(os.path.basename(f))[0] for f in grasp_files]

# Randomly shuffle the identifiers
random.shuffle(identifiers)

# Calculate the split point
split_point = int(len(identifiers) * 0.5)
print('split_point',split_point)
# split_point = int(len(identifiers) * 0.1)

# Split the identifiers
identifiers_exp = identifiers[:split_point]

# identifiers_10 = identifiers[:split_point]
# identifiers_90 = identifiers[split_point:]

def get_corresponding_files(identifier):
    # Grasp label positive
    grasp_pos = os.path.join(src_dir, 'grasp_label_positive', f"{identifier}.pt")
    
    # Grasp label negative
    grasp_neg = os.path.join(src_dir, 'grasp_label_negative', f"{identifier}.pt")
    
    # Image
    image = re.sub(r"_\d+\.pt", ".jpg", grasp_pos)
    image = image.replace("grasp_label_positive", "image")
    
    # Mask
    mask = grasp_pos.replace("grasp_label_positive", "mask").replace(".pt", ".npy")
    
    # Scene description (prompt)
    parts = grasp_pos.replace("grasp_label_positive", "scene_description").split('_')
    if len(parts) > 2:
        scene_desc = '_'.join(parts[:-1]) + '.pkl'  # Join all parts except the last one
    elif len(parts) == 2:
        scene_desc = parts[0] + '.pkl'
    else:
        raise ValueError(f"Unexpected format in grasp file name: {grasp_pos}")
    
    return grasp_pos, grasp_neg, image, mask, scene_desc

# Function to copy files based on identifiers
def copy_files(id_list, dst):
    for identifier in id_list:
        grasp_pos, grasp_neg, image, mask, scene_desc = get_corresponding_files(identifier)
        
        # Copy files
        shutil.copy2(grasp_pos, os.path.join(dst, 'grasp_label_positive', os.path.basename(grasp_pos)))
        shutil.copy2(grasp_neg, os.path.join(dst, 'grasp_label_negative', os.path.basename(grasp_neg)))
        shutil.copy2(image, os.path.join(dst, 'image', os.path.basename(image)))
        shutil.copy2(mask, os.path.join(dst, 'mask', os.path.basename(mask)))
        shutil.copy2(scene_desc, os.path.join(dst, 'scene_description', os.path.basename(scene_desc)))

# Create subdirectories in the destination folders
for folder in ['image', 'scene_description', 'grasp_label_positive', 'grasp_label_negative', 'mask']:
    os.makedirs(os.path.join(dst_dir_exp, folder), exist_ok=True)
    # os.makedirs(os.path.join(dst_dir_10, folder), exist_ok=True)
    # os.makedirs(os.path.join(dst_dir_90, folder), exist_ok=True)

copy_files(identifiers_exp, dst_dir_exp)
print(f"0.1% dataset created in {dst_dir_exp} with {len(identifiers_exp)} samples")

# # Copy files to the 10% dataset
# copy_files(identifiers_10, dst_dir_10)

# # Copy files to the 90% dataset
# copy_files(identifiers_90, dst_dir_90)

# print(f"10% dataset created in {dst_dir_10} with {len(identifiers_10)} samples")
# print(f"90% dataset created in {dst_dir_90} with {len(identifiers_90)} samples")