import glob
import os
import re

import pickle
import torch

from utils.dataset_processing import grasp, image, mask
from .language_grasp_data import LanguageGraspDatasetBase
import numpy as np
import logging

class GraspAnythingDataset(LanguageGraspDatasetBase):
    """
    Dataset wrapper for the Grasp-Anything dataset.
    """

    def __init__(self, file_path, ds_rotate=0, **kwargs):
        """
        :param file_path: Grasp-Anything Dataset directory.
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(GraspAnythingDataset, self).__init__(**kwargs)

        self.grasp_files = glob.glob(os.path.join(file_path, 'grasp_label_positive', '*.pt'))
        self.prompt_files = glob.glob(os.path.join(file_path, 'scene_description', '*.pkl'))
        self.rgb_files = glob.glob(os.path.join(file_path, 'image', '*.jpg'))
        self.mask_files = glob.glob(os.path.join(file_path, 'mask', '*.npy'))
    
        if kwargs["seen"]:
            with open(os.path.join('split/grasp-anything/seen.obj'), 'rb') as f:
                idxs = pickle.load(f)

            self.grasp_files = list(filter(lambda x: x.split('/')[-1].split('.')[0] in idxs, self.grasp_files))
        else:
            with open(os.path.join('split/grasp-anything/unseen.obj'), 'rb') as f:
                 idxs = pickle.load(f)

            self.grasp_files = list(filter(lambda x: x.split('/')[-1].split('.')[0] in idxs, self.grasp_files))


        self.grasp_files.sort()
        self.prompt_files.sort()
        self.rgb_files.sort()
        self.mask_files.sort()
        self.length = len(self.grasp_files)

        if self.length == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

        if ds_rotate:
            self.grasp_files = self.grasp_files[int(self.length * ds_rotate):] + self.grasp_files[
                                                                                 :int(self.length * ds_rotate)]
            

    def _get_crop_attrs(self, idx):
        gtbbs = grasp.GraspRectangles.load_from_grasp_anything_file(self.grasp_files[idx])
        center = gtbbs.center
        left = max(0, min(center[1] - self.output_size // 2, 416 - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, 416 - self.output_size))
        return center, left, top

    def get_gtbb(self, idx, rot=0, zoom=1.0):       
        # Jacquard try
        gtbbs = grasp.GraspRectangles.load_from_grasp_anything_file(self.grasp_files[idx], scale=self.output_size / 416.0)

        c = self.output_size // 2
        gtbbs.rotate(rot, (c, c))
        gtbbs.zoom(zoom, (c, c))

        # Cornell try
        # gtbbs = grasp.GraspRectangles.load_from_grasp_anything_file(self.grasp_files[idx])
        # center, left, top = self._get_crop_attrs(idx)
        # gtbbs.rotate(rot, center)
        # gtbbs.offset((-top, -left))
        # gtbbs.zoom(zoom, (self.output_size // 2, self.output_size // 2))
        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = image.DepthImage.from_tiff(self.depth_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        depth_img.rotate(rot, center)
        depth_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        return depth_img.img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        # Generate the corresponding .jpg file path
        rgb_file = re.sub(r"_\d+\.pt", ".jpg", self.grasp_files[idx])
        rgb_file = rgb_file.replace("grasp_label_positive", "image")

        if not os.path.exists(rgb_file):
            raise FileNotFoundError(f"RGB file {rgb_file} does not exist.")

        mask_file = self.grasp_files[idx].replace("grasp_label_positive", "mask").replace(".pt", ".npy")
        if not os.path.exists(mask_file):
            raise FileNotFoundError(f"Mask file {mask_file} does not exist.")
        
        rgb_img = image.Image.from_file(rgb_file)
        
        # Apply transformations
        rgb_img.rotate(rot)
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img
    
    def get_prompts(self, idx):
        # Adjusting the split operation to handle more than two values
        parts = self.grasp_files[idx].replace("grasp_label_positive", "scene_description").split('_')
        if len(parts) > 2:
            prompt_file = '_'.join(parts[:-1])  # Join all parts except the last one
            obj_id = parts[-1].split('.')[0]  # The last part is the object id, and we remove the extension
        elif len(parts) == 2:
            prompt_file, obj_id = parts
            obj_id = obj_id.split('.')[0]  # The object id without the extension
        else:
            raise ValueError(f"Unexpected format in grasp file name: {self.grasp_files[idx]}")
        
        prompt_file += '.pkl'
        obj_id = int(obj_id)

        with open(prompt_file, 'rb') as f:
            x = pickle.load(f)
            prompt, queries = x

        # Ensure obj_id is within the range of queries
        if obj_id >= len(queries):
            logging.warning(f"obj_id {obj_id} out of range for queries of length {len(queries)}. Using obj_id 0 instead.")
            obj_id = 0  # Use the first index if out of range

        return prompt, queries[obj_id]