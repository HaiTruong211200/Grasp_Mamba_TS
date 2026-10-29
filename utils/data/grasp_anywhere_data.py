import glob
import os
import re

import pickle
import torch
import numpy as np
from PIL import Image as PILImage
from skimage.transform import rotate, resize

from utils.dataset_processing import grasp, image, mask
from .language_grasp_data import LanguageGraspDatasetBase


def _rotate_part_mask(target, rot):
    """Exact fast path for the original square-image quarter-turn augmentation."""
    turns = rot / (np.pi / 2)
    quarter = round(turns)
    if target.shape[0] == target.shape[1] and abs(turns - quarter) < 1e-8:
        return np.rot90(target, k=quarter)
    return rotate(target, rot / np.pi * 180, order=0,
                  mode='symmetric', preserve_range=True)


class GraspAnywhereDataset(LanguageGraspDatasetBase):
    """
    Dataset wrapper for the Grasp-Anything dataset.
    """

    def __init__(self, file_path, ds_rotate=0, text_source='object', segmentation=False, eval_split=None, **kwargs):
        """
        :param file_path: Grasp-Anything Dataset directory.
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        if eval_split not in (None, 'test_seen', 'test_unseen'):
            raise ValueError('eval_split must be test_seen or test_unseen')
        if text_source not in ('object', 'instruction'):
            raise ValueError('text_source must be object or instruction')
        self.text_source = text_source
        self.segmentation = segmentation
        super(GraspAnywhereDataset, self).__init__(**kwargs)

        addition_file_path = kwargs["add_file_path"]

        self.grasp_files = glob.glob(os.path.join(addition_file_path, 'grasp_label_positive', '*.pt'))
        self.prompt_files = glob.glob(os.path.join(file_path, 'scene_description', '*.pkl'))
        self.prompt_dir = os.path.join(file_path, 'scene_description')
        self.instruction_dir = os.path.join(addition_file_path, 'grasp_instructions')
        self.part_mask_dir = os.path.join(addition_file_path, 'part_mask')
        if self.segmentation and not os.path.isdir(self.part_mask_dir):
            raise FileNotFoundError(f'Segmentation needs part_mask/: {self.part_mask_dir}')
        self.rgb_files = glob.glob(os.path.join(file_path, 'image', '*.jpg'))
        self.rgb_dir = os.path.join(file_path, 'image')
        self.mask_files = glob.glob(os.path.join(file_path, 'mask', '*.npy'))

        if eval_split is not None:
            split_name = 'seen.obj' if eval_split == 'test_seen' else 'unseen.obj'
            with open(os.path.join('split/grasp-anything++/test', split_name), 'rb') as f:
                idxs = pickle.load(f)
            self.grasp_files = list(filter(lambda x: x.split('/')[-1][:-5] in idxs, self.grasp_files))
        elif kwargs["seen"]:
            with open(os.path.join('split/grasp-anything++/train/seen.obj'), 'rb') as f:
                idxs = pickle.load(f)
            self.grasp_files = list(filter(lambda x: x.split('/')[-1][:-5] in idxs, self.grasp_files))
        else:
            with open(os.path.join('split/grasp-anything++/test/unseen.obj'), 'rb') as f:
                idxs = pickle.load(f)

            self.grasp_files = list(filter(lambda x: x.split('/')[-1][:-5] in idxs, self.grasp_files))


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
        mask_file = self.grasp_files[idx].replace("grasp_label_positive", "mask").replace(".pt", ".npy")
        # mask_img = mask.Mask.from_file(mask_file)
        rgb_file = re.sub(r"_\d{1}_\d{1}\.pt", ".jpg", self.grasp_files[idx])
        rgb_file = rgb_file.split('/')[-1]
        rgb_file = os.path.join(self.rgb_dir, rgb_file)
        rgb_img = image.Image.from_file(rgb_file)
        # rgb_img = image.Image.mask_out_image(rgb_img, mask_img)

        # Jacquard try
        rgb_img.rotate(rot)
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img

        # Cornell try
        # center, left, top = self._get_crop_attrs(idx)
        # rgb_img.rotate(rot, center)
        # rgb_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        # rgb_img.zoom(zoom)
        # rgb_img.resize((self.output_size, self.output_size))
        # if normalise:
        #     rgb_img.normalise()
        #     rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        # return rgb_img.img

    def get_prompts(self, idx):
        grasp_file = self.grasp_files[idx].split('/')[-1]
        prompt_file, obj_id, part_id = grasp_file.split('_')
        instruction_file = grasp_file.split('.')[0]
        instruction_file += '.pkl'
        instruction_file = os.path.join(self.instruction_dir, instruction_file)
        prompt_file += '.pkl'
        prompt_file = os.path.join(self.prompt_dir, prompt_file)
        obj_id = int(obj_id)
        part_id = int(part_id.split('.')[0])

        if self.text_source == 'instruction':
            with open(instruction_file, 'rb') as f:
                instruction = pickle.load(f)
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError(f'Expected a nonempty instruction string: {instruction_file}')
            return '', instruction

        with open(prompt_file, 'rb') as f:
            x = pickle.load(f)
            prompt, queries = x
        

        return prompt, queries[obj_id]
    def get_target_mask(self, idx, rot=0.0, zoom=1.0):
        """Apply original RGB geometry to the part mask, with nearest interpolation."""
        stem = os.path.splitext(os.path.basename(self.grasp_files[idx]))[0]
        path = os.path.join(self.part_mask_dir, stem + '.npy')
        target = np.load(path, allow_pickle=False)
        if target.ndim != 2 or not np.isfinite(target).all() or not np.isin(target, [0, 1]).all():
            raise ValueError(f'Part mask must be a finite binary 2D array: {path}')
        if not target.any():
            raise ValueError(f'Part mask is empty before augmentation: {path}')
        scene = stem.rsplit('_', 2)[0]
        with PILImage.open(os.path.join(self.rgb_dir, scene + '.jpg')) as rgb:
            if target.shape != (rgb.height, rgb.width):
                raise ValueError(f'Part mask and RGB dimensions differ: {path}')
        target = _rotate_part_mask(target.astype(np.float32), rot)
        original_shape = target.shape
        sr = int(target.shape[0] * (1 - zoom)) // 2
        sc = int(target.shape[1] * (1 - zoom)) // 2
        target = target[sr:target.shape[0] - sr, sc:target.shape[1] - sc]
        target = resize(target, original_shape, order=0, mode='symmetric',
                        anti_aliasing=False, preserve_range=True)
        target = resize(target, (self.output_size, self.output_size), order=0,
                        anti_aliasing=False, preserve_range=True)
        # A crop may move the target out of view; this is a valid background target.
        return self.numpy_to_torch((target >= 0.5).astype(np.float32))

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        if not getattr(self, 'segmentation', False):
            return sample
        target = self.get_target_mask(idx, rot=sample[3], zoom=sample[4])
        return (*sample, target)
