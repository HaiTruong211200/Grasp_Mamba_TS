import csv
import importlib.util
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from utils.data.grasp_anything_language_data import GraspAnythingLanguageDataset

spec = importlib.util.spec_from_file_location('prepare_pp', Path(__file__).resolve().parents[1] / 'tools/prepare_grasp_anything_pp.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class PreparationTests(unittest.TestCase):
    def test_validation_is_stable_and_groups_scenes(self):
        scenes = {f'{i:064x}' for i in range(20)}
        a = prepare.assign_validation_scenes(scenes, 0.1, 123)
        b = prepare.assign_validation_scenes(reversed(sorted(scenes)), 0.1, 123)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 2)

    def test_unverified_units_and_bad_rows_are_not_silently_scaled(self):
        with self.assertRaisesRegex(ValueError, 'unverified_subpixel'):
            prepare.validate_grasps(torch.full((1, 6), 0.5), 416, 416)
        raw = torch.tensor([[0.9, 208, 208, 60, 20, 0], [0.5]*6])
        _, indices = prepare.validate_grasps(raw, 416, 416)
        self.assertEqual(indices, [0])

    def test_builder_produces_loader_ready_disjoint_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, splits, images = root/'selected', root/'splits', root/'images'
            groups = {'train_seen': [(1,0),(1,1),(2,0),(3,0),(4,0)],
                      'test_seen': [(5,0)], 'test_unseen': [(4,0)]}
            filenames = {'train_seen':'train/seen.obj','test_seen':'test/seen.obj','test_unseen':'test/unseen.obj'}
            for group, pairs in groups.items():
                for folder in ('grasp_label_positive','part_mask','grasp_instructions'):
                    (source/group/folder).mkdir(parents=True)
                (images/('image_'+group)).mkdir(parents=True)
                with (source/group/'annotation_inventory.csv').open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=['sample_id'])
                    writer.writeheader()
                    for number, obj in pairs:
                        scene = f'{number:064x}'
                        sample = f'{scene}_{obj}_0'
                        writer.writerow({'sample_id':sample})
                        cv2.imwrite(str(images/('image_'+group)/(scene+'.jpg')), np.zeros((416,416,3), np.uint8))
                        mask = np.zeros((416,416), np.uint8); mask[180:240,180:240] = 1
                        np.save(source/group/'part_mask'/(sample+'.npy'), mask)
                        (source/group/'grasp_instructions'/(sample+'.pkl')).write_bytes(pickle.dumps('grasp the handle'))
                        raw = torch.tensor([[0.9,208,208,60,20,0],[0.5]*6]) if number != 3 else torch.full((1,6),0.5)
                        torch.save(raw, source/group/'grasp_label_positive'/(sample+'.pt'))
                (source/group/'missing_positive_object_ids.txt').write_text('')
                path = splits/filenames[group]; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(pickle.dumps([f'{number:064x}_{obj}' for number,obj in pairs]))
            output = root/'ready'
            with patch.object(prepare, 'make_preview'):
                report = prepare.build(source, output, splits, images, val_fraction=0.5)
            self.assertEqual(len(report['removed_train_scenes_shared_with_test']), 1)
            self.assertEqual(report['exclusion_reasons']['unverified_subpixel_grasp_units'], 1)
            all_scenes = set()
            for split in ('train','val','test_seen','test_unseen'):
                dataset = GraspAnythingLanguageDataset(output/'manifest.jsonl', split=split,
                                                       segmentation=True, mask_target='part', output_size=32)
                scenes = {r['scene_id'] for r in dataset.samples}
                self.assertFalse(all_scenes & scenes); all_scenes |= scenes
                self.assertEqual(dataset.samples[0]['grasp_indices'], [0])
                self.assertEqual(len(dataset.get_gtbb(0).grs), 1)
                sample = dataset[0]
                self.assertTrue(sample['target_mask'].any())
                self.assertTrue(sample['grasp_targets']['pos'].any())
                self.assertIsNone(dataset.samples[0]['object_mask_path'])


if __name__ == '__main__':
    unittest.main()
