"""Original ++ object queries and detailed instructions share the same loader contract."""
import argparse
import ast
import inspect
import io
from pathlib import Path
import pickle
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, mock_open, patch
import numpy as np
import torch
from utils.data.grasp_anywhere_data import GraspAnywhereDataset
from utils.data.language_grasp_data import LanguageGraspDatasetBase

ROOT = Path(__file__).resolve().parents[1]

class TextSourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.scene = 'a' * 64
        self.prompt_dir = self.root/'scene_description'
        self.instruction_dir = self.root/'grasp_instructions'
        self.prompt_dir.mkdir()
        self.instruction_dir.mkdir()
        self.prompt = self.prompt_dir/(self.scene+'.pkl')
        self.prompt.write_bytes(pickle.dumps(('Entire scene text', ['White mug'])))
        self.instructions = ['Lift the mug by its handle.', 'Lift the mug by its rim.']
        self.labels = [(self.root/f'{self.scene}_0_{part}.pt').as_posix() for part in (0,1)]
        for label,text in zip(self.labels,self.instructions):
            (self.instruction_dir/(Path(label).stem+'.pkl')).write_bytes(pickle.dumps(text))
    def dataset(self,mode):
        ds = GraspAnywhereDataset.__new__(GraspAnywhereDataset)
        LanguageGraspDatasetBase.__init__(ds,output_size=32,include_rgb=True,include_depth=False)
        ds.text_source = mode
        ds.prompt_dir = str(self.prompt_dir)
        ds.instruction_dir = str(self.instruction_dir)
        ds.grasp_files = self.labels
        ds.get_rgb = lambda *a,**k: np.zeros((3,32,32),dtype=np.float32)
        ds.get_gtbb = lambda *a,**k: types.SimpleNamespace(draw=lambda shape: (np.ones(shape),np.zeros(shape),np.ones(shape)))
        return ds
    def test_object_returns_original_prompt_and_query(self):
        ds = self.dataset('object')
        for file in self.instruction_dir.iterdir():
            file.unlink()
        self.assertEqual(ds.get_prompts(0), ('Entire scene text','White mug'))
        self.assertEqual(ds.get_prompts(1), ('Entire scene text','White mug'))
    def test_instructions_are_part_specific_and_need_no_scene_file(self):
        self.prompt.unlink()
        ds = self.dataset('instruction')
        self.assertEqual(ds.get_prompts(0), ('',self.instructions[0]))
        self.assertEqual(ds.get_prompts(1), ('',self.instructions[1]))
    def test_missing_instruction_never_falls_back_to_object_text(self):
        (self.instruction_dir/(Path(self.labels[0]).stem+'.pkl')).unlink()
        with self.assertRaises(FileNotFoundError):
            self.dataset('instruction').get_prompts(0)
    def test_invalid_instruction_is_rejected(self):
        file = self.instruction_dir/(Path(self.labels[0]).stem+'.pkl')
        for invalid in ('', '   ', ['text'], {'text':'instruction'}):
            with self.subTest(invalid=invalid):
                file.write_bytes(pickle.dumps(invalid))
                with self.assertRaises(ValueError):
                    self.dataset('instruction').get_prompts(0)
    def test_batch_contract_and_labels_are_equal_between_modes(self):
        batches = [next(iter(torch.utils.data.DataLoader(self.dataset(mode),batch_size=2))) for mode in ('object','instruction')]
        self.assertTrue(torch.equal(batches[0][0],batches[1][0]))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(batches[0][1],batches[1][1])))
        self.assertEqual(list(batches[0][-1]), ['White mug','White mug'])
        self.assertEqual(list(batches[1][-1]), self.instructions)
        self.assertEqual(list(batches[1][-2]), ['',''])
    def test_constructor_defaults_to_original_source_and_consumes_new_option(self):
        self.assertEqual(inspect.signature(GraspAnywhereDataset).parameters['text_source'].default,'object')
        with patch('utils.data.grasp_anywhere_data.glob.glob',side_effect=lambda pattern: self.labels if 'grasp_label_positive' in pattern else []), patch('builtins.open',mock_open(read_data=b'')), patch('utils.data.grasp_anywhere_data.pickle.load',return_value=[self.scene+'_0']):
            ds = GraspAnywhereDataset('base',add_file_path='plus',seen=1,include_rgb=True,include_depth=False,text_source='instruction')
        self.assertEqual(ds.text_source,'instruction')
        self.assertEqual(len(ds),2)
        with self.assertRaises(ValueError):
            GraspAnywhereDataset('base',text_source='concat')
    def test_train_and_eval_cli_default_and_selection(self):
        for name in ('train_network.py','evaluate_test.py'):
            tree = ast.parse((ROOT/name).read_text(encoding='utf-8'))
            fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='parse_args')
            ns = {'argparse':argparse}
            exec(compile(ast.Module(body=[fn],type_ignores=[]),name,'exec'),ns)
            for flags,expected in [([], 'object'),(['--dataset','grasp-anywhere','--text-source','instruction'],'instruction')]:
                with patch.object(sys,'argv',[name,*flags]):
                    self.assertEqual(ns['parse_args']().text_source,expected)
            with patch.object(sys,'argv',[name,'--dataset','grasp-anything','--text-source','instruction']), patch('sys.stderr',new=io.StringIO()):
                with self.assertRaises(SystemExit):
                    ns['parse_args']()
    def test_selected_source_is_forwarded_only_to_plus_loader(self):
        for name in ('train_network.py','evaluate_test.py'):
            tree=ast.parse((ROOT/name).read_text(encoding='utf-8'))
            run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
            assignments=[n for n in run.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('text_options','dataset') for t in n.targets)]
            for dataset in ('grasp-anywhere','grasp-anything'):
                args=types.SimpleNamespace(dataset=dataset,segmentation=False,text_source='instruction' if dataset=='grasp-anywhere' else 'object',dataset_path='base',add_file_path='plus',input_size=224,ds_rotate=0,use_depth=False,use_rgb=True,seen=1)
                constructor=MagicMock()
                exec(compile(ast.Module(body=assignments,type_ignores=[]),name,'exec'),{'args':args,'Dataset':constructor})
                kwargs=constructor.call_args.kwargs
                if dataset=='grasp-anywhere':
                    self.assertEqual(kwargs['text_source'],'instruction')
                else:
                    self.assertNotIn('text_source',kwargs)

if __name__ == '__main__':
    unittest.main()
