import argparse
import ast
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import mock_open, patch
from utils.data.grasp_anywhere_data import GraspAnywhereDataset

ROOT=Path(__file__).resolve().parents[1]

class OfficialTestSplitTests(unittest.TestCase):
    def dataset(self,seen=1,eval_split=None):
        ids={'train/seen.obj':'a'*64+'_0','test/seen.obj':'b'*64+'_0','test/unseen.obj':'c'*64+'_0'}
        labels=['/plus/grasp_label_positive/'+value+'_0.pt' for value in ids.values()]
        def split(f):
            path=str(f.name).replace('\\','/')
            return [next(value for suffix,value in ids.items() if path.endswith(suffix))]
        def open_split(path,*args,**kw):
            handle=io.BytesIO(b'')
            handle.name=path
            return handle
        with patch('utils.data.grasp_anywhere_data.glob.glob',side_effect=lambda path: labels if 'grasp_label_positive' in path else []),patch('builtins.open',side_effect=open_split),patch('utils.data.grasp_anywhere_data.pickle.load',side_effect=split):
            return GraspAnywhereDataset('/base',add_file_path='/plus',seen=seen,eval_split=eval_split,include_rgb=True,include_depth=False)
    def test_legacy_train_and_unseen_selection_preserved(self):
        self.assertIn('a'*64,self.dataset(seen=1).grasp_files[0])
        self.assertIn('c'*64,self.dataset(seen=0).grasp_files[0])
    def test_explicit_seen_selects_heldout_seen_even_if_seen_flag_zero(self):
        self.assertIn('b'*64,self.dataset(seen=0,eval_split='test_seen').grasp_files[0])
    def test_explicit_unseen_selects_unseen_even_if_seen_flag_one(self):
        self.assertIn('c'*64,self.dataset(seen=1,eval_split='test_unseen').grasp_files[0])
    def test_evaluator_accepts_test_seen_and_rejects_other_dataset(self):
        tree=ast.parse((ROOT/'evaluate_test.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='parse_args')
        ns={'argparse':argparse}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<parse_args>','exec'),ns)
        with patch.object(sys,'argv',['evaluate_test.py','--dataset','grasp-anywhere','--eval-split','test_seen']):
            self.assertEqual(ns['parse_args']().eval_split,'test_seen')
        with patch.object(sys,'argv',['evaluate_test.py','--dataset','grasp-anything','--eval-split','test_seen']),patch('sys.stderr',new=io.StringIO()):
            with self.assertRaises(SystemExit):
                ns['parse_args']()

if __name__=='__main__':
    unittest.main()
