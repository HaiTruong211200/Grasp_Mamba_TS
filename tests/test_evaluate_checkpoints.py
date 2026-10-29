"""Test retained-checkpoint enumeration and two-split report generation without a GPU."""
import csv
import io
import os
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from evaluate_checkpoints import evaluate_all,find_checkpoints,find_latest_checkpoint_dir,parse_args,split_result,checkpoint_read_error

class CheckpointEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory=Path(self.tmp.name)
    def files(self,count=5):
        for epoch in range(count):
            (self.directory/f'epoch_{epoch:02d}_iou_{.9-epoch*.1:.6f}.pt').write_bytes(b'checkpoint unchanged')
        return find_checkpoints(self.directory)
    def callbacks(self,mixed=False):
        calls={'model':[],'loaders':[],'validate':[],'states':[]}
        class Model:
            def load_state_dict(model,state,strict=False):
                self.assertTrue(strict)
                model.epoch=state['epoch']
                calls['states'].append(model.epoch)
        def load(path):
            epoch=int(path.name.split('_')[1])
            state={'epoch':epoch}
            if mixed and epoch==1:
                state['grasp_mamba.seg_head.layers.0.weight']=1
            return state
        def make_model(mode):
            calls['model'].append(mode)
            return Model()
        def make_loaders(mode):
            calls['loaders'].append(mode)
            return {name:types.SimpleNamespace(dataset=list(range(total)),split=name) for name,total in [('test_seen',10),('test_unseen',5)]}
        def validate(model,device,loader,threshold,plot=False):
            self.assertEqual(threshold,.25)
            self.assertFalse(plot)
            calls['validate'].append((model.epoch,loader.split))
            correct=model.epoch+1 if loader.split=='test_seen' else 5-model.epoch
            return {'correct':correct,'failed':len(loader.dataset)-correct,'loss':float(len(loader.dataset))}
        return calls,dict(load_state=load,make_model=make_model,make_loaders=make_loaders,validate=validate,device='test',iou_threshold=.25,output_dir=self.directory/'results',settings={})
    def test_finds_legacy_and_current_names_but_not_temporary_files(self):
        self.files(2)
        (self.directory/'epoch_08_iou_0.95').write_bytes(b'old style')
        (self.directory/'epoch_09_iou_0.99.tmp').write_bytes(b'partial')
        (self.directory/'commandline_args.json').write_text('{}')
        found=find_checkpoints(self.directory)
        self.assertEqual([row['epoch'] for row in found],[8,0,1])
    def test_evaluates_all_five_on_both_splits_and_keeps_validation_order(self):
        checkpoints=self.files()
        before={row['path']:row['path'].read_bytes() for row in checkpoints}
        calls,kwargs=self.callbacks()
        with patch('sys.stdout',new=io.StringIO()):
            report=evaluate_all(checkpoints,**kwargs)
        self.assertEqual(len(calls['validate']),10)
        self.assertEqual(calls['model'],[False])
        self.assertEqual(calls['loaders'],[False])
        self.assertEqual(calls['states'],list(range(5)))
        self.assertEqual([row['epoch'] for row in report['results']],list(range(5)))
        self.assertTrue(report['complete'])
        self.assertEqual(report['completed_count'],5)
        first=report['results'][0]
        self.assertAlmostEqual(first['H'],2*.1*1/(.1+1))
        self.assertEqual(first['test_seen']['total'],10)
        self.assertEqual(first['test_unseen']['total'],5)
        self.assertEqual(json.loads((kwargs['output_dir']/'checkpoint_results.json').read_text())['completed_count'],5)
        with (kwargs['output_dir']/'checkpoint_results.csv').open(newline='') as stream:
            rows=list(csv.DictReader(stream))
        self.assertEqual(len(rows),5)
        self.assertEqual(float(rows[0]['seen_accuracy_percent']),10.)
        self.assertTrue(all(path.read_bytes()==content for path,content in before.items()))
    def test_automatically_detects_segmentation_checkpoints(self):
        calls,kwargs=self.callbacks(mixed=True)
        with patch('sys.stdout',new=io.StringIO()):
            report=evaluate_all(self.files(2),**kwargs)
        self.assertEqual(calls['model'],[False,True])
        self.assertEqual(calls['loaders'],[False,True])
        self.assertEqual([row['segmentation'] for row in report['results']],[False,True])
    def test_keeps_partial_results_if_a_later_checkpoint_fails(self):
        _,kwargs=self.callbacks()
        original=kwargs['validate']
        def fail(model,*args,**kw):
            if model.epoch==1:
                raise RuntimeError('interrupted evaluation')
            return original(model,*args,**kw)
        kwargs['validate']=fail
        with patch('sys.stdout',new=io.StringIO()),self.assertRaises(RuntimeError):
            evaluate_all(self.files(2),**kwargs)
        report=json.loads((kwargs['output_dir']/'checkpoint_results.json').read_text())
        self.assertEqual(report['completed_count'],1)
        self.assertFalse(report['complete'])
    def test_empty_or_incomplete_evaluation_is_rejected(self):
        with self.assertRaises(ValueError):
            split_result({'correct':0,'failed':0})
        _,kwargs=self.callbacks()
        kwargs['validate']=lambda *a,**kw:{'correct':1,'failed':0}
        with self.assertRaises(RuntimeError):
            evaluate_all(self.files(1),**kwargs)
    def test_cli_uses_training_settings_and_explicit_override(self):
        (self.directory/'commandline_args.json').write_text(json.dumps({'text_source':'instruction','input_size':224,'vision_model':'/models/vision','text_model':'/models/text','lambda_dice':0}))
        with patch.object(sys,'argv',['evaluate_checkpoints.py','--checkpoint-dir',str(self.directory),'--dataset-path','base','--add-file-path','plus','--text-model','/override/text']):
            args=parse_args()
        self.assertEqual(args.text_source,'instruction')
        self.assertEqual(args.text_model,'/override/text')
        self.assertEqual(args.vision_model,'/models/vision')
        self.assertEqual(args.lambda_dice,0)
        with self.assertRaises(FileNotFoundError):
            find_checkpoints(self.directory)

    def test_limits_legacy_folder_to_five_checkpoints(self):
        calls,kwargs=self.callbacks()
        with patch('sys.stdout',new=io.StringIO()):
            report=evaluate_all(self.files(8),**kwargs)
        self.assertEqual(report['candidate_count'],8)
        self.assertEqual(report['checkpoint_count'],5)
        self.assertEqual(calls['states'],[0,1,2,3,4])
        self.assertEqual(len(calls['validate']),10)
    def test_corrupt_archive_is_skipped_and_next_candidate_is_evaluated(self):
        checkpoints=self.files(6)
        calls,kwargs=self.callbacks()
        original=kwargs['load_state']
        def load(path):
            if int(path.name.split('_')[1])==0:
                raise RuntimeError('PytorchStreamReader failed reading zip archive: failed finding central directory')
            return original(path)
        kwargs['load_state']=load
        with patch('sys.stdout',new=io.StringIO()):
            report=evaluate_all(checkpoints,**kwargs)
        self.assertTrue(report['complete'])
        self.assertTrue(report['finished'])
        self.assertEqual(calls['states'],[1,2,3,4,5])
        self.assertEqual(len(report['errors']),1)
        self.assertEqual(report['errors'][0]['stage'],'load')
        self.assertTrue(all(item['path'].exists() for item in checkpoints))
    def test_resume_reuses_four_completed_legacy_rows(self):
        checkpoints=self.files(6)
        _,kwargs=self.callbacks()
        original=kwargs['load_state']
        def fail(path):
            if int(path.name.split('_')[1])==4:
                raise RuntimeError('old run stopped here')
            return original(path)
        kwargs['load_state']=fail
        with patch('sys.stdout',new=io.StringIO()),self.assertRaises(RuntimeError):
            evaluate_all(checkpoints,**kwargs)
        file=kwargs['output_dir']/'checkpoint_results.json'
        old=json.loads(file.read_text())
        self.assertEqual(old['completed_count'],4)
        for row in old['results']:
            row.pop('checkpoint_fingerprint',None)
        file.write_text(json.dumps(old))
        calls,kwargs=self.callbacks()
        attempts=[]
        original=kwargs['load_state']
        def load(path):
            epoch=int(path.name.split('_')[1])
            attempts.append(epoch)
            if epoch==4:
                raise RuntimeError('PytorchStreamReader failed finding central directory')
            return original(path)
        kwargs['load_state']=load
        with patch('sys.stdout',new=io.StringIO()):
            report=evaluate_all(checkpoints,resume=True,**kwargs)
        self.assertEqual(attempts,[4,5])
        self.assertEqual(calls['states'],[5])
        self.assertEqual(len(calls['validate']),2)
        self.assertEqual([row['epoch'] for row in report['results']],[0,1,2,3,5])
        self.assertTrue(report['complete'])
    def test_not_enough_readable_files_is_explicitly_incomplete(self):
        _,kwargs=self.callbacks()
        original=kwargs['load_state']
        def load(path):
            if int(path.name.split('_')[1])==1:
                raise EOFError('truncated')
            return original(path)
        kwargs['load_state']=load
        with patch('sys.stdout',new=io.StringIO()):
            report=evaluate_all(self.files(2),**kwargs)
        self.assertFalse(report['complete'])
        self.assertTrue(report['finished'])
        self.assertEqual(report['completed_count'],1)
    def test_memory_errors_are_not_mislabeled_as_corruption(self):
        _,kwargs=self.callbacks()
        kwargs['load_state']=lambda path: (_ for _ in ()).throw(RuntimeError('CUDA out of memory'))
        with self.assertRaisesRegex(RuntimeError,'out of memory'):
            evaluate_all(self.files(1),**kwargs)
    def test_changed_resume_settings_are_rejected(self):
        checkpoints=self.files(1)
        _,kwargs=self.callbacks()
        with patch('sys.stdout',new=io.StringIO()):
            evaluate_all(checkpoints,**kwargs)
        kwargs['settings']={'text_source':'instruction'}
        with self.assertRaisesRegex(ValueError,'changed evaluation'):
            evaluate_all(checkpoints,resume=True,**kwargs)
    def test_real_truncated_torch_archive_is_recognized(self):
        import torch
        file=self.directory/'broken.pt'
        torch.save({'weight':torch.ones(2)},file)
        content=file.read_bytes()
        file.write_bytes(content[:len(content)//2])
        try:
            torch.load(file,map_location='cpu',weights_only=True)
        except Exception as error:
            self.assertTrue(checkpoint_read_error(error),repr(error))
        else:
            self.fail('Truncated archive unexpectedly loaded')


class LatestCheckpointFolderTests(unittest.TestCase):
    def test_uses_checkpoint_mtime_not_folder_name_or_other_logs(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            old=root/'999999_9999_misleading_name'
            new=root/'arbitrary_run_name'
            empty=root/'newest_empty_run'
            for directory in (old,new,empty):
                directory.mkdir()
            previous=old/'epoch_00_iou_0.99.pt'
            latest=new/'epoch_07_iou_0.70'
            previous.write_bytes(b'old')
            latest.write_bytes(b'latest')
            os.utime(previous,(100,100))
            os.utime(latest,(200,200))
            (old/'epoch_99_iou_0.999.tmp').write_bytes(b'partial')
            (empty/'log.txt').write_text('new log but no model')
            self.assertEqual(find_latest_checkpoint_dir(root),new.resolve())
    def test_auto_cli_reuses_selected_training_config_and_output_name(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            run=root/'custom_name_without_timestamp'
            run.mkdir()
            (run/'epoch_01_iou_0.50.pt').write_bytes(b'checkpoint')
            (run/'commandline_args.json').write_text(json.dumps({'text_source':'instruction'}))
            argv=['evaluate_checkpoints.py','--logs-dir',str(root),'--dataset-path','base','--add-file-path','plus']
            with patch.object(sys,'argv',argv),patch('sys.stdout',new=io.StringIO()):
                args=parse_args()
            self.assertEqual(args.checkpoint_dir,run.resolve())
            self.assertEqual(args.output_dir,Path('evaluation_results')/run.name)
            self.assertEqual(args.text_source,'instruction')
            with self.assertRaises(FileNotFoundError):
                find_latest_checkpoint_dir(root/'not-present')


if __name__=='__main__':
    unittest.main()
