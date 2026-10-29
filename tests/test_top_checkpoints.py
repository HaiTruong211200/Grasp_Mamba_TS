import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from utils.checkpoints import TopKCheckpoints


class TopCheckpointTests(unittest.TestCase):
    def test_keeps_five_best_epochs_not_only_record_highs(self):
        with tempfile.TemporaryDirectory() as folder:
            saver=TopKCheckpoints(folder)
            scores=[.9,.1,.8,.2,.7,.3,.6,.4,.5,.85]
            for epoch,score in enumerate(scores):
                saver.save({'epoch':torch.tensor(epoch)},epoch,score)
                self.assertLessEqual(len(list(Path(folder).glob('*.pt'))),5)
            self.assertEqual([entry[1] for entry in saver.saved],[0,9,2,4,6])
            self.assertEqual(len(list(Path(folder).iterdir())),5)
            for score,epoch,path in saver.saved:
                self.assertEqual(torch.load(path,weights_only=True)['epoch'].item(),epoch)
    def test_full_precision_ranking_and_ties_keep_earlier_epochs(self):
        with tempfile.TemporaryDirectory() as folder:
            saver=TopKCheckpoints(folder,k=2)
            saver.save({},0,.5000001)
            saver.save({},1,.5000002)
            saver.save({},2,.5000003)
            self.assertIsNone(saver.save({},3,.5000002))
            self.assertEqual([v[1] for v in saver.saved],[2,1])
            self.assertEqual(len(list(Path(folder).glob('*.pt'))),2)
    def test_failed_write_preserves_existing_models_and_removes_temp(self):
        with tempfile.TemporaryDirectory() as folder:
            saver=TopKCheckpoints(folder,k=1)
            original=saver.save({'value':torch.tensor(1)},0,.5)
            def fail(state,path):
                Path(path).write_bytes(b'partial checkpoint')
                raise OSError('simulated full disk')
            with patch('utils.checkpoints.torch.save',side_effect=fail):
                with self.assertRaises(OSError):
                    saver.save({},1,.8)
            self.assertTrue(original.exists())
            self.assertEqual(len(saver.saved),1)
            self.assertEqual(list(Path(folder).iterdir()),[original])
            with self.assertRaises(ValueError):
                saver.save({},2,float('nan'))

if __name__=='__main__':
    unittest.main()
