"""Keep the highest validation scores for checkpoints produced by one training run."""
import logging
import math
from pathlib import Path
import torch


class TopKCheckpoints:
    def __init__(self, directory, k=5):
        if not isinstance(k, int) or k <= 0:
            raise ValueError('k must be a positive integer')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.k = k
        self.saved = []  # (full-precision validation score, epoch, path)

    def save(self, state_dict, epoch, score):
        score = float(score)
        if not math.isfinite(score):
            raise ValueError('Checkpoint validation score must be finite')
        # Keep earlier epochs when scores tie. Do not write a non-top-k model.
        if len(self.saved) >= self.k and score <= self.saved[-1][0]:
            return None
        path = self.directory / f'epoch_{epoch:02d}_iou_{score:.6f}.pt'
        temporary = path.with_suffix('.tmp')
        try:
            torch.save(state_dict, temporary)
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
        self.saved.append((score, epoch, path))
        self.saved.sort(key=lambda item: (-item[0], item[1]))
        if len(self.saved) > self.k:
            _, old_epoch, old_path = self.saved.pop()
            old_path.unlink()
            logging.info('Removed checkpoint outside top %d: epoch %d', self.k, old_epoch)
        logging.info('Saved checkpoint: epoch %d, val IoU %.6f (top %d)', epoch, score, self.k)
        return path
