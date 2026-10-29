import argparse
import datetime
import json
import logging
import os
import random
import sys

import cv2
import numpy as np
try:
    import tensorboardX
except ModuleNotFoundError:
    from torch.utils import tensorboard as tensorboardX
import torch
import torch.optim as optim
import torch.utils.data

from hardware.device import get_device
from inference.models import get_network
from inference.post_process import post_process_output
from utils.data import get_dataset
from utils.dataset_processing import evaluation
from utils.visualisation.gridshow import gridshow
import wandb
from tqdm import tqdm
import re
import glob
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Train network')

    # Network
    parser.add_argument('--vision-model', default='nvidia/MambaVision-L-1K',
                        help='MambaVision-L Hugging Face ID or local pretrained directory')
    parser.add_argument('--text-model', default='google/siglip-base-patch16-224',
                        help='SigLIP Hugging Face ID or local model and processor directory')
    parser.add_argument('--network', type=str, default='graspmamba',
                        help='Network name in inference/models')
    parser.add_argument('--input-size', type=int, default=224,
                        help='Input image size for the network')
    parser.add_argument('--use-depth', type=int, default=1,
                        help='Use Depth image for training (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for training (1/0)')
    parser.add_argument('--use-dropout', type=int, default=1,
                        help='Use dropout for training (1/0)')
    parser.add_argument('--dropout-prob', type=float, default=0.1,
                        help='Dropout prob for training (0-1)')
    parser.add_argument('--channel-size', type=int, default=32,
                        help='Internal channel size for the network')
    parser.add_argument('--iou-threshold', type=float, default=0.25,
                        help='Threshold for IOU matching')

    # Datasets
    parser.add_argument('--dataset', type=str, default='grasp-anything',
                        help='Dataset Name ("cornell" or "jaquard")')
    parser.add_argument('--dataset-path', type=str,
                        help='Path to dataset')
    parser.add_argument('--split', type=float, default=0.9,
                        help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-shuffle', action='store_true', default=False,
                        help='Shuffle the dataset')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Dataset workers')

    # Training
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs')
    parser.add_argument('--batches-per-epoch', type=int, default=1000,
                        help='Batches per Epoch')
    parser.add_argument('--optim', type=str, default='adam',
                        help='Optmizer for the training. (adam or SGD)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')

    # Logging etc.
    parser.add_argument('--description', type=str, default='',
                        help='Training description')
    parser.add_argument('--logdir', type=str, default='logs/',
                        help='Log directory')
    parser.add_argument('--vis', action='store_true', default=False,
                        help='Visualise the training process')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')
    parser.add_argument('--random-seed', type=int, default=123,
                        help='Random seed for numpy')
    parser.add_argument('--seen', type=int, default=1,
                        help='Flag for using seen classes, only work for Grasp-Anything dataset') 
    parser.add_argument('--add-file-path', type=str, default='data/grasp-anywhere',
                        help='Specific for Grasp-Anywhere')
    
    parser.add_argument('--segmentation', action='store_true', help='Train auxiliary target segmentation')
    parser.add_argument('--lambda-seg', type=float, default=0.5)
    parser.add_argument('--lambda-dice', type=float, default=1.0)
    parser.add_argument('--grasp-loss', choices=['mse', 'smooth_l1'], default='mse')
    parser.add_argument('--mask-target', choices=['object', 'part', 'union'], default='object')
    parser.add_argument('--normalization', choices=['legacy', 'imagenet'], default='legacy')
    args = parser.parse_args()
    if args.segmentation and args.dataset != 'grasp-anything-language':
        parser.error('--segmentation requires --dataset grasp-anything-language and a JSONL manifest')
    if args.batches_per_epoch <= 0:
        parser.error('--batches-per-epoch must be positive')
    return args


def prepare_batch(batch, device):
    if isinstance(batch, dict):
        moved = dict(batch)
        moved['image'] = batch['image'].to(device)
        moved['grasp_targets'] = {k: v.to(device) for k, v in batch['grasp_targets'].items()}
        if 'target_mask' in batch:
            moved['target_mask'] = batch['target_mask'].to(device)
        return moved
    x, y, *rest = batch
    return (x.to(device), [v.to(device) for v in y], rest[-1])


def batch_loss(net, batch):
    return net.compute_loss(batch) if isinstance(batch, dict) else net.compute_loss(*batch)


def show_training_batch(batch, pred):
    """Display grasp maps and optional target/predicted masks for one sample."""
    if isinstance(batch, dict):
        image = batch['image'][0].detach().cpu().numpy().transpose(1, 2, 0)
    else:
        image = batch[0][0].detach().cpu().numpy().transpose(1, 2, 0)
    image = (image - image.min()) / max(float(image.max() - image.min()), 1e-6)
    panels = [('Input', image), ('Grasp quality', pred['pos'][0, 0].detach().cpu().numpy())]
    if 'mask_logits' in pred:
        panels += [('Target mask', batch['target_mask'][0, 0].cpu().numpy()),
                   ('Predicted mask', pred['mask_logits'][0, 0].detach().sigmoid().cpu().numpy())]
    fig = plt.figure('Training')
    fig.clear()
    for i, (title, values) in enumerate(panels, 1):
        ax = fig.add_subplot(1, len(panels), i)
        ax.imshow(values.squeeze(), cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    plt.pause(0.001)


def validate(net, device, val_data, iou_threshold, visualize=False):
    net.eval()
    results = {'correct': 0, 'failed': 0, 'loss': 0, 'losses': {}}
    count = 0
    with torch.no_grad():
        for raw in val_data:
            batch = prepare_batch(raw, device)
            lossd = batch_loss(net, batch)
            size = batch['image'].shape[0] if isinstance(batch, dict) else batch[0].shape[0]
            count += size
            results['loss'] += lossd['loss'].item() * size
            for name, value in lossd['losses'].items():
                results['losses'][name] = results['losses'].get(name, 0) + value.item() * size
            for i in range(size):
                pred = lossd['pred']
                width_scale = batch['image'].shape[-1] / 2 if isinstance(batch, dict) else None
                q, angle, width = post_process_output(
                    *(pred[k][i:i+1] for k in ('pos', 'cos', 'sin', 'width')),
                    width_scale=width_scale)
                if isinstance(batch, dict):
                    idx = batch['metadata'][i]['index']
                    gt = val_data.dataset.get_gtbb(idx)
                else:
                    idx, rot, zoom = (raw[j][i].item() for j in (2, 3, 4))
                    gt = val_data.dataset.get_gtbb(int(idx), rot, zoom)
                success = evaluation.calculate_iou_match(
                    q, angle, gt, no_grasps=1 if isinstance(batch, dict) else 15,
                    grasp_width=width, threshold=iou_threshold)
                results['correct' if success else 'failed'] += 1
                if visualize:
                    rgb = val_data.dataset.get_rgb(int(idx), normalise=False)
                    query = batch['instruction'][i] if isinstance(batch, dict) else raw[-1][i]
                    evaluation.plot_output(plt.figure(), rgb, q, angle, gt,
                                           query, '', no_grasps=1, grasp_width_img=width)
                if 'mask_logits' in pred:
                    mask = pred['mask_logits'][i].sigmoid() >= 0.5
                    target = batch['target_mask'][i].bool()
                    intersection = (mask & target).sum().item()
                    union = (mask | target).sum().item()
                    denom = mask.sum().item() + target.sum().item()
                    results['mask_iou'] = results.get('mask_iou', 0) + (intersection / union if union else 1)
                    results['mask_dice'] = results.get('mask_dice', 0) + (2 * intersection / denom if denom else 1)
    if count == 0:
        raise ValueError('Validation loader is empty')
    results['loss'] /= count
    results['losses'] = {k: v / count for k, v in results['losses'].items()}
    for key in ('mask_iou', 'mask_dice'):
        if key in results:
            results[key] /= count
    return results


def train(epoch, net, device, train_data, optimizer, batches_per_epoch, args, vis=True):
    if len(train_data) == 0 or batches_per_epoch <= 0:
        raise ValueError('Training requires a nonempty loader and positive update count')
    results = {'loss': 0, 'losses': {}}
    net.train()
    iterator = iter(train_data)
    progress = tqdm(range(batches_per_epoch), desc=f"Epoch {epoch + 1}",
                    unit="batch", dynamic_ncols=True, file=sys.stdout)
    for batch_idx in progress:
        try:
            raw = next(iterator)
        except StopIteration:
            iterator = iter(train_data)
            raw = next(iterator)
        batch = prepare_batch(raw, device)
        # Warmup is defined over actual optimizer updates, including across epochs.
        step = epoch * batches_per_epoch + batch_idx + 1
        if step <= 500:
            for group in optimizer.param_groups:
                group['lr'] = args.lr * step / 500
        optimizer.zero_grad(set_to_none=True)
        lossd = batch_loss(net, batch)
        loss = lossd['loss']
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        if vis:
            show_training_batch(batch, lossd['pred'])
        results['loss'] += loss.item()
        for name, value in lossd['losses'].items():
            results['losses'][name] = results['losses'].get(name, 0) + value.item()
        progress.set_postfix(loss=f'{loss.item():.4f}',
                             avg_loss=f"{results['loss'] / (batch_idx + 1):.4f}",
                             refresh=False)
        if (batch_idx + 1) % 100 == 0:
            logging.info('Epoch %d, update %d, loss %.4f', epoch, batch_idx + 1, loss.item())
    results['loss'] /= batches_per_epoch
    results['losses'] = {k: v / batches_per_epoch for k, v in results['losses'].items()}
    return results


def run():
    args = parse_args()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="grasp-mamba-L",
    #     # track hyperparameters and run metadata
    #     config={
    #     "learning_rate": 0.0001, # Adam default
    #     "architecture": "GraspMamba-SigLIP",
    #     "dataset": "Grasp-Anything",
    #     "epochs": 50,
    #     "weight_decay":1e-5, # default
    #     "batch_size":8,
    #     }
    # )

    # Set-up output directories
    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    net_desc = '{}_{}'.format(dt, '_'.join(args.description.split()))

    save_folder = os.path.join(args.logdir, net_desc)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    tb = tensorboardX.SummaryWriter(save_folder)

    # Save commandline args
    if args is not None:
        params_path = os.path.join(save_folder, 'commandline_args.json')
        with open(params_path, 'w') as f:
            json.dump(vars(args), f)

    # Initialize logging
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        filename="{0}/{1}.log".format(save_folder, 'log'),
        format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    # Get the compute device
    device = get_device(args.force_cpu)

    # missing_files = check_dataset_integrity(args.dataset_path)
    # if any(missing_files.values()):
    #     logging.error("There are missing files in the dataset. Please check the logs.")
    # else:
    #     logging.info("All dataset files are present. You can proceed with training.")
        
    # Load Dataset
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)
    if args.dataset == 'grasp-anything-language':
        from utils.data.grasp_anything_language_data import language_collate
        options = dict(output_size=args.input_size, segmentation=args.segmentation,
                       mask_target=args.mask_target, normalization=args.normalization)
        dataset = Dataset(args.dataset_path, split='train', **options)
        validation_dataset = Dataset(args.dataset_path, split='val', **options)
        train_data = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, collate_fn=language_collate)
        val_data = torch.utils.data.DataLoader(
            validation_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=language_collate)
    else:
        dataset = Dataset(args.dataset_path,
                          output_size=args.input_size,
                          ds_rotate=args.ds_rotate,
                          random_rotate=True,
                          random_zoom=True,
                          include_depth=args.use_depth,
                          include_rgb=args.use_rgb,
                          seen=args.seen,
                          add_file_path=args.add_file_path)
        logging.info('Dataset size is {}'.format(dataset.length))

        # Creating data indices for training and validation splits
        indices = list(range(dataset.length))
        split = int(np.floor(args.split * dataset.length))
        if args.ds_shuffle:
            np.random.seed(args.random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[:split], indices[split:]
        logging.info('Training size: {}'.format(len(train_indices)))
        logging.info('Validation size: {}'.format(len(val_indices)))

        # Creating data samplers and loaders
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

        train_data = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sampler=train_sampler,
            pin_memory=True
        )
        val_data = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=args.num_workers,
            sampler=val_sampler,
            pin_memory=True
        )
    logging.info('Done')

    # Load the network
    logging.info('Loading Network...')
    network = get_network(args.network)
    net = network(input_size=args.input_size, segmentation=args.segmentation,
                  lambda_seg=args.lambda_seg, lambda_dice=args.lambda_dice,
                  grasp_loss=args.grasp_loss,
                  vision_model=args.vision_model, text_model=args.text_model)

    net = net.to(device)
    logging.info('Done')

    if args.optim.lower() == 'adam':
        optimizer = optim.Adam(net.parameters(),lr=args.lr)
    elif args.optim.lower() == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(args.optim))

   # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    best_iou = 0.0
    for epoch in range(args.epochs):
        print('Training Epoch {:02d}'.format(epoch))
        logging.info('Beginning Epoch {:02d}'.format(epoch))
        train_results = train(epoch, net, device, train_data, optimizer, args.batches_per_epoch, args=args, vis=args.vis)

        # Log training losses to tensorboard
        tb.add_scalar('loss/train_loss', train_results['loss'], epoch)
        for n, l in train_results['losses'].items():
            tb.add_scalar('train_loss/' + n, l, epoch)
            
        # Log training losses to wandb
        if wandb.run is not None:
            wandb.log({'epoch': epoch, 'train_loss': train_results['loss'],
                       **{f'train_loss/{k}': v for k, v in train_results['losses'].items()}})
        
        # Run Validation
        logging.info('Validating...')
        test_results = validate(net, device, val_data, args.iou_threshold)
        logging.info('%d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
                                     test_results['correct'] / (test_results['correct'] + test_results['failed'])))

        # Log validation results to tensorbaord
        tb.add_scalar('loss/IOU', test_results['correct'] / (test_results['correct'] + test_results['failed']), epoch)
        tb.add_scalar('loss/val_loss', test_results['loss'], epoch)
        for n, l in test_results['losses'].items():
            tb.add_scalar('val_loss/' + n, l, epoch)
        
        # # Log validation results to wandb
        # wandb.log({'epoch': epoch, 'IOU': test_results['correct'] / (test_results['correct'] + test_results['failed']), 'val_loss': test_results['loss']})
        # for n, l in test_results['losses'].items():
        #     wandb.log({f'val_loss/{n}': l, 'epoch': epoch})

        for metric in ('mask_iou', 'mask_dice'):
            if metric in test_results:
                tb.add_scalar('validation/' + metric, test_results[metric], epoch)

        # Save best performing network
        iou = test_results['correct'] / (test_results['correct'] + test_results['failed'])

        # Update learning rate
        scheduler.step(iou)
        if iou > best_iou or epoch == 0 or (epoch % 10) == 0:
            torch.save(net.state_dict(), os.path.join(save_folder, 'epoch_%02d_iou_%0.2f' % (epoch, iou)))
            best_iou = iou
    
    # # Finish the wandb run
    # wandb.finish()


if __name__ == '__main__':
    run()
