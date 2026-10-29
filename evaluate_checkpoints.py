"""Evaluate every retained checkpoint on official ++ test seen and unseen splits.
Checkpoints stay ordered by their saved validation score; test scores are never
used to select or delete models. Reuses evaluate_test.validate (five grasps).
"""
import argparse
import csv
import gc
import json
import logging
from pathlib import Path
import re
import time

CHECKPOINT_NAME=re.compile(r'^epoch_(\d+)_iou_(\d+(?:\.\d+)?)(?:\.pt)?$')


def find_checkpoints(directory):
    directory=Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f'Checkpoint directory not found: {directory}')
    found=[]
    for path in directory.iterdir():
        match=CHECKPOINT_NAME.fullmatch(path.name)
        if path.is_file() and match:
            found.append({'path':path,'epoch':int(match[1]),'validation_accuracy':float(match[2])})
    if not found:
        raise FileNotFoundError(f'No epoch_*_iou_* checkpoints found in {directory}')
    return sorted(found,key=lambda item:(-item['validation_accuracy'],item['epoch'],item['path'].name))


def find_latest_checkpoint_dir(logs_dir):
    """Find the run containing the most recently written completed checkpoint."""
    logs_dir=Path(logs_dir)
    if not logs_dir.is_dir():
        raise FileNotFoundError(f'Logs directory not found: {logs_dir}')
    candidates=[]
    for path in logs_dir.rglob('epoch_*_iou_*'):
        if not CHECKPOINT_NAME.fullmatch(path.name):
            continue
        try:
            if path.is_file():
                candidates.append((path.stat().st_mtime_ns,str(path.parent),path.parent))
        except FileNotFoundError:
            continue
    if not candidates:
        raise FileNotFoundError(f'No completed checkpoints found under {logs_dir}')
    return max(candidates,key=lambda item:(item[0],item[1]))[2].resolve()


def split_result(metrics):
    correct=int(metrics['correct'])
    failed=int(metrics['failed'])
    total=correct+failed
    if total<=0 or correct<0 or failed<0:
        raise ValueError('Evaluation must contain nonnegative counts and at least one sample')
    return {'correct':correct,'total':total,'accuracy':correct/total,
            'mean_loss':float(metrics.get('loss',0.))/total}


def write_results(report,output_dir):
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    path=output_dir/'checkpoint_results.json'
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)
    fields=['checkpoint','epoch','validation_accuracy_percent','segmentation',
            'seen_correct','seen_total','seen_accuracy_percent',
            'unseen_correct','unseen_total','unseen_accuracy_percent','H_percent']
    path=output_dir/'checkpoint_results.csv'
    temporary=path.with_suffix('.csv.tmp')
    with temporary.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for row in report['results']:
            seen,unseen=row['test_seen'],row['test_unseen']
            writer.writerow({'checkpoint':row['checkpoint'],'epoch':row['epoch'],
                'validation_accuracy_percent':100*row['validation_accuracy_from_filename'],
                'segmentation':row['segmentation'],'seen_correct':seen['correct'],
                'seen_total':seen['total'],'seen_accuracy_percent':100*seen['accuracy'],
                'unseen_correct':unseen['correct'],'unseen_total':unseen['total'],
                'unseen_accuracy_percent':100*unseen['accuracy'],'H_percent':100*row['H']})
    temporary.replace(path)


def evaluate_all(checkpoints,*,load_state,make_model,make_loaders,validate,device,
                 iou_threshold,output_dir,settings,cleanup=lambda:None):
    report={'metric':'grasp_accuracy','checkpoint_count':len(checkpoints),
            'completed_count':0,'complete':False,
            'checkpoint_order':'saved validation score descending; not test score',
            'protocol':{'no_grasps':5,'iou_threshold':iou_threshold,
                        'angle_threshold_degrees':30,'batch_size':1},
            'settings':settings,'results':[]}
    model=None
    model_mode=None
    loaders_by_mode={}
    for index,checkpoint in enumerate(checkpoints,1):
        path=checkpoint['path']
        logging.info('[%d/%d] Testing %s',index,len(checkpoints),path.name)
        state=load_state(path)
        segmentation=any(key.startswith('grasp_mamba.seg_head.') for key in state)
        if model is None or model_mode!=segmentation:
            model=None
            gc.collect()
            cleanup()
            model=make_model(segmentation)
            model_mode=segmentation
        model.load_state_dict(state,strict=True)
        del state
        if segmentation not in loaders_by_mode:
            loaders_by_mode[segmentation]=make_loaders(segmentation)
        scores={}
        for split in ('test_seen','test_unseen'):
            loader=loaders_by_mode[segmentation][split]
            logging.info('%s: %d samples',split,len(loader.dataset))
            result=split_result(validate(model,device,loader,iou_threshold,plot=False))
            if result['total']!=len(loader.dataset):
                raise RuntimeError(f'{split}: evaluated {result["total"]}, expected {len(loader.dataset)}')
            scores[split]=result
        s,u=scores['test_seen']['accuracy'],scores['test_unseen']['accuracy']
        row={'checkpoint':str(path),'epoch':checkpoint['epoch'],
             'validation_accuracy_from_filename':checkpoint['validation_accuracy'],
             'segmentation':segmentation,**scores,'H':2*s*u/(s+u) if s+u else 0.}
        report['results'].append(row)
        report['completed_count']=index
        report['complete']=index==len(checkpoints)
        write_results(report,output_dir)
        print(f'{path.name}: seen={100*s:.2f}% ({scores["test_seen"]["correct"]}/{scores["test_seen"]["total"]}), '
              f'unseen={100*u:.2f}% ({scores["test_unseen"]["correct"]}/{scores["test_unseen"]["total"]}), H={100*row["H"]:.2f}%',flush=True)
    return report


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint-dir',type=Path,help='Explicit checkpoint run folder')
    source.add_argument('--logs-dir',type=Path,help='Automatically select the run with the newest checkpoint')
    parser.add_argument('--dataset-path',required=True)
    parser.add_argument('--add-file-path',required=True)
    parser.add_argument('--vision-model')
    parser.add_argument('--text-model')
    parser.add_argument('--text-source',choices=['object','instruction'])
    parser.add_argument('--input-size',type=int)
    parser.add_argument('--lambda-seg',type=float)
    parser.add_argument('--lambda-dice',type=float)
    parser.add_argument('--num-workers',type=int,default=1)
    parser.add_argument('--iou-threshold',type=float,default=.25)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    if args.num_workers<0:
        parser.error('--num-workers must be nonnegative')
    if args.logs_dir is not None:
        try:
            args.checkpoint_dir=find_latest_checkpoint_dir(args.logs_dir)
        except FileNotFoundError as error:
            parser.error(str(error))
        print(f'Automatically selected checkpoint folder: {args.checkpoint_dir}',flush=True)
    config_path=args.checkpoint_dir/'commandline_args.json'
    config=json.loads(config_path.read_text(encoding='utf-8')) if config_path.is_file() else {}
    defaults={'vision_model':'nvidia/MambaVision-L-1K','text_model':'google/siglip-base-patch16-224',
              'text_source':'object','input_size':224,'lambda_seg':.5,'lambda_dice':1.}
    for key,default in defaults.items():
        if getattr(args,key) is None:
            setattr(args,key,config.get(key,default))
    if args.text_source not in ('object','instruction'):
        parser.error('Training configuration has an unsupported text source; set --text-source')
    if args.output_dir is None:
        args.output_dir=Path('evaluation_results')/args.checkpoint_dir.name
    return args


def main():
    args=parse_args()
    checkpoints=find_checkpoints(args.checkpoint_dir)
    import torch
    from inference.models import get_network
    from utils.data.grasp_anywhere_data import GraspAnywhereDataset
    from evaluate_test import validate
    if not torch.cuda.is_available():
        raise RuntimeError('The original MambaVision implementation requires a CUDA GPU')
    logging.basicConfig(level=logging.INFO)
    device=torch.device('cuda')
    def make_model(segmentation):
        return get_network('graspmamba')(input_size=args.input_size,vision_model=args.vision_model,
            text_model=args.text_model,segmentation=segmentation,
            lambda_seg=args.lambda_seg,lambda_dice=args.lambda_dice).to(device)
    def make_loaders(segmentation):
        loaders={}
        for split in ('test_seen','test_unseen'):
            dataset=GraspAnywhereDataset(args.dataset_path,add_file_path=args.add_file_path,
                eval_split=split,text_source=args.text_source,segmentation=segmentation,
                output_size=args.input_size,seen=1,include_rgb=True,include_depth=False,
                random_rotate=False,random_zoom=False)
            loaders[split]=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=True,
                num_workers=args.num_workers,pin_memory=True)
        return loaders
    settings={key:getattr(args,key) for key in ('dataset_path','add_file_path','vision_model','text_model',
              'text_source','input_size','lambda_seg','lambda_dice','num_workers')}
    report=evaluate_all(checkpoints,load_state=lambda path:torch.load(path,map_location='cpu',weights_only=True),
        make_model=make_model,make_loaders=make_loaders,validate=validate,device=device,
        iou_threshold=args.iou_threshold,output_dir=args.output_dir,settings=settings,
        cleanup=torch.cuda.empty_cache)
    print('\nCheckpoint                               Val acc%   Seen acc%  Unseen acc%     H%',flush=True)
    for row in report['results']:
        print(f'{Path(row["checkpoint"]).name:40s} {100*row["validation_accuracy_from_filename"]:8.2f} '
              f'{100*row["test_seen"]["accuracy"]:10.2f} {100*row["test_unseen"]["accuracy"]:11.2f} {100*row["H"]:8.2f}',flush=True)
    print(f'Results: {args.output_dir / "checkpoint_results.csv"}\n         {args.output_dir / "checkpoint_results.json"}',flush=True)

if __name__=='__main__':
    main()
