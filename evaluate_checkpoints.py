"""Evaluate up to five readable checkpoints on official ++ test seen and unseen splits.
Checkpoints stay ordered by their saved validation score; test scores are never
used to select or delete models. Reuses evaluate_test.validate (five grasps).
"""
import argparse
import csv
import gc
import json
import logging
import pickle
import zipfile
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


def checkpoint_fingerprint(path):
    info=Path(path).stat()
    return {'size_bytes':info.st_size,'mtime_ns':info.st_mtime_ns}


def checkpoint_read_error(error):
    if isinstance(error,(OSError,EOFError,pickle.UnpicklingError,zipfile.BadZipFile)):
        return True
    if isinstance(error,RuntimeError):
        message=str(error).lower()
        return any(text in message for text in ('pytorchstreamreader','failed finding central directory',
                   'unexpected eof','invalid magic number','not a zip archive','invalid header'))
    return False


def resume_results(report_path,checkpoints,protocol,settings):
    if not report_path.is_file():
        return {}
    previous=json.loads(report_path.read_text(encoding='utf-8'))
    if previous.get('protocol')!=protocol or previous.get('settings')!=settings:
        raise ValueError('Cannot resume with changed evaluation protocol/settings; use the original settings or a new output directory')
    candidates={str(item['path'].resolve()):item['path'] for item in checkpoints}
    report_mtime=report_path.stat().st_mtime_ns
    cached={}
    for row in previous.get('results',[]):
        key=str(Path(row['checkpoint']).resolve())
        path=candidates.get(key)
        if path is None or 'test_seen' not in row or 'test_unseen' not in row:
            continue
        try:
            fingerprint=checkpoint_fingerprint(path)
        except FileNotFoundError:
            continue
        recorded=row.get('checkpoint_fingerprint')
        # Older reports lack a fingerprint: accept only files unchanged since that report.
        if (recorded is not None and recorded!=fingerprint) or (recorded is None and fingerprint['mtime_ns']>report_mtime):
            continue
        cached[key]={**row,'checkpoint':str(path),'checkpoint_fingerprint':fingerprint}
    return cached


def evaluate_all(checkpoints,*,load_state,make_model,make_loaders,validate,device,
                 iou_threshold,output_dir,settings,cleanup=lambda:None,
                 max_checkpoints=5,resume=False):
    if max_checkpoints<1:
        raise ValueError('max_checkpoints must be positive')
    target=min(max_checkpoints,len(checkpoints))
    report={'metric':'grasp_accuracy','checkpoint_count':target,'candidate_count':len(checkpoints),
            'completed_count':0,'complete':False,'finished':False,'errors':[],
            'checkpoint_order':'saved validation score descending; not test score',
            'protocol':{'no_grasps':5,'iou_threshold':iou_threshold,
                        'angle_threshold_degrees':30,'batch_size':1},
            'settings':settings,'results':[]}
    report_path=Path(output_dir)/'checkpoint_results.json'
    cached=resume_results(report_path,checkpoints,report['protocol'],settings) if resume else {}
    order={str(item['path'].resolve()):index for index,item in enumerate(checkpoints)}
    def update_report(finished=False):
        # Keep all still-compatible previous rows available if interrupted again.
        report['results']=sorted(cached.values(),key=lambda row:order[str(Path(row['checkpoint']).resolve())])[:target]
        report['completed_count']=len(report['results'])
        report['finished']=finished
        report['complete']=finished and report['completed_count']==target
        write_results(report,output_dir)
    update_report()
    model=None
    model_mode=None
    loaders_by_mode={}
    accepted=0
    for index,checkpoint in enumerate(checkpoints,1):
        path=checkpoint['path']
        key=str(path.resolve())
        if key in cached:
            accepted+=1
            logging.info('Reusing completed result: %s (%d/%d)',path.name,accepted,target)
            if accepted>=target:
                break
            continue
        logging.info('[candidate %d/%d; completed %d/%d] Testing %s',index,len(checkpoints),accepted,target,path.name)
        try:
            fingerprint=checkpoint_fingerprint(path)
            state=load_state(path)
        except Exception as error:
            if not checkpoint_read_error(error):
                raise
            report['errors'].append({'checkpoint':str(path),'stage':'load',
                                     'error_type':type(error).__name__,'message':str(error)})
            logging.warning('Skipping unreadable checkpoint %s: %s',path.name,error)
            update_report()
            continue
        segmentation=any(key.startswith('grasp_mamba.seg_head.') for key in state)
        if model is None or model_mode!=segmentation:
            model=None
            gc.collect()
            cleanup()
            model=make_model(segmentation)
            model_mode=segmentation
        # Architecture/programming errors are not treated as corrupt checkpoint files.
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
        seen,unseen=scores['test_seen']['accuracy'],scores['test_unseen']['accuracy']
        cached[key]={'checkpoint':str(path),'epoch':checkpoint['epoch'],
                     'checkpoint_fingerprint':fingerprint,
                     'validation_accuracy_from_filename':checkpoint['validation_accuracy'],
                     'segmentation':segmentation,**scores,
                     'H':2*seen*unseen/(seen+unseen) if seen+unseen else 0.}
        accepted+=1
        update_report(finished=accepted>=target)
        print(f'{path.name}: seen={100*seen:.2f}% ({scores["test_seen"]["correct"]}/{scores["test_seen"]["total"]}), '
              f'unseen={100*unseen:.2f}% ({scores["test_unseen"]["correct"]}/{scores["test_unseen"]["total"]}), H={100*cached[key]["H"]:.2f}%',flush=True)
        if accepted>=target:
            break
    update_report(finished=True)
    if not report['results']:
        raise RuntimeError(f'No checkpoint could be evaluated; read errors in {report_path}')
    if not report['complete']:
        logging.warning('Only %d/%d checkpoints evaluated; %d unreadable files recorded in JSON',report['completed_count'],target,len(report['errors']))
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
    parser.add_argument('--max-checkpoints',type=int,default=5,help='Maximum readable checkpoints to test, in validation-score order')
    parser.add_argument('--resume',action='store_true',help='Reuse compatible completed rows in the same output directory')
    args=parser.parse_args()
    if args.num_workers<0 or args.max_checkpoints<1:
        parser.error('--num-workers must be nonnegative and --max-checkpoints positive')
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
        cleanup=torch.cuda.empty_cache,max_checkpoints=args.max_checkpoints,resume=args.resume)
    print('\nCheckpoint                               Val acc%   Seen acc%  Unseen acc%     H%',flush=True)
    for row in report['results']:
        print(f'{Path(row["checkpoint"]).name:40s} {100*row["validation_accuracy_from_filename"]:8.2f} '
              f'{100*row["test_seen"]["accuracy"]:10.2f} {100*row["test_unseen"]["accuracy"]:11.2f} {100*row["H"]:8.2f}',flush=True)
    print(f'Results: {args.output_dir / "checkpoint_results.csv"}\n         {args.output_dir / "checkpoint_results.json"}',flush=True)

if __name__=='__main__':
    main()
