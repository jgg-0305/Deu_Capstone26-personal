"""Run the eight-class AI-Hub extraction, retraining, and held-out evaluation pipeline."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def completed(path, required):
    try:
        value=json.loads(path.read_text('utf-8'))
        return all(key in value for key in required)
    except (OSError, ValueError):
        return False


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['all','prepare','train','evaluate','diagnose','package'],default='all')
    parser.add_argument('--data-root',type=Path,required=True,help='AI-Hub Training/Validation ZIP root')
    parser.add_argument('--prepared',type=Path,required=True,help='Full n-pose dataset directory; created if absent')
    parser.add_argument('--output',type=Path,required=True,help='New experiment output directory')
    parser.add_argument('--previous-model',type=Path,required=True,help='Previous checkpoint for comparisons')
    parser.add_argument('--legacy-v1',type=Path,help='Optional legacy v1 checkpoint')
    parser.add_argument('--legacy-v2',type=Path,default=ROOT/'train/1_behavior/output/kpt_behavior.pth')
    parser.add_argument('--pose-n',type=Path,default=ROOT/'yolo11n-pose.pt')
    parser.add_argument('--pose-s',type=Path,default=ROOT/'yolo11s-pose.pt')
    parser.add_argument('--epochs',type=int,default=80)
    parser.add_argument('--train-per-class',type=int,default=64)
    parser.add_argument('--val-per-class',type=int,default=16)
    parser.add_argument('--test-per-class',type=int,default=16)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--monitor',action='store_true',help='Open the live training/evaluation window')
    args=parser.parse_args()
    for key in ['epochs','train_per_class','val_per_class','test_per_class']:
        if getattr(args,key)<1:
            parser.error(f'--{key.replace("_","-")} must be positive')
    env=os.environ.copy()
    paths={'BEHAVIOR_DATA_ROOT':args.data_root,'BEHAVIOR_PREPARED':args.prepared,
           'BEHAVIOR_OUTPUT':args.output,'BEHAVIOR_PREVIOUS':args.previous_model,
           'BEHAVIOR_V1':args.legacy_v1,'BEHAVIOR_V2':args.legacy_v2,
           'BEHAVIOR_POSE_N':args.pose_n,'BEHAVIOR_POSE_S':args.pose_s}
    env.update({k:str(v.resolve()) for k,v in paths.items() if v is not None})
    env['PYTHONUTF8']='1'
    env['PYTHONUNBUFFERED']='1'
    output=args.output.resolve()
    if not args.dry_run:
        if not args.data_root.is_dir() or not args.previous_model.is_file():
            parser.error('Data root and previous checkpoint must exist')
        output.mkdir(parents=True,exist_ok=True)
        env['YOLO_CONFIG_DIR']=str(output/'ultralytics-config')
    commands=[]
    def add(script,*extra):
        commands.append([sys.executable,'-u',str(HERE/script),*map(str,extra)])
    if args.stage in ['all','prepare']:
        if not completed(args.prepared/'dataset.json',['labels','counts']):
            add('prepare_aihub.py','--data-root',args.data_root.resolve(),'--output',args.prepared.resolve(),
                '--cache',output/'n-pose-cache','--pose',args.pose_n.resolve())
        if not completed(output/'mixed-dataset/dataset.json',['combined_window_counts']):
            add('prepare_mixed.py','--train-per-class',args.train_per_class,'--val-per-class',args.val_per_class)
    if args.stage in ['all','train']:
        if args.stage=='train' or not completed(output/'trained/training.json',['checkpoint_sha256','best_epoch']):
            add('train_robust.py','--epochs',args.epochs)
    if args.stage in ['all','evaluate']:
        if args.stage=='evaluate' or not completed(output/'evaluation.json',['s_pose_test','streaming_ema_50_percent']):
            add('evaluate_repair.py','--raw-test-per-class',args.test_per_class)
    if args.stage in ['all','diagnose']:
        for model,name in [(args.previous_model.resolve(),'baseline'),(output/'trained/kpt_behavior_repaired.pth','repaired')]:
            for pose,count in [('yolo11n-pose.pt',0),('yolo11s-pose.pt',4)]:
                add('diagnose.py','--model',model,'--name',name,'--pose',pose,'--smoke-count',count)
    if args.stage in ['all','package']:
        add('package_model.py')
        add('write_report.py')
        add('render_comparison.py')
    if args.dry_run:
        print(json.dumps({'environment':{k:env.get(k) for k in paths},'commands':commands},ensure_ascii=False,indent=2))
        return
    if args.monitor:
        pythonw=Path(sys.executable).with_name('pythonw.exe')
        subprocess.Popen([str(pythonw if pythonw.exists() else sys.executable),str(HERE/'monitor.py')],env=env,cwd=ROOT)
    for command in commands:
        print('RUN',subprocess.list2cmdline(command),flush=True)
        with (output/'run.log').open('a',encoding='utf-8') as log:
            child=subprocess.Popen(command,env=env,cwd=ROOT,stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,text=True,encoding='utf-8')
            try:
                for line in child.stdout:
                    log.write(line)
                    log.flush()
                    print(line,end='',flush=True)
                code=child.wait()
            except BaseException:
                if child.poll() is None:
                    child.terminate()
                    child.wait()
                raise
            if code:
                raise subprocess.CalledProcessError(code,command)


if __name__=='__main__':
    main()
