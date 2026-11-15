"""Package the selected weights with a checked input contract for the web worker."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import torch

from config import OUT as HERE, ROOT as PROJECT
sys.path.insert(0,str(PROJECT))
from backend.behavior_upload import inspect_checkpoint
from core.kpt_behavior_classifier import TrackBehaviorBuffer


def main():
    torch.set_num_threads(2)
    trained=HERE/'trained'
    source=trained/'kpt_behavior_repaired.pth'
    training=json.loads((trained/'training.json').read_text('utf-8'))
    evaluation=json.loads((HERE/'evaluation.json').read_text('utf-8'))
    if 's_pose_test' not in evaluation:
        raise RuntimeError('Complete raw-video held-out evaluation before packaging.')
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    if digest!=evaluation['checkpoint_sha256']['repaired']:
        raise RuntimeError('The evaluated checkpoint differs from the selected checkpoint.')
    target=HERE/'kpt_behavior_repaired_3fps.pth'
    config={'labels':list(training['labels']),'sample_fps':3.0,'sequence_length':30,
            'pose_models':['yolo11n-pose.pt','yolo11s-pose.pt'],'resolution':480,
            'detection_confidence':.4,'keypoint_confidence':.3}
    torch.save({'state_dict':torch.load(source,map_location='cpu',weights_only=True),
                'input_config':config,'training':{'date':'2026-11-15','source_sha256':digest,
                    'best_epoch':training['best_epoch'],'seed':42}},target)
    assert inspect_checkpoint(target)=='KeypointLSTM v2'
    original=TrackBehaviorBuffer._build_and_load(str(source)).eval()
    packaged=TrackBehaviorBuffer._build_and_load(str(target)).eval()
    torch.manual_seed(42)
    x=torch.rand(8,30,34)
    with torch.inference_mode():
        torch.testing.assert_close(original(x),packaged(x),rtol=0,atol=0)
    assert TrackBehaviorBuffer(str(target)).sample_fps==3.0
    metadata={'checkpoint_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
              'raw_weights_sha256':digest,'project_path':str(target),
              'input_config':config,'compatible_web_loader':inspect_checkpoint(target),
              'default_model_replaced':False,'registered_in_database':False}
    (HERE/'release.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(metadata,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
