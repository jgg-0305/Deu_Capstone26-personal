"""저장된 추론 결과에 영상 시각을 맞춰 전후 비교 클립을 만든다. 예측 기록을 영상에 덧그려 어떤 구간에서 결과가 바뀌는지 표시한다.
"""
import json
import shutil
import tempfile
from zipfile import ZipFile
import cv2
from diagnose import OUT, CASE, get_groups, LABELS, dump


def main():
    clips,_=get_groups()
    clip=next(c for c in clips if c.name==CASE)
    pose='yolo11s-pose.pt'
    traces={name:json.loads((OUT/f'{name}-{pose}-{CASE}-frames.json').read_text('utf-8'))
            for name in ['baseline','repaired']}
    output=OUT/'pocket-before-after.mp4'
    with tempfile.TemporaryDirectory(dir=OUT) as temp:
        from pathlib import Path
        import os
        original=Path(temp)/'clip.mp4'
        with ZipFile(clip.video_zip) as z,z.open(clip.video) as src,original.open('wb') as dst:
            shutil.copyfileobj(src,dst)
        cwd=Path.cwd()
        os.chdir(temp)
        cap=cv2.VideoCapture('clip.mp4')
        # Windows에서 OpenCV가 영상 파일을 안정적으로 열도록 이름에 ASCII 문자를 사용한다.
        writer=cv2.VideoWriter('comparison.mp4',cv2.VideoWriter_fourcc(*'mp4v'),3,(1280,410))
        if not cap.isOpened() or not writer.isOpened():
            raise RuntimeError('Could not open comparison media')
        try:
            i=0
            while True:
                ok,frame=cap.read()
                if not ok:
                    break
                panels=[]
                for name,title in [('baseline','BEFORE: previous retrained'),('repaired','AFTER: repaired candidate')]:
                    panel=cv2.resize(frame,(640,360))
                    import numpy as np
                    panel=np.vstack([np.zeros((50,640,3),np.uint8),panel])
                    cv2.putText(panel,title,(10,20),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
                    preds=traces[name][i]['predictions']
                    words=', '.join(f'ID {tid}: {LABELS[p[0]].upper()} {p[1]:.0%}' for tid,p in preds.items()) or 'Collecting keypoints'
                    cv2.putText(panel,words,(10,42),cv2.FONT_HERSHEY_SIMPLEX,.48,(80,220,255),1,cv2.LINE_AA)
                    cv2.putText(panel,f'{i/3:.2f}s',(550,75),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
                    panels.append(panel)
                writer.write(np.hstack(panels))
                i+=1
        finally:
            cap.release()
            writer.release()
            os.chdir(cwd)
        shutil.copy2(Path(temp)/'comparison.mp4',output)
    cap=cv2.VideoCapture(str(output))
    frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok,_=cap.read()
    cap.release()
    assert ok and frames==len(traces['baseline'])
    print('Comparison video verified:',output,frames,'frames')
    training=json.loads((OUT/'trained/training.json').read_text('utf-8'))
    dump(OUT/'progress.json',{'stage':'completed','best_epoch':training['best_epoch'],
                             'epochs':len(training['history']),'report':str(OUT/'REPORT.md')})


if __name__=='__main__':
    main()
