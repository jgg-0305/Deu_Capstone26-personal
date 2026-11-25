"""행동 모델 학습·평가 진행 상태를 로컬 창에 표시한다. 작업이 남긴 상태 파일을 주기적으로 읽어 단계와 진행량을 갱신한다.
"""
import json
import os
from pathlib import Path
import time
import tkinter as tk
from tkinter import ttk

from config import OUT as HERE
LABELS=['normal','fall','vandalism','arson','smoking','abandon','theft','assault']
KR=['정상','넘어짐','파손','방화','흡연','유기','절도','폭행']


def read(path):
    try:
        return json.loads(path.read_text('utf-8'))
    except (OSError,ValueError):
        return {}


def number(value):
    return '—' if value is None else f'{value*100:.1f}%'


root=tk.Tk()
root.title('Capstone | 전체 행동 학습 현황')
root.geometry('940x740')
root.minsize(820,700)
root.configure(bg='#f4f7fb')
style=ttk.Style()
style.theme_use('clam')
style.configure('TFrame',background='#f4f7fb')
style.configure('TLabel',background='#f4f7fb',font=('Malgun Gothic',10))
style.configure('Treeview',font=('Malgun Gothic',10),rowheight=26)
style.configure('Treeview.Heading',font=('Malgun Gothic',10,'bold'))
outer=ttk.Frame(root,padding=22)
outer.pack(fill='both',expand=True)
title=ttk.Label(outer,text='전체 8개 행동 · RTX 3070 재학습',font=('Malgun Gothic',18,'bold'))
title.pack(anchor='w')
status=ttk.Label(outer,text='진행 기록 읽는 중…',font=('Malgun Gothic',13,'bold'),foreground='#2259a8')
status.pack(anchor='w',pady=(12,6))
detail=ttk.Label(outer,text='',wraplength=850)
detail.pack(anchor='w')
bar=ttk.Progressbar(outer,maximum=100)
bar.pack(fill='x',pady=(12,10))
score=ttk.Label(outer,text='',font=('Malgun Gothic',11,'bold'))
score.pack(anchor='w')
caption=ttk.Label(outer,text='검증 F1 · n-pose 기준 · 최종 성능은 별도 평가 후 확정')
caption.pack(anchor='w',pady=(10,5))
table=ttk.Treeview(outer,columns=('class','before','current','change'),show='headings',height=8)
for col,text,width in [('class','행동',170),('before','이전 모델 F1',190),('current','현재 모델 F1',190),('change','변화',150)]:
    table.heading(col,text=text)
    table.column(col,width=width,anchor='center')
for label,kr in zip(LABELS,KR):
    table.insert('', 'end',iid=label,values=(kr,'—','—','—'))
table.tag_configure('up',foreground='#0b7451')
table.tag_configure('down',foreground='#b23b3b')
table.pack(fill='x')
ttk.Label(outer,text='학습 추이 · 검증 Macro-F1   파랑 n-pose / 초록 s-pose').pack(anchor='w',pady=(12,3))
canvas=tk.Canvas(outer,height=126,bg='white',highlightthickness=1,highlightbackground='#dce2ea')
canvas.pack(fill='x',expand=True)
footer=ttk.Label(outer,text='',foreground='#5e6d7b')
footer.pack(anchor='w',pady=(7,6))
buttons=ttk.Frame(outer)
buttons.pack(fill='x')
ttk.Button(buttons,text='결과 폴더 열기',command=lambda:os.startfile(str(HERE))).pack(side='left')
ttk.Button(buttons,text='전체 결과 보고서',command=lambda:os.startfile(str(HERE/'REPORT.md')) if (HERE/'REPORT.md').exists() else None).pack(side='left',padx=8)
ttk.Label(buttons,text='이 창을 닫아도 학습·평가는 계속됩니다.').pack(side='right')


def update():
    progress=read(HERE/'progress.json')
    training=read(HERE/'trained/training.json')
    evaluation=read(HERE/'evaluation.json')
    baseline=read(HERE/'trained/baseline-validation.json').get('n',{})
    before=baseline.get('per_class',{})
    current={k:{'f1':v} for k,v in progress.get('n_per_class_f1',{}).items()}
    phase=progress.get('stage','waiting')
    if (HERE/'REPORT.md').exists() and evaluation.get('s_pose_test'):
        status.config(text='전체 평가 완료 · 결과 보고서 저장됨')
        detail.config(text=f'학습 선정 epoch {training.get("best_epoch","—")} · 전체 시퀀스 + 원본 영상 + 회귀 검사 완료')
        bar['value']=100
    elif phase=='independent_raw_video_test':
        count,total=progress.get('completed',0),progress.get('total',112)
        status.config(text=f'학습 완료 · 원본 영상 평가 {count} / {total}')
        detail.config(text='학습에 넣지 않은 원본 영상에서 관절 수집 지연과 실제 표시 결과를 비교하는 중입니다.')
        bar['value']=100*count/max(1,total)
    elif training:
        status.config(text='학습 완료 · 전체 클래스 성능 비교 중')
        detail.config(text=f'검증 성능으로 선정한 epoch {training.get("best_epoch","—")} · 별도 평가 자료로 비교 중')
        bar['value']=100
    elif phase=='training':
        epoch=progress.get('epoch',0)
        status.config(text=f'모델 학습 중 · epoch {epoch} / 최대 80')
        detail.config(text='8개 클래스 Macro-F1로 모델 선정 · 12 epoch 동안 개선이 없으면 조기 종료')
        bar['value']=100*epoch/80
    elif phase=='pose_extraction':
        count,total=progress.get('completed',0),progress.get('total',1)
        status.config(text=f'추가 관절 추출 · {progress.get("split","")} {count} / {total}')
        detail.config(text='원본 영상과 XML 정답을 연결하고 인물별 관절 시퀀스를 준비하는 중입니다.')
        bar['value']=100*count/max(1,total)
    else:
        status.config(text='진행 기록 대기 중')
    if evaluation.get('test'):
        models=evaluation['test']['models']
        before=models['previous_retrained']['per_class']
        current=models['repaired']['per_class']
        caption.config(text='전체 평가 시퀀스 F1 · 학습/검증 자료와 분리된 8,005개 시퀀스')
        score.config(text=f'평가 Macro-F1  {number(models["previous_retrained"]["macro_f1"])} → {number(models["repaired"]["macro_f1"])}    |    정상→이상 오탐 {number(models["repaired"].get("normal_to_anomaly_rate"))}')
    else:
        score.config(text=f'검증 Macro-F1   n-pose {number(progress.get("n_macro_f1"))}    s-pose {number(progress.get("s_macro_f1"))}    |    Loss {progress.get("loss",0):.4f}')
    for label,kr in zip(LABELS,KR):
        old=before.get(label,{}).get('f1')
        new=current.get(label,{}).get('f1')
        delta=None if old is None or new is None else (new-old)*100
        table.item(label,values=(kr,number(old),number(new),'—' if delta is None else f'{delta:+.1f}%p'),
                   tags=() if delta is None else ('up' if delta>=0 else 'down',))
    history=training.get('history',[]) or read(HERE/'trained/training-history.json') or []
    if not history:
        log=HERE/'run.log'
        try:
            for line in log.read_text('utf-8-sig').splitlines():
                if line.startswith('{"epoch"'):
                    try: history.append(json.loads(line))
                    except ValueError: pass
        except OSError: pass
    canvas.delete('all')
    w=max(500,canvas.winfo_width())
    h=max(100,canvas.winfo_height())
    for tick in [0,.5,1]:
        y=12+(1-tick)*(h-28)
        canvas.create_line(38,y,w-10,y,fill='#e5eaf0')
        canvas.create_text(20,y,text=f'{tick:.0%}',font=('Segoe UI',8),fill='#66788a')
    xmax=max(10,max([r['epoch'] for r in history],default=1))
    for key,color in [('n_macro_f1','#2661ad'),('s_macro_f1','#168c65')]:
        coords=[]
        for row in history:
            coords.extend([38+(row['epoch']-1)*(w-50)/max(1,xmax-1),12+(1-row[key])*(h-28)])
        if len(coords)>=4: canvas.create_line(*coords,fill=color,width=2)
    try: age=int(time.time()-(HERE/'progress.json').stat().st_mtime)
    except OSError: age=0
    note=f'최근 진행 기록 {age}초 전 · 2초마다 자동 갱신'
    if age>90 and not (HERE/'REPORT.md').exists(): note+=' · 실행 상태 확인 필요'
    footer.config(text=note)
    root.after(2000,update)


root.after(100,update)
root.mainloop()
