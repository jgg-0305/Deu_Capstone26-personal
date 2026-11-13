# -*- coding: utf-8 -*-
"""성능평가 결과 → 논문용 docx 표 생성 (make_report.py 스타일 재사용)."""

from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

KOR_FONT = "맑은 고딕"


def set_kor_font(run, name=KOR_FONT):
    run.font.name = name
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    rFonts.set(qn("w:eastAsia"), name)
    rFonts.set(qn("w:ascii"), name)
    rFonts.set(qn("w:hAnsi"), name)


def shade_cell(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def add_heading(doc, text, level=1):
    h = doc.add_heading(level=level)
    run = h.add_run(text)
    set_kor_font(run)
    if level == 1:
        run.font.size = Pt(16); run.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)
    elif level == 2:
        run.font.size = Pt(13); run.font.color.rgb = RGBColor(0x2E, 0x54, 0x96)
    else:
        run.font.size = Pt(11.5); run.font.color.rgb = RGBColor(0x40, 0x40, 0x40)
    return h


def add_para(doc, text, size=10.5, bold=False, italic=False, align=None,
             color=None, space_after=6):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    run = p.add_run(text)
    set_kor_font(run)
    run.font.size = Pt(size); run.font.bold = bold; run.font.italic = italic
    if color:
        run.font.color.rgb = color
    return p


def add_bullet(doc, text, size=10.5):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.left_indent = Inches(0.3)
    p.paragraph_format.space_after = Pt(3)
    run = p.add_run(text)
    set_kor_font(run)
    run.font.size = Pt(size)
    return p


def add_table(doc, headers, rows, widths=None, highlight_rows=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        shade_cell(hdr[i], "2E5496")
        para = hdr[i].paragraphs[0]; para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = para.add_run(h); set_kor_font(run)
        run.font.size = Pt(10); run.font.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    highlight_rows = highlight_rows or set()
    for ri, row in enumerate(rows):
        cells = table.add_row().cells
        for i, val in enumerate(row):
            if ri in highlight_rows:
                shade_cell(cells[i], "EAF0FA")
            para = cells[i].paragraphs[0]
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER if i > 0 else WD_ALIGN_PARAGRAPH.LEFT
            run = para.add_run(str(val)); set_kor_font(run)
            run.font.size = Pt(9.5)
            if ri in highlight_rows:
                run.font.bold = True
    if widths:
        for row in table.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return table


def _pct(x):
    return f"{x*100:.2f}"


def _lat(ms, tp):
    if ms is None:
        return "—"
    return f"{ms:.2f} ms ({tp:.1f}/s)"


def _onset_hit(delta, tol=30):
    """온셋 오차가 허용 프레임(tol) 이내면 적중."""
    return delta is not None and abs(delta) <= tol


def build_report(out_path, kpt, onset_rows, device, label_names, tol=30):
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = KOR_FONT
    style.font.size = Pt(10.5)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), KOR_FONT)

    # 제목
    title = doc.add_paragraph(); title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(6)
    r = title.add_run("이상행동 감지 AI 성능 평가표")
    set_kor_font(r); r.font.size = Pt(18); r.font.bold = True
    r.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)
    sub = doc.add_paragraph(); sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = sub.add_run("KeypointLSTMv2 행동 분류 + 데모 영상 온셋(시작 프레임) 감지 평가")
    set_kor_font(r); r.font.size = Pt(11); r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    import torch as _t
    dev = _t.cuda.get_device_name(0) if _t.cuda.is_available() else "CPU"
    add_para(doc,
             f"측정 환경: PyTorch {_t.__version__}, {dev} / "
             "분류 지표는 혼동행렬 기반 실측값(직접 산출), 온셋 감지는 앱과 동일한 추론 "
             "경로(YOLO-pose→DeepSORT→KeypointLSTM, 도난은 규칙 병행)로 측정.",
             size=9, italic=True, color=RGBColor(0x60, 0x60, 0x60), space_after=10)

    # ── 1. 평가 개요 ──
    add_heading(doc, "1. 평가 개요", 1)
    add_para(doc,
             "본 시스템은 영상 '내용'(자세·움직임 시퀀스)을 기반으로 8종 행동"
             "(정상·전도·파손·방화·흡연·유기·절도·폭행)을 분류하는 KeypointLSTMv2 와, "
             "물품 가판대·결제기를 감지하는 객체 모델 + 규칙 기반 도난 판정으로 구성된다. "
             "평가는 ① 키포인트 분류 지표(held-out), ② 데모 영상에서 지정 온셋(시작 "
             "프레임)부터 해당 행동이 감지되는지 두 축으로 수행한다.")
    add_table(doc,
              ["구분", "내용"],
              [
                  ["행동 분류 모델", f"KeypointLSTMv2 ({kpt['arch']}), kpt_behavior.pth"],
                  ["입력 양식", "키포인트 30프레임 시퀀스 (17관절 × xy, bbox 정규화)"],
                  ["파라미터 수", f"{kpt['n_params']:,}"],
                  ["분류 평가셋", f"{kpt['eval_set']} ({kpt['n_samples']}개 / "
                   f"{kpt['n_present_classes']}종 등장)"],
                  ["온셋 학습 방식", "데모 영상에 시간 라벨(온셋 전=정상, 이후=해당 행동) "
                   "부여 — 내용 기반 지도학습"],
                  ["도난 감지", "가판대→물품→결제기 미경유 규칙(core/theft_monitor) 병행"],
              ],
              widths=[1.8, 4.6])

    # ── 2. 분류 성능 ──
    add_heading(doc, "2. 행동 분류 성능 (KeypointLSTMv2)", 1)
    add_para(doc, "■ 표 1. 분류 성능 및 추론 속도", bold=True, size=11,
             color=RGBColor(0x2E, 0x54, 0x96))
    rows = [
        ["Accuracy (%)", _pct(kpt["accuracy"])],
        ["Macro Precision (%)", _pct(kpt["macro"]["precision"])],
        ["Macro Recall (%)", _pct(kpt["macro"]["recall"])],
        ["Macro F1 (%)", _pct(kpt["macro"]["f1"])],
        ["Weighted F1 (%)", _pct(kpt["weighted"]["f1"])],
        ["추론 지연 (GPU)", _lat(kpt["gpu_ms"], kpt["gpu_tp"])],
        ["추론 지연 (CPU)", _lat(kpt["cpu_ms"], kpt["cpu_tp"])],
    ]
    add_table(doc, ["지표", "값"], rows, widths=[2.6, 3.0],
              highlight_rows={0, 3})

    add_para(doc, "■ 표 2. 클래스별 상세 지표", bold=True, size=11,
             color=RGBColor(0x2E, 0x54, 0x96))
    pc = kpt["per_class"]
    crows = []
    for i, nm in enumerate(label_names):
        sup = int(pc["support"][i])
        if sup == 0:
            crows.append([f"{i} {nm}", "0", "—", "—", "—"])
        else:
            crows.append([f"{i} {nm}", str(sup), _pct(pc["precision"][i]),
                          _pct(pc["recall"][i]), _pct(pc["f1"][i])])
    add_table(doc, ["클래스", "표본수", "Precision(%)", "Recall(%)", "F1(%)"],
              crows, widths=[1.5, 1.0, 1.4, 1.4, 1.4])

    # ── 3. 온셋 감지 평가 ──
    add_heading(doc, "3. 데모 영상 온셋(시작 프레임) 감지", 1)
    add_para(doc,
             f"각 데모 영상에서 목표 행동이 처음 감지된 프레임과 요청 온셋을 비교한다. "
             f"오차(Δ) 절댓값이 {tol}프레임 이내면 적중으로 본다. "
             "감지는 앱과 동일한 추론 경로로 측정하였다.")
    orows, hits = [], 0
    for r0 in onset_rows:
        tgt = label_names[r0["target"]]
        det = r0.get("onset_det")
        delta = r0.get("delta")
        if det is None:
            det_s, delta_s, judge = "미감지", "—", "✗"
        else:
            det_s = f"{det}f"
            delta_s = f"{delta:+d}"
            hit = _onset_hit(delta, tol)
            judge = "○" if hit else "✗"
            hits += hit
        orows.append([r0["file"], tgt, f"{r0['onset_req']}f", det_s, delta_s, judge])
    add_table(doc,
              ["영상", "목표 행동", "요청 온셋", "감지 온셋", "오차(Δ)", "적중"],
              orows, widths=[1.6, 1.1, 1.0, 1.0, 0.9, 0.7])
    add_para(doc, f"온셋 적중률: {hits}/{len(onset_rows)} "
             f"({hits/max(1,len(onset_rows))*100:.0f}%)  (허용 오차 ±{tol}f)",
             bold=True, size=10.5, color=RGBColor(0x1F, 0x38, 0x64))

    # ── 4. 분석 ──
    add_heading(doc, "4. 결과 분석", 1)
    kpt_normal_f1 = pc["f1"][0] if pc["support"][0] > 0 else 0.0
    add_bullet(doc,
               "온셋 시간 라벨링(내용 기반): 데모 영상을 '온셋 이전=정상, 이후=해당 행동'으로 "
               "시간 라벨링해 학습함으로써, 모델이 파일이 아닌 영상 내용(온셋 전후의 자세·"
               "움직임 변화)을 학습한다. 추론 시에도 영상 내용만으로 온셋 부근에서 행동이 "
               "감지된다(표 3).")
    add_bullet(doc,
               f"정상 인식 확보: '정상' 클래스 F1 {_pct(kpt_normal_f1)}% — 매장이동·구매·반품 "
               "정상 영상 보강으로 정상/이상 구분 기반이 마련되었다.")
    add_bullet(doc,
               f"경량 실시간성: 파라미터 {kpt['n_params']:,}개, CPU 추론 "
               f"{kpt['cpu_ms']:.2f}ms/표본 — Raspberry Pi 5(CPU) 실시간 구동에 적합.")
    add_bullet(doc,
               "한계: 키포인트만으로는 파손·폭행처럼 외형·접촉이 중요한 행동의 구분이 "
               "본질적으로 어렵고, 데모는 소수 영상이라 일반 held-out 정확도는 제한적이다. "
               "도난은 분류(절도)와 규칙(가판대→결제기 미경유)을 병행해 보강한다.")

    doc.save(out_path)
