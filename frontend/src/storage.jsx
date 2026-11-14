import React, { useCallback, useEffect, useId, useRef, useState } from "react";

export function sizeText(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}

export function DeleteDialog({ title, children, busy, onCancel, onConfirm }) {
  const dialog = useRef(null);
  const titleId = useId();
  useEffect(() => {
    const element = dialog.current;
    element.showModal();
    return () => element.close();
  }, []);
  return <dialog className="delete-dialog panel" ref={dialog} aria-labelledby={titleId}
    onCancel={(event) => { event.preventDefault(); if (!busy) onCancel(); }}>
    <h2 id={titleId}>{title}</h2>
    <div className="delete-description">{children}</div>
    <div className="delete-actions">
      <button autoFocus disabled={busy} onClick={onCancel}>취소</button>
      <button className="danger" disabled={busy} onClick={onConfirm}>{busy ? "삭제 중…" : "삭제 확인"}</button>
    </div>
  </dialog>;
}

export function Storage({ api, revision, canWrite, onError, notify, refresh }) {
  const [data, setData] = useState({ sources: [], total_bytes: 0 });
  const [loading, setLoading] = useState(true);
  const [target, setTarget] = useState(null);
  const [busy, setBusy] = useState(false);
  const generation = useRef(0);
  const load = useCallback(async () => {
    const sequence = ++generation.current;
    setLoading(true);
    try {
      const result = await api("/storage");
      if (sequence === generation.current) setData(result);
    } catch (error) { onError(error); }
    finally { if (sequence === generation.current) setLoading(false); }
  }, [api, onError]);
  useEffect(() => {
    load();
    return () => { generation.current += 1; };
  }, [load, revision]);

  async function purge(source) {
    setBusy(true);
    try {
      const result = await api("/storage/" + source.id, undefined, "DELETE");
      setTarget(null);
      notify(`${sizeText(result.freed_bytes)}를 정리했습니다. ` +
        (result.pending_files ? "사용 중이거나 지울 수 없는 파일이 남았습니다. 잠시 후 정리 재시도를 눌러주세요." : "영상 소스와 관련 사건이 삭제됐습니다.") +
        (result.shared_video_kept ? " 다른 소스가 사용하는 원본은 유지했습니다." : "") +
        (result.sample_kept ? " 기본 샘플 파일은 유지했습니다." : ""));
      await Promise.all([load(), refresh()]);
    } catch (error) { onError(error); }
    finally { setBusy(false); }
  }

  return <section className="panel storage-panel" aria-label="저장 공간 관리" aria-busy={loading}>
    <div className="section-heading"><div><h2>저장된 영상과 분석 결과</h2>
      <p className="muted">관제 화면에서 제거한 소스도 포함합니다. 기본 샘플 파일은 용량에서 제외합니다.</p></div>
      <strong className="storage-total">{sizeText(data.total_bytes)}</strong></div>
    <p className="muted">사건들은 원본 영상을 공유합니다. 용량을 비우려면 해당 영상과 관련 사건을 함께 삭제하세요.</p>
    <div className="storage-list">
      {data.sources.map((source) => <article className="storage-row" key={source.id}>
        <div className="storage-description"><h3>{source.name}</h3>
          <p className="muted">{source.original_filename || ({ sample: "기본 샘플", webcam: "웹캠", rtsp: "IP 카메라" })[source.source_type]}</p>
          <p className="muted">사건 {source.event_count}건 · {source.cleanup_pending ? "파일 정리 대기" : source.removed ? "관제 화면에서 제거됨" : "등록된 소스"}
            {source.source_type === "sample" && " · 샘플 원본은 유지"}</p></div>
        <div className="storage-size"><strong>{sizeText(source.size_bytes)}</strong><small>원본 {sizeText(source.video_bytes)}</small></div>
        {canWrite && <button className="danger" disabled={busy} onClick={() => source.cleanup_pending ? purge(source) : setTarget(source)}>
          {source.cleanup_pending ? "정리 재시도" : "영상·기록 삭제"}</button>}
      </article>)}
      {!data.sources.length && <p className="empty">{loading ? "저장 공간을 확인하고 있습니다…" : "정리할 영상이 없습니다."}</p>}
    </div>
    {target && <DeleteDialog title="영상과 관련 사건을 삭제할까요?" busy={busy} onCancel={() => setTarget(null)} onConfirm={() => purge(target)}>
      <p><strong>{target.name}</strong> · 사건 {target.event_count}건 · 저장 용량 {sizeText(target.size_bytes)}</p>
      <p>진행 중인 분석을 중단하고 이 소스의 모든 실행 기록에 연결된 사건, 업로드 원본과 분석 파일을 삭제합니다. 삭제한 파일은 복구할 수 없습니다.</p>
      {target.source_type === "sample" && <p>프로젝트의 기본 샘플 원본은 삭제하지 않습니다.</p>}
      <p className="muted">다른 소스가 사용하는 원본과 PC에서 처음 선택했던 파일은 유지됩니다.</p>
    </DeleteDialog>}
  </section>;
}
