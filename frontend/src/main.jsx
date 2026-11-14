import React, { useCallback, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";
import { DeleteDialog, Storage } from "./storage.jsx";
import {
  AnalysisFrame, AnalysisSettings, AnalysisStats, DisplaySettings, FileAnalysis, ModelStatus,
  initialDisplay, initialSettings, startWebcamFeed, useSnapshots,
} from "./analysis.jsx";

const types = {
  fall: "넘어짐", vandalism: "파손", arson: "방화", smoking: "흡연",
  abandon: "유기", theft_suspected: "도난 의심", assault: "폭행",
};
const methods = { model: "행동 모델", rule: "규칙 판정", hybrid: "모델 및 규칙" };
const states = {
  idle: "대기", queued: "분석 대기", starting: "준비 중", running: "분석 중",
  stopping: "종료 중", completed: "완료", failed: "실패", stopped: "중단",
};
const active = new Set(["queued", "starting", "running", "stopping"]);
const reviews = { confirmed: "확인됨", false_positive: "오탐", needs_review: "추가 검토" };

function position(ms) {
  if (ms == null) return "영상 시점 미기록";
  const seconds = Math.floor(ms / 1000);
  return Math.floor(seconds / 60) + ":" + String(seconds % 60).padStart(2, "0");
}

async function request(path, body, method = body === undefined ? "GET" : "POST") {
  const file = body instanceof File;
  const response = await fetch("/api/v1" + path, {
    method,
    headers: { "X-CCTV-Request": "1", ...(body !== undefined && { "Content-Type": file ? "application/octet-stream" : "application/json" }) },
    body: file ? body : body === undefined ? undefined : JSON.stringify(body),
  });
  if (response.status === 204) return null;
  const data = await response.json();
  if (!response.ok) {
    const message = {
      401: typeof data.detail === "string" ? data.detail : "다시 로그인하세요.", 403: typeof data.detail === "string" ? data.detail : "이 작업에는 운영자 권한이 필요합니다.",
      409: typeof data.detail === "string" ? data.detail : "이미 등록되었거나 요청된 항목입니다.",
    }[response.status];
    const error = new Error(message || (typeof data.detail === "string" ? data.detail : "요청 내용을 확인하세요."));
    error.status = response.status;
    throw error;
  }
  return data;
}

function Video({ mediaId, cameraId, time, onError }) {
  const url = cameraId ? "/api/v1/cameras/" + encodeURIComponent(cameraId) + "/video" :
    "/api/v1/media/" + encodeURIComponent(mediaId);
  return <video key={url} controls playsInline preload="metadata"
    src={url}
    onLoadedMetadata={(event) => {
      if (time != null) event.currentTarget.currentTime = Math.min(time / 1000, event.currentTarget.duration);
    }}
    onError={onError} />;
}

function Login({ onSubmit, busy }) {
  const [register, setRegister] = useState(false);
  const title = register ? "회원가입" : "로그인";
  return <section className="login panel">
    <span className="brand-mark" aria-hidden="true">◉</span>
    <h1>{title}</h1>
    <div className="switch" aria-label="접속 방식">
      <button type="button" aria-pressed={!register} disabled={busy} onClick={() => setRegister(false)}>로그인</button>
      <button type="button" aria-pressed={register} disabled={busy} onClick={() => setRegister(true)}>회원가입</button>
    </div>
    <form key={title} onSubmit={(event) => {
      event.preventDefault();
      const fields = Object.fromEntries(new FormData(event.currentTarget));
      onSubmit(register ? "register" : "login", fields);
    }}>
      {register && <>
        <label htmlFor="display-name">이름</label>
        <input id="display-name" name="display_name" required maxLength={120} autoComplete="name" />
      </>}
      <label htmlFor="username">아이디</label>
      <input id="username" name="username" required autoComplete="username" maxLength={32} />
      <label htmlFor="password">비밀번호{register && " (8자 이상)"}</label>
      <input id="password" name="password" type="password" required
        minLength={register ? 8 : 1} maxLength={128} autoComplete={register ? "new-password" : "current-password"} />
      <button className="primary" disabled={busy}>{register ? "계정 만들기" : "로그인하기"}</button>
    </form>
  </section>;
}

const liveTypes = new Set(["webcam", "rtsp"]);

function WebcamVideo({ stream }) {
  const video = useRef(null);
  useEffect(() => {
    const element = video.current;
    element.srcObject = stream;
    return () => { element.srcObject = null; };
  }, [stream]);
  return <video ref={video} autoPlay muted playsInline aria-label="웹캠 실시간 영상" />;
}

function RtspPreview({ source }) {
  const image = useRef(null);
  const [playing, setPlaying] = useState(false);
  const [ready, setReady] = useState(false);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    if (!playing) return;
    const element = image.current;
    return () => { element.removeAttribute("src"); };
  }, [playing]);
  useEffect(() => {
    if (!playing || ready) return;
    const timer = setTimeout(() => { setPlaying(false); setFailed(true); }, 18000);
    return () => clearTimeout(timer);
  }, [playing, ready]);
  return <>
    {playing && <img ref={image} className="live-feed" src={"/api/v1/cameras/" + encodeURIComponent(source.id) + "/live"}
      alt={source.name + " 실시간 영상"} hidden={!ready} onLoad={() => setReady(true)}
      onError={() => { setPlaying(false); setReady(false); setFailed(true); }} />}
    {(!playing || !ready) && <div className="tile-screen" role="status">{playing ? "카메라에 연결 중…" :
      failed ? "연결할 수 없습니다. 카메라 주소, 서버 권한과 네트워크를 확인하세요." : "미리보기가 중지되었습니다."}</div>}
    <div className="tile-footer live-actions"><span>IP 카메라 (RTSP) · 실시간 미리보기</span>
      <button type="button" onClick={() => {
        setReady(false); setFailed(false); setPlaying(!playing);
      }}>{playing ? "중지" : failed ? "다시 연결" : "연결"}</button>
    </div>
  </>;
}

function SourceTile({ source, selected, onSelect, onVideoError, remove, style, hidden, snapshot, display, pending }) {
  const live = liveTypes.has(source.source_type);
  const [overlay, setOverlay] = useState(true);
  const visibleDisplay = overlay ? display : {
    box: false, id: false, points: false, label: false, stand: false, pos: false,
  };
  return <article className={"tile" + (selected ? " selected" : "")} style={style} hidden={hidden}>
    <div className="tile-head">
      <button className="tile-name" aria-current={selected} onClick={onSelect}>{source.name}</button>
      <span className={"tile-state " + source.status}>{states[source.status] ?? source.status}</span>
      <div className="switch overlay-toggle" aria-label="영상 표시 방식">
        <button type="button" aria-pressed={!overlay} onClick={() => setOverlay(false)}>원본</button>
        <button type="button" aria-pressed={overlay} onClick={() => setOverlay(true)}>AI 분석</button>
      </div>
      {remove && <button className="tile-remove" type="button" aria-label={source.name + " 제거"}
        title="영상 소스 제거" disabled={pending} onClick={remove}>×</button>}
    </div>
    {snapshot?.image && live ? <AnalysisFrame snapshot={snapshot} display={visibleDisplay} /> :
      source.source_type === "webcam" ? source.stream ? <WebcamVideo stream={source.stream} /> :
        <div className="tile-screen">브라우저에서 웹캠을 다시 연결하세요.</div> :
      source.source_type === "rtsp" ? !hidden && <RtspPreview source={source} /> :
      !hidden && <FileAnalysis source={source} display={visibleDisplay} enabled={overlay} onError={onVideoError} />}
    <ModelStatus source={source} snapshot={snapshot} />
    <div className="tile-footer live-actions">
      <span>{live ? (source.source_type === "rtsp" ? "IP 카메라 (RTSP)" : "웹캠") :
        source.original_filename || source.sample}
      {snapshot && " · " + snapshot.frame_count + "프레임 · " + snapshot.people.length + "명"} · 사건 {source.event_count}건</span>
      {source.behavior_filename && <span title={source.behavior_sha256}>{source.behavior_filename} · {source.behavior_sha256?.slice(0, 12)}</span>}
    </div>
  </article>;
}

function Monitor({ sources, models, selection, select, api, canWrite, refresh, refreshModels, notify, onError }) {
  const [layout, setLayout] = useState("grid");
  const [sourceMode, setSourceMode] = useState("upload");
  const [settings, setSettings] = useState(initialSettings);
  const defaultChosen = useRef(false);
  const [display, setDisplay] = useState(initialDisplay);
  const [pending, setPending] = useState(false);
  const [devices, setDevices] = useState([]);
  const [webcams, setWebcams] = useState({});
  const streams = useRef(new Map());
  const mounted = useRef(true);
  const all = sources.map((source) => ({ ...source, stream: webcams[source.id] }));
  const snapshots = useSnapshots(sources, api);
  const chosen = all.find((source) => source.id === selection) ?? all[0];

  const cascade = layout === "cascade" && all.length > 1;
  const media = navigator.mediaDevices;

  useEffect(() => {
    if (defaultChosen.current || !models.length) return;
    defaultChosen.current = true;
    if (models.some((model) => model.name === "human_pose_s")) {
      setSettings((current) => ({ ...current, pose_model: "human_pose_s" }));
    }
  }, [models]);

  useEffect(() => {
    const current = streams.current;
    mounted.current = true;
    const release = () => {
      for (const [id, { stop }] of current) {
        stop();
        fetch("/api/v1/cameras/" + id, {
          method: "DELETE", headers: { "X-CCTV-Request": "1" }, keepalive: true,
        }).catch(() => {});
      }
      current.clear();
    };
    window.addEventListener("pagehide", release);
    return () => {
      mounted.current = false;
      window.removeEventListener("pagehide", release);
      release();
    };
  }, []);

  useEffect(() => {
    if (sourceMode !== "webcam" || !media?.enumerateDevices) return;
    let cancelled = false;
    const updateDevices = () => media.enumerateDevices().then((list) => {
      if (!cancelled) setDevices(list.filter((device) => device.kind === "videoinput"));
    }).catch(() => {});
    updateDevices();
    media.addEventListener("devicechange", updateDevices);
    return () => { cancelled = true; media.removeEventListener("devicechange", updateDevices); };
  }, [sourceMode, media]);

  function stopWebcam(id) {
    streams.current.get(id)?.stop();
    streams.current.delete(id);
    setWebcams((current) => {
      const remaining = { ...current };
      delete remaining[id];
      return remaining;
    });
  }

  async function removeSource(id) {
    setPending(true);
    try {
      await api("/cameras/" + id, undefined, "DELETE");
      stopWebcam(id);
      await refresh();
      notify("영상 소스를 제거했습니다.");
    } catch (error) { onError(error); }
    finally { setPending(false); }
  }

  async function addWebcam(deviceId) {
    if (!media?.getUserMedia) throw new Error("웹캠은 HTTPS 또는 localhost에서 사용할 수 있습니다.");
    let stream;
    try {
      stream = await media.getUserMedia({ video: deviceId ? { deviceId: { exact: deviceId } } : true, audio: false });
    } catch (error) {
      throw new Error(({ NotAllowedError: "브라우저 또는 운영체제에서 카메라 권한을 허용하세요.",
        NotFoundError: "사용 가능한 웹캠을 찾을 수 없습니다.",
        NotReadableError: "웹캠을 열 수 없습니다. 다른 앱에서 사용 중인지 확인하세요.",
        OverconstrainedError: "선택한 웹캠을 찾을 수 없습니다. 다시 선택하세요.",
      })[error.name] || "웹캠을 열 수 없습니다.");
    }
    const track = stream.getVideoTracks()[0];
    const deviceKey = track.getSettings().deviceId || deviceId || "default";
    let attached = false;
    try {
      if (!mounted.current) return;
      if ([...streams.current.values()].some((item) => item.deviceKey === deviceKey)) {
        throw new Error("이미 연결된 웹캠입니다.");
      }
      const name = (track.label || "웹캠").slice(0, 108) + " · " + Date.now().toString(36);
      const source = await api("/cameras/webcam", { name, settings });
      if (!mounted.current) {
        await api("/cameras/" + source.id, undefined, "DELETE").catch(() => {});
        return;
      }
      streams.current.set(source.id, { deviceKey, stop: startWebcamFeed(source.id, stream, onError) });
      attached = true;
      track.addEventListener("ended", () => {
        stopWebcam(source.id);
        api("/cameras/" + source.id, undefined, "DELETE").then(refresh).catch(onError);
        notify("웹캠 연결이 종료되었습니다.");
      }, { once: true });
      setWebcams((current) => ({ ...current, [source.id]: stream }));
      await refresh();
      select(source.id);
      notify("웹캠 분석을 요청했습니다.");
    } finally {
      if (!attached) stream.getTracks().forEach((item) => item.stop());
    }
    media.enumerateDevices().then((list) => {
      if (mounted.current) setDevices(list.filter((device) => device.kind === "videoinput"));
    }).catch(() => {});
  }

  async function add(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const { name, sample, video, url, device } = Object.fromEntries(new FormData(form));
    setPending(true);
    try {
      if (sourceMode === "webcam") {
        await addWebcam(device);
      } else {
        const source = sourceMode === "upload"
          ? await api("/cameras/upload?" + new URLSearchParams({ name, filename: video.name, ...settings }), video)
          : sourceMode === "rtsp" ? await api("/cameras/rtsp", { name, url: url.trim(), settings })
          : await api("/cameras", { name, sample, settings });
        form.reset();
        await refresh();
        select(source.id);
        notify("영상 소스를 등록하고 분석을 요청했습니다.");
      }
    } catch (error) { if (mounted.current) onError(error); }
    finally { if (mounted.current) setPending(false); }
  }
  async function analyze(event) {
    event.preventDefault();
    if (!chosen) return;
    if (chosen.source_type === "webcam" && !chosen.stream) {
      notify("이 브라우저에서 웹캠을 추가해 연결하세요.");
      return;
    }
    setPending(true);
    try {
      await api("/cameras/" + chosen.id + "/analysis", settings);
      notify("분석을 요청했습니다. 연결된 워커가 대기열에서 작업을 가져갑니다.");
      await refresh();
    } catch (error) { onError(error); }
    finally { setPending(false); }
  }
  async function attachBehavior(event) {
    const input = event.currentTarget;
    const file = input.files[0];
    if (!file) return;
    setPending(true);
    try {
      const model = await api("/models/behavior/upload?" + new URLSearchParams({ filename: file.name }), file);
      await refreshModels();
      setSettings((old) => ({ ...old, behavior_model: model.name }));
      notify("행동 모델을 선택했습니다. 기존 영상에 적용하려면 다시 분석을 누르세요.");
    } catch (error) { onError(error); }
    finally { input.value = ""; setPending(false); }
  }
  return <div className="workspace">
    <section className="panel viewer">
      <div className="section-heading"><h2>영상 소스</h2>
        <div className="switch layout-buttons" aria-label="영상 배치">
          <button aria-pressed={layout === "grid"} onClick={() => setLayout("grid")}>바둑판 정렬</button>
          <button aria-pressed={layout === "cascade"} onClick={() => setLayout("cascade")}>계단식 정렬</button>
          <button aria-pressed={layout === "focus"} disabled={!chosen} onClick={() => setLayout("focus")}>집중 보기</button>
        </div>
      </div>
      <div className={"source-grid" + (cascade ? " cascade" : layout === "focus" ? " focus" : "")}>
        {all.map((source, index) => <SourceTile key={source.id} source={source} selected={source.id === chosen?.id}
          style={cascade ? { "--step": index % 5, "--stack": Math.floor(index / 5) + 1, "--layer": source.id === chosen?.id ? 6 : index % 5 + 1 } : undefined}
          hidden={layout === "focus" && source.id !== chosen?.id}
          onSelect={() => select(source.id)} remove={canWrite ? () => removeSource(source.id) : null}
          snapshot={snapshots[source.id]} display={display} pending={pending}
          onVideoError={() => notify("영상을 열 수 없습니다. 접속 상태를 확인하세요.")} />)}
        {!all.length && <div className="empty">등록된 영상 소스가 없습니다.</div>}
      </div>
      {cascade && <div className="cascade-picker switch" aria-label="앞으로 가져올 영상">
        {all.map((source) => <button key={source.id} aria-pressed={source.id === chosen?.id}
          onClick={() => select(source.id)}>{source.name}</button>)}
      </div>}
    </section>
    <aside className="controls">
      {canWrite && <>
        <section className="panel">
          <h2>영상 소스 추가</h2>
          <div className="switch source-modes" aria-label="영상 등록 방식">
            {[["upload", "파일 추가…"], ["webcam", "웹캠 추가"], ["rtsp", "IP 카메라 (RTSP)"], ["sample", "기본 샘플"]].map(([mode, label]) =>
              <button key={mode} type="button" aria-pressed={sourceMode === mode} disabled={pending} onClick={() => setSourceMode(mode)}>{label}</button>)}
          </div>
          <form key={sourceMode} onSubmit={add}>
            {sourceMode === "upload" && <>
              <label htmlFor="video-file">MP4 영상 · 최대 256 MiB</label>
              <input id="video-file" name="video" type="file" accept=".mp4,video/mp4" required />
            </>}
            {sourceMode === "sample" && <>
              <label htmlFor="sample">샘플 영상</label>
              <select id="sample" name="sample"><option value="demo-1.mp4">샘플 영상 1</option><option value="demo-2.mp4">샘플 영상 2</option></select>
            </>}
            {sourceMode === "rtsp" && <>
              <label htmlFor="rtsp-url">서버에서 허용한 RTSP 주소</label>
              <input id="rtsp-url" name="url" type="text" required maxLength={2048} autoComplete="off"
                spellCheck={false} placeholder="rtsp://192.168.1.50:554/live" />
              <p className="muted">카메라 아이디와 비밀번호는 서버에 설정합니다.</p>
            </>}
            {sourceMode === "webcam" ? <>
              <label htmlFor="webcam-device">이 기기의 카메라</label>
              <select id="webcam-device" name="device">
                <option value="">기본 웹캠</option>
                {devices.filter((device) => device.deviceId).map((device, index) =>
                  <option key={device.deviceId} value={device.deviceId}>{device.label || "웹캠 " + (index + 1)}</option>)}
              </select>
              <p className="muted">브라우저의 카메라 권한을 허용하세요.</p>
            </> : <>
              <label htmlFor="source-name">표시 이름</label>
              <input id="source-name" name="name" maxLength={120} placeholder="예: 매장 입구" required />
            </>}
            <button className="primary" disabled={pending}>{pending ? "처리 중…" :
              sourceMode === "webcam" ? "웹캠 연결" : sourceMode === "rtsp" ? "IP 카메라 등록" : "영상 등록"}</button>
          </form>
        </section>
        <AnalysisSettings {...{ settings, setSettings, models, chosen, pending }} apply={analyze} attachBehavior={canWrite ? attachBehavior : null} />
        <DisplaySettings {...{ display, setDisplay }} />
      </>}
      <AnalysisStats {...{ sources, snapshots }} />
    </aside>
  </div>;
}

function Records({ sources, api, revision, canWrite, onError, notify, refresh, openStorage }) {
  const [camera, setCamera] = useState("");
  const [method, setMethod] = useState("");
  const [events, setEvents] = useState([]);
  const [selected, setSelected] = useState(null);
  const [loading, setLoading] = useState(true);
  const [more, setMore] = useState(false);
  const [decision, setDecision] = useState("needs_review");
  const [saving, setSaving] = useState(false);
  const [seek, setSeek] = useState(0);
  const [deleting, setDeleting] = useState(null);
  const generation = useRef(0);
  const event = events.find((entry) => entry.id === selected);

  const load = useCallback(async (cursor) => {
    const sequence = ++generation.current;
    setLoading(true);
    const params = new URLSearchParams({ limit: "50" });
    if (camera) params.set("camera_id", camera);
    if (method) params.set("method", method);
    if (cursor) {
      params.set("before", cursor.detected_at);
      params.set("before_id", cursor.id);
    }
    try {
      const rows = await api("/events?" + params);
      if (sequence !== generation.current) return;
      setEvents((old) => cursor ? [...old, ...rows] : rows);
      setMore(rows.length === 50);
    } catch (error) {
      if (sequence === generation.current) onError(error);
    } finally {
      if (sequence === generation.current) setLoading(false);
    }
  }, [api, camera, method, onError]);

  useEffect(() => {
    load();
    return () => { generation.current += 1; };
  }, [load, revision]);

  function choose(entry) {
    setSelected(entry.id);
    setDecision(entry.decision || "needs_review");
    setSeek((old) => old + 1);
  }

  async function review(submit) {
    submit.preventDefault();
    if (!event) return;
    setSaving(true);
    try {
      const saved = await api("/events/" + event.id + "/reviews", { decision });
      setEvents((rows) => rows.map((row) => row.id === event.id ? { ...row, decision: saved.decision } : row));
      notify("검토 결과를 저장했습니다.");
    } catch (error) { onError(error); }
    finally { setSaving(false); }
  }

  async function deleteEvent() {
    setSaving(true);
    try {
      await api("/events/" + deleting.id, undefined, "DELETE");
      generation.current += 1;
      setLoading(false);
      setEvents((rows) => rows.filter((row) => row.id !== deleting.id));
      setSelected(null);
      setDeleting(null);
      notify("사건을 삭제했습니다. 원본 영상 용량은 저장 공간에서 정리할 수 있습니다.");
      await refresh();
    } catch (error) { onError(error); }
    finally { setSaving(false); }
  }

  return <div className="workspace records">
    <section className="panel timeline" aria-label="감지된 사건">
      <div className="section-heading"><h2>감지된 사건</h2></div>
      <div className="filters">
        <label>영상 소스<select value={camera} onChange={(e) => { setCamera(e.target.value); setSelected(null); }}>
          <option value="">전체 소스</option>{sources.map((source) => <option key={source.id} value={source.id}>{source.name}</option>)}
        </select></label>
        <label>판정 방식<select value={method} onChange={(e) => { setMethod(e.target.value); setSelected(null); }}>
          <option value="">전체 방식</option>{Object.entries(methods).map(([key, name]) => <option key={key} value={key}>{name}</option>)}
        </select></label>
      </div>
      <div aria-busy={loading}>
        {events.map((entry) => <button className="event" key={entry.id} aria-current={entry.id === selected} onClick={() => choose(entry)}>
          <strong>{types[entry.event_type] || entry.event_type}</strong>
          <small>{entry.camera_name} · {position(entry.source_time_ms)} · {methods[entry.detection_method]}</small>
        </button>)}
        {!events.length && <p className="empty">{loading ? "사건을 불러오는 중…" : "조건에 맞는 사건이 없습니다."}</p>}
      </div>
      {more && <button disabled={loading} onClick={() => load(events.at(-1))}>이전 사건 더 보기</button>}
    </section>
    <section className="panel playback">
      <h2>{event ? types[event.event_type] || event.event_type : "영상 재생"}</h2>
      {event ? <>
        <p className="muted">{methods[event.detection_method]} · {position(event.source_time_ms)}
          {event.model_score != null && " · 모델 점수 " + Math.round(event.model_score * 100) + "%"}
          {event.evidence?.reason && " · " + ({ pos_timeout: "POS 영역 미경유", track_lost: "트랙 소실" })[event.evidence.reason]}
        </p>
        {event.media_id ? <Video key={seek} mediaId={event.media_id} time={event.source_time_ms}
          onError={() => notify("영상을 열 수 없습니다. 접속 상태를 확인하세요.")} /> : <p className="empty">연결된 영상이 없습니다.</p>}
        <p className="muted">검토 상태 · {reviews[event.decision] || "미검토"}</p>
        {canWrite && <form className="review-form" onSubmit={review}>
          <label htmlFor="decision">검토 결과</label>
          <select id="decision" value={decision} onChange={(e) => setDecision(e.target.value)}>
            {Object.entries(reviews).map(([key, name]) => <option key={key} value={key}>{name}</option>)}
          </select>
          <button disabled={saving}>저장</button>
        </form>}
        {canWrite && <div className="record-actions">
          <button className="danger" disabled={saving} onClick={() => setDeleting(event)}>이 사건 삭제</button>
          <button onClick={openStorage}>영상 용량 정리</button>
        </div>}
      </> : <p className="muted">목록에서 사건을 선택하세요.</p>}
    </section>
    {deleting && <DeleteDialog title="이 사건을 삭제할까요?" busy={saving} onCancel={() => setDeleting(null)} onConfirm={deleteEvent}>
      <p>{deleting.camera_name} · {position(deleting.source_time_ms)} · {types[deleting.event_type] || deleting.event_type}</p>
      <p>선택한 사건을 목록에서 삭제합니다. 같은 영상을 사용하는 다른 사건과 원본 영상은 유지됩니다.</p>
      <p className="muted">영상 파일까지 지워 용량을 비우려면 저장 공간에서 영상·기록 삭제를 사용하세요.</p>
    </DeleteDialog>}
  </div>;
}

function Dashboard({ user, logout }) {
  const [tab, setTab] = useState("monitor");
  const [sources, setSources] = useState([]);
  const [models, setModels] = useState([]);
  const [selection, setSelection] = useState(null);
  const [revision, setRevision] = useState(0);
  const [notice, setNotice] = useState("");
  const generation = useRef(0);
  const api = request;
  const onError = useCallback((error) => {
    if (error.status === 401) logout(error.message);
    else setNotice(error.message);
  }, [logout]);
  const refresh = useCallback(async () => {
    const sequence = ++generation.current;
    const rows = await api("/cameras");
    if (sequence !== generation.current) return;
    setSources(rows);
    setSelection((old) => rows.some((source) => source.id === old) ? old : rows[0]?.id ?? null);
  }, [api]);

  useEffect(() => {
    const poll = () => refresh().catch(onError);
    poll();
    api("/models").then(setModels).catch(onError);
    const timer = setInterval(poll, 1500);
    return () => { clearInterval(timer); generation.current += 1; };
  }, [refresh, api, onError]);

  const canWrite = user.role === "owner";
  return <>
    <header className="topbar">
      <span className="brand"><span className="brand-mark" aria-hidden="true">◉</span> 매장 영상 관제</span>
      <nav aria-label="화면 선택">
        <button className="tab" aria-pressed={tab === "monitor"} onClick={() => setTab("monitor")}>영상 관제</button>
        <button className="tab" aria-pressed={tab === "records"} onClick={() => setTab("records")}>사건 기록</button>
        <button className="tab" aria-pressed={tab === "storage"} onClick={() => setTab("storage")}>저장 공간</button>
      </nav>
      <div id="account"><span id="user-name">{user.display_name}</span>
        <button className="quiet" onClick={() => api("/auth/logout", undefined, "POST").then(() => logout()).catch(onError)}>로그아웃</button>
      </div>
    </header>
    <main>
      <div className="heading"><div><h1>{({ monitor: "영상 관제", records: "사건 기록", storage: "저장 공간" })[tab]}</h1>
        <p className="muted">{({ monitor: "영상 소스를 선택하고 감지 결과를 확인하세요.", records: "사건 시점으로 이동하고 판정 결과를 검토하세요.", storage: "필요 없는 영상과 사건을 정리해 저장 용량을 확보하세요." })[tab]}</p></div>
        <button onClick={() => {
          setNotice("");
          refresh().catch(onError);
          setRevision((old) => old + 1);
        }}>새로고침</button>
      </div>
      <div className="metrics" aria-label="분석 상태">
        <div className="panel metric"><span>등록 소스</span><strong>{sources.length}</strong></div>
        <div className="panel metric"><span>대기·분석 중</span><strong>{sources.filter((source) => active.has(source.status)).length}</strong></div>
        <div className="panel metric"><span>최근 실행의 감지 사건</span><strong>{sources.reduce((total, source) => total + source.event_count, 0)}</strong></div>
      </div>
      <div hidden={tab !== "monitor"}>
        <Monitor {...{ sources, models, selection, api, canWrite, refresh, onError }} select={setSelection} notify={setNotice}
          refreshModels={async () => setModels(await api("/models"))} />
      </div>
      {tab === "records" && <Records {...{ sources, api, revision, canWrite, onError, refresh }} notify={setNotice} openStorage={() => setTab("storage")} />}
      {tab === "storage" && <Storage {...{ api, revision, canWrite, onError, refresh }} notify={setNotice} />}
      <p id="status" role="status">{notice}</p>
    </main>
  </>;
}

function App() {
  const [session, setSession] = useState(null);
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const logout = useCallback((reason = "") => { setSession(null); setMessage(reason); }, []);
  const [checking, setChecking] = useState(true);
  useEffect(() => {
    request("/auth/me").then(setSession).catch((error) => {
      if (error.status !== 401) setMessage(error.message);
    }).finally(() => setChecking(false));
  }, []);

  async function login(action, fields) {
    setBusy(true);
    setMessage("");
    try {
      setSession(await request("/auth/" + action, fields));
    } catch (error) { setMessage(error.message); }
    finally { setBusy(false); }
  }
  if (session) return <Dashboard user={session} logout={logout} />;
  return <>
    <header className="topbar"><span className="brand"><span className="brand-mark" aria-hidden="true">◉</span> 매장 영상 관제</span></header>
    <main>{checking ? <p role="status">로그인 상태를 확인하고 있습니다.</p> : <Login onSubmit={login} busy={busy} />}<p id="status" role="status">{message}</p></main>
  </>;
}

createRoot(document.getElementById("root")).render(<App />);
