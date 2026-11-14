import React, { useEffect, useRef, useState } from "react";
import { frameAt, shouldWait } from "./analysis-timeline.js";

export const labels = ["정상", "넘어짐", "파손", "방화", "흡연", "유기", "도난 의심", "폭행"];
export const initialSettings = {
  behavior_model: "behavior", pose_model: "human_pose", resolution: 480, detection_interval: 1,
  detection_confidence: 0.4, threshold: 0.5, event_threshold: 0.6, max_age: 30,
};
export const initialDisplay = { box: true, id: true, points: true, label: true, stand: true, pos: false };
export const modelNames = {
  human_pose: "YOLO11n · 경량", human_pose_s: "YOLO11s · 균형",
  behavior: "KeypointLSTM v2 · 관절과 움직임",
  behavior_kpt_v1: "KeypointLSTM v1 · 관절",
  behavior_rgb: "ResNet50 + LSTM · 장면 전체",
};
const skeleton = [[0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],
  [11,12],[11,13],[13,15],[12,14],[14,16]];

function behaviorText(person) {
  if (person.observed === false) return "관절 미관측 · 추적 유지";
  if (person.label != null) return labels[person.label] + (person.method === "rule" ? " · 규칙" :
    person.score == null ? "" : ` ${Math.round(person.score * 100)}%`);
  const buffer = person.buffer;
  if (buffer?.state === "tracking_gap") return "관절 미관측 · 재검출 대기";
  if (buffer?.state === "uncertain") return `판정 보류 · ${labels[buffer.candidate.label]} ${Math.round(buffer.candidate.score * 100)}%`;
  if (buffer?.state === "insufficient_keypoints") return "관절 정보 부족";
  return buffer ? `관절 수집 ${buffer.collected}/${buffer.required}` : "분류 대기";
}

export function startWebcamFeed(id, stream, onError) {
  const video = document.createElement("video");
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");
  const controller = new AbortController();
  video.muted = true;
  video.playsInline = true;
  video.srcObject = stream;
  let busy = false;
  const timer = setInterval(() => {
    if (busy || !video.videoWidth || video.readyState < 2) return;
    busy = true;
    const scale = Math.min(1, 640 / video.videoWidth, 480 / video.videoHeight);
    canvas.width = Math.round(video.videoWidth * scale);
    canvas.height = Math.round(video.videoHeight * scale);
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(async (blob) => {
      try {
        if (!blob || controller.signal.aborted) return;
        const response = await fetch("/api/v1/cameras/" + id + "/frame", {
          method: "PUT", headers: { "X-CCTV-Request": "1", "Content-Type": "image/jpeg" },
          body: blob, signal: controller.signal,
        });
        if (!response.ok) {
          const error = new Error("웹캠 프레임을 서버에 보내지 못했습니다.");
          error.status = response.status;
          if ([401, 403, 404].includes(response.status)) clearInterval(timer);
          throw error;
        }
      } catch (error) {
        if (!controller.signal.aborted) onError(error);
      } finally { busy = false; }
    }, "image/jpeg", 0.65);
  }, 100);
  video.play().catch(onError);
  return () => {
    controller.abort();
    clearInterval(timer);
    video.pause();
    video.srcObject = null;
    stream.getTracks().forEach((track) => track.stop());
  };
}

export function useSnapshots(sources, api) {
  const [snapshots, setSnapshots] = useState({});
  const keys = sources.filter((source) => ["running", "completed", "failed", "stopped"].includes(source.status))
    .map((source) => source.id + ":" + source.status + ":" + (source.run_id || "")).sort().join(",");
  useEffect(() => {
    let disposed = false;
    let timer;
    const ids = keys ? keys.split(",").map((value) => value.split(":")[0]) : [];
    async function poll() {
      const rows = await Promise.all(ids.map(async (id) => {
        try { return [id, await api("/cameras/" + id + "/snapshot")]; }
        catch { return [id, { unavailable: true, people: [], frame_count: 0 }]; }
      }));
      if (disposed) return;
      setSnapshots(Object.fromEntries(rows));
      if (keys.includes(":running")) timer = setTimeout(poll, 220);
    }
    poll();
    return () => { disposed = true; clearTimeout(timer); };
  }, [keys, api]);
  return snapshots;
}

export function AnalysisFrame({ snapshot, display }) {
  return <div className="analysis-frame">
    <img src={"data:image/jpeg;base64," + snapshot.image} alt="AI 분석 프레임" />
    <AnalysisOverlay {...{ snapshot, display }} />
  </div>;
}

function AnalysisOverlay({ snapshot, display }) {
  const svg = useRef(null);
  const [size, setSize] = useState({ width: 640, height: 480 });
  useEffect(() => {
    const observer = new ResizeObserver(([entry]) => {
      if (entry.contentRect.width && entry.contentRect.height) {
        setSize({ width: entry.contentRect.width, height: entry.contentRect.height });
      }
    });
    observer.observe(svg.current);
    return () => observer.disconnect();
  }, []);
  const { width, height } = size;
  const box = (values) => values.map((value, index) => value * (index % 2 ? height : width));
  return <>
    {display.label && snapshot.scene && <span className="scene-label">
      장면 · {labels[snapshot.scene[0]]} {Math.round(snapshot.scene[1] * 100)}%
    </span>}
    <svg ref={svg} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-hidden="true">
      {(snapshot.objects || []).filter((item) => item.class_id === 0 ? display.stand : display.pos).map((item, index) => {
        const [x1, y1, x2, y2] = box(item.box);
        return <rect key={index} x={x1} y={y1} width={x2 - x1} height={y2 - y1}
          className={item.class_id === 0 ? "stand-box" : "pos-box"} />;
      })}
      {snapshot.people.map((person) => {
        const [x1, y1, x2, y2] = box(person.box);
        const points = person.points;
        const result = behaviorText(person);
        const text = [display.id && "ID " + person.id, display.label && result].filter(Boolean).join(" · ");
        return <g key={person.id} className={(person.label > 0 ? "anomaly" : "person") + (person.observed === false ? " predicted" : "")}>
          {display.box && <rect x={x1} y={y1} width={x2 - x1} height={y2 - y1} />}
          {text && <text x={Math.max(2, x1)} y={Math.max(13, y1 - 5)}>{text}</text>}
          {display.points && skeleton.map(([a, b]) => points[a]?.[2] >= 0.3 && points[b]?.[2] >= 0.3 &&
            <line key={a + ":" + b} x1={points[a][0] * width} y1={points[a][1] * height}
              x2={points[b][0] * width} y2={points[b][1] * height} />)}
          {display.points && points.map(([x, y, confidence], index) => confidence >= 0.3 &&
            <circle key={index} cx={x * width} cy={y * height} r="1.7" />)}
        </g>;
      })}
    </svg>
  </>;
}

const failures = {
  model_load_failed: "모델을 불러오지 못했습니다. 모델 등록과 워커 로그를 확인하세요.",
  source_unavailable: "입력 영상이 끊겼거나 해석 가능한 프레임이 없습니다.",
  inference_failed: "추론 중 오류가 발생했습니다. 워커 로그를 확인하세요.",
  lease_expired: "워커의 응답이 끊겼습니다. 워커를 실행한 뒤 다시 분석하세요.",
};

export function ModelStatus({ source, snapshot }) {
  const running = source.status === "running";
  let message;
  if (source.status === "failed") message = failures[source.error_code] || "분석에 실패했습니다. 워커 실행 상태와 등록된 모델을 확인하세요.";
  else if (source.status === "queued") message = "워커 배정 대기 · 계속 대기하면 분석 워커가 실행 중인지 확인하세요.";
  else if (source.status === "starting") message = "모델 파일 확인 중";
  else if (snapshot?.unavailable) message = "분석 상태를 조회할 수 없습니다. 서버 연결을 확인하세요.";
  else if (!snapshot || snapshot.run_id !== source.run_id) message = running ? "워커 연결됨 · 모델 준비 중" : "분석 기록 없음";
  else if (running && Date.now() - snapshot.updated_at_ms > 10000) message = "분석 결과 갱신 지연 · 마지막 처리 " + snapshot.frame_count + "프레임";
  else if (snapshot.phase === "loading") message = "모델 불러오는 중 · " + modelNames[snapshot.pose_model];
  else {
    const progress = snapshot.total_frames ? ` / ${snapshot.total_frames} (${Math.min(100, Math.round(snapshot.frame_count / snapshot.total_frames * 100))}%)` : "";
    message = `${source.status === "completed" ? "분석 완료" : source.status === "stopped" ? "분석 중단" : "분석 중"} · ${snapshot.frame_count}${progress}프레임`;
  }
  return <div className="model-status" role="status">
    <strong>{message}</strong>
    {snapshot?.run_id === source.run_id && !["queued", "starting"].includes(source.status) && <span>
      사람 검출 {snapshot.pose_count ?? 0}회 · 행동 추론 {snapshot.inference_count ?? 0}회
      {snapshot.fps > 0 && ` · 처리 ${snapshot.fps.toFixed(1)} FPS`}
      {snapshot.source_fps > 0 && ` · 입력 ${snapshot.source_fps.toFixed(1)} FPS`}
      {snapshot.inference_count === 0 && snapshot.phase !== "loading" &&
        (snapshot.scene_buffer ? ` · 장면 수집 ${snapshot.scene_buffer.collected}/${snapshot.scene_buffer.required}` :
          snapshot.people.length ? " · 동일 인물의 관절 30개 시점 수집 중" : " · 사람 검출 대기")}
      {snapshot.frame_duration_ms > 0 && snapshot.behavior_model !== "behavior_rgb" &&
        ` · 분류 입력 ${(snapshot.frame_duration_ms * 30 / 1000).toFixed(1)}초`}
    </span>}
    {source.pose_model && <span>{modelNames[source.pose_model]} · {modelNames[source.behavior_model] || source.behavior_filename}</span>}
  </div>;
}

export function FileAnalysis({ source, display, enabled, onError }) {
  const [timeline, setTimeline] = useState({ run: null, frames: [], offset: 0 });
  const [time, setTime] = useState(0);
  const [waiting, setWaiting] = useState(false);
  const [timelineError, setTimelineError] = useState("");
  const video = useRef(null);
  const wantsPlay = useRef(false);
  const autoPause = useRef(false);
  const cache = useRef(timeline);
  const { id, status } = source;
  const run = ["queued", "starting"].includes(status) ? null : source.run_id;
  const running = ["queued", "starting", "running"].includes(status);
  cache.current = timeline;
  useEffect(() => {
    const player = video.current;
    if (!player.requestVideoFrameCallback) return;
    let handle;
    const update = (_, frame) => {
      setTime(frame.mediaTime * 1000);
      handle = player.requestVideoFrameCallback(update);
    };
    handle = player.requestVideoFrameCallback(update);
    return () => player.cancelVideoFrameCallback(handle);
  }, []);
  useEffect(() => {
    if (!run) return;
    let cancelled = false;
    let timer;
    let offset = cache.current.run === run ? cache.current.offset : 0;
    const controller = new AbortController();
    async function poll() {
      try {
        const params = new URLSearchParams({ run_id: run, offset });
        const response = await fetch("/api/v1/cameras/" + id + "/timeline?" + params, { signal: controller.signal });
        if (!response.ok) throw new Error("분석 결과를 불러오지 못했습니다. 연결 상태를 확인하세요.");
        const data = await response.json();
        if (cancelled) return;
        setTimelineError("");
        offset = data.offset;
        if (data.frames.length) {
          setTimeline((current) => ({ run, frames: [
            ...(current.run === run ? current.frames : []), ...data.frames,
          ], offset }));
        }
        if (data.frames.length || running) timer = setTimeout(poll, data.frames.length ? 0 : 300);
      } catch (error) {
        if (!cancelled) {
          setTimelineError(error.message);
          timer = setTimeout(poll, 1500);
        }
      }
    }
    if (cache.current.run !== run) setTimeline({ run, frames: [], offset: 0 });
    poll();
    return () => { cancelled = true; clearTimeout(timer); controller.abort(); };
  }, [id, run, running]);
  const frames = timeline.run === run ? timeline.frames : [];
  const current = frameAt(frames, time);
  // AI 화면은 분석이 끝난 시점까지만 재생하고 결과가 도착하면 이어서 재생한다.
  useEffect(() => {
    const player = video.current;
    if (!wantsPlay.current) return;
    const wait = enabled && shouldWait(frames, time, running, waiting);
    if (wait) {
      if (!player.paused) { autoPause.current = true; player.pause(); }
      if (!waiting) setWaiting(true);
    } else if (waiting) {
      setWaiting(false);
      player.play().catch(() => { wantsPlay.current = false; });
    }
  }, [frames, time, running, waiting, enabled]);
  const update = (event) => setTime(event.currentTarget.currentTime * 1000);
  return <div className="file-analysis">
    <div className="analysis-frame synchronized-video">
      <video ref={video} controls playsInline preload="metadata" src={"/api/v1/cameras/" + id + "/video"}
        onTimeUpdate={update} onSeeking={update} onError={onError}
        onPlay={() => {
          wantsPlay.current = true;
          if (enabled && shouldWait(frames, video.current.currentTime * 1000, running, false)) {
            autoPause.current = true;
            video.current.pause();
            setWaiting(true);
          }
        }}
        onPause={() => {
          if (autoPause.current) { autoPause.current = false; return; }
          wantsPlay.current = false;
          setWaiting(false);
        }} />
      {enabled && current && <AnalysisOverlay snapshot={current} display={display} />}
    </div>
    <div className="playback-status">
      <span>{!enabled ? "원본 재생" : waiting ? "AI 분석을 기다리는 중 · 준비되면 자동 재생" :
        current ? `AI 분석 · 영상 ${(time / 1000).toFixed(1)}초 · 행동 추론 ${current.inference_count ?? 0}회` :
        "이 시점의 분석 결과 없음"}</span>
      {waiting && <button type="button" onClick={() => { wantsPlay.current = false; setWaiting(false); }}>대기 취소</button>}
      {timelineError && <span role="alert">{timelineError}</span>}
      {enabled && current && <span className="playback-results">{current.people.length ? current.people.map((person) =>
        `ID ${person.id}: ${behaviorText(person)}`).join(" · ") : "이 시점에 감지된 인물 없음"}</span>}
    </div>
  </div>;
}

export function AnalysisSettings({ settings, setSettings, models, chosen, apply, pending, attachBehavior }) {
  const update = (event) => {
    const { name, value } = event.target;
    setSettings({ ...settings, [name]: ["behavior_model", "pose_model"].includes(name) ? value : Number(value) });
  };
  const behaviorModels = models.filter((model) => model.role === "behavior");
  const selectedBehavior = behaviorModels.find((model) => model.name === settings.behavior_model);
  const poseModels = models.filter((model) => model.role === "pose" && modelNames[model.name]);
  const demoAvailable = poseModels.some((model) => model.name === "human_pose_s");
  return <section className="panel">
    <h2>모델 설정</h2>
    <div className="switch model-presets">
      <button type="button" disabled={pending} onClick={() => setSettings({ ...initialSettings,
        pose_model: demoAvailable ? "human_pose_s" : "human_pose" })}>원본 영상 기본값</button>
      <button type="button" disabled={pending || !demoAvailable} onClick={() => setSettings({ ...initialSettings,
        pose_model: "human_pose_s", detection_confidence: 0.9 })}>데모 비교용 설정</button>
    </div>
    {!demoAvailable && <p className="muted">YOLO11s를 설치하고 모델을 등록하면 데모 설정을 선택할 수 있습니다.</p>}
    <form onSubmit={apply}>
      <label htmlFor="pose-model">사람·관절 검출 모델</label>
      <select id="pose-model" name="pose_model" value={settings.pose_model} onChange={update}>
        {poseModels.map((model) => <option key={model.id} value={model.name}>{modelNames[model.name]}</option>)}
      </select>
      <label htmlFor="behavior-model">이상행동 판별 모델</label>
      <select id="behavior-model" name="behavior_model" value={settings.behavior_model} onChange={update} disabled={pending}>
        {behaviorModels.map((model) => <option key={model.id} value={model.name}>
          {modelNames[model.name] || (model.filename + " · " + model.sha256.slice(0, 8))}
        </option>)}
      </select>
      {selectedBehavior && <p className="muted" title={selectedBehavior.sha256}>
        선택한 파일 · {selectedBehavior.storage_key || selectedBehavior.filename} · {selectedBehavior.sha256.slice(0, 12)}
      </p>}
      {attachBehavior && <>
        <label htmlFor="behavior-file">행동 모델 파일 첨부 (.pth)</label>
        <input id="behavior-file" type="file" accept=".pth" disabled={pending} onChange={attachBehavior} />
        <p className="muted">KeypointLSTM v1/v2 가중치 또는 프로젝트의 장면 모델 model_final.pth를 선택하세요.</p>
      </>}
      <label htmlFor="resolution">처리 해상도</label>
      <select id="resolution" name="resolution" value={settings.resolution} onChange={update}>
        <option value="384">384 · 빠름</option><option value="480">480 · 권장</option><option value="640">640 · 정밀</option>
      </select>
      <label htmlFor="detection-interval">검출 간격</label>
      <select id="detection-interval" name="detection_interval" value={settings.detection_interval} onChange={update}>
        <option value="1">1 · 매 프레임</option><option value="2">2 · 두 프레임마다</option><option value="3">3 · 세 프레임마다</option>
      </select>
      <label htmlFor="detect-confidence">포즈 신뢰도 임계값 <output>{Math.round(settings.detection_confidence * 100)}%</output></label>
      <input id="detect-confidence" name="detection_confidence" type="range" min=".1" max=".95" step=".01"
        value={settings.detection_confidence} onChange={update} />
      <label htmlFor="threshold">행동 표시 기준 <output>{Math.round(settings.threshold * 100)}%</output></label>
      <input id="threshold" name="threshold" type="range" min=".5" max="1" step=".01" value={settings.threshold} onChange={update} />
      <label htmlFor="event-threshold">사건 저장 기준 <output>{Math.round(settings.event_threshold * 100)}%</output></label>
      <input id="event-threshold" name="event_threshold" type="range" min=".5" max="1" step=".01" value={settings.event_threshold} onChange={update} />
      <p className="muted">원본 영상 기본값: 사람 검출 40%, 행동 표시 50%, 사건 저장 60%. 사람 검출 90%는 관절 추적이 끊길 수 있어 데모 비교에만 사용합니다. 3 FPS 영상의 30프레임 입력은 10초입니다.</p>
      <label htmlFor="max-age">DeepSORT 최대 소실 프레임</label>
      <input id="max-age" name="max_age" type="number" min="1" max="120" required value={settings.max_age} onChange={update} />
      <p className="muted">새 소스에 적용됩니다. 기존 소스는 선택한 뒤 다시 분석하세요.</p>
      <button className="primary" disabled={pending || !chosen?.is_enabled}>{chosen ? chosen.name + " 다시 분석" : "영상 소스를 선택하세요"}</button>
    </form>
  </section>;
}

export function DisplaySettings({ display, setDisplay }) {
  const options = { box: "바운딩 박스", id: "트랙 ID", points: "관절 키포인트",
    label: "행동 분류", stand: "물품 가판대 감지", pos: "결제기(POS) 감지" };
  return <section className="panel display-settings">
    <h2>표시 옵션</h2>
    {Object.entries(options).map(([key, name]) => <label key={key}>
      <input type="checkbox" checked={display[key]} onChange={(event) => setDisplay({ ...display, [key]: event.target.checked })} />
      {name}
    </label>)}
  </section>;
}

export function AnalysisStats({ sources, snapshots }) {
  const available = sources.filter((source) => snapshots[source.id]?.run_id === source.run_id && source.run_id
    && !["queued", "starting"].includes(source.status));
  const running = available.filter((source) => source.status === "running");
  const live = (running.length ? running : available)
    .map((source) => snapshots[source.id]);
  const people = live.flatMap((snapshot) => snapshot.people).filter((person) => person.observed !== false);
  const scenes = live.flatMap(({ scene }) => scene ? [{ label: scene[0], score: scene[1] }] : []);
  const result = [...people, ...scenes].filter((person) => person.label != null).sort((a, b) =>
    Number(b.label > 0) - Number(a.label > 0) || (b.score ?? 0) - (a.score ?? 0))[0];
  return <section className="panel analysis-stats">
    <h2>분석 통계</h2>
    <p className="muted">{running.length ? "분석 중인 소스의 최근 결과" : "완료·중단된 분석의 마지막 결과"}</p>
    <dl>
      <div><dt>감지된 인원</dt><dd>{people.length}명</dd></div>
      <div><dt>행동 분류</dt><dd>{result ? labels[result.label] + (result.score == null ? "" :
        " (" + Math.round(result.score * 100) + "%)") : live.length ?
        (people.length ? "관절 수집 중" : "최근 프레임에 인물 없음") : "분석 대기"}</dd></div>
      <div><dt>평균 처리 FPS</dt><dd>{live.length ?
        (live.reduce((sum, snapshot) => sum + snapshot.fps, 0) / live.length).toFixed(1) : "—"}</dd></div>
      <div><dt>행동 추론 실행</dt><dd>{live.reduce((sum, snapshot) => sum + (snapshot.inference_count || 0), 0)}회</dd></div>
    </dl>
  </section>;
}
