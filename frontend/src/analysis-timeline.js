// 재생 시점보다 뒤의 결과나 너무 오래된 결과를 현재 화면에 붙이지 않는다.
export function frameAt(frames, time) {
  let low = 0;
  let high = frames.length;
  while (low < high) {
    const mid = (low + high) >>> 1;
    if (frames[mid].time_ms <= time + 1) low = mid + 1;
    else high = mid;
  }
  const frame = frames[low - 1];
  if (!frame) return null;
  const next = frames[low];
  // 3 FPS 영상의 프레임은 333ms 동안 보인다. 실제 영상 시각 간격까지 결과를 유지한다.
  // 새 메타데이터는 검출 간격을 포함한 유효 시간을 제공한다.
  const duration = frame.frame_duration_ms ?? (next ? next.time_ms - frame.time_ms : 220);
  return time - frame.time_ms <= duration + 1 ? frame : null;
}

export function shouldWait(frames, time, running, waiting) {
  const frontier = frames.at(-1)?.time_ms ?? -1;
  // 다시 재생할 때 약간의 여유를 확보해 짧은 재생과 정지를 반복하지 않는다.
  return running && (frontier < 0 || frontier - time < (waiting ? 600 : 120));
}
