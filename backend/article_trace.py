# 상품 추적 진단을 조회하는 명령줄 도구다. 실행 ID에 해당하는 결과를 읽어 요청한 영상 시간 범위만 출력한다. 상품 검출, 손 연결,
# 가방과 은닉 판정 중 어디에서 후보가 사라졌는지 비교할 때 쓴다.
"""저장된 영상 분석에서 상품·가방 추적 진단만 조회한다."""

import argparse
import json
from uuid import UUID

from backend.analysis_media import frame_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", type=UUID)
    parser.add_argument("--from-sec", type=float, default=0)
    parser.add_argument("--to-sec", type=float, default=float("inf"))
    args = parser.parse_args()
    path = frame_path(args.run_id, "timeline.jsonl")
    if not path.is_file():
        parser.error("이 실행 ID에 해당하는 영상 타임라인이 없습니다.")
    previous = None
    count = 0
    with path.open(encoding="utf-8") as timeline:
        for line in timeline:
            # 실행 중인 워커가 마지막 줄을 기록하는 중이면 다음 조회에서 읽는다.
            if not line.endswith("\n"):
                break
            row = json.loads(line)
            trace = row.get("article_trace")
            if not trace:
                continue
            stamp = trace["sample_time_ms"]
            if stamp == previous or not args.from_sec <= stamp / 1000 <= args.to_sec:
                continue
            previous = stamp
            if not count:
                print(json.dumps({key: row.get(key) for key in (
                    "runtime", "pose_model", "behavior_model", "auxiliary_models",
                    "behavior_threshold", "event_threshold", "source_fps",
                )}))
            print(json.dumps(trace))
            count += 1
    if not count:
        parser.error("해당 구간에 상품 진단이 없습니다. 변경된 워커로 상품 모델을 켜고 다시 분석하세요.")


if __name__ == "__main__":
    main()
