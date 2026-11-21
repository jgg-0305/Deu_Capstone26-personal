"""프로젝트에서 검증한 YOLO 포즈 가중치의 등록과 저장 경로."""

import json
from functools import lru_cache

from psycopg.types.json import Jsonb

from backend.behavior_upload import MODEL_DIR
from backend.samples import ROOT


@lru_cache(maxsize=1)
def pose_artifacts():
    manifest = json.loads((ROOT / "docs/design/model-manifest.json").read_text(encoding="utf-8"))
    return {model["sha256"]: model for model in manifest["models"]
            if model["role"] == "human_and_keypoint_detection"}


def uploaded_pose_path(artifact):
    return MODEL_DIR / f"{artifact['sha256']}.pt"


def register_pose_upload(db, artifact, temp):
    digest = artifact["sha256"]
    db.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (digest,))
    path = uploaded_pose_path(artifact)
    db.execute("""
        INSERT INTO model_artifacts
            (role, name, version, sha256, storage_key, size_bytes, labels, preprocessing)
        VALUES ('pose', %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (sha256) DO NOTHING
    """, (artifact["id"], artifact.get("version", "verified"), digest, artifact["path"],
          artifact["bytes"], Jsonb(artifact["classes"]), Jsonb({
              "architecture": artifact["architecture"], "keypoint_shape": artifact["keypoint_shape"],
          })))
    # 해시로 검증한 파일로 원자적으로 교체해 손상된 로컬 복사본도 복구한다.
    temp.replace(path)
    return {"name": artifact["id"], "sha256": digest}
