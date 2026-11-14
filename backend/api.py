"""인증이 필요한 웹 API와 분석 워커용 내부 API."""

import hashlib
import json
import os
import secrets
import subprocess
import sys
from typing import Annotated, Literal
from uuid import UUID, uuid4

import anyio
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from psycopg import Connection, Error, OperationalError
from psycopg.errors import ForeignKeyViolation, UndefinedColumn, UndefinedTable, UniqueViolation
from psycopg.types.json import Jsonb
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

from backend.analysis_media import FRAME_LIMIT, frame_path, read_snapshot, replace_frame
from backend.behavior_upload import MAX_MODEL_BYTES, MODEL_DIR, UPLOAD_PREFIX, bundled_rgb_artifact
from backend.db import DatabaseConfigurationError, connect
from backend.live import BOUNDARY, frames, public_rtsp_url, resolve_camera
from backend.passwords import hasher, test_admin_enabled, verify_password
from backend.samples import MAX_VIDEO_BYTES, ROOT, UPLOAD_DIR, asset_path, source_path
from backend.storage import cleanup_camera, inventory, prepare_purge

app = FastAPI(title="CCTV API", docs_url=None, redoc_url=None, openapi_url=None)
FRONTEND_DIST = ROOT / "frontend" / "dist"

bearer = HTTPBearer(auto_error=False)
SESSION_COOKIE = "cctv_session"
SESSION_HOURS = 24
MODEL_VALIDATION_LIMITER = anyio.CapacityLimiter(1)


@app.get("/", include_in_schema=False)
def dashboard():
    index = FRONTEND_DIST / "index.html"
    if not index.is_file():
        raise HTTPException(503, "Build the React frontend with npm --prefix frontend run build")
    return FileResponse(index)

Credentials = Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]
ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]
ModelRole = Literal["pose", "object", "behavior", "reid"]
ACTIVE_RUNS = ("starting", "running", "stopping")


@app.get("/assets/{filename}", include_in_schema=False)
def frontend_asset(filename: str):
    path = FRONTEND_DIST / "assets" / filename
    if not filename.endswith((".js", ".css")) or not path.is_file():
        raise HTTPException(404, "Asset not found")
    return FileResponse(path)


@app.middleware("http")
async def private_response(request: Request, call_next):
    # 사용자 API의 변경 요청은 브라우저의 동일 출처 스크립트에서만 허용한다.
    if (request.url.path.startswith("/api/v1/") and request.method not in {"GET", "HEAD", "OPTIONS"}
            and request.headers.get("X-CCTV-Request") != "1"):
        response = JSONResponse({"detail": "허용되지 않은 요청입니다."}, status_code=403)
    else:
        response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.exception_handler(RequestValidationError)
async def invalid_input(request: Request, exc: RequestValidationError):
    return JSONResponse({"detail": [
        {"loc": error["loc"], "msg": error["msg"]} for error in exc.errors()
    ]}, status_code=422)


@app.exception_handler(DatabaseConfigurationError)
async def database_configuration_error(request: Request, exc: DatabaseConfigurationError):
    return JSONResponse({"detail": str(exc)}, status_code=503)


@app.exception_handler(Error)
async def database_error(request: Request, exc: Error):
    if isinstance(exc, (UndefinedTable, UndefinedColumn)):
        detail = "현재 데이터베이스에 마이그레이션이 필요합니다. python -m backend.manage migrate를 실행하세요."
    elif isinstance(exc, OperationalError):
        detail = "PostgreSQL에 연결할 수 없습니다. 서버 실행 상태와 CCTV_DATABASE_URL의 주소, 계정, 데이터베이스 이름을 확인하세요."
    else:
        detail = "데이터베이스 요청을 처리하지 못했습니다."
    return JSONResponse({"detail": detail}, status_code=503)


def database():
    with connect() as connection:
        yield connection


Db = Annotated[Connection, Depends(database, scope="function")]


def unauthorized():
    return HTTPException(401, "Invalid or missing bearer token",
                         headers={"WWW-Authenticate": "Bearer"})


def session_digest(request: Request):
    token = request.cookies.get(SESSION_COOKIE, "")
    if not token or len(token) > 256:
        raise HTTPException(401, "로그인이 필요합니다.")
    return hashlib.sha256(token.encode()).hexdigest()


def lookup_user(db, digest):
    user = db.execute("""
        SELECT u.id, u.display_name, u.username, u.role, u.is_test_admin, s.id AS session_id
        FROM auth_sessions s JOIN users u ON u.id = s.user_id
        WHERE s.token_hash = %s AND s.revoked_at IS NULL AND s.expires_at > now()
          AND u.is_active AND u.disabled_at IS NULL
    """, (digest,)).fetchone()
    if user is None or (user["is_test_admin"] and not test_admin_enabled()):
        raise HTTPException(401, "로그인이 필요합니다.")
    return user


def current_user(digest: Annotated[str, Depends(session_digest)], db: Db):
    return lookup_user(db, digest)


User = Annotated[dict, Depends(current_user)]


def owner(user: User):
    if user["role"] != "owner":
        raise HTTPException(403, "Owner access is required")
    return user


Owner = Annotated[dict, Depends(owner)]


def worker(credentials: Credentials):
    token = os.environ.get("CCTV_WORKER_TOKEN", "")
    if len(token) < 32:
        raise HTTPException(503, "Worker ingestion is disabled")
    if credentials is None or not secrets.compare_digest(
        token.encode(), credentials.credentials.encode()
    ):
        raise unauthorized()


Worker = Annotated[None, Depends(worker)]


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


Username = Annotated[str, StringConstraints(strip_whitespace=True, to_lower=True, pattern=r"^[a-zA-Z0-9_]{3,32}$")]


class LoginInput(Input):
    username: Username
    password: str = Field(min_length=1, max_length=128)


class RegisterInput(LoginInput):
    display_name: ShortText
    password: str = Field(min_length=8, max_length=128)


BehaviorModel = Annotated[str, StringConstraints(
    pattern=r"^(behavior|behavior_kpt_v1|behavior_rgb|upload_[0-9a-f]{64})$"
)]
PoseModel = Literal["human_pose", "human_pose_s"]


class AnalysisInput(Input):
    behavior_model: BehaviorModel = "behavior"
    pose_model: PoseModel = "human_pose"
    threshold: float = Field(default=0.5, ge=0.5, le=1)
    event_threshold: float = Field(default=0.6, ge=0.5, le=1)
    resolution: int = 480
    detection_interval: int = Field(default=1, ge=1, le=3)
    detection_confidence: float = Field(default=0.4, ge=0.1, le=0.95)
    max_age: int = Field(default=30, ge=1, le=120)

    @field_validator("resolution")
    @classmethod
    def supported_resolution(cls, value):
        if value not in (384, 480, 640):
            raise ValueError("처리 해상도는 384, 480, 640 중 하나여야 합니다.")
        return value


class WebcamInput(Input):
    name: ShortText
    settings: AnalysisInput = Field(default_factory=AnalysisInput)


class CameraInput(Input):
    name: ShortText
    sample: Literal["demo-1.mp4", "demo-2.mp4"]
    settings: AnalysisInput = Field(default_factory=AnalysisInput)


class RtspInput(Input):
    name: ShortText
    url: str = Field(min_length=1, max_length=2048)
    settings: AnalysisInput = Field(default_factory=AnalysisInput)


class RunInput(Input):
    camera_id: UUID
    worker_id: ShortText
    models: dict[ModelRole, UUID]
    rule_version: ShortText
    job_id: UUID | None = None
    claim_token: UUID | None = None
    resolution: Literal[384, 480, 640] = 480
    detection_interval: int = Field(default=1, ge=1, le=60)
    detection_confidence: float = Field(default=0.4, ge=0.1, le=0.95)
    max_age: int = Field(default=30, ge=1, le=120)
    behavior_threshold: float = Field(default=0.5, ge=0.5, le=1)
    event_threshold: float = Field(default=0.6, ge=0.5, le=1)
    alert_threshold: float = Field(default=0.9, ge=0.5, le=1)

    @model_validator(mode="after")
    def required_models(self):
        if (self.job_id is None) != (self.claim_token is None):
            raise ValueError("Job ID and claim token must be supplied together")
        if not {"pose", "object", "behavior"} <= self.models.keys():
            raise ValueError("pose, object and behavior models are required")
        return self


class Lease(Input):
    lease_token: UUID


class RunFinish(Lease):
    status: Literal["completed", "stopped", "failed"]
    error_code: Literal["model_load_failed", "source_unavailable", "inference_failed"] | None = None


class RuleEvidence(Input):
    reason: Literal["pos_timeout", "track_lost"]
    stand_contact_ms: int | None = Field(default=None, ge=0, le=2**63 - 1)


class EventInput(Lease):
    producer_event_key: ShortText
    event_type: Literal["fall", "vandalism", "arson", "smoking", "abandon", "theft_suspected", "assault"]
    track_id: ShortText | None = None
    detection_method: Literal["model", "rule", "hybrid"]
    model_score: float | None = Field(default=None, ge=0, le=1)
    score_model_role: ModelRole | None = None
    rule_code: ShortText | None = None
    rule_version: ShortText | None = None
    evidence: RuleEvidence | None = None
    detected_at: AwareDatetime
    occurred_at: AwareDatetime | None = None
    source_time_ms: int | None = Field(default=None, ge=0, le=2**63 - 1)
    source_frame_index: int | None = Field(default=None, ge=0, le=2**63 - 1)
    source_end_time_ms: int | None = Field(default=None, ge=0, le=2**63 - 1)

    @model_validator(mode="after")
    def validate_detection(self):
        scored = self.model_score is not None and self.score_model_role is not None
        if self.detection_method == "rule":
            if self.model_score is not None or self.score_model_role is not None:
                raise ValueError("Rule events cannot carry model scores")
        elif not scored:
            raise ValueError("Model events require a score and model role")
        if self.detection_method != "model" and not (self.rule_code and self.rule_version):
            raise ValueError("Rule events require a rule code and version")
        if self.detection_method == "model" and (self.rule_code or self.rule_version or self.evidence):
            raise ValueError("Model-only events cannot carry rule evidence")
        if self.source_end_time_ms is not None and (
            self.source_time_ms is None or self.source_end_time_ms < self.source_time_ms
        ):
            raise ValueError("The event end must be at or after the source start")
        return self


class JobClaim(Input):
    claim_token: UUID


class ReviewInput(Input):
    decision: Literal["confirmed", "false_positive", "needs_review"]
    note: str = Field(default="", max_length=2000)


@app.get("/health/live")
def live():
    return {"status": "ok"}


@app.get("/health/ready")
def ready(db: Db):
    if db.execute("SELECT 1 FROM schema_migrations WHERE name = %s",
                  ("008_behavior_uploads.sql",)).fetchone() is None:
        raise HTTPException(503, "Database migration is required")
    return {"status": "ready"}


def public_user(user):
    return {key: user[key] for key in ("id", "display_name", "username", "role")}


def sign_in(request, response, db, user):
    # 로그인할 때 새 세션을 만들고 같은 브라우저의 이전 세션을 폐기한다.
    old = request.cookies.get(SESSION_COOKIE, "")
    if old and len(old) <= 256:
        db.execute("UPDATE auth_sessions SET revoked_at = now() WHERE token_hash = %s AND revoked_at IS NULL",
                   (hashlib.sha256(old.encode()).hexdigest(),))
    token = secrets.token_urlsafe(32)
    db.execute("""
        INSERT INTO auth_sessions (user_id, token_hash, expires_at)
        VALUES (%s, %s, now() + %s * interval '1 hour')
    """, (user["id"], hashlib.sha256(token.encode()).hexdigest(), SESSION_HOURS))
    response.set_cookie(SESSION_COOKIE, token, httponly=True, samesite="strict",
                        secure=request.url.scheme == "https", path="/api/v1",
                        max_age=SESSION_HOURS * 3600)
    return public_user(user)


@app.post("/api/v1/auth/register", status_code=201)
def register(data: RegisterInput, request: Request, response: Response, db: Db):
    if data.username == "admin":
        raise HTTPException(409, "예약된 아이디입니다.")
    password_hash = hasher.hash(data.password)
    try:
        user = db.execute("""
            INSERT INTO users (username, display_name, password_hash, role)
            VALUES (%s, %s, %s, 'owner') RETURNING id, display_name, username, role
        """, (data.username, data.display_name, password_hash)).fetchone()
    except UniqueViolation:
        raise HTTPException(409, "이미 사용 중인 아이디입니다.") from None
    return sign_in(request, response, db, user)


@app.post("/api/v1/auth/login")
def login(data: LoginInput, request: Request, response: Response, db: Db):
    user = db.execute("""
        SELECT id, username, display_name, role, password_hash, is_test_admin
        FROM users WHERE username = %s AND is_active AND disabled_at IS NULL
    """, (data.username,)).fetchone()
    if not verify_password(data.password, user["password_hash"] if user else None):
        raise HTTPException(401, "아이디 또는 비밀번호가 올바르지 않습니다.")
    if user["is_test_admin"] and not test_admin_enabled():
        raise HTTPException(401, "아이디 또는 비밀번호가 올바르지 않습니다.")
    return sign_in(request, response, db, user)


@app.get("/api/v1/auth/me")
def me(user: User):
    return public_user(user)


@app.post("/api/v1/auth/logout", status_code=204)
def logout(user: User, db: Db):
    db.execute("UPDATE auth_sessions SET revoked_at = now() WHERE id = %s",
               (user["session_id"],))
    webcams = db.execute("""
        UPDATE cameras SET deleted_at = now(), is_enabled = false
        WHERE source_type = 'webcam' AND webcam_session_id = %s AND deleted_at IS NULL
        RETURNING id
    """, (user["session_id"],)).fetchall()
    for camera in webcams:
        stop_analysis(db, camera["id"])
        frame_path(camera["id"], "webcam.jpg").unlink(missing_ok=True)
        frame_path(camera["id"], "analysis.json").unlink(missing_ok=True)
    response = Response(status_code=204)
    response.delete_cookie(SESSION_COOKIE, path="/api/v1")
    return response


def enqueue(db, camera_id, settings):
    if db.execute("""
        SELECT 1 FROM model_artifacts a JOIN cameras c ON c.id = %s
        WHERE a.role = 'behavior' AND a.name = %s
          AND (a.name IN ('behavior', 'behavior_kpt_v1', 'behavior_rgb') OR EXISTS (
              SELECT 1 FROM behavior_uploads u WHERE u.artifact_id = a.id AND u.owner_id = c.owner_id
          ))
    """, (camera_id, settings.behavior_model)).fetchone() is None:
        raise HTTPException(409, "선택한 행동 모델을 사용할 수 없습니다. 파일을 첨부하거나 기본 모델을 선택하세요.")
    if db.execute("SELECT 1 FROM model_artifacts WHERE role = 'pose' AND name = %s",
                  (settings.pose_model,)).fetchone() is None:
        raise HTTPException(409, "선택한 포즈 모델을 설치하고 등록하세요. python -m backend.manage register-models")
    return db.execute("""
        INSERT INTO analysis_jobs (camera_id, threshold, settings) VALUES (%s, %s, %s)
        RETURNING id, status
    """, (camera_id, settings.threshold, Jsonb(settings.model_dump(exclude={"threshold"})))).fetchone()


def registered_source(db, row, settings):
    enqueue(db, row["id"], settings)
    return {**row, "status": "queued"}


@app.post("/api/v1/cameras", status_code=201)
def create_camera(data: CameraInput, user: Owner, db: Db):
    try:
        row = db.execute("""
            INSERT INTO cameras (owner_id, name, source_type, source_storage_key)
            VALUES (%s, %s, 'sample', %s)
            RETURNING id, name, source_type, is_enabled, created_at
        """, (user["id"], data.name, data.sample)).fetchone()
        return registered_source(db, row, data.settings)
    except UniqueViolation:
        raise HTTPException(409, "A source with that name already exists") from None


def uploaded_camera(db, user_id, name, filename, key, settings):
    row = db.execute("""
        INSERT INTO cameras (owner_id, name, source_type, source_storage_key, original_filename)
        VALUES (%s, %s, 'file', %s, %s)
        RETURNING id, name, source_type, is_enabled, created_at
    """, (user_id, name, key, filename)).fetchone()
    row = registered_source(db, row, settings)
    db.commit()
    return row


@app.post("/api/v1/cameras/upload", status_code=201)
async def upload_camera(request: Request, user: Owner, db: Db,
                        name: ShortText, filename: Annotated[str, Query(min_length=5, max_length=255)],
                        resolution: Literal["384", "480", "640"] = "480",
                        detection_interval: Annotated[int, Query(ge=1, le=3)] = 1,
                        behavior_model: BehaviorModel = "behavior",
                        pose_model: PoseModel = "human_pose",
                        detection_confidence: Annotated[float, Query(ge=0.1, le=0.95)] = 0.4,
                        max_age: Annotated[int, Query(ge=1, le=120)] = 30,
                        threshold: Annotated[float, Query(ge=0.5, le=1)] = 0.5,
                        event_threshold: Annotated[float, Query(ge=0.5, le=1)] = 0.6):
    settings = AnalysisInput(resolution=int(resolution), detection_interval=detection_interval,
                             behavior_model=behavior_model, pose_model=pose_model, event_threshold=event_threshold,
                             detection_confidence=detection_confidence, max_age=max_age, threshold=threshold)
    filename = filename.replace(chr(92), "/").rsplit("/", 1)[-1]
    if not filename.lower().endswith(".mp4") or any(ord(char) < 32 for char in filename):
        raise HTTPException(415, "압축을 푼 MP4 영상 파일을 선택하세요.")
    content_length = request.headers.get("content-length")
    if content_length and content_length.isdecimal() and int(content_length) > MAX_VIDEO_BYTES:
        raise HTTPException(413, "영상은 최대 256 MiB까지 등록할 수 있습니다.")
    if UPLOAD_DIR.is_symlink():
        raise HTTPException(503, "영상 저장소를 사용할 수 없습니다.")
    UPLOAD_DIR.mkdir(mode=0o700, exist_ok=True)
    identifier = uuid4()
    temp = UPLOAD_DIR / f"{identifier}.part"
    path = UPLOAD_DIR / f"{identifier}.mp4"
    registered = False
    size = 0
    header = bytearray()
    try:
        async with await anyio.open_file(temp, "xb") as output:
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_VIDEO_BYTES:
                    raise HTTPException(413, "영상은 최대 256 MiB까지 등록할 수 있습니다.")
                if len(header) < 12:
                    header.extend(chunk[:12 - len(header)])
                if len(header) == 12 and header[4:8] != b"ftyp":
                    raise HTTPException(415, "MP4 영상 파일을 확인하세요.")
                await output.write(chunk)
        if len(header) < 12:
            raise HTTPException(415, "MP4 영상 파일을 확인하세요.")
        temp.replace(path)
        # 요청이 끊겨도 파일과 등록 결과가 일치하도록 DB 기록을 마무리한다.
        with anyio.CancelScope(shield=True):
            row = await anyio.to_thread.run_sync(
                uploaded_camera, db, user["id"], name, filename, f"uploads/{path.name}", settings
            )
            registered = True
        return row
    except UniqueViolation:
        raise HTTPException(409, "같은 이름으로 등록된 영상 소스가 있습니다.") from None
    finally:
        temp.unlink(missing_ok=True)
        if not registered:
            path.unlink(missing_ok=True)


@app.post("/api/v1/cameras/rtsp", status_code=201)
def create_rtsp_camera(data: RtspInput, user: Owner, db: Db):
    url = public_rtsp_url(data.url)
    resolve_camera(user["username"], url)
    try:
        row = db.execute("""
            INSERT INTO cameras (owner_id, name, source_type, source_uri)
            VALUES (%s, %s, 'rtsp', %s)
            RETURNING id, name, source_type, is_enabled, created_at
        """, (user["id"], data.name, url)).fetchone()
        return registered_source(db, row, data.settings)
    except UniqueViolation:
        raise HTTPException(409, "같은 이름으로 등록된 영상 소스가 있습니다.") from None


@app.post("/api/v1/cameras/webcam", status_code=201)
def create_webcam(data: WebcamInput, user: Owner, db: Db):
    try:
        row = db.execute("""
            INSERT INTO cameras (owner_id, name, source_type, webcam_session_id)
            VALUES (%s, %s, 'webcam', %s)
            RETURNING id, name, source_type, is_enabled, created_at
        """, (user["id"], data.name, user["session_id"])).fetchone()
        return registered_source(db, row, data.settings)
    except UniqueViolation:
        raise HTTPException(409, "같은 이름의 웹캠이 이미 있습니다.") from None


@app.put("/api/v1/cameras/{camera_id}/frame", status_code=204)
async def webcam_frame(camera_id: UUID, request: Request, user: Owner, db: Db):
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > FRAME_LIMIT:
            raise HTTPException(413, "웹캠 프레임이 너무 큽니다.")
        data.extend(chunk)
    if not data.startswith(b"\xff\xd8") or not data.endswith(b"\xff\xd9"):
        raise HTTPException(415, "웹캠 프레임은 JPEG 형식이어야 합니다.")

    def save():
        row = db.execute("""
            SELECT id FROM cameras WHERE id = %s AND owner_id = %s AND webcam_session_id = %s
              AND source_type = 'webcam' AND is_enabled AND deleted_at IS NULL FOR SHARE
        """, (camera_id, user["id"], user["session_id"])).fetchone()
        if row is None:
            raise HTTPException(404, "연결된 웹캠을 찾을 수 없습니다.")
        replace_frame(camera_id, "webcam.jpg", data)

    await anyio.to_thread.run_sync(save)
    return Response(status_code=204)


def stop_analysis(db, camera_id):
    db.execute("""
        UPDATE inference_runs SET status = 'stopped', ended_at = clock_timestamp()
        WHERE camera_id = %s AND status = ANY(%s)
    """, (camera_id, list(ACTIVE_RUNS)))
    db.execute("""
        UPDATE analysis_jobs SET status = 'stopped', claim_token = NULL, claim_expires_at = NULL
        WHERE camera_id = %s AND status IN ('queued', 'claimed')
    """, (camera_id,))


@app.delete("/api/v1/cameras/{camera_id}", status_code=204)
def remove_camera(camera_id: UUID, user: Owner, db: Db):
    row = db.execute("""
        UPDATE cameras SET deleted_at = now(), is_enabled = false
        WHERE id = %s AND owner_id = %s AND deleted_at IS NULL
        RETURNING source_type
    """, (camera_id, user["id"])).fetchone()
    if row is None:
        raise HTTPException(404, "영상 소스를 찾을 수 없습니다.")
    stop_analysis(db, camera_id)
    frame_path(camera_id, "webcam.jpg").unlink(missing_ok=True)
    frame_path(camera_id, "analysis.json").unlink(missing_ok=True)
    for run in db.execute("SELECT id FROM inference_runs WHERE camera_id = %s", (camera_id,)):
        frame_path(run["id"], "timeline.jsonl").unlink(missing_ok=True)
    return Response(status_code=204)


@app.get("/api/v1/cameras/{camera_id}/snapshot")
def snapshot(camera_id: UUID, user: User, db: Db):
    row = db.execute("""
        SELECT r.id FROM cameras c JOIN inference_runs r ON r.camera_id = c.id
        WHERE c.id = %s AND c.owner_id = %s AND c.deleted_at IS NULL
          AND (c.source_type <> 'webcam' OR c.webcam_session_id = %s)
        ORDER BY r.started_at DESC, r.id DESC LIMIT 1
    """, (camera_id, user["id"], user["session_id"])).fetchone()
    return read_snapshot(camera_id, row["id"]) if row else None


@app.get("/api/v1/cameras/{camera_id}/timeline")
def timeline(camera_id: UUID, run_id: UUID, user: User, db: Db,
             offset: Annotated[int, Query(ge=0, le=2**63 - 1)] = 0):
    row = db.execute("""
        SELECT r.id FROM inference_runs r JOIN cameras c ON c.id = r.camera_id
        WHERE r.id = %s AND c.id = %s AND c.owner_id = %s AND c.deleted_at IS NULL
          AND c.source_type IN ('file', 'sample')
    """, (run_id, camera_id, user["id"])).fetchone()
    if row is None:
        raise HTTPException(404, "분석 영상을 찾을 수 없습니다.")
    entries = []
    try:
        with frame_path(run_id, "timeline.jsonl").open("rb") as source:
            if offset:
                source.seek(offset - 1)
                if source.read(1) != b"\n":
                    raise HTTPException(416, "분석 영상 위치를 확인하세요.")
            end_offset = offset
            while source.tell() - offset < 512 * 1024:
                line = source.readline(1024 * 1024 + 1)
                if not line.endswith(b"\n"):
                    break
                entries.append(json.loads(line))
                end_offset = source.tell()
            return {"frames": entries, "offset": end_offset}
    except FileNotFoundError:
        return {"frames": [], "offset": offset}


@app.get("/api/v1/cameras/{camera_id}/live")
def camera_live(camera_id: UUID, user: Owner, db: Db,
                digest: Annotated[str, Depends(session_digest)]):
    row = db.execute("""
        SELECT source_uri FROM cameras WHERE id = %s AND owner_id = %s
          AND source_type = 'rtsp' AND is_enabled AND deleted_at IS NULL
    """, (camera_id, user["id"])).fetchone()
    if row is None:
        raise HTTPException(404, "IP 카메라를 찾을 수 없습니다.")
    url = resolve_camera(user["username"], row["source_uri"])

    def authorized():
        try:
            with connect() as connection:
                current = lookup_user(connection, digest)
            return current["role"] == "owner"
        except (HTTPException, Error, DatabaseConfigurationError):
            return False

    return StreamingResponse(frames(url, authorized),
                             media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
                             headers={"X-Accel-Buffering": "no"})


@app.get("/api/v1/cameras/{camera_id}/video")
def camera_video(camera_id: UUID, user: User, db: Db):
    row = db.execute("""
        SELECT source_storage_key FROM cameras WHERE id = %s AND owner_id = %s
          AND source_type IN ('sample', 'file') AND deleted_at IS NULL
    """, (camera_id, user["id"])).fetchone()
    if row is None:
        raise HTTPException(404, "영상을 찾을 수 없습니다.")
    try:
        path = source_path(row["source_storage_key"])
    except (ValueError, FileNotFoundError):
        raise HTTPException(404, "영상 파일을 찾을 수 없습니다.") from None
    return FileResponse(path, media_type="video/mp4")


@app.get("/api/v1/cameras")
def cameras(user: User, db: Db, limit: Annotated[int, Query(ge=1, le=100)] = 50):
    return db.execute("""
        SELECT c.id, c.name, c.source_type, c.is_enabled, c.created_at,
               CASE WHEN c.source_type = 'sample' THEN c.source_storage_key END AS sample, c.original_filename,
               CASE WHEN j.status = 'queued' THEN 'queued'
                    WHEN j.status = 'claimed' AND j.run_id IS NULL
                         AND j.claim_expires_at > now() THEN 'starting'
                    WHEN j.status = 'claimed' AND j.run_id IS NULL THEN 'failed'
                    WHEN j.status = 'failed' AND j.run_id IS NULL
                         AND (r.started_at IS NULL OR r.started_at <= j.requested_at) THEN 'failed'
                    WHEN r.status IN ('starting', 'running', 'stopping')
                         AND r.lease_expires_at <= now() THEN 'failed'
                    ELSE COALESCE(r.status, 'idle') END AS status,
               r.id AS run_id, r.behavior_model, r.pose_model, r.config_snapshot, r.error_code,
               r.behavior_filename, r.behavior_sha256,
               video.id AS media_id, COALESCE(r.event_count, 0) AS event_count
        FROM cameras c
        LEFT JOIN LATERAL (
            SELECT status, run_id, claim_expires_at, requested_at FROM analysis_jobs
            WHERE camera_id = c.id
            ORDER BY requested_at DESC, id DESC LIMIT 1
        ) j ON true
        LEFT JOIN LATERAL (
            SELECT ir.id, ir.status, ir.lease_expires_at, ir.started_at, ir.config_snapshot, ir.error_code,
                (SELECT ma.name FROM run_models rm JOIN model_artifacts ma ON ma.id = rm.model_artifact_id
                 WHERE rm.run_id = ir.id AND rm.role = 'pose') AS pose_model,
                a.name AS behavior_model, a.sha256 AS behavior_sha256,
                COALESCE(u.filename, regexp_replace(a.storage_key, '^.*/', '')) AS behavior_filename,
                (SELECT count(*) FROM detection_events WHERE run_id = ir.id AND deleted_at IS NULL) AS event_count
            FROM inference_runs ir
            LEFT JOIN run_models rm ON rm.run_id = ir.id AND rm.role = 'behavior'
            LEFT JOIN model_artifacts a ON a.id = rm.model_artifact_id
            LEFT JOIN behavior_uploads u ON u.artifact_id = a.id AND u.owner_id = c.owner_id
            WHERE ir.camera_id = c.id
            ORDER BY ir.started_at DESC, ir.id DESC LIMIT 1
        ) r ON true
        LEFT JOIN media_assets video ON video.run_id = r.id AND video.kind = 'source'
            AND video.status = 'ready' AND (video.expires_at IS NULL OR video.expires_at > now())
        WHERE c.owner_id = %s AND c.deleted_at IS NULL
          AND (c.source_type <> 'webcam' OR c.webcam_session_id = %s)
        ORDER BY c.created_at DESC, c.id DESC LIMIT %s
    """, (user["id"], user["session_id"], limit)).fetchall()


def expire_jobs(db, camera_id=None):
    db.execute("""
        UPDATE analysis_jobs j SET status = 'failed', claim_token = NULL, claim_expires_at = NULL
        WHERE j.status = 'claimed' AND (%s::uuid IS NULL OR j.camera_id = %s)
          AND ((j.run_id IS NULL AND j.claim_expires_at <= clock_timestamp()) OR EXISTS (
              SELECT 1 FROM inference_runs r WHERE r.id = j.run_id
                AND r.lease_expires_at <= clock_timestamp()
          ))
    """, (camera_id, camera_id))


@app.post("/api/v1/cameras/{camera_id}/analysis", status_code=202)
def queue_analysis(camera_id: UUID, data: AnalysisInput, user: Owner, db: Db):
    source = db.execute("""
        SELECT source_type, source_uri FROM cameras WHERE id = %s AND owner_id = %s
          AND (source_type <> 'webcam' OR webcam_session_id = %s)
          AND is_enabled AND deleted_at IS NULL FOR UPDATE
    """, (camera_id, user["id"], user["session_id"])).fetchone()
    if source is None:
        raise HTTPException(404, "영상 소스를 찾을 수 없습니다.")
    if source["source_type"] == "rtsp":
        resolve_camera(user["username"], source["source_uri"])
    stop_analysis(db, camera_id)
    return enqueue(db, camera_id, data)


@app.get("/api/v1/models")
def models(user: User, db: Db):
    return db.execute("""
        SELECT a.id, a.role, a.name, a.version, a.sha256, a.labels, a.preprocessing,
               COALESCE(u.filename, regexp_replace(a.storage_key, '^.*/', '')) AS filename,
               CASE WHEN left(a.name, 7) <> 'upload_' THEN a.storage_key END AS storage_key
        FROM model_artifacts a
        LEFT JOIN behavior_uploads u ON u.artifact_id = a.id AND u.owner_id = %s
        WHERE left(a.name, 7) <> 'upload_' OR u.owner_id IS NOT NULL
        ORDER BY a.role, a.created_at DESC
    """, (user["id"],)).fetchall()


def register_behavior_upload(db, owner_id, filename, temp, size, digest, architecture):
    from core.labels import BEHAVIOR_LABELS

    # 같은 해시를 동시에 첨부해도 모델을 한 번만 저장한다.
    db.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (digest,))
    artifact = db.execute("""
        SELECT id, role, name FROM model_artifacts WHERE sha256 = %s
    """, (digest,)).fetchone()
    path = MODEL_DIR / f"{digest}.pth"
    created_file = False
    try:
        if artifact is None:
            artifact = db.execute("""
                INSERT INTO model_artifacts (role, name, version, sha256, storage_key, size_bytes, labels, preprocessing)
                VALUES ('behavior', %s, 'upload', %s, %s, %s, %s, %s) RETURNING id, role, name
            """, (UPLOAD_PREFIX + digest, digest, path.name, size,
                  Jsonb({i: label[1].lower() for i, label in enumerate(BEHAVIOR_LABELS)}),
                  Jsonb({"architecture": architecture, "input_shape": [1, 30, 34]}))).fetchone()
        if artifact["role"] != "behavior":
            raise HTTPException(415, "이 파일은 행동 분류 가중치가 아닙니다.")
        if not artifact["name"].startswith(UPLOAD_PREFIX):
            return {"name": artifact["name"], "sha256": digest}
        if not path.exists():
            temp.replace(path)
            created_file = True
        db.execute("""
            INSERT INTO behavior_uploads (owner_id, artifact_id, filename) VALUES (%s, %s, %s)
            ON CONFLICT (owner_id, artifact_id) DO NOTHING
        """, (owner_id, artifact["id"], filename))
        db.commit()
    except Exception:
        if created_file:
            path.unlink(missing_ok=True)
        raise
    return {"name": artifact["name"], "sha256": digest}


@app.post("/api/v1/models/behavior/upload", status_code=201)
async def upload_behavior_model(request: Request, user: Owner, db: Db,
                                filename: Annotated[str, Query(min_length=5, max_length=255)]):
    if not filename.lower().endswith(".pth") or any(c in filename for c in "/\\") or any(ord(c) < 32 for c in filename):
        raise HTTPException(415, "행동 모델의 .pth 파일을 선택하세요.")
    rgb = bundled_rgb_artifact()
    content_length = request.headers.get("content-length", "")
    if content_length.isdecimal() and int(content_length) > MAX_MODEL_BYTES and int(content_length) != rgb["bytes"]:
        raise HTTPException(413, "새 키포인트 모델은 최대 8 MiB입니다. 장면 모델은 프로젝트의 model_final.pth를 선택하세요.")
    if MODEL_DIR.is_symlink():
        raise HTTPException(503, "모델 저장소를 사용할 수 없습니다.")
    MODEL_DIR.mkdir(mode=0o700, exist_ok=True)
    temp = MODEL_DIR / f"{uuid4()}.part"
    size = 0
    digest = hashlib.sha256()
    try:
        async with await anyio.open_file(temp, "xb") as output:
            async for chunk in request.stream():
                size += len(chunk)
                if size > rgb["bytes"]:
                    raise HTTPException(413, "행동 모델 파일이 너무 큽니다.")
                digest.update(chunk)
                if size <= MAX_MODEL_BYTES:
                    await output.write(chunk)
        if size > MAX_MODEL_BYTES:
            if size != rgb["bytes"] or digest.hexdigest() != rgb["sha256"]:
                raise HTTPException(415, "프로젝트에 등록된 model_final.pth와 일치하지 않습니다. 새로 첨부하는 가중치는 KeypointLSTM v1/v2만 지원합니다.")
            registered = db.execute("""
                SELECT 1 FROM model_artifacts WHERE role = 'behavior' AND name = %s AND sha256 = %s
            """, (rgb["id"], rgb["sha256"])).fetchone()
            if registered is None or not (ROOT / rgb["path"]).is_file():
                raise HTTPException(409, "서버에 장면 모델을 설치하고 python -m backend.manage register-models를 실행하세요.")
            return {"name": rgb["id"], "sha256": rgb["sha256"]}
        try:
            result = await anyio.to_thread.run_sync(lambda: subprocess.run(
                [sys.executable, "-m", "backend.behavior_upload", str(temp)], cwd=ROOT,
                capture_output=True, text=True, timeout=20, check=True,
            ), limiter=MODEL_VALIDATION_LIMITER)
        except (subprocess.SubprocessError, OSError):
            raise HTTPException(415, "원본 GUI와 호환되는 KeypointLSTM v1/v2의 8개 클래스 .pth 가중치를 선택하세요.") from None
        return await anyio.to_thread.run_sync(
            register_behavior_upload, db, user["id"], filename, temp, size, digest.hexdigest(), result.stdout.strip()
        )
    finally:
        temp.unlink(missing_ok=True)


@app.get("/api/v1/cameras/{camera_id}/runs")
def runs(camera_id: UUID, user: User, db: Db, limit: Annotated[int, Query(ge=1, le=100)] = 50):
    if db.execute("""
        SELECT id FROM cameras WHERE id = %s AND owner_id = %s AND deleted_at IS NULL
    """, (camera_id, user["id"])).fetchone() is None:
        raise HTTPException(404, "Source not found")
    return db.execute("""
        SELECT id, status, started_at, ended_at, config_snapshot FROM inference_runs
        WHERE camera_id = %s ORDER BY started_at DESC, id DESC LIMIT %s
    """, (camera_id, limit)).fetchall()


@app.get("/api/v1/events")
def events(user: User, db: Db, camera_id: UUID | None = None,
           before: AwareDatetime | None = None, before_id: UUID | None = None,
           method: Literal["model", "rule", "hybrid"] | None = None,
           limit: Annotated[int, Query(ge=1, le=100)] = 50):
    if (before is None) != (before_id is None):
        raise HTTPException(422, "Provide before and before_id together")
    return db.execute("""
        SELECT e.id, e.run_id, r.camera_id, c.name AS camera_name, e.event_type, e.track_id,
               e.detection_method, e.model_score, e.score_model_role, e.rule_code,
               e.rule_version, e.evidence, e.detected_at, e.occurred_at, e.source_time_ms,
               e.source_frame_index, e.source_end_time_ms, review.decision,
               (SELECT id FROM media_assets WHERE run_id = r.id
                AND kind = 'source' AND status = 'ready') AS media_id
        FROM detection_events e
        JOIN inference_runs r ON r.id = e.run_id JOIN cameras c ON c.id = r.camera_id
        LEFT JOIN LATERAL (
            SELECT decision FROM event_reviews WHERE event_id = e.id
            ORDER BY created_at DESC, id DESC LIMIT 1
        ) review ON true
        WHERE c.owner_id = %s AND e.deleted_at IS NULL
          AND (%s::uuid IS NULL OR c.id = %s)
          AND (%s::text IS NULL OR e.detection_method = %s)
          AND (%s::timestamptz IS NULL OR (e.detected_at, e.id) < (%s, %s))
        ORDER BY e.detected_at DESC, e.id DESC LIMIT %s
    """, (user["id"], camera_id, camera_id, method, method, before, before, before_id, limit)).fetchall()


@app.post("/api/v1/events/{event_id}/reviews", status_code=201)
def review(event_id: UUID, data: ReviewInput, user: Owner, db: Db):
    row = db.execute("""
        INSERT INTO event_reviews (event_id, reviewer_id, decision, note)
        SELECT e.id, c.owner_id, %s, %s FROM detection_events e
        JOIN inference_runs r ON r.id = e.run_id JOIN cameras c ON c.id = r.camera_id
        WHERE e.id = %s AND c.owner_id = %s AND e.deleted_at IS NULL
        RETURNING id, event_id, decision, note, created_at
    """, (data.decision, data.note, event_id, user["id"])).fetchone()
    if row is None:
        raise HTTPException(404, "Event not found")
    return row


@app.delete("/api/v1/events/{event_id}", status_code=204)
def delete_event(event_id: UUID, user: Owner, db: Db):
    row = db.execute('''
        UPDATE detection_events e SET deleted_at = COALESCE(e.deleted_at, now()), evidence = '{}'::jsonb
        FROM inference_runs r, cameras c
        WHERE e.id = %s AND e.run_id = r.id AND r.camera_id = c.id AND c.owner_id = %s
        RETURNING e.id
    ''', (event_id, user['id'])).fetchone()
    if row is None:
        raise HTTPException(404, "사건을 찾을 수 없습니다.")
    db.execute('''UPDATE notification_deliveries SET status = 'cancelled', lease_token = NULL, lease_expires_at = NULL
                  WHERE event_id = %s AND status IN ('queued', 'retry_wait', 'delivering')''', (event_id,))
    return Response(status_code=204)


@app.get("/api/v1/storage")
def storage(user: User, db: Db):
    return inventory(db, user['id'])


@app.delete("/api/v1/storage/{camera_id}")
def delete_storage(camera_id: UUID, user: Owner, db: Db):
    camera = db.execute('SELECT * FROM cameras WHERE id = %s AND owner_id = %s FOR UPDATE',
                        (camera_id, user['id'])).fetchone()
    if camera is None:
        raise HTTPException(404, "영상 소스를 찾을 수 없습니다.")
    stop_analysis(db, camera_id)
    shared = prepare_purge(db, camera)
    # 실패하더라도 재시도할 수 있도록 삭제 요청과 접근 차단을 파일 삭제보다 먼저 확정한다.
    db.commit()
    result = cleanup_camera(db, camera_id)
    return {**result, 'shared_video_kept': shared, 'sample_kept': camera['source_type'] == 'sample'}


@app.get("/api/v1/media/{media_id}")
def media(media_id: UUID, user: User, db: Db):
    row = db.execute("""
        SELECT m.storage_key FROM media_assets m
        JOIN inference_runs r ON r.id = m.run_id JOIN cameras c ON c.id = r.camera_id
        WHERE m.id = %s AND m.kind = 'source' AND m.status = 'ready'
          AND (m.expires_at IS NULL OR m.expires_at > now())
          AND c.owner_id = %s
    """, (media_id, user["id"])).fetchone()
    if row is None:
        raise HTTPException(404, "Video not found")
    try:
        path = asset_path(row["storage_key"])
    except (ValueError, FileNotFoundError):
        raise HTTPException(404, "Video not found") from None
    return FileResponse(path, media_type="video/mp4")


@app.get("/internal/v1/models")
def worker_models(authorized: Worker, db: Db):
    return db.execute("""
        SELECT id, role, name, sha256 FROM model_artifacts
        ORDER BY role, name
    """).fetchall()


@app.post("/internal/v1/jobs/claim")
def claim_job(authorized: Worker, db: Db, kind: Literal["file", "live"] = "file"):
    expire_jobs(db)
    return db.execute("""
        WITH candidate AS (
            SELECT j.id FROM analysis_jobs j JOIN cameras c ON c.id = j.camera_id
            WHERE j.status = 'queued' AND c.is_enabled AND c.deleted_at IS NULL
              AND (c.source_type IN ('webcam', 'rtsp')) = %s
              AND (c.source_type <> 'webcam' OR EXISTS (
                  SELECT 1 FROM auth_sessions s WHERE s.id = c.webcam_session_id
                    AND s.revoked_at IS NULL AND s.expires_at > now()
              ))
            ORDER BY j.requested_at, j.id LIMIT 1 FOR UPDATE OF j SKIP LOCKED
        )
        UPDATE analysis_jobs j SET status = 'claimed', claim_token = gen_random_uuid(),
            claim_expires_at = clock_timestamp() + interval '5 minutes'
        FROM candidate WHERE j.id = candidate.id
        RETURNING j.id, j.camera_id, j.threshold, j.claim_token, j.settings
    """, (kind == "live",)).fetchone()


@app.post("/internal/v1/jobs/{job_id}/fail", status_code=204)
def fail_job(job_id: UUID, data: JobClaim, authorized: Worker, db: Db):
    db.execute("""
        UPDATE analysis_jobs SET status = 'failed', claim_token = NULL, claim_expires_at = NULL
        WHERE id = %s AND claim_token = %s AND run_id IS NULL AND status = 'claimed'
    """, (job_id, data.claim_token))
    return Response(status_code=204)


@app.post("/internal/v1/runs", status_code=201)
def start_run(data: RunInput, authorized: Worker, db: Db):
    source = db.execute("""
        SELECT c.source_storage_key, c.source_type, c.source_uri, u.username
        FROM cameras c JOIN users u ON u.id = c.owner_id
        WHERE c.id = %s AND c.is_enabled AND c.deleted_at IS NULL FOR UPDATE OF c
    """, (data.camera_id,)).fetchone()
    if source is None:
        raise HTTPException(404, "영상 소스를 찾을 수 없습니다.")
    sample = None
    url = None
    if source["source_type"] == "rtsp":
        url = resolve_camera(source["username"], source["source_uri"])
    elif source["source_storage_key"]:
        try:
            sample = source_path(source["source_storage_key"])
        except (ValueError, FileNotFoundError):
            raise HTTPException(404, "영상 파일을 찾을 수 없습니다.") from None
    if data.job_id is not None:
        job = db.execute("""
            SELECT id FROM analysis_jobs WHERE id = %s AND camera_id = %s
              AND claim_token = %s AND status = 'claimed' AND run_id IS NULL
              AND threshold = %s AND claim_expires_at > clock_timestamp() FOR UPDATE
        """, (data.job_id, data.camera_id, data.claim_token, data.behavior_threshold)).fetchone()
        if job is None:
            raise HTTPException(409, "Analysis request is no longer active")
    elif db.execute("""
        SELECT id FROM analysis_jobs WHERE camera_id = %s AND status IN ('queued', 'claimed')
    """, (data.camera_id,)).fetchone():
        raise HTTPException(409, "This source already has a dashboard analysis request")
    if sample:
        with sample.open("rb") as media_file:
            digest = hashlib.file_digest(media_file, "sha256").hexdigest()
    db.execute("""
        UPDATE inference_runs SET status = 'failed', ended_at = clock_timestamp(),
            error_code = 'lease_expired'
        WHERE camera_id = %s AND status = ANY(%s) AND lease_expires_at <= clock_timestamp()
    """, (data.camera_id, list(ACTIVE_RUNS)))
    try:
        run = db.execute("""
            INSERT INTO inference_runs (camera_id, worker_id, status, lease_expires_at, config_snapshot)
            VALUES (%s, %s, 'running', now() + interval '60 seconds', %s)
            RETURNING id, lease_token, lease_expires_at
        """, (data.camera_id, data.worker_id, Jsonb(data.model_dump(
            mode="json", exclude={"camera_id", "worker_id", "models", "job_id", "claim_token"})))).fetchone()
        for role, model_id in data.models.items():
            db.execute("INSERT INTO run_models VALUES (%s, %s, %s)", (run["id"], role, model_id))
    except UniqueViolation:
        raise HTTPException(409, "An active run already exists for this source") from None
    except ForeignKeyViolation:
        raise HTTPException(422, "Invalid model registration or role") from None
    if sample:
        db.execute("""
            INSERT INTO media_assets (run_id, kind, storage_key, mime_type, size_bytes, sha256, status, ready_at)
            VALUES (%s, 'source', %s, 'video/mp4', %s, %s, 'ready', now())
        """, (run["id"], f"{'samples' if source['source_type'] == 'sample' else 'uploads'}/{run['id']}/{sample.name}", sample.stat().st_size, digest))
    if data.job_id is not None:
        db.execute("UPDATE analysis_jobs SET run_id = %s WHERE id = %s", (run["id"], data.job_id))
    return {**run, "source_type": source["source_type"],
            "source_storage_key": source["source_storage_key"], "rtsp_url": url}


def lock_run(db, run_id, lease_token, *, active=True):
    row = db.execute("""
        SELECT r.status, r.config_snapshot, c.owner_id, r.camera_id,
               r.lease_expires_at > clock_timestamp() AS lease_valid
        FROM inference_runs r JOIN cameras c ON c.id = r.camera_id
        WHERE r.id = %s AND r.lease_token = %s FOR UPDATE OF r
    """, (run_id, lease_token)).fetchone()
    if row is None:
        raise HTTPException(404, "Run or lease not found")
    if active and (row["status"] != "running" or not row["lease_valid"]):
        raise HTTPException(409, "Run lease is no longer active")
    return row


class SnapshotInput(Lease):
    snapshot: dict
    timeline: bool = False


@app.post("/internal/v1/runs/{run_id}/snapshot", status_code=204)
def save_snapshot(run_id: UUID, data: SnapshotInput, authorized: Worker, db: Db):
    run = lock_run(db, run_id, data.lease_token)
    payload = json.dumps({**data.snapshot, "run_id": str(run_id)}, allow_nan=False).encode()
    if len(payload) > 1024 * 1024:
        raise HTTPException(413, "분석 미리보기가 너무 큽니다.")
    replace_frame(run["camera_id"], "analysis.json", payload)
    if data.timeline:
        with frame_path(run_id, "timeline.jsonl").open("ab") as output:
            output.write(payload + b"\n")
    return Response(status_code=204)


@app.post("/internal/v1/runs/{run_id}/heartbeat")
def heartbeat(run_id: UUID, data: Lease, authorized: Worker, db: Db):
    lock_run(db, run_id, data.lease_token)
    return db.execute("""
        UPDATE inference_runs SET heartbeat_at = clock_timestamp(),
            lease_expires_at = clock_timestamp() + interval '60 seconds'
        WHERE id = %s RETURNING lease_expires_at
    """, (run_id,)).fetchone()


@app.post("/internal/v1/runs/{run_id}/finish", status_code=204)
def finish(run_id: UUID, data: RunFinish, authorized: Worker, db: Db):
    run = lock_run(db, run_id, data.lease_token, active=False)
    if run["status"] != data.status:
        if run["status"] != "running" or not run["lease_valid"]:
            raise HTTPException(409, "Run lease is no longer active")
        db.execute("UPDATE inference_runs SET status = %s, error_code = %s, ended_at = clock_timestamp() WHERE id = %s",
                   (data.status, data.error_code if data.status == "failed" else None, run_id))
        db.execute("""
            UPDATE analysis_jobs SET status = %s, claim_token = NULL, claim_expires_at = NULL
            WHERE run_id = %s AND status = 'claimed'
        """, (data.status, run_id))
    return Response(status_code=204)


@app.post("/internal/v1/runs/{run_id}/events")
def ingest(run_id: UUID, data: EventInput, authorized: Worker, db: Db, response: Response):
    run = lock_run(db, run_id, data.lease_token, active=False)
    payload = data.model_dump(mode="json", exclude={"lease_token"})
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    existing = db.execute("""
        SELECT id, payload_sha256 FROM detection_events
        WHERE run_id = %s AND producer_event_key = %s
    """, (run_id, data.producer_event_key)).fetchone()
    if existing is not None:
        if existing["payload_sha256"] != digest:
            raise HTTPException(409, "Event key was used for a different payload")
        return {"id": existing["id"]}
    if run["status"] != "running" or not run["lease_valid"]:
        raise HTTPException(409, "Run lease is no longer active")
    if data.detection_method != "model" and data.rule_version != run["config_snapshot"]["rule_version"]:
        raise HTTPException(422, "Rule version differs from the run configuration")
    try:
        event = db.execute("""
            INSERT INTO detection_events (
                run_id, producer_event_key, payload_sha256, event_type, track_id,
                detection_method, model_score, score_model_role, rule_code, rule_version,
                detected_at, occurred_at, source_time_ms, source_frame_index, source_end_time_ms, evidence
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (run_id, data.producer_event_key, digest, data.event_type, data.track_id,
              data.detection_method, data.model_score, data.score_model_role,
              data.rule_code, data.rule_version, data.detected_at, data.occurred_at,
              data.source_time_ms, data.source_frame_index, data.source_end_time_ms,
              Jsonb(payload["evidence"] or {}))).fetchone()
    except ForeignKeyViolation:
        raise HTTPException(422, "The scoring model is not registered on this run") from None
    if data.detection_method != "model" or data.model_score >= run["config_snapshot"].get("alert_threshold", 0.9):
        db.execute("""
            INSERT INTO notification_deliveries
                (event_id, recipient_user_id, oauth_account_id, channel)
            SELECT %s, user_id, id, provider FROM oauth_accounts
            WHERE user_id = %s AND is_enabled AND provider = 'kakao'
            ORDER BY created_at, id LIMIT 1
        """, (event["id"], run["owner_id"]))
    response.status_code = 201
    return event
