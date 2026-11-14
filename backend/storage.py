"""사용자 영상의 용량 조회와 커밋된 파일 삭제 요청을 처리한다."""

import re
from uuid import UUID

from backend.samples import UPLOAD_DIR

_NAME = re.compile(r'([0-9a-f-]{36})\.(mp4|analysis\.json|webcam\.jpg|timeline\.jsonl)')


def managed_path(filename):
    """서버가 만든 UUID 파일만 허용하고 저장소 밖 경로·심볼릭 링크를 거부한다."""
    match = _NAME.fullmatch(filename)
    if match is None or str(UUID(match[1])) != match[1]:
        raise ValueError('허용되지 않은 저장 파일 이름입니다.')
    path = UPLOAD_DIR / filename
    if UPLOAD_DIR.is_symlink() or path.is_symlink():
        raise ValueError('심볼릭 링크 파일은 정리할 수 없습니다.')
    return path


def source_filename(camera):
    key = camera.get('source_storage_key') or ''
    if camera['source_type'] != 'file' or not key.startswith('uploads/'):
        return None
    filename = key.removeprefix('uploads/')
    if not filename.endswith('.mp4'):
        raise ValueError('허용되지 않은 영상 저장 경로입니다.')
    managed_path(filename)
    return filename


def file_names(camera, run_ids):
    names = {f"{camera['id']}.analysis.json", f"{camera['id']}.webcam.jpg"}
    names.update(f'{run_id}.timeline.jsonl' for run_id in run_ids)
    original = source_filename(camera)
    if original:
        names.add(original)
    return names


def file_size(filename):
    try:
        return managed_path(filename).stat().st_size
    except (OSError, ValueError):
        return 0


def inventory(db, owner_id):
    cameras = db.execute('''
        SELECT c.id, c.name, c.original_filename, c.source_type, c.source_storage_key,
               c.deleted_at, c.purge_requested_at, c.created_at,
               ARRAY(SELECT id FROM inference_runs WHERE camera_id = c.id) AS run_ids,
               (SELECT count(*) FROM detection_events e JOIN inference_runs r ON r.id = e.run_id
                WHERE r.camera_id = c.id AND e.deleted_at IS NULL) AS event_count,
               (SELECT count(*) FROM storage_deletions d WHERE d.camera_id = c.id
                AND d.status = 'pending') AS pending_files
        FROM cameras c WHERE c.owner_id = %s AND (c.purge_requested_at IS NULL OR EXISTS (
            SELECT 1 FROM storage_deletions d WHERE d.camera_id = c.id AND d.status = 'pending'))
        ORDER BY c.created_at DESC, c.id DESC
    ''', (owner_id,)).fetchall()
    all_files = set()
    rows = []
    for camera in cameras:
        names = file_names(camera, camera['run_ids'])
        all_files.update(names)
        original = source_filename(camera)
        rows.append({key: camera[key] for key in (
            'id', 'name', 'original_filename', 'source_type', 'created_at', 'event_count', 'pending_files')})
        rows[-1].update(size_bytes=sum(file_size(name) for name in names),
                        video_bytes=file_size(original) if original else 0,
                        removed=camera['deleted_at'] is not None,
                        cleanup_pending=camera['purge_requested_at'] is not None)
    return {'sources': rows, 'total_bytes': sum(file_size(name) for name in all_files)}


def prepare_purge(db, camera):
    """카메라 잠금·분석 중단 후 삭제 의도를 기록한다. 여기서는 파일을 지우지 않는다."""
    run_ids = [row['id'] for row in db.execute('SELECT id FROM inference_runs WHERE camera_id = %s', (camera['id'],))]
    names = file_names(camera, run_ids)
    original = source_filename(camera)
    shared = False
    if original:
        # 같은 원본을 참조하는 소스의 동시 삭제가 파일을 남기지 않게 직렬화한다.
        db.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))', (camera['source_storage_key'],))
        shared = db.execute('''SELECT 1 FROM cameras WHERE id <> %s AND source_storage_key = %s
                               AND purge_requested_at IS NULL LIMIT 1''',
                            (camera['id'], camera['source_storage_key'])).fetchone() is not None
        if shared:
            names.discard(original)
    db.execute('''UPDATE cameras SET deleted_at = COALESCE(deleted_at, now()), is_enabled = false,
                  purge_requested_at = COALESCE(purge_requested_at, now()) WHERE id = %s''', (camera['id'],))
    db.execute('''UPDATE detection_events SET deleted_at = COALESCE(deleted_at, now()), evidence = '{}'::jsonb
                  WHERE run_id = ANY(%s)''', (run_ids,))
    db.execute('''UPDATE notification_deliveries SET status = 'cancelled', lease_token = NULL, lease_expires_at = NULL
                  WHERE event_id IN (SELECT id FROM detection_events WHERE run_id = ANY(%s))
                    AND status IN ('queued', 'retry_wait', 'delivering')''', (run_ids,))
    db.execute("UPDATE media_assets SET status = 'deleting' WHERE run_id = ANY(%s) AND status <> 'deleted'", (run_ids,))
    for filename in sorted(names):
        db.execute('''INSERT INTO storage_deletions(camera_id, filename) VALUES (%s, %s)
                      ON CONFLICT (camera_id, filename) DO NOTHING''', (camera['id'], filename))
    return shared


def cleanup_camera(db, camera_id):
    """커밋된 요청만 호출한다. Windows의 열린 파일은 보류하고 다음 요청에서 재시도한다."""
    freed = 0
    jobs = db.execute('''SELECT id, filename FROM storage_deletions
                        WHERE camera_id = %s AND status = 'pending' ORDER BY filename FOR UPDATE''',
                      (camera_id,)).fetchall()
    for job in jobs:
        try:
            path = managed_path(job['filename'])
            size = path.stat().st_size if path.exists() else 0
            path.unlink(missing_ok=True)
        except (OSError, ValueError):
            db.execute("""UPDATE storage_deletions SET attempts = attempts + 1, error_code = 'file_unavailable'
                          WHERE id = %s""", (job['id'],))
        else:
            freed += size
            db.execute("""UPDATE storage_deletions SET status = 'deleted', deleted_at = now(),
                          attempts = attempts + 1, freed_bytes = %s, error_code = NULL WHERE id = %s""",
                       (size, job['id']))
    pending = db.execute("SELECT count(*) AS n FROM storage_deletions WHERE camera_id = %s AND status = 'pending'",
                         (camera_id,)).fetchone()['n']
    if not pending:
        db.execute("""UPDATE media_assets SET status = 'deleted', deleted_at = now()
                      WHERE run_id IN (SELECT id FROM inference_runs WHERE camera_id = %s)
                      AND status = 'deleting'""", (camera_id,))
    return {'freed_bytes': freed, 'pending_files': pending}


def cleanup_pending():
    """관리 명령에서 실패·중단됐던 파일 정리를 다시 실행한다."""
    from backend.db import connect
    with connect() as db:
        ids = [row['camera_id'] for row in db.execute("SELECT DISTINCT camera_id FROM storage_deletions WHERE status = 'pending'")]
    for camera_id in ids:
        with connect() as db:
            cleanup_camera(db, camera_id)
