"""독립 PostgreSQL 스키마와 임시 파일로 삭제 권한·공유 영상·실패 복구를 검증한다."""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from fastapi.testclient import TestClient

from backend.api import app, database
from backend.storage import managed_path, prepare_purge


@unittest.skipUnless(os.environ.get('CCTV_TEST_DATABASE_URL'), '별도 테스트 DB 연결이 필요합니다.')
class StorageCleanupTests(unittest.TestCase):
    def setUp(self):
        self.dsn = os.environ['CCTV_TEST_DATABASE_URL']
        self.schema = 'test_storage_' + uuid4().hex
        with psycopg.connect(self.dsn) as db:
            db.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(self.schema)))
        self.addCleanup(self.drop_schema)
        with self.db() as db:
            for migration in sorted(Path('backend/migrations').glob('*.sql')):
                db.execute(migration.read_text())
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.folder = Path(self.directory.name)
        for name in ('backend.storage.UPLOAD_DIR', 'backend.samples.UPLOAD_DIR'):
            guard = patch(name, self.folder)
            guard.start()
            self.addCleanup(guard.stop)
        token = patch.dict(os.environ, CCTV_WORKER_TOKEN='worker-token-for-storage-tests-only')
        token.start()
        self.addCleanup(token.stop)

        def connection():
            with self.db() as db:
                yield db
        app.dependency_overrides[database] = connection
        self.addCleanup(app.dependency_overrides.pop, database)
        self.client = TestClient(app, raise_server_exceptions=False)
        self.addCleanup(self.client.close)
        self.client.headers['X-CCTV-Request'] = '1'
        self.owner, self.token = self.user('owner')
        self.client.cookies.set('cctv_session', self.token)

    @contextmanager
    def db(self):
        with psycopg.connect(self.dsn, row_factory=dict_row) as db:
            db.execute(sql.SQL('SET search_path TO {}, public').format(sql.Identifier(self.schema)))
            yield db

    def drop_schema(self):
        with psycopg.connect(self.dsn) as db:
            db.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(self.schema)))

    def user(self, role):
        token = uuid4().hex
        with self.db() as db:
            user = db.execute('INSERT INTO users(display_name, role) VALUES (%s, %s) RETURNING id', ('삭제 검증 계정', role)).fetchone()['id']
            db.execute("INSERT INTO auth_sessions(user_id, token_hash, expires_at) VALUES (%s, %s, now() + interval '1 hour')",
                       (user, hashlib.sha256(token.encode()).hexdigest()))
        return user, token

    def source(self, filename=None, sample=False):
        filename = filename or f'{uuid4()}.mp4'
        if not sample:
            (self.folder / filename).write_bytes(b'original-video' * 20)
        with self.db() as db:
            camera = db.execute('''INSERT INTO cameras(owner_id, name, source_type, source_storage_key)
                                   VALUES (%s, %s, %s, %s) RETURNING id''',
                                (self.owner, '삭제 검증 ' + uuid4().hex, 'sample' if sample else 'file',
                                 'demo-1.mp4' if sample else 'uploads/' + filename)).fetchone()['id']
            run = db.execute('''INSERT INTO inference_runs(camera_id, worker_id, status, lease_expires_at, config_snapshot)
                                VALUES (%s, 'test', 'running', now() + interval '1 hour',
                                        '{"rule_version":"stand-pos-v1"}') RETURNING id, lease_token''', (camera,)).fetchone()
            media = db.execute('''INSERT INTO media_assets(run_id, kind, storage_key, mime_type, size_bytes, sha256, status, ready_at)
                                  VALUES (%s, 'source', %s, 'video/mp4', 280, %s, 'ready', now()) RETURNING id''',
                               (run['id'], f"samples/{run['id']}/demo-1.mp4" if sample else f"uploads/{run['id']}/{filename}", 'a' * 64)).fetchone()['id']
        (self.folder / f'{camera}.analysis.json').write_bytes(b'{}')
        (self.folder / f"{run['id']}.timeline.jsonl").write_bytes(b'{}\n')
        return {'camera': str(camera), 'run': str(run['id']), 'lease': str(run['lease_token']), 'media': str(media), 'filename': filename}

    def event(self, source):
        body = {'lease_token': source['lease'], 'producer_event_key': uuid4().hex,
                'event_type': 'theft_suspected', 'detection_method': 'rule', 'rule_code': 'stand_pos',
                'rule_version': 'stand-pos-v1', 'detected_at': datetime.now(timezone.utc).isoformat(),
                'evidence': {'reason': 'pos_timeout'}}
        response = self.client.post('/internal/v1/runs/' + source['run'] + '/events', json=body,
                                    headers={'Authorization': 'Bearer worker-token-for-storage-tests-only'})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()['id'], body

    def test_event_delete_keeps_video_and_retry_cannot_restore_event(self):
        source = self.source()
        first, body = self.event(source)
        second, _ = self.event(source)
        self.assertEqual(self.client.post('/api/v1/events/' + first + '/reviews', json={'decision': 'false_positive'}).status_code, 201)
        for _ in range(2):
            self.assertEqual(self.client.delete('/api/v1/events/' + first).status_code, 204)
        self.assertEqual([e['id'] for e in self.client.get('/api/v1/events').json()], [second])
        self.assertEqual(self.client.get('/api/v1/cameras').json()[0]['event_count'], 1)
        self.assertEqual(self.client.get('/api/v1/media/' + source['media']).status_code, 200)
        self.assertTrue((self.folder / source['filename']).is_file())
        self.assertEqual(self.client.post('/api/v1/events/' + first + '/reviews', json={'decision': 'confirmed'}).status_code, 404)
        response = self.client.post('/internal/v1/runs/' + source['run'] + '/events', json=body,
                                    headers={'Authorization': 'Bearer worker-token-for-storage-tests-only'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self.client.get('/api/v1/events').json()), 1)

    def test_purge_frees_real_files_and_stops_queued_and_running_analysis(self):
        source = self.source()
        self.event(source)
        with self.db() as db:
            db.execute('INSERT INTO analysis_jobs(camera_id, threshold) VALUES (%s, .6)', (source['camera'],))
        before = sum(p.stat().st_size for p in self.folder.iterdir())
        result = self.client.delete('/api/v1/storage/' + source['camera'])
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()['freed_bytes'], before)
        self.assertEqual(list(self.folder.iterdir()), [])
        self.assertEqual(self.client.get('/api/v1/events').json(), [])
        self.assertEqual(self.client.get('/api/v1/storage').json()['total_bytes'], 0)
        self.assertEqual(self.client.get('/api/v1/media/' + source['media']).status_code, 404)
        self.assertEqual(self.client.post('/api/v1/cameras/' + source['camera'] + '/analysis', json={}).status_code, 404)
        blocked = self.client.post('/internal/v1/runs/' + source['run'] + '/snapshot',
                                   json={'lease_token': source['lease'], 'snapshot': {}},
                                   headers={'Authorization': 'Bearer worker-token-for-storage-tests-only'})
        self.assertEqual(blocked.status_code, 409)
        with self.db() as db:
            self.assertEqual(db.execute('SELECT status FROM analysis_jobs').fetchone()['status'], 'stopped')
        self.assertEqual(self.client.delete('/api/v1/storage/' + source['camera']).json()['freed_bytes'], 0)

    def test_permissions_and_csrf_protect_events_and_files(self):
        source = self.source()
        event, _ = self.event(source)
        routes = ['/api/v1/storage/' + source['camera'], '/api/v1/events/' + event]
        _, stranger = self.user('owner')
        self.client.cookies.set('cctv_session', stranger)
        for route in routes:
            self.assertEqual(self.client.delete(route).status_code, 404)
        self.assertEqual(self.client.get('/api/v1/storage').json()['sources'], [])
        _, viewer = self.user('demo_viewer')
        self.client.cookies.set('cctv_session', viewer)
        for route in routes:
            self.assertEqual(self.client.delete(route).status_code, 403)
        self.client.cookies.set('cctv_session', self.token)
        del self.client.headers['X-CCTV-Request']
        self.assertEqual(self.client.delete(routes[0]).status_code, 403)
        self.client.headers['X-CCTV-Request'] = '1'
        self.client.cookies.clear()
        self.assertEqual(self.client.delete(routes[0]).status_code, 401)
        self.assertTrue((self.folder / source['filename']).exists())

    def test_shared_original_is_counted_once_and_kept_until_last_source(self):
        first = self.source()
        second = self.source(filename=first['filename'])
        actual_bytes = sum(p.stat().st_size for p in self.folder.iterdir())
        self.assertEqual(self.client.get('/api/v1/storage').json()['total_bytes'], actual_bytes)
        result = self.client.delete('/api/v1/storage/' + first['camera']).json()
        self.assertTrue(result['shared_video_kept'])
        self.assertTrue((self.folder / second['filename']).exists())
        self.assertEqual(self.client.get('/api/v1/media/' + second['media']).status_code, 200)
        self.client.delete('/api/v1/storage/' + second['camera'])
        self.assertFalse((self.folder / second['filename']).exists())

    def test_file_in_use_is_retried_without_restoring_access(self):
        source = self.source()
        real_unlink = Path.unlink
        def fail_video(path, *args, **kwargs):
            if path.name == source['filename']:
                raise PermissionError('사용 중')
            return real_unlink(path, *args, **kwargs)
        with patch.object(Path, 'unlink', fail_video):
            result = self.client.delete('/api/v1/storage/' + source['camera'])
        self.assertEqual(result.json()['pending_files'], 1)
        self.assertEqual(self.client.get('/api/v1/media/' + source['media']).status_code, 404)
        self.assertTrue(self.client.get('/api/v1/storage').json()['sources'][0]['cleanup_pending'])
        self.assertEqual(self.client.delete('/api/v1/storage/' + source['camera']).json()['pending_files'], 0)
        self.assertFalse((self.folder / source['filename']).exists())

    def test_sample_and_unrelated_original_are_not_deleted(self):
        sample = self.folder / 'demo-1.mp4'
        sample.write_bytes(b'protected-sample')
        unrelated = self.folder / 'dataset-original.mp4'
        unrelated.write_bytes(b'protected-dataset')
        source = self.source(sample=True)
        result = self.client.delete('/api/v1/storage/' + source['camera'])
        self.assertEqual(result.status_code, 200, result.text)
        self.assertTrue(result.json()['sample_kept'])
        self.assertEqual(sample.read_bytes(), b'protected-sample')
        self.assertEqual(unrelated.read_bytes(), b'protected-dataset')

    def test_transaction_failure_cannot_delete_file(self):
        source = self.source()
        def fail_after_prepare(db, camera):
            prepare_purge(db, camera)
            raise RuntimeError('커밋 전 실패')
        with patch('backend.api.prepare_purge', fail_after_prepare):
            self.assertEqual(self.client.delete('/api/v1/storage/' + source['camera']).status_code, 500)
        self.assertTrue((self.folder / source['filename']).exists())
        with self.db() as db:
            self.assertIsNone(db.execute('SELECT purge_requested_at FROM cameras').fetchone()['purge_requested_at'])
            self.assertEqual(db.execute('SELECT count(*) AS n FROM storage_deletions').fetchone()['n'], 0)
            self.assertEqual(db.execute('SELECT status FROM inference_runs').fetchone()['status'], 'running')


class StoragePathTests(unittest.TestCase):
    def test_only_owned_storage_file_formats_are_accepted(self):
        for name in ('../demo-1.mp4', 'demo-1.mp4', '/tmp/video.mp4', f'{uuid4()}.pth'):
            with self.assertRaises(ValueError):
                managed_path(name)
        with tempfile.TemporaryDirectory() as directory, patch('backend.storage.UPLOAD_DIR', Path(directory)):
            target = Path(directory) / 'original'
            target.write_bytes(b'original')
            link = Path(directory) / f'{uuid4()}.mp4'
            try:
                link.symlink_to(target)
            except OSError:
                return  # Windows에서 심볼릭 링크 생성 권한이 없는 경우 경로 검사만 완료한다.
            with self.assertRaises(ValueError):
                managed_path(link.name)
            self.assertEqual(target.read_bytes(), b'original')
