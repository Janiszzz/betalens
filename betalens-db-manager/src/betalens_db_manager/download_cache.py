"""Retire downloaded payloads without deleting typed data or rollback evidence."""
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import json
import os

from .util import digest, now


TERMINAL = {'completed', 'rolled_back'}


@contextmanager
def cache_lock(root):
    """Serialize deletion with queue admission and archive reuse."""
    with (Path(root) / 'download-cache.lock').open('a+') as handle:
        if os.name == 'nt':
            import msvcrt
            handle.write('0'); handle.flush(); handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def cache_guard(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        service = getattr(self, 's', self)
        state = getattr(service, 'state', service)
        with cache_lock(state.root):
            return method(self, *args, **kwargs)
    return guarded


def cleanup_download_cache(service, *, job=None, dry_run=False):
    """Clean terminal task families; shared payloads stay until all users finish.

    Only task directories directly below the configured archive directory are
    visited. SQLite plans, slice hashes, reference audits, typed snapshots and
    before/after journals remain available. Interrupted/partial jobs and rejected
    cohorts retain their inputs for recovery and human review.
    """
    with cache_lock(service.state.root):
        return _cleanup(service, job, dry_run)


def _cleanup(service, job, dry_run):
    state = service.state
    archive = Path(service.settings.archive_dir or service.settings.state_dir / 'archive').resolve()
    summary = {'dry_run': dry_run, 'tasks': 0, 'files': 0, 'bytes': 0, 'disk_bytes': 0,
               'retained_files': 0, 'errors': [], 'results': []}
    if not archive.is_dir():
        return summary
    with state.connect() as conn:
        jobs = {r['id']: dict(r) for r in conn.execute("""SELECT id,status,
            json_extract(plan,'$.pipeline_parent') AS parent FROM jobs""")}

        def root_of(identity):
            seen = set()
            while identity in jobs and jobs[identity]['parent']:
                if identity in seen:
                    return None
                seen.add(identity)
                identity = jobs[identity]['parent']
            return identity if identity in jobs else None

        roots = {identity: root_of(identity) for identity in jobs}
        blocked = {roots.get(r[0]) for r in conn.execute(
            "SELECT DISTINCT job FROM tasks WHERE status IN ('queued','running')")}
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'pipeline_units' in tables:
            blocked.update(roots.get(r[0]) for r in conn.execute(
                "SELECT DISTINCT job FROM pipeline_units WHERE kind='normalize' AND status='failed'"))
        eligible = {identity for identity, r in jobs.items()
                    if not r['parent'] and r['status'] in TERMINAL and identity not in blocked}
        selected = eligible & {job} if job else eligible
        directories = {}
        for identity, root in roots.items():
            if root not in selected or Path(identity).name != identity or identity in {'.', '..'}:
                continue
            directory = archive / identity
            if directory.is_dir() and not directory.is_symlink():
                directories[identity] = root
        if not directories:
            return summary
        def signature(current):
            return digest([sorted(eligible), sorted(blocked, key=str), sorted(current), job])
        if not dry_run and getattr(service, '_download_cache_signature', None) == signature(directories):
            return summary

        # A child can still be marked downloaded even after its parent completed.
        # Classify by the root lifecycle, never by the child's own status.
        conn.execute('CREATE TEMP TABLE cache_finished_jobs(id TEXT PRIMARY KEY)')
        conn.executemany('INSERT INTO cache_finished_jobs VALUES(?)',
                         ((identity,) for identity, root in roots.items() if root in eligible))
        protected = set()

        def protect(value):
            if not value:
                return
            path = Path(value)
            if not path.is_absolute():
                path = state.root / path
            path = path.resolve()
            try:
                relative = path.relative_to(archive)
            except ValueError:
                return
            if relative.parts and relative.parts[0] in directories:
                protected.add(path)
                protected.add(path.with_suffix('.meta.json'))
                protected.add(path.with_suffix('.tmp'))
                if path.name.endswith('.json.gz'):
                    protected.add(path.with_name(path.name[:-8] + '.response.json.gz'))

        # One bounded SQL scan rather than a full slice scan for each file.
        for row in conn.execute("""SELECT DISTINCT archive FROM slices s
            LEFT JOIN cache_finished_jobs f ON f.id=s.job
            WHERE f.id IS NULL AND archive IS NOT NULL"""):
            protect(row[0])
        for row in conn.execute("""SELECT value FROM ledger l
            LEFT JOIN cache_finished_jobs f ON f.id=json_extract(l.value,'$.task_id')
            WHERE l.table_name='file_ingestion' AND f.id IS NULL"""):
            protect(json.loads(row[0]).get('source_path'))

    results = {}
    if not dry_run:
        for root in set(directories.values()):
            previous = state.ledger_get('download_cache_cleanup', root) or {}
            state.ledger_put('download_cache_cleanup', root,
                             {**previous, 'task_id': root, 'status': 'cleaning', 'checked_at': now()})
    for identity, root in directories.items():
        result = results.setdefault(root, {'task_id': root, 'files': 0, 'bytes': 0, 'disk_bytes': 0,
                                          'retained_files': 0, 'errors': []})
        directory = archive / identity
        def walk_error(exc):
            if len(result['errors']) < 20:
                result['errors'].append({'path': str(exc.filename), 'error': str(exc)})
        # Do not follow links even when a file/directory resolves within archive.
        for current, dirs, files in os.walk(directory, followlinks=False, onerror=walk_error):
            result['retained_files'] += sum((Path(current) / d).is_symlink() for d in dirs)
            dirs[:] = [d for d in dirs if not (Path(current) / d).is_symlink()]
            for name in files:
                path = Path(current) / name
                if path in protected or path.is_symlink():
                    result['retained_files'] += 1
                    continue
                try:
                    stat = path.stat()
                    if not dry_run:
                        path.unlink()
                    result['files'] += 1
                    result['bytes'] += stat.st_size
                    result['disk_bytes'] += getattr(stat, 'st_blocks', 0) * 512 if hasattr(stat, 'st_blocks') else stat.st_size
                except FileNotFoundError:
                    pass  # A previous interrupted sweep may already have removed it.
                except OSError as exc:
                    if len(result['errors']) < 20:
                        result['errors'].append({'path': str(path), 'error': str(exc)})
        if not dry_run:
            for current, _, _ in os.walk(directory, topdown=False, followlinks=False):
                try:
                    Path(current).rmdir()
                except OSError:
                    pass

    for root, result in results.items():
        summary['tasks'] += 1
        for key in ('files', 'bytes', 'disk_bytes', 'retained_files'):
            summary[key] += result[key]
        summary['errors'].extend(result['errors'])
        summary['results'].append(result)
        if not dry_run:
            previous = state.ledger_get('download_cache_cleanup', root) or {}
            record = {**result, 'checked_at': now(),
                      'status': 'failed' if result['errors'] else 'shared' if result['retained_files'] else 'cleaned',
                      'files': previous.get('files', 0) + result['files'],
                      'bytes': previous.get('bytes', 0) + result['bytes'],
                      'disk_bytes': previous.get('disk_bytes', 0) + result['disk_bytes']}
            state.ledger_put('download_cache_cleanup', root, record)
            if result['files'] or result['errors']:
                state.log(root, 'download_cache_cleanup', **{k: v for k, v in result.items() if k != 'task_id'})
    if not dry_run and not summary['errors']:
        service._download_cache_signature = signature(
            [identity for identity in directories if (archive / identity).is_dir()])
    return summary
