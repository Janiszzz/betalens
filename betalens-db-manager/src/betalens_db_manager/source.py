from __future__ import annotations

import gzip
import hashlib
import json
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from decimal import Decimal
from pathlib import Path

import requests

from .config import Settings
from .state import State
from .util import atomic_json, now


class SourceError(RuntimeError):
    pass


class TushareSource:
    url = "https://api.tushare.pro"

    def __init__(self, settings: Settings, state: State):
        self.settings, self.state = settings, state

    def _wait(self, seconds):
        deadline = time.monotonic() + max(0, seconds)
        while True:
            job = getattr(self, 'active_job', None)
            if job: self.state.check_cancel(job)
            remaining = deadline - time.monotonic()
            if remaining <= 0: return
            time.sleep(min(remaining, .2))

    def _rate(self, endpoint=''):
        from .retrieval import request_rate, tushare_policy
        rpm = request_rate(getattr(self, 'requests_per_minute', self.settings.requests_per_minute))
        policies = getattr(self, 'retrieval_policies', {})
        policy = policies.get('tushare:'+endpoint) or (tushare_policy(endpoint) if endpoint else {})
        cap = policy.get('endpoint_rpm')
        limits = {'tushare': rpm}
        if cap: limits['tushare:'+endpoint] = min(rpm, cap)
        logged = False
        while True:
            job = getattr(self, 'active_job', None)
            if job: self.state.check_cancel(job)
            with self.state.connect() as conn:
                conn.execute('BEGIN IMMEDIATE')
                instant = time.time()
                assigned = instant
                for scope in limits:
                    row = conn.execute('SELECT next FROM request_rate WHERE scope=?', (scope,)).fetchone()
                    if row: assigned = max(assigned, row[0])
                wait = assigned-instant
                if wait <= 0:
                    for scope, rate in limits.items():
                        conn.execute('INSERT OR REPLACE INTO request_rate VALUES(?,?)', (scope, instant+60/rate))
                    return min(limits.values())
            if job and not logged:
                self.state.log(job, 'rate_limit_wait', endpoint=endpoint, wait_seconds=round(wait, 3), effective_rpm=min(limits.values()))
                logged = True
            self._wait(min(wait, .5))

    def _cooldown(self, seconds, endpoint, reason):
        # Shared with every request/retry using this subscription and state directory.
        with self.state.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            row = conn.execute("SELECT next FROM request_rate WHERE scope='tushare'").fetchone()
            conn.execute('INSERT OR REPLACE INTO request_rate VALUES(?,?)', ('tushare', max(time.time()+seconds, row[0] if row else 0)))
        job = getattr(self, 'active_job', None)
        if job: self.state.log(job, 'rate_limit_cooldown', endpoint=endpoint, wait_seconds=seconds, reason=reason)

    @staticmethod
    def retry_after(value):
        try: return max(0, float(value))
        except (TypeError, ValueError):
            try: return max(0, (parsedate_to_datetime(value)-datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError): return 60

    def fetch(self, endpoint: str, params: dict, fields: list[str], archive: Path) -> tuple[dict, str]:
        if not self.settings.token: raise SourceError("尚未配置 TUSHARE_TOKEN")
        body = {"api_name": endpoint, "token": self.settings.token, "params": params, "fields": ",".join(fields)}
        for attempt in range(4):
            effective_rpm = self._rate(endpoint)
            job=getattr(self,'active_job',None)
            if job:self.state.log(job,'request_started',source_id='tushare',endpoint=endpoint,parameters=params,attempt=attempt+1,effective_rpm=effective_rpm)
            started=time.monotonic()
            try:
                response = requests.post(self.url, json=body, timeout=self.settings.timeout, allow_redirects=False)
                if job:self.state.log(job,'http_response',source_id='tushare',endpoint=endpoint,http_status=response.status_code,elapsed_ms=int((time.monotonic()-started)*1000))
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt == 3: raise SourceError(f"Tushare HTTP {response.status_code}")
                    if response.status_code == 429:
                        self._cooldown(max(1, self.retry_after(response.headers.get('Retry-After'))), endpoint, 'HTTP 429')
                    else:self._wait(2 ** attempt)
                    continue
                if response.status_code != 200: raise SourceError(f"Tushare HTTP {response.status_code}")
                raw = response.content
                result = json.loads(raw, parse_float=Decimal,parse_constant=str)
                if result.get("code") != 0:
                    message = str(result.get("msg", "接口拒绝请求")).replace(self.settings.token, "[redacted]")
                    if any(word in message for word in ('每天','每日','每小时','今日','总量')):
                        raise SourceError(f'Tushare 配额不足，停止请求并保留检查点：{message}')
                    if any(word in message for word in ('频','每分钟')) and attempt < 3:
                        self._cooldown(60, endpoint, '源端分钟限频')
                        continue
                    raise SourceError(f"Tushare {result.get('code')}: {message}")
                data = result.get("data")
                if not isinstance(data, dict) or not isinstance(data.get("fields"), list) or not isinstance(data.get("items"), list):
                    raise SourceError("接口响应缺少 fields/items")
                archive.parent.mkdir(parents=True, exist_ok=True)
                tmp = archive.with_suffix(".tmp")
                with gzip.open(tmp, "wb") as handle: handle.write(raw)
                tmp.replace(archive)
                sha = hashlib.sha256(raw).hexdigest()
                atomic_json(archive.with_suffix(".meta.json"), dict(endpoint=endpoint, params=params, fields=fields,
                            fetched_at=now(), sha256=sha, rows=len(data["items"])))
                return data, sha
            except (requests.Timeout, requests.ConnectionError):
                if attempt == 3: raise SourceError("Tushare 网络连接失败，已保留成功分片") from None
                self._wait(2 ** attempt)
        raise SourceError("下载失败")


def read_archive(path: Path, expected_sha: str) -> dict:
    with gzip.open(path, "rb") as handle: raw = handle.read()
    if hashlib.sha256(raw).hexdigest() != expected_sha: raise ValueError("原始归档哈希不匹配")
    return json.loads(raw, parse_float=Decimal,parse_constant=str)["data"]
