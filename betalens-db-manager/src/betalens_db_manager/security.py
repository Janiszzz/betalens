"""Credential references, request boundaries and public-data redaction.

Field-name checks are a conservative guard, not a detector for arbitrary secrets.
Resolved credentials are remembered only in process memory so their source echoes
can be removed before logs, errors or public exports are emitted.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from decimal import Decimal
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import quote, quote_plus

REDACTED = '[redacted]'
_secrets: set[str] = set()
_resolved_names: set[str] = set()


def remember_secrets(*values):
    for value in values:
        if isinstance(value, str) and value:
            _secrets.add(value)


def _key(value):
    return re.sub(r'[^a-z0-9]', '', str(value).lower())


def sensitive_key(value):
    name = _key(value)
    return name in {'auth', 'cookie', 'setcookie'} or any(
        marker in name for marker in ('password', 'passwd', 'secret', 'apikey',
                                     'accesskey', 'authorization', 'credential', 'token'))


def _secret_variants(values):
    variants = set()
    for secret in values:
        if isinstance(secret, str) and secret:
            variants.update((secret, quote(secret, safe=''), quote_plus(secret), json.dumps(secret, ensure_ascii=True)[1:-1]))
    return sorted(variants, key=len, reverse=True)


def _contains_secret(message, secret):
    # A one-character password is not evidence that every word containing that
    # character is a credential. Exact scalar echoes and delimited text still
    # match; explicitly named credential fields are always protected separately.
    if len(secret) <= 3:
        return re.search(r'(?<!\w)' + re.escape(secret) + r'(?!\w)', message) is not None
    return secret in message


def contains_known_secret(value, *, secrets=()):
    """Check a string without exposing the in-memory credential collection."""
    if not isinstance(value, str):
        return False
    known = set(_secrets) | {secret for secret in secrets if isinstance(secret, str) and secret}
    return any(_contains_secret(value, secret) for secret in _secret_variants(known))


def auth_reference(auth):
    """Normalize an explicit reference; retain the legacy env/header syntax."""
    if auth is None or auth == {}:
        return None
    if not isinstance(auth, Mapping) or set(auth) - {'env', 'location', 'name', 'header', 'query', 'body', 'prefix'}:
        raise ValueError('鉴权仅允许环境变量引用及 header/query/body 注入位置')
    env = auth.get('env')
    if not isinstance(env, str) or not re.fullmatch(r'[A-Z][A-Z0-9_]*', env):
        raise ValueError('鉴权环境变量名称无效')
    legacy = [kind for kind in ('header', 'query', 'body') if kind in auth]
    if len(legacy) > 1 or (legacy and ('location' in auth or 'name' in auth)):
        raise ValueError('鉴权只能指定一个注入位置')
    location = legacy[0] if legacy else auth.get('location', 'header')
    name = auth[legacy[0]] if legacy else auth.get('name', 'Authorization' if location == 'header' else None)
    if location not in {'header', 'query', 'body'} or not isinstance(name, str) or not name:
        raise ValueError('鉴权需要有效的 header/query/body 位置和字段名')
    if not re.fullmatch(r"[A-Za-z0-9_!#$%&'*+.^`|~-]+", name) or len(name) > 256:
        raise ValueError('鉴权字段名无效')
    if location == 'body' and any(not part for part in name.split('.')):
        raise ValueError('body 鉴权路径无效')
    prefix = auth.get('prefix', '')
    if not isinstance(prefix, str) or len(prefix) > 64 or any(c in prefix for c in ('\r', '\n', '\x00')):
        raise ValueError('鉴权前缀无效')
    return {'env': env, 'location': location, 'name': name, 'prefix': prefix}


def validate_parameters(value, *, path='parameters', allowed=None, auth=None):
    """Reject embedded credentials before parameters reach a plan or archive."""
    if not isinstance(value, Mapping):
        raise ValueError(f'{path}: 参数必须为对象')
    reference = auth_reference(auth)
    reserved = {_key(reference['name'].split('.')[-1])} if reference else set()
    if allowed is not None:
        if not isinstance(allowed, (Mapping, list, tuple, set)):
            raise ValueError('参数合同必须为字段对象或字段名列表')
        unknown = set(value) - set(allowed)
        if unknown:
            raise ValueError(f'{path}: 包含未登记业务参数')

    def visit(item, location):
        if isinstance(item, Mapping):
            for name, nested in item.items():
                if sensitive_key(name) or _key(name) in reserved:
                    # Field locations are useful; values must never be printed.
                    raise ValueError(f'{location}.{name}: 不能持久化鉴权值，请使用 auth.env 引用')
                visit(nested, f'{location}.{name}')
        elif isinstance(item, (list, tuple)):
            for index, nested in enumerate(item):
                visit(nested, f'{location}[{index}]')
        elif isinstance(item,str) and contains_known_secret(item):
            raise ValueError(f'{location}: 不能持久化已解析的凭据值')
    visit(value, path)
    return value


def credential_configured(settings, name):
    """Report reference availability without resolving a private-file value."""
    return name in getattr(settings, 'credential_names', ()) or name in os.environ or name in _resolved_names


def request_auth(auth, settings, params, method):
    """Resolve secrets only at the outgoing HTTP request boundary."""
    if method not in {'GET', 'POST'}:
        raise ValueError('HTTP 方法须为 GET/POST')
    validate_parameters(params, auth=auth)
    reference = auth_reference(auth)
    query = deepcopy(params) if method == 'GET' else {}
    body = deepcopy(params) if method == 'POST' else None
    headers = {}
    if reference is None:
        return headers, query or None, body
    from dotenv import dotenv_values
    path = getattr(settings, 'env_file', None)
    environment = dict(os.environ)
    if path and Path(path).exists():
        environment.update({k: v for k, v in dotenv_values(path, interpolate=False).items() if v is not None})
    secret = environment.get(reference['env'])
    if not secret:
        raise ValueError('来源鉴权环境变量未配置：' + reference['env'])
    remember_secrets(secret)
    _resolved_names.add(reference['env'])
    value = reference['prefix'] + secret
    remember_secrets(value)
    if reference['location'] == 'header':
        headers[reference['name']] = value
    elif reference['location'] == 'query':
        query[reference['name']] = value
    else:
        if method != 'POST':
            raise ValueError('body 鉴权只支持 POST 请求')
        current = body
        pieces = reference['name'].split('.')
        for name in pieces[:-1]:
            if name in current and not isinstance(current[name], dict):
                raise ValueError('body 鉴权路径与业务参数冲突')
            current = current.setdefault(name, {})
        current[pieces[-1]] = value
    return headers, query or None, body


_assignment = re.compile(
    r'''(?ix)(["']?(?:[a-z0-9_-]*(?:api[_-]?key|access[_-]?key|password|passwd|secret|token|authorization|credential)[a-z0-9_-]*)["']?\s*[:=]\s*)(?:"[^"\r\n]*"|'[^'\r\n]*'|[^\s&,;}]+)''')
_authorization = re.compile(r'''(?ix)(["']?(?:proxy[-_])?authorization["']?\s*[:=]\s*)(?:"[^"\r\n]*"|'[^'\r\n]*'|[^\r\n,;}]+)''')
_bearer = re.compile(r'(?i)(\bBearer\s+)[^\s,;\}\]"\']+')


def redact(value, *, secrets=(), sensitive_names=(), preserve_tokens=False):
    known = set(_secrets) | {v for v in secrets if isinstance(v, str) and v}
    variants = _secret_variants(known)
    names = {_key(name) for name in sensitive_names}

    def text(message):
        for secret in variants:
            if len(secret) <= 3:
                message = re.sub(r'(?<!\w)' + re.escape(secret) + r'(?!\w)', lambda _: REDACTED, message)
            else:
                message = message.replace(secret, REDACTED)
        message = _authorization.sub(lambda match: match[1] + REDACTED, message)
        message = _bearer.sub(lambda match: match[1] + REDACTED, message)
        return _assignment.sub(lambda match: match[1] + REDACTED, message)

    def walk(item):
        if isinstance(item, Mapping):
            confirmation = False
            if preserve_tokens and isinstance(item.get('token'), str):
                from .util import digest
                confirmation = item['token'] not in known and item['token'] == digest({k: v for k, v in item.items() if k != 'token'})
            result = {}
            for name, nested in item.items():
                descriptor = False
                if name == 'auth' and isinstance(nested, Mapping):
                    try:descriptor = auth_reference(nested) is not None
                    except ValueError:pass
                mask = (sensitive_key(name) or _key(name) in names) and not descriptor
                if _key(name).endswith('configured') and isinstance(nested, bool):mask = False
                public_name = text(name) if isinstance(name, str) else name
                if name == 'token' and confirmation:
                    result[public_name] = nested
                elif mask:
                    result[public_name] = REDACTED
                else:
                    result[public_name] = walk(nested)
            # Source payloads are often column-oriented rather than row objects.
            fields = item.get('fields')
            if isinstance(fields, list) and all(isinstance(f, str) for f in fields) and isinstance(item.get('items'), list):
                protected = {i for i, name in enumerate(fields) if sensitive_key(name) or _key(name) in names}
                result['items'] = [[REDACTED if i in protected else walk(cell) for i, cell in enumerate(row)]
                                   if isinstance(row, (list, tuple)) else walk(row) for row in item['items']]
            if any(sensitive_key(item.get(label, '')) or _key(item.get(label, '')) in names
                   for label in ('source_field', 'canonical_column', 'column')):
                for field in ('raw_value', 'converted_value', 'value', 'before', 'after'):
                    if field in result:
                        result[field] = REDACTED
            return result
        if isinstance(item, (list, tuple)):
            return [walk(nested) for nested in item]
        if isinstance(item, str):
            return text(item)
        return item
    return walk(value)


def public_data(value, settings=None, definitions=()):
    secrets = tuple(getattr(settings, name, '') for name in ('token', 'pg_password', 'ch_password'))
    names = set()
    by_source = {}
    for definition in definitions:
        try:
            reference = auth_reference(definition.get('auth'))
        except ValueError:
            continue
        if reference:
            name = reference['name'].split('.')[-1]
            names.add(name)
            by_source[definition.get('source_id')] = {name}

    def scoped(item, active=None):
        if isinstance(item, Mapping):
            active = by_source.get(item.get('source_id'), active)
            selected = active if active is not None else names
            result = {}
            for key, nested in item.items():
                if key in {'parameters', 'params', 'headers', 'query', 'body'}:
                    result[key] = redact(nested, secrets=secrets, sensitive_names=selected)
                else:
                    result[key] = scoped(nested, active)
            if (isinstance(item.get('fields'), list) and isinstance(item.get('items'), list)) or (
                'source_row' in item and 'dataset' in item) or 'raw_value' in item:
                return redact(result, secrets=secrets, sensitive_names=selected, preserve_tokens=True)
            return result
        if isinstance(item, (list, tuple)):
            return [scoped(nested, active) for nested in item]
        return item

    # Verify confirmation tokens against the original object before scoped
    # redaction changes any nested data used by its digest.
    baseline = redact(value, secrets=secrets, preserve_tokens=True)
    return scoped(baseline)


def safe_error(exc, settings=None, definitions=()):
    return public_data(str(exc), settings, definitions)[:2000]


def write_private_wire(path, content):
    """Preserve exact wire bytes in a private gzip file, never a public export."""
    from .private import private_path, restrict
    path = private_path(Path(path))
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    restrict(path.parent, True)
    fd, name = tempfile.mkstemp(prefix='.wire-', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as handle:
            restrict(temporary)
            handle.write(gzip.compress(content, mtime=0))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def archive_export(path, settings=None, definitions=()):
    """Keep immutable source bytes intact and export a separately hashed copy."""
    from .util import dumps
    raw = gzip.decompress(Path(path).read_bytes())
    value = json.loads(raw, parse_float=Decimal)
    definitions = list(definitions)
    names = []
    for definition in definitions:
        try:reference = auth_reference(definition.get('auth'))
        except ValueError:continue
        if reference:names.append(reference['name'].split('.')[-1])
    # An archive is source data, not a task envelope: declared auth field names
    # apply throughout this export even when they are generic names such as id.
    exported = dumps(redact(public_data(value, settings, definitions), sensitive_names=names)).encode()
    return gzip.compress(exported, mtime=0), hashlib.sha256(raw).hexdigest(), hashlib.sha256(exported).hexdigest()
