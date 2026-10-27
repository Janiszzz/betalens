"""Safe YAML parsing with LibYAML acceleration when PyYAML provides it.

Keep the loader local to this component. Never patch PyYAML globally and never
use its generic CLoader/Loader, which accept Python object construction tags.
"""
from __future__ import annotations

import yaml


_SafeLoader = getattr(yaml, 'CSafeLoader', yaml.SafeLoader)
_SafeDumper = getattr(yaml, 'CSafeDumper', yaml.SafeDumper)
YAMLError = yaml.YAMLError


def safe_load(stream):
    return yaml.load(stream, Loader=_SafeLoader)


def safe_dump(data, stream=None, **kwargs):
    return yaml.dump(data, stream=stream, Dumper=_SafeDumper, **kwargs)
