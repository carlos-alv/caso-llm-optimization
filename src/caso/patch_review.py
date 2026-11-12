from __future__ import annotations
import re
from typing import List
_PATTERNS = [('cache_decorator', re.compile('@\\s*(functools\\.)?(lru_cache|cache|cached_property)\\b')), ('function_attribute_state', re.compile('getattr\\(\\s*\\w+\\s*,\\s*[\\"\']_\\w*[\\"\']\\s*,\\s*(\\{\\}|\\[\\]|dict\\(\\)|set\\(\\))')), ('global_statement', re.compile('^\\s*global\\s+\\w+')), ('mutable_default_argument', re.compile('def\\s+\\w+\\(.*=\\s*(\\{\\}|\\[\\]|dict\\(\\)|set\\(\\))')), ('cache_or_memo_container', re.compile('\\b\\w*(cache|memo(?!ry))\\w*\\s*(\\[[^\\]]+\\]\\s*=|=\\s*(\\{\\}|dict\\(\\)|\\[\\]))', re.I))]

def persistent_state_flags(diff: str) -> List[str]:
    flags = []
    for line in diff.splitlines():
        if not line.startswith('+') or line.startswith('+++'):
            continue
        code = line[1:].split('#', 1)[0]
        for name, pattern in _PATTERNS:
            if pattern.search(code) and name not in flags:
                flags.append(name)
    return flags
