"""Small helpers shared across modules."""

import os
import time
from typing import Optional


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def dir_size_bytes(path: str) -> Optional[int]:
    """Total size of every file under `path`, or None if it can't be read."""
    try:
        total = 0
        for root, _dirs, files in os.walk(path):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
        return total or None
    except Exception:
        return None
