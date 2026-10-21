"""Shared JSON store of key results, used by later steps and the slides."""

import json

import numpy as np

from h12eva import config as cfg

PATH = cfg.PROC_DIR / "key_results.json"


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return "inf" if np.isinf(v) else (None if np.isnan(v) else v)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def load():
    """Load all stored results (empty dict if none)."""
    if PATH.exists():
        return json.loads(PATH.read_text())
    return {}


def update(section, values):
    """Store values under a section name, replacing older ones."""
    data = load()
    data[section] = _clean(values)
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(data, indent=2))


def as_float(value):
    """Convert a stored value back to float ('inf' becomes inf)."""
    return np.inf if value == "inf" else float(value)
