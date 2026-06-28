"""
Persistent Memory Store
Remembers three things across sessions:
  - aliases   : named command shortcuts  ("my dev server" -> intent + params)
  - notes     : free-text references     ("my project path" -> "C:/Projects/app")
  - prefs     : auto-learned defaults    (most-used port, app, dir, etc.)
"""

import json
import os
import re
from typing import Optional

MEMORY_PATH = os.path.join(os.path.dirname(__file__), "data", "memory.json")

# How many times a param value must appear before it becomes a default
PREF_THRESHOLD = 3

_REMEMBER_RE = re.compile(
    r'^(?:remember|save|store|define)\s+(.+?)\s+as\s+(.+)$', re.I
)
_REMEMBER_EQ_RE = re.compile(
    r'^(?:remember|save|store|define)\s+([\w\s]+?)\s*=\s*(.+)$', re.I
)
_FORGET_RE = re.compile(
    r'^(?:forget|delete|remove)\s+(?:memory\s+)?(.+)$', re.I
)

# Params that are worth learning defaults for
TRACKABLE_PARAMS = {"port", "app", "dir", "url", "file", "branch"}


# ---------------------------------------------------------------------------
# Load / Save
# ---------------------------------------------------------------------------

def _empty() -> dict:
    return {"aliases": {}, "notes": {}, "prefs": {}, "usage": {}}


def load() -> dict:
    if not os.path.exists(MEMORY_PATH):
        return _empty()
    try:
        with open(MEMORY_PATH, encoding="utf-8") as f:
            data = json.load(f)
        for key in _empty():
            data.setdefault(key, {})
        return data
    except (json.JSONDecodeError, OSError):
        return _empty()


def save(data: dict) -> None:
    os.makedirs(os.path.dirname(MEMORY_PATH), exist_ok=True)
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Alias management
# ---------------------------------------------------------------------------

def set_alias(name: str, intent: str, params: dict, raw: str = "") -> None:
    data = load()
    data["aliases"][name.lower()] = {
        "intent": intent,
        "params": params,
        "raw": raw,
    }
    save(data)


def set_note(name: str, value: str) -> None:
    data = load()
    data["notes"][name.lower()] = value
    save(data)


def forget(name: str) -> bool:
    """Remove from aliases or notes. Returns True if something was deleted."""
    data = load()
    key = name.lower().strip()
    found = False
    if key in data["aliases"]:
        del data["aliases"][key]
        found = True
    if key in data["notes"]:
        del data["notes"][key]
        found = True
    if found:
        save(data)
    return found


def get_alias(text: str) -> Optional[dict]:
    """Return stored prediction dict if text matches a saved alias, else None."""
    data = load()
    key = text.lower().strip()
    if key in data["aliases"]:
        entry = data["aliases"][key]
        return {
            "text": text,
            "intent": entry["intent"],
            "params": entry["params"],
            "confidence": 1.0,
            "top3": [{"intent": entry["intent"], "score": 1.0}],
            "_from_memory": True,
            "_alias_name": key,
        }
    return None


def get_note(name: str) -> Optional[str]:
    data = load()
    return data["notes"].get(name.lower().strip())


# ---------------------------------------------------------------------------
# Preference defaults
# ---------------------------------------------------------------------------

def track_usage(intent: str, params: dict) -> None:
    """Record param values used after approval to learn defaults over time."""
    data = load()
    usage = data["usage"]

    for param, value in params.items():
        if param not in TRACKABLE_PARAMS or not value or value.startswith("<"):
            continue
        slot_key = f"{intent}.{param}"
        if slot_key not in usage:
            usage[slot_key] = {}
        usage[slot_key][value] = usage[slot_key].get(value, 0) + 1

        # Promote to preference once threshold is hit
        top_val, top_count = max(usage[slot_key].items(), key=lambda x: x[1])
        if top_count >= PREF_THRESHOLD:
            data["prefs"][slot_key] = top_val

    save(data)


def fill_defaults(prediction: dict) -> dict:
    """
    Fill any missing or empty params using learned preferences.
    Returns an updated copy of prediction.
    """
    data = load()
    prefs = data.get("prefs", {})
    if not prefs:
        return prediction

    intent = prediction.get("intent", "")
    params = dict(prediction.get("params", {}))

    for param in TRACKABLE_PARAMS:
        slot_key = f"{intent}.{param}"
        if slot_key in prefs and not params.get(param):
            params[param] = prefs[slot_key]

    return {**prediction, "params": params}


# ---------------------------------------------------------------------------
# Command parsing
# ---------------------------------------------------------------------------

def parse_remember(text: str) -> Optional[tuple[str, str]]:
    """
    Detect 'remember X as Y' or 'remember Y = X'.
    Returns (name, value) or None.
    """
    m = _REMEMBER_RE.match(text.strip())
    if m:
        return m.group(2).strip(), m.group(1).strip()

    m = _REMEMBER_EQ_RE.match(text.strip())
    if m:
        return m.group(1).strip(), m.group(2).strip()

    return None


def parse_forget(text: str) -> Optional[str]:
    """Detect 'forget X'. Returns name or None."""
    m = _FORGET_RE.match(text.strip())
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def list_all() -> dict:
    return load()


def summary() -> str:
    data = load()
    a = len(data["aliases"])
    n = len(data["notes"])
    p = len(data["prefs"])
    return f"{a} alias(es), {n} note(s), {p} learned preference(s)"
