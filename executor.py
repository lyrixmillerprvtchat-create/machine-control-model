"""
Executor — danger-aware, streaming, no friction for safe operations.
LOW    → runs immediately, prints output inline
MEDIUM → one-line warning, runs after brief pause
HIGH   → simple y/n
CRITICAL → hard block with explicit y/n
"""

import os
import re
import subprocess
import webbrowser
import platform
import datetime
from typing import Optional

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import pyautogui
    HAS_PYAUTOGUI = True
except ImportError:
    HAS_PYAUTOGUI = False

try:
    from colorama import Fore, Style
    def _c(t, c):
        m = {"cyan": Fore.CYAN, "green": Fore.GREEN, "yellow": Fore.YELLOW,
             "red": Fore.RED, "bold": Style.BRIGHT, "dim": Style.DIM, "white": Fore.WHITE}
        return m.get(c, "") + t + Style.RESET_ALL
except ImportError:
    def _c(t, _): return t

LOG_PATH = os.path.join(os.path.dirname(__file__), "data", "execution.log")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

DANGER_PATTERNS = [
    (re.compile(r'\brm\s+-rf\b|\brmdir\s+/s\b', re.I), "HIGH"),
    (re.compile(r'\bformat\b|\bdiskpart\b|\bfdisk\b', re.I), "CRITICAL"),
    (re.compile(r'\bshutdown\b|\breboot\b|\brestart\b', re.I), "HIGH"),
    (re.compile(r'\bdel\b.+/[fqs]', re.I), "HIGH"),
    (re.compile(r'\bkill\b|\btaskkill\b', re.I), "MEDIUM"),
    (re.compile(r'\bgit\s+push\b|\bgit\s+reset\b', re.I), "MEDIUM"),
    (re.compile(r'\bfile_op_delete\b', re.I), "MEDIUM"),
]

def _danger(cmd: str, intent: str) -> str:
    check = cmd + " " + intent
    for pat, level in DANGER_PATTERNS:
        if pat.search(check):
            return level
    return "LOW"

def _log(intent: str, cmd: str, ran: bool):
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] [{'RAN' if ran else 'SKIPPED'}] intent={intent} cmd={cmd!r}\n")

def _shell(cmd: str) -> int:
    print(_c(f"  $ {cmd}", "dim"))
    result = subprocess.run(cmd, shell=True, text=True)
    return result.returncode

def _gate(action: str, cmd: str, danger: str) -> bool:
    """Simple one-line gate for HIGH/CRITICAL only."""
    color = "red" if danger == "CRITICAL" else "yellow"
    print(_c(f"\n  [{danger}] {action}", color))
    print(_c(f"  Command: {cmd}", "white"))
    try:
        ans = input(_c("  Run it? [y/N] > ", "bold")).strip().lower()
    except (KeyboardInterrupt, EOFError):
        return False
    return ans == "y"

def execute(prediction: dict, registry=None) -> bool:
    intent  = prediction.get("intent", "unknown")
    params  = prediction.get("params", {})
    text    = prediction.get("text", "")

    # ── project function ──────────────────────────────────────────────
    if intent.startswith("project::") and registry and registry.is_project_tool(intent):
        info = registry.get_tool_info(intent)
        fn   = info.get("function", intent)
        src  = os.path.basename(info.get("source", ""))
        cmd_repr = f"{src}::{fn}()"
        danger = _danger(cmd_repr, intent)
        if danger in ("HIGH", "CRITICAL"):
            if not _gate(f"Call {fn}()", cmd_repr, danger):
                _log(intent, cmd_repr, False)
                return False
        else:
            print(_c(f"  Calling {fn}() ...", "cyan"))
        try:
            result = registry.call_project_tool(intent, params)
            if result is not None:
                print(_c(f"  => {result}", "green"))
        except Exception as e:
            print(_c(f"  Error: {e}", "red"))
        _log(intent, cmd_repr, True)
        return True

    # ── build action + command ────────────────────────────────────────
    action, cmd = _build(intent, params, text)
    danger = _danger(cmd, intent)

    # HIGH / CRITICAL — ask
    if danger in ("HIGH", "CRITICAL"):
        if not _gate(action, cmd, danger):
            _log(intent, cmd, False)
            return False

    # MEDIUM — warn but proceed
    elif danger == "MEDIUM":
        print(_c(f"  ! {action}", "yellow"))

    # LOW — silent, just do it
    _log(intent, cmd, True)
    _run(intent, params, cmd)
    return True


def _build(intent: str, params: dict, text: str) -> tuple[str, str]:
    """Return (human-readable action, shell command / repr)."""
    is_win = platform.system() == "Windows"

    if intent == "sys_command":
        cmd = params.get("cmd", params.get("raw_text", text))
        return f"Run: {cmd}", cmd

    if intent == "dev_server":
        port = params.get("port", "3000")
        cmd  = f"npm run dev -- --port {port}"
        return f"Start dev server on :{port}", cmd

    if intent == "open_app":
        app = params.get("app", params.get("raw_text", text))
        cmd = f"start {app}" if is_win else f"open -a '{app}'"
        return f"Open {app}", cmd

    if intent == "file_op_open":
        f = params.get("file", params.get("raw_text", text))
        cmd = f"start {f}" if is_win else f"open '{f}'"
        return f"Open file {f}", cmd

    if intent == "file_op_delete":
        f = params.get("file", params.get("raw_text", text))
        cmd = f"del {f}" if is_win else f"rm '{f}'"
        return f"Delete {f}", cmd

    if intent == "file_op_create":
        f = params.get("file", params.get("raw_text", text))
        return f"Create {f}", f"New-Item {f}"

    if intent == "dir_op":
        d = params.get("dir", params.get("raw_text", text))
        cmd = f"explorer {d}" if is_win else f"open '{d}'"
        return f"Open folder {d}", cmd

    if intent == "browser_open":
        url = params.get("url", params.get("raw_text", text))
        if not url.startswith("http"):
            url = "https://" + url
        return f"Open {url}", f"webbrowser.open({url!r})"

    if intent == "browser_search":
        q = params.get("query", params.get("raw_text", text))
        url = f"https://www.google.com/search?q={q.replace(' ', '+')}"
        return f"Search: {q}", f"webbrowser.open({url!r})"

    if intent == "system_info":
        return "System info", "psutil"

    if intent == "kill_process":
        proc = params.get("process", params.get("raw_text", text))
        cmd  = f"taskkill /IM {proc}.exe /F" if is_win else f"pkill {proc}"
        return f"Kill {proc}", cmd

    if intent == "volume_control":
        return "Volume control", "volume"

    if intent == "screenshot":
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.expanduser(f"~/Desktop/screenshot_{ts}.png")
        return "Screenshot", f"screenshot -> {path}"

    if intent == "git_op":
        raw = params.get("raw_text", text).lower()
        cmd = next((v for k, v in {
            "status": "git status", "pull": "git pull",
            "push": "git push", "log": "git log --oneline -10",
            "diff": "git diff",
        }.items() if k in raw), None)
        branch = params.get("branch")
        msg    = params.get("msg")
        if branch:
            cmd = f"git checkout -b {branch}" if "new" in raw else f"git checkout {branch}"
        elif msg:
            cmd = f"git commit -m {msg}"
        elif not cmd:
            cmd = f"git {raw}"
        return f"Git: {cmd}", cmd

    if intent == "sleep_shutdown":
        raw = params.get("raw_text", text).lower()
        if "shutdown" in raw or "turn off" in raw:
            cmd = "shutdown /s /t 30" if is_win else "shutdown -h now"
        elif "restart" in raw or "reboot" in raw:
            cmd = "shutdown /r /t 30" if is_win else "shutdown -r now"
        elif "sleep" in raw:
            cmd = "rundll32.exe powrprof.dll,SetSuspendState 0,1,0" if is_win else "pmset sleepnow"
        else:
            cmd = "rundll32.exe user32.dll,LockWorkStation" if is_win else "loginctl lock-session"
        return f"Power: {cmd}", cmd

    if intent == "clipboard":
        return "Clipboard", "clipboard"

    return f"Unknown: {intent}", str(params)


def _run(intent: str, params: dict, cmd: str):
    """Execute after any gating is done."""
    is_win = platform.system() == "Windows"

    if intent in ("sys_command", "dev_server", "open_app",
                  "file_op_open", "file_op_delete", "dir_op",
                  "kill_process", "git_op", "sleep_shutdown"):
        rc = _shell(cmd)
        if rc != 0:
            print(_c(f"  exited {rc}", "yellow"))

    elif intent == "file_op_create":
        f = params.get("file", "")
        if f and not os.path.exists(f):
            open(f, "w").close()
            print(_c(f"  Created {f}", "green"))
        else:
            print(_c(f"  Already exists: {f}", "yellow"))

    elif intent in ("browser_open", "browser_search"):
        url = params.get("url") or params.get("query", "")
        if intent == "browser_search":
            url = f"https://www.google.com/search?q={url.replace(' ', '+')}"
        elif not url.startswith("http"):
            url = "https://" + url
        webbrowser.open(url)
        print(_c(f"  Opened {url}", "green"))

    elif intent == "system_info":
        print(_c(f"  OS   : {platform.system()} {platform.release()}", "white"))
        if HAS_PSUTIL:
            cpu  = psutil.cpu_percent(interval=0.3)
            mem  = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            print(_c(f"  CPU  : {cpu}%", "white"))
            print(_c(f"  RAM  : {mem.percent}%  ({mem.used//1024**2}MB / {mem.total//1024**2}MB)", "white"))
            print(_c(f"  Disk : {disk.percent}%  ({disk.used//1024**3}GB / {disk.total//1024**3}GB)", "white"))

    elif intent == "screenshot":
        if HAS_PYAUTOGUI:
            ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.expanduser(f"~/Desktop/screenshot_{ts}.png")
            pyautogui.screenshot().save(path)
            print(_c(f"  Saved {path}", "green"))
        else:
            print(_c("  pip install pyautogui", "yellow"))

    elif intent == "volume_control":
        print(_c("  pip install pycaw for full volume control", "yellow"))
