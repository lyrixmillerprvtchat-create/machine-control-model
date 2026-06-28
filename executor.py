"""
Phase 4: Execution Sandbox & Human Gatekeeper
All predicted commands route here. Nothing executes without explicit Y/N terminal approval.
"""

import os
import re
import sys
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
    from colorama import init, Fore, Style
    init(autoreset=True)
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False

LOG_PATH = os.path.join(os.path.dirname(__file__), "data", "execution.log")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# ---------------------------------------------------------------------------
# Danger tier classification — determines warning level shown to user
# ---------------------------------------------------------------------------

DANGER_PATTERNS = [
    (re.compile(r'\brm\s+-rf\b|\brmdir\s+/s\b', re.I), "HIGH"),
    (re.compile(r'\bformat\b|\bdiskpart\b|\bfdisk\b', re.I), "CRITICAL"),
    (re.compile(r'\bshutdown\b|\breboot\b|\brestart\b', re.I), "HIGH"),
    (re.compile(r'\bdel\b.+/[fqs]', re.I), "HIGH"),
    (re.compile(r'\bkill\b|\btaskkill\b', re.I), "MEDIUM"),
    (re.compile(r'\bgit\s+push\b|\bgit\s+reset\b', re.I), "MEDIUM"),
]


def _danger_level(command_string: str) -> str:
    for pattern, level in DANGER_PATTERNS:
        if pattern.search(command_string):
            return level
    return "LOW"


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def _c(text: str, color: str) -> str:
    if not HAS_COLOR:
        return text
    colors = {
        "red": Fore.RED, "yellow": Fore.YELLOW, "green": Fore.GREEN,
        "cyan": Fore.CYAN, "white": Fore.WHITE, "bold": Style.BRIGHT,
    }
    return colors.get(color, "") + text + Style.RESET_ALL


def _box(lines: list[str], width: int = 62) -> str:
    border = "+" + "-" * width + "+"
    rows = [border]
    for line in lines:
        padded = f"| {line:<{width - 2}} |"
        rows.append(padded)
    rows.append(border)
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# THE GATEKEEPER — cannot be bypassed
# ---------------------------------------------------------------------------

def _gatekeeper(prediction: dict, action_description: str, command_repr: str) -> bool:
    """
    Displays full prediction details and demands explicit 'y' from the user.
    Returns True only on confirmed 'y'. Any other input aborts.
    """
    danger = _danger_level(command_repr)
    danger_colors = {"LOW": "green", "MEDIUM": "yellow", "HIGH": "red", "CRITICAL": "red"}
    danger_str = _c(f"[{danger}]", danger_colors[danger])

    print("\n" + _c("=" * 64, "cyan"))
    print(_c("  MACHINE CONTROL — AWAITING YOUR APPROVAL", "bold"))
    print(_c("=" * 64, "cyan"))

    info_lines = [
        f"Intent    : {prediction.get('intent', 'unknown')}",
        f"Confidence: {prediction.get('confidence', 0):.1%}",
        f"Action    : {action_description}",
        f"Danger    : {danger}",
        "",
        f"Command   : {command_repr}",
    ]
    if prediction.get("params"):
        for k, v in prediction["params"].items():
            if k != "raw_text":
                info_lines.append(f"  param.{k:<8}: {v}")

    for line in info_lines:
        print(f"  {line}")

    print(_c("=" * 64, "cyan"))

    if danger in ("HIGH", "CRITICAL"):
        print(_c(f"\n  WARNING: This is a {danger}-risk operation.", "red"))

    top3 = prediction.get("top3", [])
    if len(top3) > 1:
        print(f"\n  Other candidates:")
        for alt in top3[1:]:
            print(f"    {alt['intent']:<25} ({alt['score']:.1%})")

    print()
    try:
        answer = input("  Proceed? [y/N] > ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print("\n  Aborted.")
        return False

    if answer == "y":
        return True

    print(_c("  [BLOCKED] Execution cancelled by user.", "yellow"))
    return False


def _log(intent: str, command: str, approved: bool) -> None:
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    status = "APPROVED" if approved else "BLOCKED"
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] [{status}] intent={intent} cmd={command!r}\n")


# ---------------------------------------------------------------------------
# Intent handlers
# ---------------------------------------------------------------------------

def _shell(cmd: str) -> None:
    print(_c(f"\n[>] {cmd}", "cyan"))
    result = subprocess.run(cmd, shell=True, text=True, capture_output=False)
    if result.returncode != 0:
        print(_c(f"[!] Exit code {result.returncode}", "yellow"))


def execute(prediction: dict, registry=None) -> bool:
    """
    Routes a model prediction to the correct handler.
    Returns True if the action was approved and executed, False if blocked.
    """
    intent = prediction.get("intent", "unknown")
    params = prediction.get("params", {})

    # ------------------------------------------------------------------
    # Project function invocation (registry:: tools)
    # ------------------------------------------------------------------
    if intent.startswith("project::") and registry is not None and registry.is_project_tool(intent):
        info = registry.get_tool_info(intent)
        func_name = info.get("function", intent)
        source = info.get("source", "unknown")
        desc = info.get("description", "")
        action = f"Call project function: {func_name}()"
        command_repr = f"{os.path.basename(source)}::{func_name}({', '.join(f'{k}={v!r}' for k, v in params.items() if k != 'raw_text')})"

        approved = _gatekeeper(prediction, action, command_repr)
        _log(intent, command_repr, approved)
        if not approved:
            return False

        print(_c("\n[+] Executing...", "green"))
        try:
            result = registry.call_project_tool(intent, params)
            if result is not None:
                print(_c(f"[+] Return value: {result}", "green"))
        except Exception as exc:
            print(_c(f"[!] Error calling {func_name}: {exc}", "red"))
        print(_c("\n[OK] Done.\n", "green"))
        return True

    # ------------------------------------------------------------------
    # Build the action description and command representation
    # ------------------------------------------------------------------
    action = "unknown"
    command_repr = ""

    if intent == "sys_command":
        cmd = params.get("cmd", params.get("raw_text", ""))
        action = f"Run shell command: {cmd}"
        command_repr = cmd

    elif intent == "dev_server":
        port = params.get("port", "3000")
        cmd = f"npm run dev -- --port {port}"
        action = f"Start development server on port {port}"
        command_repr = cmd

    elif intent == "open_app":
        app = params.get("app", params.get("raw_text", ""))
        action = f"Open application: {app}"
        command_repr = f"start {app}" if platform.system() == "Windows" else f"open -a '{app}'"

    elif intent == "file_op_open":
        path = params.get("file", params.get("raw_text", ""))
        action = f"Open file: {path}"
        command_repr = f"start {path}" if platform.system() == "Windows" else f"open '{path}'"

    elif intent == "file_op_delete":
        path = params.get("file", params.get("raw_text", ""))
        action = f"DELETE file: {path}"
        command_repr = f"del {path}" if platform.system() == "Windows" else f"rm '{path}'"

    elif intent == "file_op_create":
        path = params.get("file", params.get("raw_text", ""))
        action = f"Create new file: {path}"
        command_repr = f"New-Item {path}"

    elif intent == "dir_op":
        d = params.get("dir", params.get("raw_text", ""))
        action = f"Directory operation on: {d}"
        command_repr = f"explorer {d}" if platform.system() == "Windows" else f"open '{d}'"

    elif intent == "browser_open":
        url = params.get("url", params.get("raw_text", ""))
        if not url.startswith("http"):
            url = "https://" + url
        action = f"Open in browser: {url}"
        command_repr = f"webbrowser.open({url!r})"

    elif intent == "browser_search":
        query = params.get("query", params.get("raw_text", ""))
        url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        action = f"Web search: {query}"
        command_repr = f"webbrowser.open({url!r})"

    elif intent == "system_info":
        action = "Display system resource information"
        command_repr = "psutil + platform info"

    elif intent == "kill_process":
        proc = params.get("process", params.get("raw_text", ""))
        action = f"Terminate process: {proc}"
        command_repr = f"taskkill /IM {proc}.exe /F" if platform.system() == "Windows" else f"pkill {proc}"

    elif intent == "volume_control":
        action = f"Adjust system volume"
        command_repr = f"nircmd.exe setsysvolume / pyautogui volume keys"

    elif intent == "screenshot":
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.expanduser(f"~/Desktop/screenshot_{ts}.png")
        action = "Capture screenshot"
        command_repr = f"pyautogui.screenshot -> {save_path}"

    elif intent == "git_op":
        raw = params.get("raw_text", prediction.get("text", ""))
        git_map = {
            "status": "git status",
            "pull": "git pull",
            "push": "git push",
            "log": "git log --oneline -10",
            "diff": "git diff",
        }
        cmd = next((v for k, v in git_map.items() if k in raw.lower()), f"git {raw}")
        branch = params.get("branch")
        msg = params.get("msg")
        if branch:
            cmd = f"git checkout -b {branch}" if "new" in raw.lower() else f"git checkout {branch}"
        if msg:
            cmd = f'git commit -m {msg}'
        action = f"Git: {cmd}"
        command_repr = cmd

    elif intent == "sleep_shutdown":
        raw = params.get("raw_text", prediction.get("text", "")).lower()
        if "shutdown" in raw or "turn off" in raw:
            cmd = "shutdown /s /t 30" if platform.system() == "Windows" else "shutdown -h now"
        elif "restart" in raw or "reboot" in raw:
            cmd = "shutdown /r /t 30" if platform.system() == "Windows" else "shutdown -r now"
        elif "sleep" in raw:
            cmd = "rundll32.exe powrprof.dll,SetSuspendState 0,1,0" if platform.system() == "Windows" else "pmset sleepnow"
        elif "lock" in raw:
            cmd = "rundll32.exe user32.dll,LockWorkStation" if platform.system() == "Windows" else "loginctl lock-session"
        else:
            cmd = raw
        action = f"System power: {cmd}"
        command_repr = cmd

    elif intent == "clipboard":
        action = "Clipboard operation"
        command_repr = "win32clipboard / pyperclip"

    else:
        action = f"Unknown intent handler: {intent}"
        command_repr = str(params)

    # ------------------------------------------------------------------
    # GATEKEEPER — mandatory, cannot be skipped
    # ------------------------------------------------------------------
    approved = _gatekeeper(prediction, action, command_repr)
    _log(intent, command_repr, approved)

    if not approved:
        return False

    # ------------------------------------------------------------------
    # Execute (only reached after explicit user approval)
    # ------------------------------------------------------------------
    print(_c("\n[+] Executing...", "green"))

    if intent == "sys_command":
        _shell(params.get("cmd", params.get("raw_text", "")))

    elif intent == "dev_server":
        _shell(command_repr)

    elif intent == "open_app":
        _shell(command_repr)

    elif intent in ("file_op_open", "dir_op"):
        _shell(command_repr)

    elif intent == "file_op_delete":
        _shell(command_repr)

    elif intent == "file_op_create":
        path = params.get("file", "")
        if path and not os.path.exists(path):
            open(path, "w").close()
            print(_c(f"[+] Created: {path}", "green"))
        else:
            print(_c(f"[!] File already exists or path missing: {path}", "yellow"))

    elif intent in ("browser_open", "browser_search"):
        url = params.get("url") or params.get("query", "")
        if intent == "browser_search":
            url = f"https://www.google.com/search?q={url.replace(' ', '+')}"
        elif not url.startswith("http"):
            url = "https://" + url
        webbrowser.open(url)
        print(_c(f"[+] Opened: {url}", "green"))

    elif intent == "system_info":
        _print_system_info()

    elif intent == "kill_process":
        _shell(command_repr)

    elif intent == "git_op":
        _shell(command_repr)

    elif intent == "sleep_shutdown":
        _shell(command_repr)

    elif intent == "screenshot":
        if HAS_PYAUTOGUI:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.expanduser(f"~/Desktop/screenshot_{ts}.png")
            img = pyautogui.screenshot()
            img.save(path)
            print(_c(f"[+] Screenshot saved: {path}", "green"))
        else:
            print(_c("[!] pyautogui not installed. Run: pip install pyautogui", "yellow"))

    else:
        print(_c(f"[!] No handler implemented yet for intent: {intent}", "yellow"))

    print(_c("\n[OK] Done.\n", "green"))
    return True


def _print_system_info() -> None:
    print(_c("\n  -- System Info --", "cyan"))
    print(f"  OS       : {platform.system()} {platform.release()} {platform.version()}")
    print(f"  Machine  : {platform.machine()}")
    if HAS_PSUTIL:
        cpu = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        print(f"  CPU      : {cpu}%")
        print(f"  RAM      : {mem.percent}% used ({mem.used // 1024**2}MB / {mem.total // 1024**2}MB)")
        print(f"  Disk     : {disk.percent}% used ({disk.used // 1024**3}GB / {disk.total // 1024**3}GB)")
    else:
        print("  (install psutil for full resource info)")
