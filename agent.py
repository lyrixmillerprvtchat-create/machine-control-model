"""
JB Agent — phi3 is the brain.
Reads the user's message, decides what to do, executes tools, responds naturally.
"""

import json
import os
import re
import subprocess
import webbrowser
import platform
import datetime

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from colorama import Fore, Style
    def _c(t, c):
        m = {"cyan": Fore.CYAN, "green": Fore.GREEN, "yellow": Fore.YELLOW,
             "red": Fore.RED, "bold": Style.BRIGHT, "dim": Style.DIM, "white": Fore.WHITE}
        return m.get(c, "") + t + Style.RESET_ALL
except ImportError:
    def _c(t, _): return t

OLLAMA_URL    = "http://localhost:11434/api/chat"
HISTORY_PATH  = os.path.join(os.path.dirname(__file__), "data", "chat_history.json")
MODEL_PATH    = os.path.join(os.path.dirname(__file__), "data", "chat_config.json")
LOG_PATH      = os.path.join(os.path.dirname(__file__), "data", "execution.log")
MAX_HISTORY   = 40
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# ── system prompt ─────────────────────────────────────────────────────────────
SYSTEM = """You are JB, a local AI assistant that controls this Windows PC and holds conversations.

For computer actions output JSON tool calls (one per line), then one short sentence:
{"tool": "open_app", "params": {"app": "NAME"}}
{"tool": "browser_search", "params": {"query": "TERMS"}}
{"tool": "browser_open", "params": {"url": "URL"}}
{"tool": "run_command", "params": {"cmd": "CMD"}}
{"tool": "git", "params": {"cmd": "git status"}}
{"tool": "system_info", "params": {}}
{"tool": "screenshot", "params": {}}
{"tool": "file_open", "params": {"file": "FILE"}}
{"tool": "file_create", "params": {"file": "FILE"}}
{"tool": "file_delete", "params": {"file": "FILE"}}
{"tool": "kill_process", "params": {"process": "NAME"}}
{"tool": "dev_server", "params": {"port": "3000"}}
{"tool": "shutdown", "params": {"action": "shutdown|restart|sleep|lock"}}

For questions or chat, just reply normally — no tool calls. Be concise. You are JB."""

# ── tool call extraction ──────────────────────────────────────────────────────
_TOOL_RE   = re.compile(r'\{[^{}]*"tool"\s*:[^{}]*\}', re.DOTALL)
_BLOCK_RE  = re.compile(r'```(?:json)?\s*(.*?)\s*```', re.DOTALL)

def _parse_obj(s: str) -> dict | None:
    try:
        obj = json.loads(s.strip())
        if isinstance(obj, dict) and "tool" in obj:
            return obj
    except json.JSONDecodeError:
        pass
    return None

def extract_tools(text: str) -> list[dict]:
    calls = []
    seen  = set()

    # pull from ```json ... ``` blocks first
    for block in _BLOCK_RE.finditer(text):
        content = block.group(1)
        # may contain multiple JSON objects
        for m in _TOOL_RE.finditer(content):
            obj = _parse_obj(m.group())
            if obj:
                key = json.dumps(obj, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    calls.append(obj)
        # or the whole block might be one JSON
        if not calls:
            obj = _parse_obj(content)
            if obj:
                key = json.dumps(obj, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    calls.append(obj)

    # also scan raw text outside code blocks
    for m in _TOOL_RE.finditer(text):
        obj = _parse_obj(m.group())
        if obj:
            key = json.dumps(obj, sort_keys=True)
            if key not in seen:
                seen.add(key)
                calls.append(obj)

    return calls

def clean_text(text: str) -> str:
    """Strip tool call JSON and code fences from display text."""
    t = _BLOCK_RE.sub("", text)
    t = _TOOL_RE.sub("", t)
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t.strip()

# ── conversation history ──────────────────────────────────────────────────────
def load_history() -> list:
    if not os.path.exists(HISTORY_PATH):
        return []
    try:
        with open(HISTORY_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_history(h: list):
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(h[-MAX_HISTORY:], f, indent=2)

def clear_history():
    if os.path.exists(HISTORY_PATH):
        os.remove(HISTORY_PATH)

# ── ollama call ───────────────────────────────────────────────────────────────
def _get_model() -> str:
    if os.path.exists(MODEL_PATH):
        try:
            return json.load(open(MODEL_PATH))["model"]
        except Exception:
            pass
    return "phi3"

def _call(messages: list) -> str:
    """Non-streaming — used when we need the full response before acting."""
    model = _get_model()
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": model, "messages": messages, "stream": False},
            timeout=90,
        )
        return resp.json().get("message", {}).get("content", "")
    except requests.exceptions.ConnectionError:
        return "[!] Ollama not running — type: ollama serve"
    except Exception as e:
        return f"[!] {e}"

def _stream(messages: list) -> str:
    """Stream tokens to terminal as they arrive. Returns full collected reply."""
    model = _get_model()
    full  = []
    try:
        with requests.post(
            OLLAMA_URL,
            json={"model": model, "messages": messages, "stream": True},
            stream=True,
            timeout=90,
        ) as resp:
            print(_c("  JB: ", "cyan"), end="", flush=True)
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = chunk.get("message", {}).get("content", "")
                if token:
                    print(token, end="", flush=True)
                    full.append(token)
                if chunk.get("done"):
                    break
            print()
        return "".join(full)
    except requests.exceptions.ConnectionError:
        print(_c("\n  [!] Ollama not running — type: ollama serve", "red"))
        return ""
    except Exception as e:
        print(_c(f"\n  [!] {e}", "red"))
        return ""

# ── tool executor ─────────────────────────────────────────────────────────────
DANGER = {
    "file_delete": "MEDIUM",
    "shutdown":    "HIGH",
    "kill_process": "MEDIUM",
}

def _gate(action: str) -> bool:
    try:
        ans = input(_c(f"  {action} — run it? [y/N] > ", "yellow")).strip().lower()
        return ans == "y"
    except (KeyboardInterrupt, EOFError):
        return False

def _shell(cmd: str) -> str:
    print(_c(f"  $ {cmd}", "dim"))
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    out = (r.stdout + r.stderr).strip()
    if out:
        for line in out.splitlines()[:20]:
            print(f"    {line}")
    return out

def run_tool(tool: str, params: dict) -> bool:
    danger = DANGER.get(tool)
    is_win = platform.system() == "Windows"

    if danger == "HIGH":
        if not _gate(f"[{danger}] {tool}({params})"):
            return False
    elif danger == "MEDIUM":
        print(_c(f"  [{danger}] {tool}", "yellow"))

    _log(tool, params)

    if tool == "open_app":
        app = params.get("app", "")
        _shell(f"start {app}" if is_win else f"open -a '{app}'")

    elif tool == "browser_search":
        q = params.get("query", "")
        webbrowser.open(f"https://www.google.com/search?q={q.replace(' ', '+')}")
        print(_c(f"  Searching: {q}", "green"))

    elif tool == "browser_open":
        url = params.get("url", "")
        if not url.startswith("http"):
            url = "https://" + url
        webbrowser.open(url)
        print(_c(f"  Opened: {url}", "green"))

    elif tool == "run_command":
        _shell(params.get("cmd", ""))

    elif tool == "git":
        _shell(params.get("cmd", "git status"))

    elif tool == "system_info":
        print(_c(f"  OS   : {platform.system()} {platform.release()}", "white"))
        if HAS_PSUTIL:
            cpu  = psutil.cpu_percent(interval=0.3)
            mem  = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            print(_c(f"  CPU  : {cpu:.1f}%", "white"))
            print(_c(f"  RAM  : {mem.percent:.1f}%  ({mem.used//1024**2}MB / {mem.total//1024**2}MB)", "white"))
            print(_c(f"  Disk : {disk.percent:.1f}%  ({disk.used//1024**3}GB / {disk.total//1024**3}GB)", "white"))

    elif tool == "screenshot":
        try:
            import pyautogui
            ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.expanduser(f"~/Desktop/screenshot_{ts}.png")
            pyautogui.screenshot().save(path)
            print(_c(f"  Saved: {path}", "green"))
        except ImportError:
            print(_c("  Run: pip install pyautogui", "yellow"))

    elif tool == "file_open":
        f = params.get("file", "")
        _shell(f"start {f}" if is_win else f"open '{f}'")

    elif tool == "file_create":
        f = params.get("file", "")
        if f:
            open(f, "w").close()
            print(_c(f"  Created: {f}", "green"))

    elif tool == "file_delete":
        f = params.get("file", "")
        _shell(f"del {f}" if is_win else f"rm '{f}'")

    elif tool == "kill_process":
        proc = params.get("process", "")
        _shell(f"taskkill /IM {proc}.exe /F" if is_win else f"pkill {proc}")

    elif tool == "dev_server":
        port = params.get("port", "3000")
        _shell(f"npm run dev -- --port {port}")

    elif tool == "volume":
        action = params.get("action", "")
        print(_c(f"  Volume: {action} (install pycaw for full control)", "yellow"))

    elif tool == "shutdown":
        action = params.get("action", "shutdown").lower()
        cmds = {
            "shutdown": "shutdown /s /t 10",
            "restart":  "shutdown /r /t 10",
            "sleep":    "rundll32.exe powrprof.dll,SetSuspendState 0,1,0",
            "lock":     "rundll32.exe user32.dll,LockWorkStation",
        }
        _shell(cmds.get(action, f"shutdown /s /t 10"))

    return True

def _log(tool: str, params: dict):
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {tool} {params}\n")

# ── main agent call ───────────────────────────────────────────────────────────
def respond(user_message: str) -> None:
    history = load_history()
    history.append({"role": "user", "content": user_message})
    messages = [{"role": "system", "content": SYSTEM}] + history[-20:]

    # Use streaming so the user sees tokens as they arrive
    reply = _stream(messages)

    if not reply:
        return

    # After streaming, check if phi3 embedded tool calls
    tools  = extract_tools(reply)
    spoken = clean_text(reply)

    if tools:
        # Erase the streamed line — we'll show clean tool output instead
        print("\033[A\033[2K", end="")
        for call in tools:
            run_tool(call["tool"], call.get("params", {}))
        if spoken:
            print(_c(f"  {spoken}", "white"))
    # else: the streamed text IS the response — already printed

    print()
    history.append({"role": "assistant", "content": reply})
    save_history(history)
