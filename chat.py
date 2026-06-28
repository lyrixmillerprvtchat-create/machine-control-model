"""
Local Chat Engine — wraps Ollama for fully offline conversational AI.
Maintains persistent conversation history across sessions.
"""

import json
import os
import sys

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    def _c(text, color):
        colors = {"cyan": Fore.CYAN, "green": Fore.GREEN, "yellow": Fore.YELLOW,
                  "red": Fore.RED, "bold": Style.BRIGHT, "dim": Style.DIM}
        return colors.get(color, "") + text + Style.RESET_ALL
except ImportError:
    def _c(text, _): return text

HISTORY_PATH  = os.path.join(os.path.dirname(__file__), "data", "chat_history.json")
CONFIG_PATH   = os.path.join(os.path.dirname(__file__), "data", "chat_config.json")
OLLAMA_BASE   = "http://localhost:11434"
DEFAULT_MODEL = "phi3"
MAX_HISTORY   = 40   # messages (20 exchanges)

SYSTEM_PROMPT = (
    "You are JB, a smart local AI assistant running entirely on this machine "
    "with no internet connection for inference. You help with both computer control "
    "tasks (opening apps, running commands, managing files, git, screenshots, etc.) "
    "and general conversation. Be concise, direct, and helpful. "
    "When asked what you can do, mention you can control the computer AND chat. "
    "Never mention Ollama, Claude, or any underlying model — you are simply JB."
)


# ---------------------------------------------------------------------------
# Config (persists chosen model)
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"model": DEFAULT_MODEL}


def _save_config(cfg: dict) -> None:
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def get_model() -> str:
    return _load_config().get("model", DEFAULT_MODEL)


def set_model(name: str) -> None:
    cfg = _load_config()
    cfg["model"] = name
    _save_config(cfg)


# ---------------------------------------------------------------------------
# Ollama health + model discovery
# ---------------------------------------------------------------------------

def is_available() -> bool:
    if not HAS_REQUESTS:
        return False
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def list_models() -> list[str]:
    if not HAS_REQUESTS:
        return []
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=3)
        data = r.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def best_available_model() -> str:
    """Return the configured model if available, else first installed model."""
    models = list_models()
    if not models:
        return DEFAULT_MODEL
    configured = get_model()
    # Prefer exact match, then prefix match, then first available
    for m in models:
        if m == configured or m.startswith(configured.split(":")[0]):
            return m
    return models[0]


# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------

def load_history() -> list[dict]:
    if not os.path.exists(HISTORY_PATH):
        return []
    try:
        with open(HISTORY_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_history(history: list[dict]) -> None:
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    trimmed = history[-MAX_HISTORY:]
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(trimmed, f, indent=2)


def clear_history() -> None:
    if os.path.exists(HISTORY_PATH):
        os.remove(HISTORY_PATH)
    print(_c("  [Chat] Conversation history cleared.", "green"))


def history_summary() -> str:
    h = load_history()
    exchanges = len(h) // 2
    return f"{exchanges} exchange(s) in memory"


# ---------------------------------------------------------------------------
# Core chat call (streaming)
# ---------------------------------------------------------------------------

def chat(user_message: str) -> str:
    if not HAS_REQUESTS:
        return "[!] 'requests' not installed. Run: pip install requests"

    if not is_available():
        return (
            "[!] Ollama is not running.\n"
            "    Install: https://ollama.com/download\n"
            "    Then run: ollama pull llama3.1\n"
            "    Then run: ollama serve"
        )

    model = best_available_model()
    history = load_history()
    history.append({"role": "user", "content": user_message})

    payload = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + history,
        "stream": True,
    }

    print(_c(f"\n  JB ", "bold") + _c(f"({model})", "dim") + "  ", end="", flush=True)

    response_text = ""
    try:
        with requests.post(
            f"{OLLAMA_BASE}/api/chat",
            json=payload,
            stream=True,
            timeout=120,
        ) as resp:
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                try:
                    chunk = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                token = chunk.get("message", {}).get("content", "")
                if token:
                    response_text += token
                    print(token, end="", flush=True)
                if chunk.get("done"):
                    break

    except requests.exceptions.ConnectionError:
        print()
        return _c("[!] Lost connection to Ollama.", "red")
    except requests.exceptions.Timeout:
        print()
        return _c("[!] Ollama timed out.", "red")

    print("\n")

    if response_text:
        history.append({"role": "assistant", "content": response_text})
        save_history(history)

    return response_text


# ---------------------------------------------------------------------------
# Install guidance
# ---------------------------------------------------------------------------

INSTALL_GUIDE = """
  Ollama is not running. To enable chat:

  1. Download Ollama:   https://ollama.com/download  (Windows installer)
  2. Open a terminal and run:
       ollama pull llama3.1
  3. Ollama will auto-start on future boots. If not, run:
       ollama serve

  Recommended models (pick one based on your RAM):
    ollama pull phi3            # 3B  — needs ~4GB RAM  (default, fastest)
    ollama pull mistral         # 7B  — needs ~6GB RAM
    ollama pull llama3.1        # 8B  — needs ~8GB RAM

  Once installed, 'jb' will detect it automatically.
"""
