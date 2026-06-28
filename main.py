"""
JB — Machine Control Model
Type what you want done. JB figures it out and does it.
"""

import os
import re
import sys

PYTHON  = sys.executable
PROJECT = os.path.dirname(os.path.abspath(__file__))

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    def _c(t, c):
        m = {"cyan": Fore.CYAN, "green": Fore.GREEN, "yellow": Fore.YELLOW,
             "red": Fore.RED, "bold": Style.BRIGHT, "dim": Style.DIM}
        return m.get(c, "") + t + Style.RESET_ALL
except ImportError:
    def _c(t, _): return t

# ── multi-task parser ─────────────────────────────────────────────────────────
# Splits on: then / and then / after that / ; / , (before a verb) / \n
_SPLIT = re.compile(
    r'\s*(?:and\s+then|after\s+that|after\s+which|then|;|\n)\s*'
    r'|,\s*(?=(?:open|launch|start|run|search|go|take|kill|delete|create|show|'
    r'git|check|close|stop|navigate|make|get|list|push|pull|commit|grab|capture)\b)',
    re.IGNORECASE,
)

VERBS = re.compile(
    r'^(?:open|launch|start|run|search|go|take|kill|delete|create|show|git|'
    r'check|close|stop|navigate|make|get|list|push|pull|commit|grab|capture|'
    r'explain|tell|what|how|who|why|can|i|help)\b',
    re.IGNORECASE,
)

def parse_tasks(text: str) -> list[str]:
    parts = [p.strip() for p in _SPLIT.split(text)]
    return [p for p in parts if p and len(p) > 1]


# ── model (lazy, loaded once) ─────────────────────────────────────────────────
_model = None

def get_model():
    global _model
    if _model is None:
        from matcher_model import MatcherModel, MODEL_PATH
        if not os.path.exists(MODEL_PATH):
            print(_c("[!] No model found. Run:  jb --setup", "yellow"))
            sys.exit(1)
        _model = MatcherModel()
    return _model


# ── run a batch of tasks ──────────────────────────────────────────────────────
def run(text: str, registry=None):
    import memory
    import chat as chat_mod
    from executor import execute

    model  = get_model()
    tasks  = parse_tasks(text)

    if len(tasks) > 1:
        print(_c(f"\n  {len(tasks)} tasks detected:", "cyan"))
        for i, t in enumerate(tasks, 1):
            print(_c(f"  {i}. {t}", "dim"))
        print()

    for task in tasks:
        # memory alias shortcut
        prediction = memory.get_alias(task)
        if prediction:
            print(_c(f"  [{prediction['_alias_name']}]", "yellow"))
        else:
            prediction = model.predict(task)
            prediction = memory.fill_defaults(prediction)

        intent     = prediction.get("intent")
        confidence = prediction.get("confidence", 1.0)

        # low-confidence or chat intent → phi3
        if intent == "chat" or confidence < 0.40:
            chat_mod.chat(task)
            continue

        # show what we understood (one clean line)
        conf_color = "green" if confidence > 0.75 else "yellow"
        print(_c(f"  {intent}", "bold") + "  " + _c(f"{confidence:.0%}", conf_color) +
              _c(f"  {task}", "dim"))

        ok = execute(prediction, registry=registry)
        if ok:
            memory.track_usage(intent, prediction.get("params", {}))
        print()


# ── REPL ──────────────────────────────────────────────────────────────────────
def repl(registry=None):
    import memory

    print(_c("\n  JB  ready. Type what you want done. 'help' for commands.\n", "bold"))

    while True:
        try:
            raw = input(_c("jb> ", "cyan")).strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not raw:
            continue

        low = raw.lower()

        if low in ("exit", "quit", "q"):
            break

        if low == "help":
            print(_c("\n  Commands", "bold"))
            print("  help          this list")
            print("  memories      show saved aliases, notes, preferences")
            print("  history       last 20 executed commands")
            print("  tools         list registered project tools")
            print("  remember X as Y  save last command as alias Y")
            print("  forget Y      delete alias or note")
            print("  recall Y      look up a saved alias or note")
            print("  clear history wipe conversation history with phi3")
            print("  chat models   list installed Ollama models")
            print("  --setup       regenerate dataset and retrain model")
            print("  exit          quit\n")
            continue

        if low == "memories":
            import memory as mem
            d = mem.list_all()
            if d["aliases"]:
                print(_c(f"\n  Aliases:", "cyan"))
                for n, e in d["aliases"].items():
                    print(f"    {n}  ->  {e['intent']}  {e['params']}")
            if d["notes"]:
                print(_c(f"\n  Notes:", "cyan"))
                for n, v in d["notes"].items():
                    print(f"    {n}  =  {v}")
            if d["prefs"]:
                print(_c(f"\n  Learned:", "cyan"))
                for k, v in d["prefs"].items():
                    print(f"    {k}  ->  {v}")
            if not any(d.values()):
                print("  Nothing saved yet.")
            print()
            continue

        if low == "history":
            from executor import LOG_PATH
            if os.path.exists(LOG_PATH):
                lines = open(LOG_PATH, encoding="utf-8").readlines()
                for line in lines[-20:]:
                    print(f"  {line.rstrip()}")
            else:
                print("  No history yet.")
            print()
            continue

        if low == "tools":
            if registry:
                for t in registry.list_tools():
                    if not t["builtin"]:
                        print(f"  {t['name']:<45} {t['description'][:40]}")
            print()
            continue

        if low == "clear history":
            import chat as cm; cm.clear_history()
            continue

        if low == "chat models":
            import chat as cm
            for m in cm.list_models():
                print(f"  {m}")
            print()
            continue

        if low.startswith("remember "):
            import memory as mem
            result = mem.parse_remember(raw)
            if result:
                name, value = result
                mem.set_note(name, value)
                print(_c(f"  Saved: '{name}'", "green"))
            continue

        if low.startswith("forget "):
            import memory as mem
            name = mem.parse_forget(raw)
            if name and mem.forget(name):
                print(_c(f"  Removed: '{name}'", "green"))
            continue

        if low.startswith("recall "):
            import memory as mem
            name = low[7:].strip()
            d = mem.list_all()
            entry = d["aliases"].get(name) or d["notes"].get(name)
            print(f"  {entry}" if entry else _c(f"  Nothing for '{name}'", "yellow"))
            print()
            continue

        run(raw, registry=registry)


# ── entry point ───────────────────────────────────────────────────────────────
def main():
    args = sys.argv[1:]

    if "--setup" in args:
        print(_c("Generating dataset...", "cyan"))
        import generate_dataset; generate_dataset.main()
        print(_c("Training model...", "cyan"))
        from matcher_model import train; train()
        print(_c("Done. Run: jb", "green"))
        return

    if "--train" in args:
        from matcher_model import train; train()
        return

    if "--scan" in args:
        idx  = args.index("--scan")
        root = args[idx + 1] if idx + 1 < len(args) else os.getcwd()
        from registry import initialize
        initialize(project_root=root, scan=True)
        return

    # lazy-load registry only once
    from registry import initialize
    registry = initialize(project_root=PROJECT, scan=False)

    # warm model in background so first command is instant
    get_model()

    if args:
        run(" ".join(args), registry=registry)
    else:
        repl(registry=registry)


if __name__ == "__main__":
    main()
