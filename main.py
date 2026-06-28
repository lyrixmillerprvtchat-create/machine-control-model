"""
JB — hybrid brain.
Commands → sklearn classifier → instant execution (no LLM needed).
Conversation → phi3 → streaming response.
"""

import os
import sys

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    def _c(t, c):
        m = {"cyan": Fore.CYAN, "green": Fore.GREEN, "yellow": Fore.YELLOW,
             "red": Fore.RED, "bold": Style.BRIGHT, "dim": Style.DIM}
        return m.get(c, "") + t + Style.RESET_ALL
except ImportError:
    def _c(t, _): return t

PROJECT  = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(PROJECT, "data", "execution.log")

# ── lazy model load ───────────────────────────────────────────────────────────
_clf = None

def _get_clf():
    global _clf
    if _clf is None:
        from matcher_model import MatcherModel
        _clf = MatcherModel()
    return _clf

# ── multi-task splitter ───────────────────────────────────────────────────────
import re

_SPLIT = re.compile(
    r'\s*(?:and\s+then|after\s+that|after\s+which|then|;|\n)\s*'
    r'|,\s*(?=(?:open|launch|start|run|search|go|take|kill|delete|create|show|'
    r'git|check|close|stop|push|pull|commit|navigate|make|get|list)\b)',
    re.IGNORECASE,
)

CHAT_INTENTS = {"chat"}
COMMAND_THRESHOLD = 0.52   # below this → treat as chat

def split_tasks(text: str) -> list[str]:
    parts = [p.strip() for p in _SPLIT.split(text)]
    return [p for p in parts if len(p) > 1]

# ── route one task ────────────────────────────────────────────────────────────
def route(task: str, registry=None) -> None:
    import memory as mem
    from executor import execute
    import agent

    # check memory alias first
    prediction = mem.get_alias(task)
    if prediction:
        print(_c(f"  [{prediction.get('_alias_name', 'alias')}]", "yellow"))
        execute(prediction, registry=registry)
        return

    # classifier — instant
    clf        = _get_clf()
    prediction = clf.predict(task)
    prediction = mem.fill_defaults(prediction)

    intent     = prediction.get("intent", "chat")
    confidence = prediction.get("confidence", 0.0)

    # low confidence or chat intent → phi3
    if intent in CHAT_INTENTS or confidence < COMMAND_THRESHOLD:
        agent.respond(task)
        return

    # command path — execute directly, no LLM
    ok = execute(prediction, registry=registry)
    if ok:
        mem.track_usage(intent, prediction.get("params", {}))

# ── handle a full user message (may be multi-task) ───────────────────────────
def handle(text: str, registry=None) -> None:
    tasks = split_tasks(text)

    if len(tasks) > 1:
        print(_c(f"\n  {len(tasks)} tasks:", "cyan"))
        for i, t in enumerate(tasks, 1):
            print(_c(f"    {i}. {t}", "dim"))
        print()

    for task in tasks:
        route(task, registry=registry)


# ── REPL ──────────────────────────────────────────────────────────────────────
def repl(registry=None):
    print(_c("\n  JB  ready.\n", "bold"))

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
            print(_c("\n  Just talk to JB naturally.", "bold"))
            print("  Examples:")
            print("    open spotify")
            print("    search for pytorch tutorials")
            print("    what is my CPU usage")
            print("    git pull then run npm install")
            print("    how do I reverse a string in Python")
            print()
            print("  Commands:  history | clear history | exit\n")
            continue

        if low == "history":
            if os.path.exists(LOG_PATH):
                lines = open(LOG_PATH, encoding="utf-8").readlines()
                for line in lines[-20:]:
                    print(f"  {line.rstrip()}")
            else:
                print("  No history yet.")
            print()
            continue

        if low == "clear history":
            import agent
            agent.clear_history()
            print(_c("  Cleared.", "green"))
            continue

        if low == "memories":
            import memory as mem
            d = mem.list_all()
            for section, items in d.items():
                if items:
                    print(_c(f"\n  {section.title()}:", "cyan"))
                    for k, v in items.items():
                        print(f"    {k}  ->  {v}")
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

        handle(raw, registry=registry)


# ── entry ─────────────────────────────────────────────────────────────────────
def main():
    args = sys.argv[1:]

    if "--setup" in args:
        print(_c("Generating dataset...", "cyan"))
        import generate_dataset; generate_dataset.main()
        print(_c("Training model...", "cyan"))
        from matcher_model import train; train()
        print(_c("Done.", "green"))
        return

    if "--train" in args:
        from matcher_model import train; train()
        return

    from registry import initialize
    registry = initialize(project_root=PROJECT, scan=False)

    # warm classifier now so first command is instant
    _get_clf()

    if args:
        handle(" ".join(args), registry=registry)
    else:
        repl(registry=registry)


if __name__ == "__main__":
    main()
