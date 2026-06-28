"""
Machine Control Model — Main Entry Point

Usage:
  python main.py --setup          # Phase 1+2: generate dataset & train model
  python main.py                  # Interactive REPL
  python main.py "open chrome"    # Single command
  python main.py --scan /path     # Scan a project directory and register its tools
  python main.py --train          # Retrain only (dataset must already exist)
"""

import os
import re
import sys

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    def _c(text, color):
        colors = {
            "cyan": Fore.CYAN, "green": Fore.GREEN,
            "yellow": Fore.YELLOW, "red": Fore.RED, "bold": Style.BRIGHT,
        }
        return colors.get(color, "") + text + Style.RESET_ALL
except ImportError:
    def _c(text, _color): return text

BANNER = r"""
  __  __  ____  __  __
 |  \/  |/ ___|  \/  |
 | |\/| | |   | |\/| |
 | |  | | |___| |  | |
 |_|  |_|\____|_|  |_|   Machine Control Model v1.2
"""

_CHAIN_RE = re.compile(
    r'\s*(?:then|and then|after that|after which|;)\s*',
    re.IGNORECASE,
)


def parse_chain(text: str) -> list[str]:
    parts = [p.strip() for p in _CHAIN_RE.split(text)]
    return [p for p in parts if p]


def require_model():
    from matcher_model import MatcherModel, MODEL_PATH
    if not os.path.exists(MODEL_PATH):
        print(_c("[!] Model not found. Run: python main.py --setup", "yellow"))
        sys.exit(1)
    return MatcherModel()


def setup():
    print(_c("\n[Phase 1] Generating training dataset...", "cyan"))
    import generate_dataset
    generate_dataset.main()
    print(_c("\n[Phase 2] Training model...", "cyan"))
    from matcher_model import train
    train()
    print(_c("\n[+] Setup complete. Run `python main.py` to start.", "green"))


# ---------------------------------------------------------------------------
# Core query runner
# ---------------------------------------------------------------------------

def run_query(model, query: str, registry=None, ask_correction: bool = True) -> None:
    from executor import execute
    import corrections
    import memory
    import chat as chat_module

    # Confidence threshold below which we fall back to chat
    CHAT_FALLBACK_THRESHOLD = 0.40

    steps = parse_chain(query)
    if len(steps) > 1:
        print(_c(f"\n[Chain] {len(steps)} steps detected.", "cyan"))

    for i, step in enumerate(steps, 1):
        if len(steps) > 1:
            print(_c(f"\n--- Step {i}/{len(steps)}: {step!r} ---", "bold"))

        # 1. Check memory alias first — skip model entirely
        prediction = memory.get_alias(step)

        if prediction:
            print(_c(f"  [Memory] Alias matched: '{prediction['_alias_name']}'", "yellow"))
        else:
            # 2. Normal model prediction
            prediction = model.predict(step)
            # 3. Fill any missing params from learned preferences
            prediction = memory.fill_defaults(prediction)

        intent = prediction.get("intent")
        confidence = prediction.get("confidence", 1.0)

        # 4. Route to chat if: intent is chat, OR confidence is too low to act on
        if intent == "chat" or confidence < CHAT_FALLBACK_THRESHOLD:
            chat_module.chat(step)
            # No correction prompt for chat — it's conversational
            continue

        approved = execute(prediction, registry=registry)

        # 5. Track usage for auto-learning after every approval
        if approved:
            memory.track_usage(intent, prediction.get("params", {}))

        # 6. Correction prompt (REPL only)
        if ask_correction:
            corrections.prompt_correction(step, intent)


# ---------------------------------------------------------------------------
# Memory command handlers (called from REPL before routing to model)
# ---------------------------------------------------------------------------

def _handle_remember(text: str, model, last_prediction: dict) -> bool:
    """
    Handle 'remember X as Y' — saves the most recent prediction under name Y,
    OR saves a free-text note if no recent prediction is available.
    Returns True if handled.
    """
    import memory
    result = memory.parse_remember(text)
    if result is None:
        return False

    name, value = result

    # If value looks like a plain phrase (not a known intent), save as note
    if last_prediction and last_prediction.get("intent"):
        memory.set_alias(name, last_prediction["intent"], last_prediction.get("params", {}), value)
        print(_c(f"  [Memory] Saved alias '{name}' -> intent={last_prediction['intent']}", "green"))
    else:
        memory.set_note(name, value)
        print(_c(f"  [Memory] Saved note '{name}' = '{value}'", "green"))

    return True


def _handle_forget(text: str) -> bool:
    import memory
    name = memory.parse_forget(text)
    if name is None:
        return False
    if memory.forget(name):
        print(_c(f"  [Memory] Forgot '{name}'", "green"))
    else:
        print(_c(f"  [Memory] Nothing found for '{name}'", "yellow"))
    return True


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

def repl(model, registry=None):
    import memory
    import chat as chat_module

    print(_c(BANNER, "cyan"))

    mem_summary = memory.summary()
    print(_c(f"  Memory loaded: {mem_summary}", "yellow"))

    if chat_module.is_available():
        active_model = chat_module.best_available_model()
        hist_summary = chat_module.history_summary()
        print(_c(f"  Chat online:   {active_model}  |  {hist_summary}", "green"))
    else:
        print(_c("  Chat offline:  Ollama not running (type 'chat setup' for instructions)", "red"))

    print(_c("  Type in plain English. Chain steps with 'then'. Type 'help' for commands.\n", "bold"))

    last_prediction: dict = {}

    while True:
        try:
            query = input(_c("MCM > ", "cyan")).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not query:
            continue

        low = query.lower()

        # ── built-in REPL commands ─────────────────────────────────────────

        if low in ("exit", "quit", "q"):
            print("Goodbye.")
            break

        if low == "help":
            print(_c("\n  Commands:", "bold"))
            print("    tools                    list all registered tools")
            print("    intents                  list all built-in intent names")
            print("    history                  show last 20 execution log entries")
            print("    memories                 show everything in memory")
            print("    corrections              show saved corrections")
            print("    retrain                  retrain model with current corrections")
            print("    remember X as Y          save last command as alias Y")
            print("    remember Y = <note>      save a free-text note")
            print("    forget Y                 delete alias or note Y")
            print("    recall Y                 look up what Y means")
            print("    clear history            wipe conversation history")
            print("    chat model <name>        switch local chat model")
            print("    chat models              list installed Ollama models")
            print("    chat setup               show Ollama install instructions")
            print("    exit / quit              exit\n")
            continue

        if low == "tools":
            if registry:
                builtins = [t for t in registry.list_tools() if t["builtin"]]
                projects = [t for t in registry.list_tools() if not t["builtin"]]
                print(_c(f"\n  Built-in ({len(builtins)}):", "cyan"))
                for t in builtins:
                    print(f"    {t['name']:<30} {t['description'][:45]}")
                if projects:
                    print(_c(f"\n  Project tools ({len(projects)}):", "cyan"))
                    for t in projects:
                        print(f"    {t['name']:<45} {t['description'][:35]}")
            print()
            continue

        if low == "intents":
            import corrections as corr
            print(_c("\n  Available intents:", "cyan"))
            for i, intent in enumerate(corr.ALL_INTENTS, 1):
                print(f"    {i:>2}. {intent}")
            print()
            continue

        if low == "history":
            from executor import LOG_PATH
            if os.path.exists(LOG_PATH):
                with open(LOG_PATH, encoding="utf-8") as f:
                    lines = f.readlines()
                print(_c(f"\n  Last {min(20, len(lines))} entries:", "cyan"))
                for line in lines[-20:]:
                    print(f"    {line.rstrip()}")
            else:
                print("  No history yet.")
            print()
            continue

        if low == "memories":
            data = memory.list_all()
            aliases = data["aliases"]
            notes = data["notes"]
            prefs = data["prefs"]

            if aliases:
                print(_c(f"\n  Aliases ({len(aliases)}):", "cyan"))
                for name, entry in aliases.items():
                    print(f"    '{name}'  ->  intent={entry['intent']}  params={entry['params']}")
            if notes:
                print(_c(f"\n  Notes ({len(notes)}):", "cyan"))
                for name, val in notes.items():
                    print(f"    '{name}'  =  '{val}'")
            if prefs:
                print(_c(f"\n  Learned preferences ({len(prefs)}):", "cyan"))
                for key, val in prefs.items():
                    print(f"    {key}  ->  '{val}'")
            if not aliases and not notes and not prefs:
                print("  Memory is empty. Run commands and say 'remember X as Y'.")
            print()
            continue

        if low.startswith("recall "):
            name = low[7:].strip()
            data = memory.list_all()
            if name in data["aliases"]:
                e = data["aliases"][name]
                print(_c(f"\n  '{name}'  ->  intent={e['intent']}  params={e['params']}\n", "cyan"))
            elif name in data["notes"]:
                print(_c(f"\n  '{name}'  =  '{data['notes'][name]}'\n", "cyan"))
            else:
                print(_c(f"  Nothing found for '{name}'\n", "yellow"))
            continue

        if low == "corrections":
            import corrections as corr
            data = corr.load()
            if data:
                print(_c(f"\n  {len(data)} corrections saved:", "cyan"))
                for c in data:
                    print(f"    '{c['text']}'  {c['wrong_intent']} -> {c['correct_intent']}")
            else:
                print("  No corrections saved yet.")
            print()
            continue

        if low == "retrain":
            import corrections as corr
            corr.apply_and_retrain()
            model = require_model()
            continue

        if low == "clear history":
            chat_module.clear_history()
            continue

        if low == "chat setup":
            print(chat_module.INSTALL_GUIDE)
            continue

        if low == "chat models":
            models = chat_module.list_models()
            if models:
                current = chat_module.get_model()
                print(_c(f"\n  Installed Ollama models:", "cyan"))
                for m in models:
                    marker = " <-- active" if m.startswith(current.split(":")[0]) else ""
                    print(f"    {m}{marker}")
            else:
                print(_c("  No models found. Is Ollama running?", "yellow"))
            print()
            continue

        if low.startswith("chat model "):
            name = query[11:].strip()
            chat_module.set_model(name)
            print(_c(f"  [Chat] Model set to '{name}'", "green"))
            continue

        # ── memory shortcut commands ───────────────────────────────────────

        if _handle_forget(query):
            continue

        if _handle_remember(query, model, last_prediction):
            continue

        # ── normal command routing ─────────────────────────────────────────

        # Capture prediction for 'remember last command as X'
        steps = parse_chain(query)
        if len(steps) == 1:
            alias_hit = memory.get_alias(query)
            if alias_hit:
                last_prediction = alias_hit
            else:
                last_prediction = model.predict(query)

        run_query(model, query, registry=registry, ask_correction=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]

    if "--setup" in args:
        setup()
        return

    if "--scan" in args:
        idx = args.index("--scan")
        root = args[idx + 1] if idx + 1 < len(args) else os.getcwd()
        from registry import initialize
        initialize(project_root=root, scan=True)
        return

    if "--train" in args:
        from matcher_model import train
        train()
        return

    from registry import initialize
    registry = initialize(
        project_root=os.path.dirname(os.path.abspath(__file__)),
        scan=False,
    )

    model = require_model()

    if args:
        query = " ".join(args)
        run_query(model, query, registry=registry, ask_correction=False)
    else:
        repl(model, registry)


if __name__ == "__main__":
    main()
