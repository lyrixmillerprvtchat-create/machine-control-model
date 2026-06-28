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
 |_|  |_|\____|_|  |_|   Machine Control Model v1.1
"""

# Separators that signal a chained multi-step command
_CHAIN_RE = re.compile(
    r'\s*(?:then|and then|after that|after which|;)\s*',
    re.IGNORECASE,
)


def parse_chain(text: str) -> list[str]:
    """Split a compound command into individual steps."""
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


def run_query(model, query: str, registry=None, ask_correction: bool = True) -> None:
    """Predict, execute, then optionally ask for correction feedback."""
    from executor import execute
    import corrections

    steps = parse_chain(query)

    if len(steps) > 1:
        print(_c(f"\n[Chain] {len(steps)} steps detected.", "cyan"))

    for i, step in enumerate(steps, 1):
        if len(steps) > 1:
            print(_c(f"\n--- Step {i}/{len(steps)}: {step!r} ---", "bold"))

        prediction = model.predict(step)
        approved = execute(prediction, registry=registry)

        # Correction prompt — only in REPL (not single-shot CLI mode)
        if ask_correction:
            corrections.prompt_correction(step, prediction["intent"])


def repl(model, registry=None):
    print(_c(BANNER, "cyan"))
    print(_c("  Type in plain English. Chain steps with 'then'. Type 'help' for commands.\n", "bold"))

    while True:
        try:
            query = input(_c("MCM > ", "cyan")).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not query:
            continue

        low = query.lower()

        if low in ("exit", "quit", "q"):
            print("Goodbye.")
            break

        if low == "help":
            print(_c("\n  Commands:", "bold"))
            print("    tools          — list all registered tools")
            print("    intents        — list all built-in intent names")
            print("    history        — show execution log")
            print("    corrections    — show saved corrections")
            print("    retrain        — retrain model with current corrections")
            print("    exit / quit    — exit\n")
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
            # Reload the model in-place after retraining
            model = require_model()
            continue

        run_query(model, query, registry=registry, ask_correction=True)


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
