"""
Machine Control Model — Main Entry Point
Ties all 4 phases into a single interactive CLI.

Usage:
  python main.py --setup          # Phase 1+2: generate dataset & train model
  python main.py                  # Interactive REPL mode
  python main.py "open chrome"    # Single command mode
  python main.py --scan /path     # Phase 3: scan a project directory
"""

import os
import sys

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    def _c(text, color):
        colors = {"cyan": Fore.CYAN, "green": Fore.GREEN, "yellow": Fore.YELLOW, "red": Fore.RED, "bold": Style.BRIGHT}
        return colors.get(color, "") + text + Style.RESET_ALL
except ImportError:
    def _c(text, _color): return text

BANNER = r"""
  __  __  ____  __  __
 |  \/  |/ ___|  \/  |
 | |\/| | |   | |\/| |
 | |  | | |___| |  | |
 |_|  |_|\____|_|  |_|   Machine Control Model v1.0
"""


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


def run_query(model, query: str, registry=None) -> bool:
    from executor import execute
    prediction = model.predict(query)
    return execute(prediction)


def repl(model, registry=None):
    print(_c(BANNER, "cyan"))
    print(_c("  Type a command in plain English. Type 'exit' to quit.\n", "bold"))

    while True:
        try:
            query = input(_c("MCM > ", "cyan")).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("Goodbye.")
            break
        if query.lower() == "tools":
            if registry:
                for t in registry.list_tools():
                    flag = "(builtin)" if t["builtin"] else "(project)"
                    print(f"  {flag:<12} {t['name']:<35} {t['description'][:40]}")
            continue

        run_query(model, query, registry)


def main():
    args = sys.argv[1:]

    if "--setup" in args:
        setup()
        return

    if "--scan" in args:
        idx = args.index("--scan")
        root = args[idx + 1] if idx + 1 < len(args) else os.getcwd()
        from registry import initialize
        reg = initialize(project_root=root, scan=True)
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
        run_query(model, query, registry)
    else:
        repl(model, registry)


if __name__ == "__main__":
    main()
