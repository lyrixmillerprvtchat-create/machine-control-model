"""
Phase 3: Dynamic Tool Registry
Scans the host project directory and auto-registers callable scripts and functions
as tools the local model can route to.
"""

import os
import ast
import json
import importlib.util
from typing import Callable, Optional

REGISTRY_CACHE = os.path.join(os.path.dirname(__file__), "data", "registry.json")

# Built-in intents that always exist regardless of project scan
BUILTIN_INTENTS: dict[str, dict] = {
    "sys_command":    {"description": "Run an arbitrary shell command", "builtin": True},
    "dev_server":     {"description": "Start a local development server", "builtin": True},
    "open_app":       {"description": "Open an application by name", "builtin": True},
    "file_op_open":   {"description": "Open or read a file", "builtin": True},
    "file_op_delete": {"description": "Delete a file", "builtin": True},
    "file_op_create": {"description": "Create a new file", "builtin": True},
    "dir_op":         {"description": "Directory navigation or creation", "builtin": True},
    "browser_open":   {"description": "Open a URL in the default browser", "builtin": True},
    "browser_search": {"description": "Search the web for a query", "builtin": True},
    "system_info":    {"description": "Display system resource information", "builtin": True},
    "kill_process":   {"description": "Terminate a running process by name", "builtin": True},
    "volume_control": {"description": "Adjust system audio volume", "builtin": True},
    "screenshot":     {"description": "Capture a screenshot of the desktop", "builtin": True},
    "clipboard":      {"description": "Read or write the system clipboard", "builtin": True},
    "git_op":         {"description": "Execute a git operation", "builtin": True},
    "sleep_shutdown": {"description": "Sleep, shutdown, restart, or lock the system", "builtin": True},
}


class ToolRegistry:
    def __init__(self, project_root: Optional[str] = None):
        self._tools: dict[str, dict] = dict(BUILTIN_INTENTS)
        self._handlers: dict[str, Callable] = {}
        self._project_root = project_root

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: str, handler: Callable, description: str = "") -> None:
        self._tools[name] = {"description": description, "builtin": False, "handler": True}
        self._handlers[name] = handler

    def has_handler(self, intent: str) -> bool:
        return intent in self._handlers

    def get_handler(self, intent: str) -> Optional[Callable]:
        return self._handlers.get(intent)

    def list_tools(self) -> list[dict]:
        return [
            {"name": k, "description": v["description"], "builtin": v.get("builtin", False)}
            for k, v in self._tools.items()
        ]

    # ------------------------------------------------------------------
    # Dynamic project scan
    # ------------------------------------------------------------------

    def scan_project(self, root: str) -> int:
        """
        Walk `root`, parse Python files, and auto-register any top-level
        function whose name doesn't start with _ as a discovered tool.
        Returns count of newly discovered tools.
        """
        discovered = 0
        for dirpath, _dirs, filenames in os.walk(root):
            _dirs[:] = [d for d in _dirs if d not in {"__pycache__", ".git", "node_modules", ".venv", "venv"}]
            for fname in filenames:
                if not fname.endswith(".py"):
                    continue
                fpath = os.path.join(dirpath, fname)
                funcs = self._extract_functions(fpath)
                for func_name, docstring in funcs:
                    tool_name = f"project::{fname[:-3]}::{func_name}"
                    self._tools[tool_name] = {
                        "description": docstring or f"Function {func_name} in {fname}",
                        "builtin": False,
                        "source": fpath,
                    }
                    discovered += 1
        return discovered

    @staticmethod
    def _extract_functions(filepath: str) -> list[tuple[str, str]]:
        results = []
        try:
            with open(filepath, encoding="utf-8", errors="ignore") as f:
                source = f.read()
            tree = ast.parse(source, filename=filepath)
        except (SyntaxError, OSError):
            return results
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                doc = ast.get_docstring(node) or ""
                results.append((node.name, doc.strip().split("\n")[0]))
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str = REGISTRY_CACHE) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        serializable = {
            k: {kk: vv for kk, vv in v.items() if kk != "handler"}
            for k, v in self._tools.items()
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

    def load(self, path: str = REGISTRY_CACHE) -> None:
        if not os.path.exists(path):
            return
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)
        for k, v in saved.items():
            if k not in self._tools:
                self._tools[k] = v

    def __repr__(self) -> str:
        return f"<ToolRegistry tools={len(self._tools)}>"


# ---------------------------------------------------------------------------
# Module-level singleton + initializer called by main.py
# ---------------------------------------------------------------------------

_registry: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


def initialize(project_root: Optional[str] = None, scan: bool = True) -> ToolRegistry:
    global _registry
    _registry = ToolRegistry(project_root)
    _registry.load()

    if scan and project_root and os.path.isdir(project_root):
        count = _registry.scan_project(project_root)
        if count:
            print(f"[+] Registry: discovered {count} project functions in '{project_root}'")
        _registry.save()

    print(f"[+] Registry: {len(_registry.list_tools())} total tools loaded")
    return _registry


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(__file__)
    reg = initialize(project_root=root, scan=True)
    print("\nRegistered tools:")
    for tool in reg.list_tools():
        flag = "(builtin)" if tool["builtin"] else "(project)"
        print(f"  [{flag}] {tool['name']}: {tool['description'][:60]}")
