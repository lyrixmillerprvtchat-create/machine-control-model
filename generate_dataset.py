"""
Phase 1: Synthetic Training Dataset Generator
Produces thousands of labeled text-to-intent examples covering all supported system operations.
"""

import json
import random
import os
from itertools import product

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "generated", "dataset.json")

# ---------------------------------------------------------------------------
# Template bank: (text_template, intent, param_extractor_hint)
# {port}, {app}, {path}, {url}, {query}, {file}, {dir}, {process} are slot vars
# ---------------------------------------------------------------------------

TEMPLATES = {
    "sys_command": [
        ("run {cmd}", {"cmd": None}),
        ("execute {cmd}", {"cmd": None}),
        ("run the command {cmd}", {"cmd": None}),
        ("shell {cmd}", {"cmd": None}),
        ("terminal: {cmd}", {"cmd": None}),
    ],
    "dev_server": [
        ("start dev server", {}),
        ("start the dev server", {}),
        ("launch dev server on port {port}", {"port": None}),
        ("open my dev server on port {port}", {"port": None}),
        ("spin up localhost on {port}", {"port": None}),
        ("npm run dev", {}),
        ("start next.js server", {}),
        ("run development server", {}),
        ("fire up the server", {}),
        ("boot the backend", {}),
        ("start the api server", {}),
        ("launch flask on {port}", {"port": None}),
        ("uvicorn on port {port}", {"port": None}),
        ("start react app", {}),
        ("vite dev", {}),
    ],
    "open_app": [
        ("open {app}", {"app": None}),
        ("launch {app}", {"app": None}),
        ("start {app}", {"app": None}),
        ("open up {app}", {"app": None}),
        ("pull up {app}", {"app": None}),
        ("run {app}", {"app": None}),
        ("bring up {app}", {"app": None}),
        ("fire up {app}", {"app": None}),
        ("get {app} open", {"app": None}),
        ("i need {app}", {"app": None}),
    ],
    "file_op_open": [
        ("open the file {file}", {"file": None}),
        ("open {file}", {"file": None}),
        ("read {file}", {"file": None}),
        ("show me {file}", {"file": None}),
        ("display {file}", {"file": None}),
        ("view {file}", {"file": None}),
        ("cat {file}", {"file": None}),
        ("load {file}", {"file": None}),
    ],
    "file_op_delete": [
        ("delete {file}", {"file": None}),
        ("remove {file}", {"file": None}),
        ("trash {file}", {"file": None}),
        ("erase {file}", {"file": None}),
        ("get rid of {file}", {"file": None}),
        ("delete the file {file}", {"file": None}),
    ],
    "file_op_create": [
        ("create a file called {file}", {"file": None}),
        ("make a new file {file}", {"file": None}),
        ("touch {file}", {"file": None}),
        ("new file {file}", {"file": None}),
        ("create {file}", {"file": None}),
    ],
    "dir_op": [
        ("open folder {dir}", {"dir": None}),
        ("navigate to {dir}", {"dir": None}),
        ("go to directory {dir}", {"dir": None}),
        ("cd into {dir}", {"dir": None}),
        ("change directory to {dir}", {"dir": None}),
        ("list files in {dir}", {"dir": None}),
        ("show contents of {dir}", {"dir": None}),
        ("make a folder called {dir}", {"dir": None}),
        ("create directory {dir}", {"dir": None}),
        ("mkdir {dir}", {"dir": None}),
    ],
    "browser_open": [
        ("open the browser", {}),
        ("open chrome", {}),
        ("launch firefox", {}),
        ("open edge", {}),
        ("open a browser", {}),
        ("go to {url}", {"url": None}),
        ("navigate to {url}", {"url": None}),
        ("open {url} in browser", {"url": None}),
        ("browse to {url}", {"url": None}),
        ("take me to {url}", {"url": None}),
    ],
    "browser_search": [
        ("search for {query}", {"query": None}),
        ("google {query}", {"query": None}),
        ("look up {query}", {"query": None}),
        ("search {query}", {"query": None}),
        ("find {query} online", {"query": None}),
        ("web search {query}", {"query": None}),
        ("bing {query}", {"query": None}),
    ],
    "system_info": [
        ("what's my cpu usage", {}),
        ("show cpu usage", {}),
        ("how much ram am i using", {}),
        ("check memory usage", {}),
        ("show disk space", {}),
        ("check disk usage", {}),
        ("system status", {}),
        ("what processes are running", {}),
        ("show running processes", {}),
        ("list active processes", {}),
        ("what is my ip address", {}),
        ("show network info", {}),
        ("check battery", {}),
        ("system info", {}),
        ("resource usage", {}),
    ],
    "kill_process": [
        ("kill {process}", {"process": None}),
        ("stop {process}", {"process": None}),
        ("terminate {process}", {"process": None}),
        ("end the {process} process", {"process": None}),
        ("close {process}", {"process": None}),
        ("force quit {process}", {"process": None}),
        ("end task {process}", {"process": None}),
        ("kill process {process}", {"process": None}),
    ],
    "volume_control": [
        ("mute the volume", {}),
        ("unmute", {}),
        ("turn up the volume", {}),
        ("increase volume", {}),
        ("lower the volume", {}),
        ("decrease volume", {}),
        ("set volume to {num}%", {"num": None}),
        ("volume up", {}),
        ("volume down", {}),
        ("max volume", {}),
        ("silence", {}),
    ],
    "screenshot": [
        ("take a screenshot", {}),
        ("screenshot", {}),
        ("capture screen", {}),
        ("grab a screenshot", {}),
        ("save screen capture", {}),
        ("print screen", {}),
        ("snap the screen", {}),
    ],
    "clipboard": [
        ("copy {text} to clipboard", {"text": None}),
        ("what's in my clipboard", {}),
        ("paste clipboard", {}),
        ("clear clipboard", {}),
        ("show clipboard contents", {}),
    ],
    "git_op": [
        ("git status", {}),
        ("show git status", {}),
        ("git pull", {}),
        ("pull latest changes", {}),
        ("git push", {}),
        ("push to remote", {}),
        ("commit with message {msg}", {"msg": None}),
        ("git commit {msg}", {"msg": None}),
        ("create a new branch called {branch}", {"branch": None}),
        ("checkout branch {branch}", {"branch": None}),
        ("switch to branch {branch}", {"branch": None}),
        ("git log", {}),
        ("show commit history", {}),
        ("git diff", {}),
        ("show changes", {}),
    ],
    "sleep_shutdown": [
        ("shutdown the computer", {}),
        ("turn off the computer", {}),
        ("restart the computer", {}),
        ("reboot", {}),
        ("sleep mode", {}),
        ("put computer to sleep", {}),
        ("lock the screen", {}),
        ("log out", {}),
        ("sign out", {}),
    ],
}

# Slot fill values for realistic variation
SLOT_VALUES = {
    "port": ["3000", "3001", "4000", "5000", "8000", "8080", "8888", "9000"],
    "app": [
        "notepad", "calculator", "vs code", "visual studio code", "chrome",
        "firefox", "spotify", "discord", "slack", "terminal", "file explorer",
        "task manager", "paint", "word", "excel", "powershell", "cmd",
    ],
    "file": [
        "notes.txt", "config.json", "main.py", "index.html", "README.md",
        "requirements.txt", "data.csv", "output.log", "settings.yaml",
    ],
    "dir": [
        "C:/Users/DON COMPUTER/Documents", "C:/Projects", "Downloads",
        "Desktop", "C:/Users/DON COMPUTER/machine-control-model",
    ],
    "url": [
        "github.com", "localhost:3000", "localhost:8080",
        "google.com", "stackoverflow.com",
    ],
    "query": [
        "python regex examples", "best practices for REST APIs",
        "neural network architecture", "scikit-learn tutorial",
        "pytorch custom model", "windows automation python",
    ],
    "process": [
        "node", "python", "chrome", "notepad", "explorer",
        "discord", "spotify", "firefox",
    ],
    "num": ["10", "25", "50", "75", "100"],
    "cmd": [
        "npm install", "pip install -r requirements.txt",
        "python main.py", "ls -la", "dir",
    ],
    "msg": ['"initial commit"', '"fix: hotfix login bug"', '"feat: add registry"'],
    "branch": ["main", "dev", "feature/auth", "hotfix/login"],
    "text": ["hello world", "test string", "project alpha"],
}


def fill_slots(template: str, params: dict) -> tuple[str, dict]:
    filled_params = {}
    result = template
    for slot in params:
        if slot in SLOT_VALUES:
            val = random.choice(SLOT_VALUES[slot])
            result = result.replace("{" + slot + "}", val)
            filled_params[slot] = val
        else:
            result = result.replace("{" + slot + "}", f"<{slot}>")
            filled_params[slot] = f"<{slot}>"
    return result, filled_params


def apply_surface_variation(text: str) -> str:
    """Apply minor textual noise: capitalization, punctuation, filler words."""
    variations = [
        text,
        text.capitalize(),
        text.upper() if len(text) < 20 else text,
        "hey, " + text,
        "please " + text,
        "can you " + text + "?",
        "i want to " + text,
        text + " now",
        text + " please",
        "quickly " + text,
    ]
    return random.choice(variations)


def generate(samples_per_intent: int = 80) -> list[dict]:
    dataset = []
    for intent, templates in TEMPLATES.items():
        count = 0
        seen = set()
        attempts = 0
        while count < samples_per_intent and attempts < samples_per_intent * 10:
            attempts += 1
            template, params = random.choice(templates)
            text, filled = fill_slots(template, params)
            text = apply_surface_variation(text)
            if text.lower() in seen:
                continue
            seen.add(text.lower())
            dataset.append({
                "text": text,
                "intent": intent,
                "params": filled,
            })
            count += 1
    random.shuffle(dataset)
    return dataset


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    data = generate(samples_per_intent=90)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    intent_counts = {}
    for item in data:
        intent_counts[item["intent"]] = intent_counts.get(item["intent"], 0) + 1
    print(f"[+] Generated {len(data)} training samples -> {OUTPUT_PATH}")
    print("[+] Intent distribution:")
    for intent, count in sorted(intent_counts.items()):
        print(f"    {intent:<25} {count}")


if __name__ == "__main__":
    main()
