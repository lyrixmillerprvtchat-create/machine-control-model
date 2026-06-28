"""
Runtime Correction & Incremental Learning
When the model predicts wrong, the user names the correct intent.
The correction is saved and the model retrains in-place (<1 second).
"""

import json
import os
from typing import Optional

CORRECTIONS_PATH = os.path.join(os.path.dirname(__file__), "data", "corrections.json")

ALL_INTENTS = [
    "sys_command", "dev_server", "open_app", "file_op_open", "file_op_delete",
    "file_op_create", "dir_op", "browser_open", "browser_search", "system_info",
    "kill_process", "volume_control", "screenshot", "clipboard", "git_op",
    "sleep_shutdown",
]


def load() -> list[dict]:
    if not os.path.exists(CORRECTIONS_PATH):
        return []
    with open(CORRECTIONS_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_correction(original_text: str, wrong_intent: str, correct_intent: str) -> None:
    data = load()
    data.append({
        "text": original_text,
        "wrong_intent": wrong_intent,
        "correct_intent": correct_intent,
        "params": {},
    })
    os.makedirs(os.path.dirname(CORRECTIONS_PATH), exist_ok=True)
    with open(CORRECTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def apply_and_retrain() -> None:
    """Merge corrections into dataset and retrain the model immediately."""
    from matcher_model import train, DATASET_PATH

    corrections = load()
    if not corrections:
        return

    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    # Each correction adds 8 copies (weight boost) so the classifier learns it
    for c in corrections:
        for _ in range(8):
            dataset.append({"text": c["text"], "intent": c["correct_intent"], "params": {}})

    augmented_path = DATASET_PATH.replace(".json", "_augmented.json")
    with open(augmented_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f)

    train(dataset_path=augmented_path)
    print(f"[+] Model retrained with {len(corrections)} correction(s).")


def prompt_correction(original_text: str, predicted_intent: str) -> Optional[str]:
    """
    Ask the user if the prediction was wrong.
    Returns the correct intent string if they provide one, else None.
    """
    print(f"\n  Was that right? (enter=yes  |  ? for intent list  |  type correct intent)")
    try:
        answer = input("  > ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        return None

    if not answer:
        return None

    if answer == "?":
        print("\n  Available intents:")
        for i, intent in enumerate(ALL_INTENTS, 1):
            print(f"    {i:>2}. {intent}")
        try:
            answer = input("  > ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            return None

    # Accept intent by number
    if answer.isdigit():
        idx = int(answer) - 1
        if 0 <= idx < len(ALL_INTENTS):
            answer = ALL_INTENTS[idx]

    if answer in ALL_INTENTS and answer != predicted_intent:
        save_correction(original_text, predicted_intent, answer)
        apply_and_retrain()
        return answer

    return None
