"""
Phase 2: Local Intent Classifier & NER Model
Custom TF-IDF + Logistic Regression classifier with regex-based parameter extraction.
Runs fully offline, CPU-only, under 100MB on disk.
"""

import json
import os
import re
from typing import Optional

import joblib

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "model.joblib")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "data", "generated", "dataset.json")

# ---------------------------------------------------------------------------
# Slot extraction patterns (NER layer — regex-based, no external deps)
# ---------------------------------------------------------------------------

SLOT_PATTERNS = {
    "port": re.compile(r"\b(\d{2,5})\b"),
    "url": re.compile(
        r"((?:https?://)?(?:localhost|[\w.-]+\.(?:com|org|net|io|dev|co))(?::\d+)?(?:/[\w./?=&%-]*)?)"
    ),
    "file": re.compile(r"([\w.\-]+\.(?:txt|json|py|js|ts|html|css|md|csv|log|yaml|yml|sh|bat|env))"),
    "dir": re.compile(r"([A-Za-z]:\\[\\\w\s\-\.]+|\.{0,2}/[\w\-/\.]+|\bDesktop\b|\bDownloads\b|\bDocuments\b)"),
    "branch": re.compile(r"(?:branch|checkout|switch to)\s+([\w/\-]+)"),
    "process": re.compile(r"(?:kill|stop|terminate|end|close|quit)\s+([\w.]+)"),
    "msg": re.compile(r'"([^"]+)"'),
    "query": re.compile(r'(?:search|google|look up|find|bing)\s+(?:for\s+)?(.+)$', re.IGNORECASE),
    "cmd": re.compile(r'(?:run|execute|shell|terminal[:\s]+)\s+(.+)$', re.IGNORECASE),
    "num": re.compile(r'(\d+)\s*%'),
}


# ---------------------------------------------------------------------------
# Custom tokenizer: lower, strip punctuation, handle camelCase
# ---------------------------------------------------------------------------

class CustomTokenizer:
    """Picklable tokenizer: lowercase, strip punctuation, split camelCase."""

    def __call__(self, text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        raw_tokens = text.split()
        tokens = []
        for t in raw_tokens:
            tokens.extend(re.sub(r'([a-z])([A-Z])', r'\1 \2', t).split())
        return [t for t in tokens if len(t) > 1]


# ---------------------------------------------------------------------------
# Model: TF-IDF pipeline
# ---------------------------------------------------------------------------

def build_pipeline() -> Pipeline:
    vectorizer = TfidfVectorizer(
        tokenizer=CustomTokenizer(),
        ngram_range=(1, 3),
        max_features=8000,
        sublinear_tf=True,
        min_df=1,
    )
    classifier = LogisticRegression(
        C=4.0,
        max_iter=500,
        solver="lbfgs",
    )
    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(dataset_path: str = DATASET_PATH, model_path: str = MODEL_PATH) -> None:
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    texts = [d["text"] for d in data]
    labels = [d["intent"] for d in data]

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("\n[+] Classification Report:")
    print(classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)

    size_kb = os.path.getsize(model_path) / 1024
    print(f"[+] Model saved -> {model_path}  ({size_kb:.1f} KB)")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

class MatcherModel:
    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run: python main.py --setup"
            )
        self._pipeline: Pipeline = joblib.load(model_path)

    def predict(self, text: str) -> dict:
        intent = self._pipeline.predict([text])[0]
        proba = self._pipeline.predict_proba([text])[0]
        classes = self._pipeline.classes_
        confidence = float(np.max(proba))
        top3 = sorted(
            zip(classes, proba.tolist()), key=lambda x: x[1], reverse=True
        )[:3]

        params = self._extract_params(text, intent)
        return {
            "text": text,
            "intent": intent,
            "confidence": round(confidence, 4),
            "params": params,
            "top3": [{"intent": i, "score": round(s, 4)} for i, s in top3],
        }

    def _extract_params(self, text: str, intent: str) -> dict:
        params = {}
        for slot, pattern in SLOT_PATTERNS.items():
            match = pattern.search(text)
            if match:
                params[slot] = match.group(1)
        if not params:
            params["raw_text"] = text
        return params


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    args = sys.argv[1:]

    if "--train" in args:
        print("[*] Training model...")
        train()
    elif args:
        query = " ".join(args)
        model = MatcherModel()
        result = model.predict(query)
        print(json.dumps(result, indent=2))
    else:
        print("Usage:")
        print("  python matcher_model.py --train")
        print("  python matcher_model.py open chrome")
