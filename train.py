"""Train and evaluate the three models, then write reports/evaluation.md.

    .venv/bin/python train.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from emotion_detector.model import TextClassifier

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

TASKS = {
    # name: (hub dataset, config, metric used to pick C)
    "emotion": ("dair-ai/emotion", "split"),
    "sentiment": ("cardiffnlp/tweet_eval", "sentiment"),
    "sarcasm": ("cardiffnlp/tweet_eval", "irony"),
}
C_GRID = [0.5, 1, 2, 4, 8]


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: does 70% confidence mean right 70% of the time?"""
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def train_task(name: str, hub: str, config: str) -> dict:
    ds = load_dataset(hub, config)
    names = ds["train"].features["label"].names
    if name == "sarcasm":
        names = ["not sarcastic", "sarcastic"]
    tr, va, te = ({"text": list(ds[s]["text"]), "label": list(ds[s]["label"])}
                  for s in ("train", "validation", "test"))

    # 1. pick regularisation strength on the validation split
    best_c, best_f1, model = None, -1.0, None
    for c in C_GRID:
        m = TextClassifier(C=c).fit(tr["text"], tr["label"], names)
        f1 = f1_score(va["label"], m.predict(va["text"]), average="macro")
        print(f"  [{name}] C={c:<4} val macro-F1={f1:.3f}")
        if f1 > best_f1:
            best_c, best_f1, model = c, f1, m

    # 2. calibrate confidence on validation, then evaluate once on the untouched test split
    model.calibrate(va["text"], va["label"])
    print(f"  [{name}] temperature = {model.temperature:.2f}")
    probs = model.predict_proba(te["text"])
    y = np.array(te["label"])
    pred = model.clf.classes_[probs.argmax(axis=1)]
    conf = probs.max(axis=1)

    joblib.dump(model, MODELS / f"{name}.joblib")

    majority = np.bincount(y).max() / len(y)
    # accuracy when the model is confident vs. when it is not
    hi = conf >= 0.6
    wrong = np.where(pred != y)[0]
    confident_wrong = wrong[np.argsort(-conf[wrong])][:8]

    return {
        "name": name,
        "dataset": f"{hub} ({config})",
        "C": best_c,
        "T": model.temperature,
        "sizes": (len(tr["text"]), len(va["text"]), len(te["text"])),
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "majority_baseline": majority,
        "ece": ece(probs, y),
        "acc_confident": float((pred[hi] == y[hi]).mean()),
        "share_confident": float(hi.mean()),
        "acc_unsure": float((pred[~hi] == y[~hi]).mean()) if (~hi).any() else float("nan"),
        "report": classification_report(y, pred, target_names=model.labels, digits=3),
        "confusion": confusion_matrix(y, pred).tolist(),
        "labels": model.labels,
        "mistakes": [
            (te["text"][i], model.labels[y[i]], model.labels[pred[i]], float(conf[i]))
            for i in confident_wrong
        ],
    }


def to_markdown(r: dict) -> str:
    lab = r["labels"]
    cm = "| true \\ predicted | " + " | ".join(lab) + " |\n|" + "---|" * (len(lab) + 1) + "\n"
    for name, row in zip(lab, r["confusion"]):
        cm += f"| **{name}** | " + " | ".join(str(v) for v in row) + " |\n"
    mistakes = "\n".join(
        f"| {t[:90].replace('|', '/')!s} | {gold} | {p} | {c:.0%} |" for t, gold, p, c in r["mistakes"]
    )
    tr, va, te = r["sizes"]
    return f"""## {r['name'].title()}

Dataset: `{r['dataset']}` — train {tr}, validation {va}, test {te}. Chosen C = {r['C']}, calibration temperature T = {r['T']:.2f}.

| metric | value |
|---|---|
| **Accuracy (test)** | **{r['accuracy']:.3f}** |
| Macro-F1 (test) | {r['macro_f1']:.3f} |
| Majority-class baseline | {r['majority_baseline']:.3f} |
| Calibration error (ECE, lower is better) | {r['ece']:.3f} |
| Accuracy when confidence ≥ 60% ({r['share_confident']:.0%} of test) | {r['acc_confident']:.3f} |
| Accuracy when confidence < 60% | {r['acc_unsure']:.3f} |

```
{r['report']}```

Confusion matrix:

{cm}
Most confident mistakes:

| text | true | predicted | confidence |
|---|---|---|---|
{mistakes}
"""


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    results = []
    for name, (hub, config) in TASKS.items():
        print(f"Training {name} ...")
        r = train_task(name, hub, config)
        print(f"  -> test accuracy {r['accuracy']:.3f}, macro-F1 {r['macro_f1']:.3f}")
        results.append(r)

    summary = {r["name"]: {k: r[k] for k in ("accuracy", "macro_f1", "ece", "C")} for r in results}
    (REPORTS / "metrics.json").write_text(json.dumps(summary, indent=2))
    body = "# Evaluation (auto-generated by train.py)\n\n" + "\n".join(to_markdown(r) for r in results)
    (REPORTS / "evaluation.md").write_text(body)
    print(f"Wrote {REPORTS / 'evaluation.md'}")


if __name__ == "__main__":
    main()
