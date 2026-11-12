"""Command-line interface.

    .venv/bin/python cli.py "I can't believe I got the job!"
    .venv/bin/python cli.py "Great, just great." --context "My flight got cancelled again."
    .venv/bin/python cli.py            # interactive mode
    .venv/bin/python cli.py --json "..."
"""
import argparse
import json

from emotion_detector.predictor import EmotionAnalyzer, hedge


def bar(p: float, width: int = 20) -> str:
    n = round(p * width)
    return "█" * n + "·" * (width - n)


def show(a) -> None:
    print(f"\n  “{a.text}”")
    print(f"  → {a.summary()}\n")

    print("  Emotion")
    for label, p in a.emotion.probabilities.items():
        print(f"    {label:<9} {bar(p)} {p:6.1%}")
    print(f"\n  Sentiment: {a.sentiment.label} ({a.sentiment.confidence:.0%}, {hedge(a.sentiment.confidence)})")
    print(f"  Sarcasm:   {a.sarcasm_probability:.0%} chance")

    if a.emotion.evidence:
        print(f"\n  Word influence on '{a.emotion.label}' (+ for, − against):")
        for word, w in a.emotion.evidence:
            sign = "+" if w > 0 else "−"
            print(f"    {sign} {word:<20} ({w:+.2f})")
        if all(w <= 0 for _, w in a.emotion.evidence):
            reason = "the surrounding context" if a.context_used else "the model's prior"
            print(f"    (no word in the sentence supports '{a.emotion.label}'; it comes from {reason})")
    if a.sentiment.evidence:
        words = ", ".join(w for w, v in a.sentiment.evidence if v > 0)
        if words:
            print(f"  Words that pushed toward '{a.sentiment.label}': {words}")

    for n in a.notes:
        print(f"\n  ⚠ {n}")
    print("\n  (These are statistical guesses about likely emotions, not a reading of what anyone truly means.)\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict the likely emotion and sentiment of a sentence.")
    ap.add_argument("text", nargs="?", help="sentence to analyse (omit for interactive mode)")
    ap.add_argument("--context", "-c", help="preceding text/conversation, to help interpret the sentence")
    ap.add_argument("--json", action="store_true", help="print raw JSON")
    args = ap.parse_args()

    analyzer = EmotionAnalyzer()
    if args.text:
        a = analyzer.analyze(args.text, args.context)
        print(json.dumps(a.to_dict(), indent=2)) if args.json else show(a)
        return

    print("Type a sentence (empty line to quit). Prefix with 'ctx:' to set context for the next sentence.")
    context = None
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            break
        if line.lower().startswith("ctx:"):
            context = line[4:].strip()
            print(f"  (context set: “{context}”)")
            continue
        show(analyzer.analyze(line, context))
        context = None


if __name__ == "__main__":
    main()
