"""Combines the sentiment, emotion and sarcasm models into one analysis."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np

from .model import Prediction, TextClassifier

MODELS = Path(__file__).resolve().parent.parent / "models"

CONTEXT_WEIGHT = 0.3       # how much the surrounding text can shift the reading
SARCASM_THRESHOLD = 0.75   # sarcasm model is weak (~63% accuracy), so warn only when fairly sure
EMOTION_POLARITY = {"joy": "positive", "love": "positive", "sadness": "negative",
                    "anger": "negative", "fear": "negative", "surprise": None}


def hedge(conf: float) -> str:
    """Words for a calibrated probability. We never say 'is', only 'likely'."""
    if conf >= 0.85:
        return "very likely"
    if conf >= 0.65:
        return "likely"
    if conf >= 0.45:
        return "possibly"
    return "unclear"


@dataclass
class Analysis:
    text: str
    sentiment: Prediction
    emotion: Prediction
    sarcasm_probability: float
    context_used: bool
    known_words: int
    emotion_words: int   # known words that are not grammar words like "you", "are", "the"
    unknown_words: list[str]   # meaningful words neither model saw in training
    notes: list[str] = field(default_factory=list)

    @property
    def no_clear_emotion(self) -> bool:
        # The emotion dataset has no "neutral" class, so it always picks *something*.
        # When the sentiment model is fairly sure the text is neutral, believe it instead.
        s = self.sentiment
        return s.label == "neutral" and s.confidence >= 0.6 and self.emotion.confidence < 0.65

    def summary(self) -> str:
        e, s = self.emotion, self.sentiment
        if self.no_clear_emotion:
            emo = f"no strong emotion (weak lean: {e.label})"
        elif hedge(e.confidence) == "unclear" or self.emotion_words == 0:
            top2 = list(e.probabilities)[:2]
            emo = f"emotion is unclear (maybe {top2[0]} or {top2[1]})"
        else:
            emo = f"{hedge(e.confidence)} {e.label}"
        sent = "sentiment unclear" if hedge(s.confidence) == "unclear" else f"sentiment {hedge(s.confidence)} {s.label}"
        return f"{emo}; {sent}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["summary"] = self.summary()
        d["no_clear_emotion"] = self.no_clear_emotion
        return d


class EmotionAnalyzer:
    def __init__(self, models_dir: Path = MODELS):
        missing = [n for n in ("emotion", "sentiment", "sarcasm") if not (models_dir / f"{n}.joblib").exists()]
        if missing:
            raise FileNotFoundError(f"Missing models {missing} in {models_dir}. Run: python train.py")
        self.emotion: TextClassifier = joblib.load(models_dir / "emotion.joblib")
        self.sentiment: TextClassifier = joblib.load(models_dir / "sentiment.joblib")
        self.sarcasm: TextClassifier = joblib.load(models_dir / "sarcasm.joblib")

    def _probs(self, model: TextClassifier, text: str, context: str | None) -> np.ndarray:
        p = model.predict_proba([text])[0]
        if context:
            # Read the sentence together with what came before it, and blend. A normal
            # sentence still dominates; a short one ("Fine.") leans more on its context.
            evidence = float(model.evidence_weight([text])[0])
            w = CONTEXT_WEIGHT + 0.4 * (1 - evidence)
            # Classify the context on its own and use it as a prior. (Joining the two texts
            # lets one strong word like "fine" drown the context out.)
            p_ctx = model.predict_proba([context])[0]
            p = (1 - w) * p + w * p_ctx
        return p

    def analyze(self, text: str, context: str | None = None) -> Analysis:
        text = text.strip()
        context = (context or "").strip() or None

        emo = self.emotion.explain(text, probs=self._probs(self.emotion, text, context))
        sent = self.sentiment.explain(text, probs=self._probs(self.sentiment, text, context))
        sarc = float(self._probs(self.sarcasm, text, context)[self.sarcasm.labels.index("sarcastic")])
        words = self.emotion.known_words(text)
        content = [w for w in words if self.emotion.is_content_word(w)]
        known, n_content = len(words), len(content)
        # A model that found no meaningful words fell back to its prior, so word
        # "influence" lists would only show noise. Don't present them as explanations.
        if n_content == 0:
            emo.evidence = []
        if self.sentiment.evidence_weight([text])[0] == 0:
            sent.evidence = []
        unk_emo = self.emotion.unknown_words(text)
        unk_sent = set(self.sentiment.unknown_words(text))
        unknown = [w for w in unk_emo if w in unk_sent]          # neither model knows it
        emo_only = [w for w in unk_emo if w not in unk_sent]     # only the emotion model lacks it

        notes: list[str] = []
        if known == 0:
            notes.append("None of these words appeared in training data, so the prediction is close to a guess.")
        elif n_content == 0:
            notes.append(f"The emotion model recognised only grammar words here ({', '.join(words[:4])}), "
                         "which say nothing about emotion, so it falls back to a neutral prior (no real guess).")
        elif known <= 2:
            notes.append("Very short or unfamiliar text: there is little evidence to go on.")
        if unknown:
            notes.append(f"Neither model has seen: {', '.join(unknown)}. If that word carries the meaning, "
                         "both readings are unreliable.")
        if emo_only:
            notes.append(f"The emotion model has never seen: {', '.join(emo_only)} "
                         "(the sentiment model has, so trust the sentiment more).")
        if hedge(emo.confidence) == "unclear" and n_content > 0:
            notes.append("No emotion stands out clearly. The text may be mixed, neutral or ambiguous.")
        if sarc >= SARCASM_THRESHOLD:
            flip = {"positive": " (the intended meaning may be negative)",
                    "negative": " (the intended meaning may be positive)"}.get(sent.label, "")
            notes.append(f"Possible sarcasm ({sarc:.0%}): the literal words may mean the opposite{flip}, "
                         "so read the sentiment/emotion with caution.")
        negated = [w for w, v in emo.evidence[:3] if v > 0 and "not-" in w]
        if negated and emo.label in ("joy", "love") and sent.label == "negative":
            notes.append(f"The emotion model is being driven by a negated phrase ({negated[0]}). Its training "
                         "data handles negation poorly, so the sentiment reading is more trustworthy here.")
        polarity = EMOTION_POLARITY.get(emo.label)
        if (not negated and n_content > 0 and polarity and sent.label != "neutral" and polarity != sent.label
                and sent.confidence >= 0.5):
            if emo.confidence >= 0.5:
                notes.append(f"Mixed signals: the emotion ({emo.label}) and sentiment ({sent.label}) disagree. "
                             "This often happens with sarcasm, irony or bittersweet statements.")
            elif sarc < SARCASM_THRESHOLD:  # with sarcasm, the *sentiment* may be the one fooled
                notes.append(f"Contradiction: the weak emotion guess ({emo.label}) doesn't fit the "
                             f"{sent.label} sentiment ({sent.confidence:.0%}). The emotion guess is probably wrong.")
        if context:
            notes.append("Surrounding context was taken into account.")

        return Analysis(text, sent, emo, sarc, bool(context), known, n_content, unknown + emo_only, notes)
