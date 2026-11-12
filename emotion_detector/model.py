"""TF-IDF + Logistic Regression classifier with confidence and word-level explanations."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .text import CONTRACTION_STOPWORDS, pretty_feature, tokenize

# How much to trust a prediction, by number of *distinct known words* in the text.
# 1 word ("Fine.") is barely evidence; 3+ words is a normal sentence and gets full trust.
# Anything below full trust is pulled toward the class prior. Stops "Fine." -> joy 100%
# without penalising short but complete sentences like "I am depressed".
EVIDENCE_BY_WORDS = {0: 0.0, 1: 0.15, 2: 0.5}


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]     # every class, sorted high -> low
    evidence: list[tuple[str, float]]   # (word/phrase, weight toward `label`)


class TextClassifier:
    """A linear model, so every prediction can be explained by summing word weights."""

    def __init__(self, C: float = 1.0, class_weight: str | None = "balanced"):
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize,
            lowercase=False,
            token_pattern=None,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )
        self.clf = LogisticRegression(C=C, max_iter=3000, class_weight=class_weight)
        self.labels: list[str] = []
        self.temperature = 1.0

    def fit(self, texts: list[str], y: list[int], label_names: list[str]) -> "TextClassifier":
        X = self.vectorizer.fit_transform(texts)
        self.clf.fit(X, y)
        self.labels = [label_names[i] for i in self.clf.classes_]
        counts = np.bincount(np.searchsorted(self.clf.classes_, y), minlength=len(self.clf.classes_))
        self.prior = counts / counts.sum()
        self._features = self.vectorizer.get_feature_names_out()
        return self

    def _logits(self, texts: list[str]) -> np.ndarray:
        z = self.clf.decision_function(self.vectorizer.transform(texts))
        if z.ndim == 1:  # binary -> two columns
            z = np.column_stack([np.zeros_like(z), z])
        return z

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def calibrate(self, texts: list[str], y: list[int]) -> float:
        """Temperature scaling: find T so softmax(logits / T) matches held-out accuracy.

        Only changes how confident the model says it is, never which label wins.
        """
        z = self._logits(texts)
        a = self.evidence_weight(texts)[:, None]
        idx = np.searchsorted(self.clf.classes_, y)
        best_t, best_nll = 1.0, np.inf
        for t in np.exp(np.linspace(np.log(0.05), np.log(5), 200)):
            p = (a * self._softmax(z / t) + (1 - a) * self.prior)[np.arange(len(idx)), idx]
            nll = -np.log(np.clip(p, 1e-12, None)).mean()
            if nll < best_nll:
                best_t, best_nll = float(t), nll
        self.temperature = best_t
        return best_t

    def known_words(self, text: str) -> list[str]:
        """Single words (not word pairs) in `text` that the model learned weights for."""
        x = self.vectorizer.transform([text])
        return [self._features[i] for i in x.indices if " " not in self._features[i]]

    @staticmethod
    def is_content_word(w: str) -> bool:
        """A word that can carry meaning, i.e. not "you", "are", "the", "!" ..."""
        w = w.removeprefix("NOT_")
        return w[:1].isalpha() and w not in ENGLISH_STOP_WORDS and w not in CONTRACTION_STOPWORDS

    def unknown_words(self, text: str) -> list[str]:
        """Meaningful words in `text` that the model never saw in training."""
        vocab = self.vectorizer.vocabulary_
        seen, out = set(), []
        for t in tokenize(text):
            base = t.removeprefix("NOT_")
            if self.is_content_word(base) and t not in vocab and base not in vocab and base not in seen:
                seen.add(base)
                out.append(base)
        return out

    def evidence_weight(self, texts: list[str]) -> np.ndarray:
        """1.0 for a normal sentence, smaller when few words are known to the model.

        Grammar words alone ("you are a ...") are not evidence of anything, so a text
        whose only known words are grammar words gets weight 0, i.e. the class prior.
        """
        X = self.vectorizer.transform(texts)
        if not hasattr(self, "_unigram"):
            self._unigram = np.array([" " not in f for f in self._features])
            self._content = self._unigram & np.array([self.is_content_word(f) for f in self._features])
        out = []
        for a, b in zip(X.indptr[:-1], X.indptr[1:]):
            idx = X.indices[a:b]
            if not self._content[idx].any():
                out.append(0.0)
            else:
                out.append(EVIDENCE_BY_WORDS.get(int(self._unigram[idx].sum()), 1.0))
        return np.array(out)

    def _shrunk(self, texts: list[str], t: float) -> np.ndarray:
        a = self.evidence_weight(texts)[:, None]
        return a * self._softmax(self._logits(texts) / t) + (1 - a) * self.prior

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self._shrunk(texts, self.temperature)

    def predict(self, texts: list[str]) -> np.ndarray:
        return self.clf.classes_[self.predict_proba(texts).argmax(axis=1)]

    def explain(self, text: str, top_k: int = 5, probs: np.ndarray | None = None) -> Prediction:
        x = self.vectorizer.transform([text])
        if probs is None:
            probs = self.predict_proba([text])[0]
        best = int(np.argmax(probs))

        # Contribution of each word = tf-idf value x coefficient.
        # For multi-class LR we explain "why this class rather than the others",
        # so we subtract the mean coefficient across classes.
        coef = self.clf.coef_
        if coef.shape[0] == 1:  # binary: one row, sign tells the direction
            direction = coef[0] if best == 1 else -coef[0]
        else:
            direction = coef[best] - coef.mean(axis=0)
        idx = x.indices
        contrib = x.data * direction[idx]
        order = np.argsort(-np.abs(contrib))[:top_k]
        evidence = [
            (pretty_feature(self._features[idx[i]]), float(contrib[i]))
            for i in order
            if abs(contrib[i]) > 1e-3
        ]

        ranked = sorted(zip(self.labels, probs), key=lambda p: -p[1])
        return Prediction(
            label=self.labels[best],
            confidence=float(probs[best]),
            probabilities={k: float(v) for k, v in ranked},
            evidence=evidence,
        )
