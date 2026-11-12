# Emotion Detector

Reads a sentence and predicts its **likely** sentiment (positive / negative / neutral) and emotion
(joy, sadness, anger, fear, love, surprise), each with a calibrated confidence score. It also
estimates the chance of sarcasm, shows which words influenced the prediction, and can use
surrounding context.

> The system predicts *likely* emotions. It does not claim to know what a person truly means.
> Every output is hedged ("very likely", "possibly", "unclear") and comes with warnings when
> the evidence is thin or contradictory.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python train.py            # downloads datasets, trains, writes reports/evaluation.md (~1 min)

.venv/bin/python cli.py "I can't believe I finally got the job!"
.venv/bin/python cli.py "Fine." --context "My dog died this morning."
.venv/bin/python cli.py              # interactive mode ("ctx: ..." sets context)
.venv/bin/python cli.py --json "..." # machine-readable output

.venv/bin/python app.py              # web page at http://localhost:8000
```

## How it works

| Task | Public dataset | Model |
|---|---|---|
| Emotion | `dair-ai/emotion` (16k train) | TF-IDF (1–2 grams) + Logistic Regression |
| Sentiment | `cardiffnlp/tweet_eval` sentiment (45k train) | same |
| Sarcasm (bonus) | `cardiffnlp/tweet_eval` irony (2.9k train) | same |

- **Preprocessing** (`emotion_detector/text.py`): lower-casing, URL/@user replacement, squashing
  "soooo" to "soo", removing apostrophes (the emotion dataset writes "im", "dont"), and **negation marking**: the 3 words after "not", "never", "n't" etc. get a
  `NOT_` prefix, so "not happy" is a different feature from "happy".
- **Model choice:** regularisation strength C is picked by macro-F1 on the validation split. The test
  split is used once, for the final numbers.
- **Confidence you can trust:** raw LR probabilities were badly calibrated for emotion (ECE 0.19).
  **Temperature scaling** fitted on validation brings this down to 0.017, so "70% confident" really
  means roughly 70% correct.
- **Evidence shrinkage:** texts with only 1–2 distinct known words have their probabilities pulled
  toward the class prior (1 word keeps 15% of the model's opinion, 2 words keep 50%). Without
  this, "Fine." was classified as joy with 100% confidence. Short but complete sentences like
  "I am depressed" (3 words) are trusted fully.
- **Grammar-only check:** grammar words ("you", "are", "a") are not evidence. If they are the only
  words a model recognises ("you are a sadist": neither model has seen "sadist"), that model falls
  back to its prior, so the bars show no confident emotion, and a warning names the unknown word. If a weak emotion guess contradicts confident sentiment, that is flagged
  too, unless sarcasm is suspected, in which case the sentiment may be the reading that's wrong.
- **Explanations (bonus):** the model is linear, so each word's contribution is
  `tf-idf × (coef[class] − mean coef)`. The CLI and web page list the words that pushed toward the
  prediction, and the ones that pushed against it.
- **Context (bonus):** the context is classified on its own and blended in as a prior. It gets 30%
  weight for a normal sentence and up to 70% for a very short one. Example: "Fine." alone →
  *unclear*; "Fine." after "My dog died this morning…" → *likely sadness*.
- **Uncertainty rules** (`emotion_detector/predictor.py`): the system adds a warning for few known
  words, no clear winning emotion, likely sarcasm (≥ 75%), emotion and sentiment disagreeing, or a
  negated phrase driving the emotion. Because the emotion dataset has no neutral class, the system
  reports *no strong emotion* when the sentiment model is confident the text is neutral.

## Evaluation (test splits)

Full report with per-class scores, confusion matrices and the most confident mistakes:
[`reports/evaluation.md`](reports/evaluation.md).

| Task | Accuracy | Macro-F1 | Majority baseline | Calibration error (ECE) |
|---|---|---|---|---|
| Emotion (6 classes) | **0.858** | 0.809 | 0.347 | 0.022 |
| Sentiment (3 classes) | **0.607** | 0.604 | 0.483 | 0.043 |
| Sarcasm (2 classes) | **0.625** | 0.617 | 0.603 | 0.022 |

Confidence carries real information. For emotion, predictions at ≥ 60% confidence (88% of the test
set) are 90.4% accurate, while those below 60% are only 51% accurate.

## Where it fails

1. **Negation in emotion.** `dair-ai/emotion` was labelled from the emotion word in each sentence,
   so "i'm not feeling good" is labelled *joy* (4 of 4 such training examples). The model
   inherits this: "I am not happy" → *possibly joy*. The sentiment model (tweet data, real labels)
   correctly says *negative* at 97%, and the system warns that the sentiment reading is more
   trustworthy here.
2. **Neighbouring emotions blur.** The most common confusions are joy ↔ love (45 + 23 test errors)
   and fear → surprise. Words like "passionate", "accepted" and "loved" sit on the boundary, and the
   dataset's labels are themselves debatable.
3. **Surprise is weak** (F1 0.70, only 66 test examples). "Wow, I did not see that coming" doesn't
   score surprise as its top-2 emotion, because the training data's surprise examples are phrased as
   "i feel amazed / shocked".
4. **Sentiment on tweets is hard for bag-of-words** (61%). Most errors are neutral ↔ negative. The
   most confident mistakes are hashtag-heavy political or promotional tweets ("#draintheswamp GO
   RUDY!" → positive, gold neutral) where word sentiment doesn't match the labelled stance.
5. **Sarcasm is barely above the majority baseline** (62.5% vs 60%). Sarcasm usually depends on world
   knowledge and contrast, not individual words. The system therefore only warns at ≥ 75% and never
   flips a label automatically. It catches stock patterns ("Oh great, another Monday…" → 88%) but
   also gives 66% to the sincere "I finally got the job, I'm so happy!".
6. **Vocabulary gaps.** The emotion data contains almost no insults or slang aimed at someone
   else. "you are dumbass" is scored mostly as *love*, because "you are" appears in affectionate
   sentences, and "sadist" is unknown to both models. The system now reports these as *unclear*
   and names the unknown words, but it can't name the right emotion (anger). Dataset quirks also
   show up: "you are amazing" → *surprise*, because the data labels "amazed/amazing" as surprise. A pretrained model would be needed for that.
7. **Domain shift.** Emotion is trained on "i feel …" blog sentences and sentiment on tweets. Formal
   text, long paragraphs and other languages fall outside what either model has seen.

## Possible next steps

- Fine-tune a BERT-style model (e.g. DistilRoBERTa) on the same datasets. Expect roughly 93% emotion
  accuracy and roughly 70% tweet-sentiment accuracy, at the cost of easy word-level explanations.
- Train sarcasm on a larger corpus (e.g. the SARC Reddit dataset), with the parent comment as context.

## Layout

```
emotion_detector/text.py       tokenizer with negation marking
emotion_detector/model.py      TF-IDF + LR, temperature scaling, shrinkage, explanations
emotion_detector/predictor.py  combines the 3 models, context, uncertainty notes
train.py                       training + evaluation -> models/, reports/
cli.py                         command-line interface
app.py, web/index.html         small web page (standard library only)
```
