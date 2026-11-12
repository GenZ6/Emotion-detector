# Demo script (about 3 minutes)

**Setup, before you present:**
```bash
cd ~/Desktop/Emotion_detector
.venv/bin/python app.py        # open http://localhost:8000
```
Every sentence below is a clickable example chip on the page, in this order.
(Backup if the browser fails: `.venv/bin/python cli.py "sentence"`.)

---

### 1. What it does (30 s)
> "It reads a sentence and predicts the *likely* emotion and sentiment, how sure it is, and which
> words made it decide."

**Type:** `I just got accepted into my dream university!`
→ **very likely joy** (89%), positive.
**Point at:** the word chips: *accepted*, *university*, *got*. "It's a linear model, so we can see
exactly which words pushed it there."

### 2. It recognises different emotions (20 s)
- `I'm terrified of the exam tomorrow` → **fear** (*terrified*, *exam*)
- `I was shocked when I saw the results` → **surprise** (*shocked*)

### 3. Sarcasm (30 s)
**Type:** `Oh great, another Monday. Just what I needed.`
→ the sentiment model reads it as **positive** (because of "great"), but the sarcasm model gives
**87%** and the page warns that the intended meaning may be negative.
> "It doesn't silently flip the answer. It tells you it might be sarcasm."

### 4. Context changes the reading (30 s)
- `Fine.` on its own → **unclear**. One word isn't enough evidence, and it says so.
- `Fine.` with context *"My dog died this morning."* → **likely sadness**.
> "Same word, different meaning, depending on what came before."

### 5. It says when there's no emotion (15 s)
`The meeting is at 3pm.` → **no strong emotion**, neutral sentiment.

### 6. Where it fails, and admits it (45 s)
This is the most important part of the brief: *it predicts likely emotions; it doesn't claim to know.*

| Sentence | What happens | Why |
|---|---|---|
| `I'm not happy with my grade` | emotion says **joy**, sentiment says **negative**, and a warning appears | The emotion dataset labels "not happy" sentences as joy, so the model learned that. The warning tells you to trust the sentiment here. |
| `you are a sadist` | **unclear**, with "Neither model has seen: *sadist*" | The model only knows words from its 16k training sentences. |
| `Thanks a lot for ruining my weekend` | sarcasm only **59%**, below the warning threshold | The sarcasm model is weak (62.5% accuracy). It's the hardest task. |

### 7. Numbers (20 s)
| | Accuracy | Baseline |
|---|---|---|
| Emotion (6 classes) | **85.8%** | 34.7% |
| Sentiment (tweets) | 60.7% | 48.3% |
| Sarcasm | 62.5% | 60.3% |

> "And the confidence numbers are calibrated. When it says 90%, it's right about 90% of the time.
> That's why we can trust it when it says *unclear*."

---

**Likely questions**
- *Why not BERT?* Classic TF-IDF + LR is fast, runs anywhere and is fully explainable. BERT would fix
  the unknown-word problem (e.g. "sadist") but loses the clear word-level explanations.
- *What datasets?* `dair-ai/emotion` (16k sentences), `tweet_eval` sentiment (45k tweets) and
  `tweet_eval` irony (2.9k tweets), all public on Hugging Face.
- *How is confidence calibrated?* Temperature scaling on the validation set. Calibration error for
  emotion dropped from 0.19 to 0.02.
