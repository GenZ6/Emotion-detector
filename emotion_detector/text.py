"""Text normalisation shared by training and prediction."""
import re

_URL = re.compile(r"https?://\S+|www\.\S+")
_USER = re.compile(r"@\w+")
_REPEAT = re.compile(r"(.)\1{2,}")          # "soooo" -> "soo"
_TOKEN = re.compile(r"[a-z]+|[!?]+|[\U0001F300-\U0001FAFF☀-➿]")
_NEGATORS = {"not", "no", "never", "nothing", "nobody", "nor", "neither", "cannot", "without",
             "dont", "didnt", "doesnt", "isnt", "wasnt", "arent", "werent", "cant", "couldnt",
             "wont", "wouldnt", "shouldnt", "havent", "hasnt", "hadnt", "aint", "nt"}
# Contractions (after apostrophes are removed) that are grammar, not meaning.
CONTRACTION_STOPWORDS = {"im", "ive", "id", "youre", "youve", "youd", "hes", "shes", "its", "were",
                         "theyre", "theyve", "thats", "theres", "whats", "lets", "ill", "youll"} | _NEGATORS
_CLAUSE_END = re.compile(r"^[!?]+$")
_NEG_SCOPE = 3  # how many words after a negator get marked


def tokenize(text: str) -> list[str]:
    """Lower-case, strip URLs/handles, and mark negated words.

    "I am not happy at all" -> ["i", "am", "not", "NOT_happy", "NOT_at", "NOT_all"]
    so the model can learn that "happy" after "not" means something else.
    """
    text = text.lower()
    text = _URL.sub(" url ", text)
    text = _USER.sub(" user ", text)
    text = _REPEAT.sub(r"\1\1", text)
    # The emotion dataset has no apostrophes ("im", "dont"), so drop them everywhere.
    text = text.replace("’", "'").replace("'", "")

    tokens: list[str] = []
    negate = 0
    # split on sentence punctuation so negation stops at ".", ",", ";"
    for clause in re.split(r"[.,;:]", text):
        negate = 0
        for tok in _TOKEN.findall(clause):
            if _CLAUSE_END.match(tok):
                tokens.append(tok)
                negate = 0
                continue
            if negate:
                tokens.append("NOT_" + tok)
                negate -= 1
            else:
                tokens.append(tok)
            if tok in _NEGATORS:
                negate = _NEG_SCOPE
    return tokens


def pretty_feature(feature: str) -> str:
    """Turn an internal feature name back into something readable."""
    return " ".join(
        f"not-{w[4:]}" if w.startswith("NOT_") else w for w in feature.split(" ")
    )
