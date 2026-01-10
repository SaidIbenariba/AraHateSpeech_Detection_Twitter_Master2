import re
import csv
from pathlib import Path

ARABIC_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
URL = re.compile(r"https?://\S+|www\.\S+")
MENTION = re.compile(r"@\w+")
HASHTAG = re.compile(r"#\w+")
MULTISPACE = re.compile(r"\s+")
TATWEEL = "\u0640"

def load_stopwords(*paths: str) -> set:
    sw = set()
    for p in paths:
        if not p:
            continue
        path = Path(p)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                word = row[0].strip()
                if word:
                    sw.add(word)
    return sw

def normalize_arabic(text: str) -> str:
    text = re.sub(ARABIC_DIACRITICS, "", text)
    text = text.replace(TATWEEL, "")
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ئ", "ي", text)
    text = re.sub(r"ة", "ه", text)
    return text

def preprocess(text: str, stopwords: set | None = None, keep_latin: bool = True) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(URL, " ", text)
    text = re.sub(MENTION, " ", text)
    text = re.sub(HASHTAG, " ", text)

    text = normalize_arabic(text)

    # garder arabe + (optionnel) latin + chiffres + espaces
    if keep_latin:
        text = re.sub(r"[^0-9A-Za-z\u0600-\u06FF\s]", " ", text)
    else:
        text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)

    text = re.sub(MULTISPACE, " ", text).strip()

    if stopwords:
        tokens = [t for t in text.split() if t not in stopwords and len(t) > 1]
        text = " ".join(tokens)

    return text
