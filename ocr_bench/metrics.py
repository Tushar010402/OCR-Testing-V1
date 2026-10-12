"""Accuracy metrics: character and word error rates against ground truth."""

import re
import string
import unicodedata

import jiwer

_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(text: str, *, loose: bool = False) -> str:
    """Collapse whitespace; with ``loose`` also lowercase and drop punctuation.

    Strict scoring keeps case and punctuation, which matter for things like
    invoice numbers. Loose scoring answers "did it read the words right?".
    """
    text = unicodedata.normalize("NFKC", text or "")
    if loose:
        text = text.lower().translate(_PUNCT)
    return re.sub(r"\s+", " ", text).strip()


def _rate(fn, reference: str, hypothesis: str) -> float:
    if not reference:
        # Nothing to read: any output at all counts as a full error.
        return 0.0 if not hypothesis else 1.0
    return fn(reference, hypothesis)


def score(reference: str, hypothesis: str) -> dict:
    """Return CER/WER (strict and loose) plus exact-match flags."""
    out = {}
    for suffix, loose in (("", False), ("_loose", True)):
        ref = normalize(reference, loose=loose)
        hyp = normalize(hypothesis, loose=loose)
        out[f"cer{suffix}"] = _rate(jiwer.cer, ref, hyp)
        out[f"wer{suffix}"] = _rate(jiwer.wer, ref, hyp)
        out[f"exact{suffix}"] = ref == hyp
    return out
