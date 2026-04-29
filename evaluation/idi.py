import numpy as np
from scipy.optimize import curve_fit
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


def extract_facts(text):
    """
    Extract facts from text robustly.
    Short answers (single word/phrase < 60 chars) returned as-is.
    Longer text split into sentences.
    This fixes the bug where 'Paris' was filtered out by 10-char minimum.
    """
    if not text or not text.strip():
        return []
    text = text.strip()
    if len(text) < 60:
        return [text]
    sentences = nltk.sent_tokenize(text)
    return [s.strip() for s in sentences if len(s.strip()) > 5]


def keyword_overlap(pred, gold):
    """
    Soft match: checks if gold content appears in prediction.
    For short gold answers (< 4 words), does substring check first.
    For longer gold, uses keyword overlap ratio.
    """
    STOPWORDS = {'that', 'this', 'with', 'from', 'they',
                 'have', 'been', 'were', 'which', 'their',
                 'also', 'more', 'about', 'when', 'than'}

    pred_lower = pred.lower()
    gold_lower = gold.lower().strip()

    # Short answer: direct substring match first
    if len(gold.split()) <= 4:
        if gold_lower in pred_lower:
            return 1.0

    gold_words = set(
        w.lower() for w in gold.split()
        if len(w) > 3 and w.lower() not in STOPWORDS
    )
    pred_words = set(
        w.lower() for w in pred.split()
        if len(w) > 3 and w.lower() not in STOPWORDS
    )
    if not gold_words:
        return 1.0 if gold_lower in pred_lower else 0.0
    overlap = gold_words & pred_words
    return len(overlap) / len(gold_words)


def compute_irr(output_k, gold_answer):
    """
    IRR(k) = |facts(O_k) ∩ F*| / |F*|
    Measures what fraction of the gold answer's content
    survived into this hop's output.
    Range: 0 (nothing retained) to 1 (everything retained).
    """
    if not output_k or not gold_answer:
        return 0.0
    gold_facts = extract_facts(gold_answer)
    pred_facts = extract_facts(output_k)
    if not gold_facts:
        return 0.0

    total_score = 0.0
    for gf in gold_facts:
        best = max(
            (keyword_overlap(pf, gf) for pf in pred_facts),
            default=0.0
        )
        total_score += best
    return total_score / len(gold_facts)


def compute_idi_series(outputs, gold_answer):
    """
    Returns [IDI(1), IDI(2), ..., IDI(n)]
    IDI(k) = 1 - IRR(k)
    0 = perfect retention, 1 = complete fact loss.
    """
    return [1.0 - compute_irr(o, gold_answer) for o in outputs]


def exponential_decay(k, lam):
    """IDI(k) = 1 - e^(-lambda * k)"""
    return 1.0 - np.exp(-lam * k)


def fit_decay_model(idi_series):
    """
    Fits IDI(k) ≈ 1 - e^(-λk) using scipy curve_fit.
    Returns (lambda, R²).
    lambda = decay rate. Lower = better pipeline.
    R² = how well exponential model fits (> 0.8 = good fit).
    """
    if len(idi_series) < 3:
        return None, None
    k_vals = np.arange(1, len(idi_series) + 1, dtype=float)
    idi_arr = np.array(idi_series, dtype=float)
    if np.std(idi_arr) < 0.01:
        return None, None  # Flat — curve fitting meaningless
    try:
        popt, _ = curve_fit(
            exponential_decay, k_vals, idi_arr,
            p0=[0.3], bounds=(0.001, 5.0), maxfev=2000
        )
        lam = float(popt[0])
        predicted = exponential_decay(k_vals, lam)
        ss_res = np.sum((idi_arr - predicted) ** 2)
        ss_tot = np.sum((idi_arr - np.mean(idi_arr)) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-10)
        return lam, float(r2)
    except Exception:
        return None, None
    