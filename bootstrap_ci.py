"""
Bootstrap confidence intervals for InfoDecay results.

Computes 95% CIs on the % reduction in IDI@h6 and mean hallucination rate
between fixed and adaptive pipelines, paired per question, per model.

Usage:
    Save this file in the InfoDecay project root (next to the `results/` folder)
    and run: python bootstrap_ci.py
"""

import json
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("results")
N_BOOTSTRAP = 10000
SEED = 42

MODELS = {
    "LLaMA-4-Scout-17B": (
        "meta_llama_llama_4_scout_17b_16e_instruct_fixed.json",
        "meta_llama_llama_4_scout_17b_16e_instruct_adaptive.json",
    ),
    "Qwen3-32B": (
        "qwen_qwen3_32b_fixed.json",
        "qwen_qwen3_32b_adaptive.json",
    ),
    "GPT-OSS-120B": (
        "openai_gpt_oss_120b_fixed.json",
        "openai_gpt_oss_120b_adaptive.json",
    ),
}


def load_results(filename):
    with open(RESULTS_DIR / filename, encoding="utf-8") as f:
        return json.load(f)


def pair_by_question(fixed_records, adaptive_records):
    """Pair fixed and adaptive results on the same question text."""
    fixed_by_q = {r["question"]: r for r in fixed_records}
    adaptive_by_q = {r["question"]: r for r in adaptive_records}
    common = sorted(set(fixed_by_q) & set(adaptive_by_q))
    return [(fixed_by_q[q], adaptive_by_q[q]) for q in common]


def bootstrap_pct_reduction(fixed_vals, adaptive_vals, n_boot=N_BOOTSTRAP, seed=SEED):
    """Paired bootstrap on % reduction: (fixed_mean - adaptive_mean) / fixed_mean * 100."""
    rng = np.random.default_rng(seed)
    fixed_vals = np.asarray(fixed_vals, dtype=float)
    adaptive_vals = np.asarray(adaptive_vals, dtype=float)
    n = len(fixed_vals)

    fixed_mean = fixed_vals.mean()
    adaptive_mean = adaptive_vals.mean()
    point = (fixed_mean - adaptive_mean) / fixed_mean * 100 if fixed_mean > 0 else np.nan

    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        f = fixed_vals[idx].mean()
        a = adaptive_vals[idx].mean()
        deltas[i] = (f - a) / f * 100 if f > 0 else np.nan

    deltas = deltas[~np.isnan(deltas)]
    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    return point, ci_low, ci_high, n, fixed_mean, adaptive_mean


def main():
    print(f"\n{'Model':<22} {'Metric':<22} {'Fixed':>8} {'Adapt':>8} {'Reduction':>10} {'95% CI':>22} {'n':>4}")
    print("-" * 100)

    for model_name, (fixed_file, adaptive_file) in MODELS.items():
        try:
            fixed_records = load_results(fixed_file)
            adaptive_records = load_results(adaptive_file)
        except FileNotFoundError as e:
            print(f"  [skip] {model_name}: {e}")
            continue

        pairs = pair_by_question(fixed_records, adaptive_records)
        if not pairs:
            print(f"  [skip] {model_name}: no paired questions")
            continue

        # IDI at hop 6 (last hop)
        idi_fixed = [p[0]["idi"][5] for p in pairs]
        idi_adaptive = [p[1]["idi"][5] for p in pairs]

        # Mean hallucination rate across all 6 hops, per question
        hi_fixed = [float(np.mean(p[0]["hi_rate"])) for p in pairs]
        hi_adaptive = [float(np.mean(p[1]["hi_rate"])) for p in pairs]

        idi_pt, idi_lo, idi_hi, n, idi_f_mean, idi_a_mean = bootstrap_pct_reduction(
            idi_fixed, idi_adaptive
        )
        hi_pt, hi_lo, hi_hi, _, hi_f_mean, hi_a_mean = bootstrap_pct_reduction(
            hi_fixed, hi_adaptive
        )

        ci_str_idi = f"({idi_lo:+.1f}%, {idi_hi:+.1f}%)"
        ci_str_hi = f"({hi_lo:+.1f}%, {hi_hi:+.1f}%)"

        print(f"{model_name:<22} {'IDI@h6':<22} {idi_f_mean:>8.3f} {idi_a_mean:>8.3f} {idi_pt:>9.1f}% {ci_str_idi:>22} {n:>4}")
        print(f"{model_name:<22} {'Hallucination (mean)':<22} {hi_f_mean:>8.3f} {hi_a_mean:>8.3f} {hi_pt:>9.1f}% {ci_str_hi:>22} {n:>4}")
        print()

    print("Notes:")
    print("- 'Reduction' is % decrease from fixed to adaptive (positive = adaptive better)")
    print("- 95% CI from 10,000 paired bootstrap resamples")
    print("- IDI@h6 = information decay at final (6th) hop, lower is better")
    print("- Hallucination = mean hi_rate across all 6 hops, lower is better")


if __name__ == "__main__":
    main()
