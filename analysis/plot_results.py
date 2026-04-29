"""
plot_results.py — generates all figures for InfoDecay paper/poster
Run after run_all.py completes.
Saves figures/ as PDF (vector) and PNG (preview).
"""

import json, glob, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

# ── AAAI-style settings ───────────────────────────────────────────────────────
mpl.rcParams.update({
    'font.family'     : 'serif',
    'font.size'       : 11,
    'axes.spines.top' : False,
    'axes.spines.right': False,
    'figure.dpi'      : 300,
    'savefig.bbox'    : 'tight',
    'savefig.facecolor': 'white',
})

COLORS = {
    'meta-llama/llama-4-scout-17b-16e-instruct' : '#05897B',  # teal
    'qwen/qwen3-32b'                            : '#D97706',  # amber
    'openai/gpt-oss-120b'                       : '#1A4F8A',  # blue
}
LABELS = {
    'meta-llama/llama-4-scout-17b-16e-instruct' : 'LLaMA-4-Scout-17B (Meta)',
    'qwen/qwen3-32b'                            : 'Qwen3-32B (Alibaba)',
    'openai/gpt-oss-120b'                       : 'GPT-OSS-120B (OpenAI)',
}
HOPS = list(range(1, 7))
os.makedirs('figures', exist_ok=True)

# ── Load results ──────────────────────────────────────────────────────────────

def load_all():
    rows = []
    for path in glob.glob('results/*.json'):
        try:
            data = json.load(open(path, encoding='utf-8'))
        except Exception:
            continue
        for d in data:
            if not d.get('idi') or len(d['idi']) < 6:
                continue
            for hop_idx in range(6):
                rows.append({
                    'model'        : d['model'],
                    'pipeline_type': d['pipeline_type'],
                    'hop'          : hop_idx + 1,
                    'idi'          : d['idi'][hop_idx],
                    'hi_rate'      : d['hi_rate'][hop_idx]
                                     if d.get('hi_rate') else 0,
                    'lambda'       : d.get('lambda'),
                    'r2'           : d.get('r2'),
                    'n_switches'   : d.get('n_switches', 0),
                })
    return pd.DataFrame(rows)

# ── Figure 1 — IDI decay curves by model, fixed pipeline (RQ1) ───────────────

def plot_fig1(df):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    fixed = df[df.pipeline_type == 'fixed']

    for model, color in COLORS.items():
        sub = fixed[fixed.model == model].groupby('hop')['idi'].mean()
        if sub.empty:
            continue
        ax.plot(sub.index, sub.values, 'o-',
                color=color, linewidth=2, markersize=5,
                label=LABELS[model])

    ax.set_xlabel('Pipeline hop depth')
    ax.set_ylabel('IDI(k) — Information Decay Index')
    ax.set_xticks(HOPS)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title('Fig 1 — IDI across hops (RQ1: Is decay monotonic?)',
                 fontsize=10, pad=8)
    plt.tight_layout()
    plt.savefig('figures/fig1_idi_decay.pdf')
    plt.savefig('figures/fig1_idi_decay.png', dpi=300)
    print("Saved fig1_idi_decay")
    plt.close()

# ── Figure 2 — Lambda by model (RQ2) ─────────────────────────────────────────

def plot_fig2(df):
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    fixed = df[df.pipeline_type == 'fixed']
    models = list(COLORS.keys())

    lambdas, colors, labels = [], [], []
    for m in models:
        lam = fixed[fixed.model == m]['lambda'].dropna().mean()
        if not np.isnan(lam):
            lambdas.append(lam)
            colors.append(COLORS[m])
            labels.append(LABELS[m])

    bars = ax.bar(range(len(labels)), lambdas, color=colors,
                  width=0.5, alpha=0.88)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, rotation=10)
    ax.set_ylabel('Decay rate λ (lower = better)')

    for bar, val in zip(bars, lambdas):
        ax.text(bar.get_x() + bar.get_width()/2,
                val + 0.003, f'{val:.3f}',
                ha='center', va='bottom', fontsize=9)

    ax.set_title('Fig 2 — Decay rate λ by model (RQ2)',
                 fontsize=10, pad=8)
    plt.tight_layout()
    plt.savefig('figures/fig2_lambda_by_model.pdf')
    plt.savefig('figures/fig2_lambda_by_model.png', dpi=300)
    print("Saved fig2_lambda_by_model")
    plt.close()

# ── Figure 3 — Adaptive vs Fixed IDI at each hop (RQ3 — your contribution) ───

def plot_fig3(df):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)

    for ax, model in zip(axes, COLORS.keys()):
        adaptive = df[(df.model == model) &
                      (df.pipeline_type == 'adaptive')
                      ].groupby('hop')['idi'].mean()
        fixed = df[(df.model == model) &
                   (df.pipeline_type == 'fixed')
                   ].groupby('hop')['idi'].mean()

        if not adaptive.empty:
            ax.plot(adaptive.index, adaptive.values, 'o-',
                    color=COLORS[model], linewidth=2,
                    markersize=5, label='Adaptive (ours)')
        if not fixed.empty:
            ax.plot(fixed.index, fixed.values, 's--',
                    color=COLORS[model], linewidth=1.5,
                    markersize=5, alpha=0.6, label='Fixed BM25')

        ax.set_title(LABELS[model], fontsize=9)
        ax.set_xlabel('Hop depth')
        ax.set_xticks(HOPS)
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=False, fontsize=8)

    axes[0].set_ylabel('IDI(k)')
    fig.suptitle(
        'Fig 3 — Adaptive switching vs Fixed BM25 (RQ3: Does IDI-driven '
        'switching reduce decay?)',
        fontsize=10)
    plt.tight_layout()
    plt.savefig('figures/fig3_adaptive_vs_fixed.pdf')
    plt.savefig('figures/fig3_adaptive_vs_fixed.png', dpi=300)
    print("Saved fig3_adaptive_vs_fixed")
    plt.close()

# ── Figure 4 — Hallucination rate across hops (RQ4) ──────────────────────────

def plot_fig4(df):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    fixed = df[df.pipeline_type == 'fixed']
    hi_by_hop = fixed.groupby('hop')['hi_rate'].mean()

    ax.fill_between(hi_by_hop.index, hi_by_hop.values,
                    alpha=0.15, color='#B91C1C')
    ax.plot(hi_by_hop.index, hi_by_hop.values,
            'o-', color='#B91C1C', linewidth=2, markersize=6)

    for hop, val in zip(hi_by_hop.index, hi_by_hop.values):
        ax.text(hop, val + 0.01, f'{val:.2f}',
                ha='center', fontsize=9, color='#B91C1C')

    # Mark inflection point
    diffs = np.diff(hi_by_hop.values)
    if len(diffs) > 0:
        inflection = int(np.argmax(diffs)) + 2
        ax.axvline(x=inflection, color='#888', linestyle=':', alpha=0.7)
        ax.text(inflection + 0.1, hi_by_hop.max() * 0.75,
                f'Inflection\nhop {inflection}', fontsize=8, color='#666')

    ax.set_xlabel('Pipeline hop depth')
    ax.set_ylabel('Hallucination rate')
    ax.set_xticks(HOPS)
    ax.set_ylim(0, min(1.05, hi_by_hop.max() * 1.4 + 0.1))
    ax.set_title('Fig 4 — Hallucination grows with hop depth (RQ4)',
                 fontsize=10, pad=8)
    plt.tight_layout()
    plt.savefig('figures/fig4_hallucination.pdf')
    plt.savefig('figures/fig4_hallucination.png', dpi=300)
    print("Saved fig4_hallucination")
    plt.close()

# ── Figure 5 — Lambda: Adaptive vs Fixed per model (RQ3 summary) ─────────────

def plot_fig5(df):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    models = list(COLORS.keys())
    x = np.arange(len(models))
    w = 0.35

    lam_fixed    = [df[(df.model==m) & (df.pipeline_type=='fixed')
                       ]['lambda'].dropna().mean() for m in models]
    lam_adaptive = [df[(df.model==m) & (df.pipeline_type=='adaptive')
                       ]['lambda'].dropna().mean() for m in models]

    b1 = ax.bar(x - w/2, lam_fixed,    w, label='Fixed BM25',
                color='#888780', alpha=0.85)
    b2 = ax.bar(x + w/2, lam_adaptive, w, label='Adaptive (ours)',
                color='#1D9E75', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in models], fontsize=8, rotation=10)
    ax.set_ylabel('Decay rate λ (lower = better)')
    ax.legend(frameon=False, fontsize=9)
    ax.set_title('Fig 5 — Adaptive switching reduces λ (RQ3 summary)',
                 fontsize=10, pad=8)
    plt.tight_layout()
    plt.savefig('figures/fig5_lambda_comparison.pdf')
    plt.savefig('figures/fig5_lambda_comparison.png', dpi=300)
    print("Saved fig5_lambda_comparison")
    plt.close()

# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(df):
    print("\n" + "="*80)
    print("RESULTS SUMMARY TABLE")
    print("="*80)
    print(f"{'Model':<28} {'Pipeline':<10} {'λ':>6} {'R²':>7} "
          f"{'IDI@h1':>7} {'IDI@h6':>7} {'HI@h6':>7} {'Switches':>9}")
    print("-"*80)

    for model in COLORS:
        for pt in ['fixed', 'adaptive']:
            sub = df[(df.model == model) & (df.pipeline_type == pt)]
            if sub.empty:
                continue
            lam  = sub['lambda'].dropna().mean()
            r2   = sub['r2'].dropna().mean()
            idi1 = sub[sub.hop == 1]['idi'].mean()
            idi6 = sub[sub.hop == 6]['idi'].mean()
            hi6  = sub[sub.hop == 6]['hi_rate'].mean()
            # avg switches per question
            raw  = [d.get('n_switches', 0)
                    for f in glob.glob('results/*.json')
                    for d in json.load(open(f, encoding='utf-8'))
                    if d.get('model') == model
                    and d.get('pipeline_type') == pt]
            avg_sw = np.mean(raw) if raw else 0

            lam_str = f'{lam:.3f}' if not np.isnan(lam) else ' N/A'
            r2_str  = f'{r2:.3f}'  if not np.isnan(r2)  else ' N/A'
            print(f"{LABELS[model]:<28} {pt:<10} {lam_str:>6} {r2_str:>7} "
                  f"{idi1:>7.3f} {idi6:>7.3f} {hi6:>7.3f} {avg_sw:>9.1f}")
    print("="*80)

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    df = load_all()
    if df.empty:
        print("No results found. Run experiments/run_all.py first.")
        exit(1)

    print(f"Loaded {len(df)} rows — "
          f"{df['model'].nunique()} models, "
          f"{df['pipeline_type'].nunique()} pipeline types")

    plot_fig1(df)
    plot_fig2(df)
    plot_fig3(df)
    plot_fig4(df)
    plot_fig5(df)
    print_summary(df)
    print("\nAll figures saved to figures/")
    print("Use PDF versions for poster (vector, crisp at A0 scale)")
