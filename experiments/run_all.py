"""
run_all.py — InfoDecay experiment runner

Conditions:
  3 models × 2 pipeline types (adaptive vs fixed-BM25) × 50 questions × 6 hops
  = 6 conditions, 1800 total API calls (~60 minutes on free tier)

Pipeline types:
  adaptive  — IDI-driven retriever switching (our contribution)
  fixed     — BM25 only for all hops (baseline)

Results saved after every question — crash safe.
Re-running resumes from where it left off.
"""

import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tqdm import tqdm
from pipeline.retriever import BM25Retriever, FAISSRetriever
from pipeline.generator import GroqGenerator
from pipeline.multihop import run_multihop_pipeline
from evaluation.idi import compute_idi_series, fit_decay_model
from evaluation.hallucination import compute_hallucination_rate

# ── Configuration ─────────────────────────────────────────────────────────────

GROQ_API_KEY = "paste_your_key_here"

N_SAMPLES  = 50
N_HOPS     = 6
SLEEP_SEC  = 2.5   # pause between API calls to stay under rate limit

MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",       # Meta 70B — large
    "qwen/qwen3-32b",                # Alibaba 32B — medium, different architecture
    "openai/gpt-oss-120b",           # OpenAI 120B — largest
]

PIPELINE_TYPES = ["adaptive", "fixed"]
# adaptive = IDI-driven switching between BM25 and FAISS (our contribution)
# fixed    = BM25 only for all hops (baseline to compare against)

DATA_FILE = "data/hotpotqa_dev_500.jsonl"

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_samples(path, n):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(l) for l in f][:n]

def get_corpus(sample):
    """Extract passage strings from HotpotQA distractor sample."""
    passages = []
    ctx = sample.get('context', {})
    titles = ctx.get('title', [])
    sentences = ctx.get('sentences', [])
    for title, sents in zip(titles, sentences):
        for sent in sents:
            if sent.strip():
                passages.append(f"{title}: {sent.strip()}")
    return passages if passages else [sample.get('question', '')]

def result_path(model, pipeline_type):
    safe = model.replace('/', '_').replace('-', '_').replace('.', '_')
    os.makedirs('results', exist_ok=True)
    return f"results/{safe}_{pipeline_type}.json"

def load_existing(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

# ── Single condition runner ───────────────────────────────────────────────────

def run_condition(model, pipeline_type, samples):
    out_path = result_path(model, pipeline_type)
    results  = load_existing(out_path)
    done_qs  = {r['question'] for r in results}

    print(f"\n{'='*60}")
    print(f"  Model         : {model}")
    print(f"  Pipeline type : {pipeline_type.upper()}")
    print(f"  Samples       : {N_SAMPLES}  |  Hops: {N_HOPS}")
    print(f"  Resuming from : {len(results)} completed")
    print(f"{'='*60}")

    generator = GroqGenerator(api_key=GROQ_API_KEY, model=model)
    is_adaptive = (pipeline_type == 'adaptive')

    for sample in tqdm(samples, desc=f"{model[:25]}/{pipeline_type}"):
        question = sample.get('question', '').strip()
        if not question or question in done_qs:
            continue

        gold_answer = str(sample.get('answer', '')).strip()
        corpus = get_corpus(sample)
        if not corpus or not gold_answer:
            continue

        try:
            bm25  = BM25Retriever(corpus)
            faiss = FAISSRetriever(corpus)
        except Exception as e:
            print(f"\nRetriever error: {e}")
            continue

        try:
            result = run_multihop_pipeline(
                question, corpus, bm25, faiss, generator,
                n_hops=N_HOPS, adaptive=is_adaptive
            )
        except Exception as e:
            print(f"\nPipeline error: {e}")
            time.sleep(5)
            continue

        outputs        = result['outputs']
        retrievers_used = result['retrievers_used']
        switch_log     = result['switch_log']

        idi_series = compute_idi_series(outputs, gold_answer)
        lam, r2    = fit_decay_model(idi_series)
        hi_series  = [compute_hallucination_rate(o, corpus)
                      for o in outputs]
        n_switches = sum(
            1 for i in range(1, len(retrievers_used))
            if retrievers_used[i] != retrievers_used[i-1]
        )

        results.append({
            'question'       : question,
            'gold_answer'    : gold_answer,
            'model'          : model,
            'pipeline_type'  : pipeline_type,
            'outputs'        : outputs,
            'retrievers_used': retrievers_used,
            'switch_log'     : switch_log,
            'idi'            : idi_series,
            'lambda'         : lam,
            'r2'             : r2,
            'hi_rate'        : hi_series,
            'n_switches'     : n_switches,
        })

        # Save after every question — crash safe
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

        time.sleep(SLEEP_SEC)

    print(f"  Saved {len(results)} results → {out_path}")
    return results

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("InfoDecay Experiment Runner")
    print(f"Models: {len(MODELS)} | Pipelines: {len(PIPELINE_TYPES)}"
          f" | Samples: {N_SAMPLES} | Hops: {N_HOPS}")
    print(f"Total API calls: ~{len(MODELS)*len(PIPELINE_TYPES)*N_SAMPLES*N_HOPS}")

    samples = load_samples(DATA_FILE, N_SAMPLES)
    print(f"Loaded {len(samples)} samples\n")

    for model in MODELS:
        for pipeline_type in PIPELINE_TYPES:
            try:
                run_condition(model, pipeline_type, samples)
            except KeyboardInterrupt:
                print("\nInterrupted — progress saved. Re-run to resume.")
                sys.exit(0)
            except Exception as e:
                print(f"\nCondition failed ({model}/{pipeline_type}): {e}")
                print("Continuing to next condition...")
                time.sleep(10)
                continue

    print("\n" + "="*60)
    print("ALL CONDITIONS COMPLETE")
    print("Now run: python analysis/plot_results.py")
    print("="*60)

if __name__ == '__main__':
    main()