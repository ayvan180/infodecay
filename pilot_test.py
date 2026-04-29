import sys
sys.path.insert(0, '.')

from pipeline.retriever import BM25Retriever, FAISSRetriever
from pipeline.generator import GroqGenerator
from pipeline.multihop import run_multihop_pipeline
from evaluation.idi import compute_idi_series, fit_decay_model
from evaluation.hallucination import compute_hallucination_rate

GROQ_API_KEY = "paste_your_key_here"

question = 'Who was the US president when the Eiffel Tower was built?'
corpus = [
    'The Eiffel Tower was built between 1887 and 1889 in Paris, France.',
    'Grover Cleveland was the 22nd and 24th President of the United States.',
    'Cleveland served as president from 1885 to 1889.',
    'The Eiffel Tower was designed by Gustave Eiffel and opened in 1889.',
    'Benjamin Harrison served as president from 1889 to 1893.',
    'The tower stands 330 metres tall on the Champ de Mars in Paris.',
    'Cleveland lost the 1888 election to Harrison but won again in 1892.',
]
gold_answer = 'Grover Cleveland was the US president when the Eiffel Tower was built in 1889.'
gold_docs = corpus

print('Setting up retrievers...')
bm25 = BM25Retriever(corpus)
faiss = FAISSRetriever(corpus)
print('OK')

print('Setting up generator (LLaMA-3.3-70B via Groq)...')
gen = GroqGenerator(api_key=GROQ_API_KEY)
print('OK')

print('Running ADAPTIVE pipeline (6 hops)...')
result = run_multihop_pipeline(
    question, corpus, bm25, faiss, gen,
    n_hops=6, adaptive=True
)
print('OK')

print()
print('=' * 60)
idi = compute_idi_series(result['outputs'], gold_answer)
lam, r2 = fit_decay_model(idi)

for k, (out, idi_k, ret) in enumerate(
        zip(result['outputs'], idi, result['retrievers_used']), 1):
    hi = compute_hallucination_rate(out, gold_docs)
    print(f"Hop {k} [{ret.upper():5s}] IDI:{idi_k:.3f} "
          f"HI:{hi:.3f} | {out[:60]}...")

print('=' * 60)
print()
print('Switching log:')
for log in result['switch_log']:
    print(' ', log)
print()
print(f'Lambda : {lam:.3f}' if lam else 'Lambda: flat series')
print(f'R²     : {r2:.3f}' if r2 else '')
print()
print('PILOT TEST PASSED')