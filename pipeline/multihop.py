import numpy as np
from pipeline.retriever import BM25Retriever, FAISSRetriever

SWITCH_THRESHOLD = 0.15  # If IDI jumps more than this in one hop, switch to FAISS


def cosine_sim(v1, v2):
    denom = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10
    return float(np.dot(v1, v2) / denom)


def build_prompt(question, passages, context):
    passages_text = '\n'.join(
        [f"Passage {i+1}: {p}" for i, p in enumerate(passages)]
    )
    return f"""Context passages:
{passages_text}

Previous answer: {context if context else "None yet"}

Question: {question}

Answer using ONLY the passages above. Be concise:"""


def run_multihop_pipeline(question, corpus, bm25_retriever,
                          faiss_retriever, generator,
                          n_hops=6, adaptive=True):
    """
    Runs n_hops of retrieve -> generate.

    If adaptive=True: uses IDI-driven retriever switching.
      - Start with BM25
      - After each hop, compute IDI delta
      - If IDI jumped > SWITCH_THRESHOLD, switch to FAISS
        (semantic retrieval handles vocabulary drift better)
      - If IDI stable, revert to BM25

    If adaptive=False: uses BM25 for all hops (baseline).

    Returns dict with:
      outputs       -- LLM answer at each hop
      retrievers_used -- which retriever was used at each hop
      idi_series    -- IDI value at each hop (computed externally)
      switch_log    -- hop-by-hop switching decisions
    """
    outputs = []
    context = ""
    retrievers_used = []
    switch_log = []
    current_retriever = 'bm25'
    prev_idi = 0.0

    # Encode original question for drift measurement
    q_vec = bm25_retriever.encode(question)

    for k in range(1, n_hops + 1):
        query = question if k == 1 else outputs[-1]

        # Select retriever for this hop
        if adaptive and k > 1:
            # Compute semantic drift: how far has query drifted
            # from original question?
            q_k_vec = bm25_retriever.encode(query)
            drift = 1.0 - cosine_sim(q_vec, q_k_vec)

            # Approximate IDI delta from drift
            # High drift = BM25 will fail = switch to FAISS
            if drift > SWITCH_THRESHOLD:
                current_retriever = 'faiss'
                switch_log.append(
                    f"Hop {k}: drift={drift:.3f} > {SWITCH_THRESHOLD}"
                    f" → switched to FAISS"
                )
            else:
                current_retriever = 'bm25'
                switch_log.append(
                    f"Hop {k}: drift={drift:.3f} ≤ {SWITCH_THRESHOLD}"
                    f" → stayed with BM25"
                )
        else:
            switch_log.append(
                f"Hop {k}: {'adaptive start' if k==1 else 'fixed BM25'}"
            )

        # Retrieve using selected retriever
        if current_retriever == 'faiss':
            passages = faiss_retriever.retrieve(query, top_k=5)
        else:
            passages = bm25_retriever.retrieve(query, top_k=5)

        retrievers_used.append(current_retriever)

        # Generate
        prompt = build_prompt(question, passages, context)
        output_k = generator.generate(prompt)
        outputs.append(output_k)
        context = output_k

    return {
        'outputs': outputs,
        'retrievers_used': retrievers_used,
        'switch_log': switch_log
    }