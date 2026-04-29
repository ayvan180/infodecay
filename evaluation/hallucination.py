from evaluation.idi import extract_facts, keyword_overlap


def compute_hallucination_rate(output_k, gold_docs):
    """
    Hallucination rate = fraction of output facts NOT supported
    by any gold document.
    
    'Not found in context' is treated as abstention, not hallucination.
    Abstention rate is returned separately as 0 hallucination.
    """
    if not output_k or not gold_docs:
        return 0.0

    output_facts = extract_facts(output_k)
    if not output_facts:
        return 0.0

    hallucinated = 0
    for fact in output_facts:
        max_support = max(
            (keyword_overlap(fact, doc) for doc in gold_docs),
            default=0.0
        )
        if max_support < 0.25:
            hallucinated += 1

    return hallucinated / len(output_facts)