import streamlit as st
import sys
import pandas as pd
import numpy as np
sys.path.insert(0, '.')

from pipeline.retriever import BM25Retriever, FAISSRetriever
from pipeline.generator import GroqGenerator
from pipeline.multihop import run_multihop_pipeline
from evaluation.idi import compute_idi_series, fit_decay_model, extract_facts
from evaluation.hallucination import compute_hallucination_rate

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InfoDecay — Multi-Hop RAG Pipeline",
    layout="wide",
    page_icon="📉"
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.hero {
    background: linear-gradient(135deg, #0A1628, #102040);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    border-bottom: 3px solid #00A896;
}
.hero h1 { font-size: 1.8rem; font-weight: 700; color: #FFF; margin-bottom: 4px; }
.hero h1 em { color: #3DD4C2; font-style: normal; }
.hero p { font-size: 0.85rem; color: rgba(255,255,255,0.5); margin: 0; }
.hero .tags { display: flex; gap: 8px; margin-top: 10px; flex-wrap: wrap; }
.tag { background: rgba(0,168,150,0.15); border: 1px solid rgba(0,168,150,0.35);
       color: #3DD4C2; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem;
       font-family: 'JetBrains Mono', monospace; }

.metric-card { background: #F7F9FB; border: 1px solid #D8E0EA; border-radius: 10px;
               padding: 1rem; text-align: center; }
.metric-card .lbl { font-size: 0.72rem; color: #536070; font-weight: 600;
                    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
.metric-card .val { font-size: 1.6rem; font-weight: 700; color: #0A1628; line-height: 1; }
.metric-card .val.good { color: #007A6B; }
.metric-card .val.warn { color: #A85500; }
.metric-card .val.bad  { color: #B83025; }

.hop-card { background: #FFF; border: 1px solid #D8E0EA; border-radius: 10px;
            padding: 1rem 1.2rem; margin-bottom: 0.7rem; }
.hop-card:hover { border-color: #00A896; }
.hop-header { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
.hop-num { width: 30px; height: 30px; border-radius: 50%; background: #0A1628;
           color: #3DD4C2; display: flex; align-items: center; justify-content: center;
           font-weight: 700; font-size: 0.85rem; flex-shrink: 0;
           font-family: 'JetBrains Mono', monospace; }
.hop-num.warn { background: #FDF0E0; color: #A85500; }
.hop-num.bad  { background: #FDECEA; color: #B83025; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 20px;
         font-size: 0.7rem; font-weight: 600; margin-right: 4px; }
.badge-idi { background: #E6F4F2; color: #005048; }
.badge-idi-high { background: #FDECEA; color: #7A1E17; }
.badge-ret-bm25 { background: #EBF2FA; color: #0C2850; }
.badge-ret-faiss { background: #FDF0E0; color: #6B3500; }
.badge-hi { background: #FDECEA; color: #7A1E17; }
.hop-text { font-size: 0.875rem; color: #28394A; line-height: 1.6;
            border-left: 3px solid #D8E0EA; padding-left: 12px; margin-top: 6px; }
.switch-box { background: #FDF0E0; border: 1px solid #F0C080; border-radius: 8px;
              padding: 10px 14px; margin-top: 1rem; }
.switch-box .stitle { font-size: 0.75rem; font-weight: 700; color: #A85500;
                      text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; }
.switch-item { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
               color: #6B3500; line-height: 1.8; }
.insight-box { background: #E6F4F2; border: 1px solid #B8D8D5; border-radius: 8px;
               padding: 12px 16px; margin-top: 1rem; font-size: 0.875rem;
               color: #004A42; line-height: 1.6; }
.insight-box strong { color: #007A6B; }

div[data-testid="stSidebar"] { background: #F0F4F8; border-right: 1px solid #D8E0EA; }
.stButton > button { background: linear-gradient(135deg, #007A6B, #005048);
                     color: white; border: none; border-radius: 8px;
                     padding: 0.6rem 2rem; font-weight: 600; font-size: 0.95rem;
                     width: 100%; }
.stButton > button:hover { opacity: 0.9; border: none; }
footer { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>Info<em>Decay</em> — Multi-Hop RAG Pipeline</h1>
  <p>Measuring factual information degradation across retrieval-augmented generation hops</p>
  <div class="tags">
    <span class="tag">CS F469 Information Retrieval</span>
    <span class="tag">Navya Agarwal · 2023A7PS0359G</span>
    <span class="tag">Pawani Purwar · 2023A7PS0452G</span>
    <span class="tag">BITS Pilani Goa</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get a free key at console.groq.com"
    )
    st.caption("🔑 [Get free key → console.groq.com](https://console.groq.com)")
    st.divider()

    model = st.selectbox(
    "LLM Model",
    [
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "qwen/qwen3-32b",
        "openai/gpt-oss-120b",
    ],
    help="Same three models used in experiments"
)
    st.divider()
    n_hops = st.slider("Pipeline hops", min_value=1, max_value=6, value=4)
    pipeline_type = st.radio(
        "Pipeline type",
        ["Adaptive (IDI-driven switching)", "Fixed BM25 (baseline)"],
        help="Adaptive switches between BM25 and FAISS based on query semantic drift"
    )
    adaptive = pipeline_type.startswith("Adaptive")

    st.divider()
    st.markdown("### 📐 IDI Metric")
    st.markdown("""
**IRR(k)** = |facts(Oₖ) ∩ F\\*| / |F\\*|

**IDI(k)** = 1 − IRR(k)

**Model:** IDI(k) ≈ 1 − e^(−λk)

- IDI = 0 → all facts kept
- IDI = 1 → all facts lost
- **λ** = decay rate scalar
- **Negative R²** → adaptive switching disrupted monotonic decay (active interruption)
""")

# ── Main layout ───────────────────────────────────────────────────────────────
left, right = st.columns([1, 1.1], gap="large")

with left:
    st.subheader("Input")

    question = st.text_input(
        "Multi-hop question",
        value="Who was the US president when the Eiffel Tower was built?",
        help="Ask a factoid question that requires chaining multiple pieces of evidence"
    )

    passages_raw = st.text_area(
        "Context passages — one per line",
        value=(
            "The Eiffel Tower was built between 1887 and 1889 in Paris, France.\n"
            "Grover Cleveland was the 22nd and 24th President of the United States.\n"
            "Cleveland served as president from 1885 to 1889.\n"
            "The Eiffel Tower was designed by Gustave Eiffel and opened in 1889.\n"
            "Benjamin Harrison served as president from 1889 to 1893.\n"
            "The tower stands 330 metres tall on the Champ de Mars in Paris.\n"
            "Cleveland lost the 1888 election to Harrison but won again in 1892.\n"
            "The Eiffel Tower was the world's tallest structure when it was built.\n"
            "Grover Cleveland was born in New Jersey in 1837."
        ),
        height=220,
        help="Paste relevant context passages. The pipeline retrieves from these."
    )

    gold_answer = st.text_input(
        "Gold answer (used to compute IDI)",
        value="Grover Cleveland was the US president when the Eiffel Tower was built in 1889.",
        help="The correct answer. IDI measures how much of this survives across hops."
    )

    run = st.button("▶ Run Pipeline", type="primary")

# ── Results ───────────────────────────────────────────────────────────────────
with right:
    st.subheader("Results")

    if run:
        if not api_key:
            st.error("Enter your Groq API key in the sidebar.")
            st.stop()

        corpus = [l.strip() for l in passages_raw.split("\n") if l.strip()]
        if len(corpus) < 2:
            st.error("Add at least 2 context passages.")
            st.stop()

        with st.spinner(f"Running {n_hops}-hop {'adaptive' if adaptive else 'fixed'} pipeline..."):
            try:
                bm25 = BM25Retriever(corpus)
                faiss_r = FAISSRetriever(corpus)
                generator = GroqGenerator(api_key=api_key, model=model)
                gold_facts = extract_facts(gold_answer)

                result = run_multihop_pipeline(
                    question, corpus, bm25, faiss_r, generator,
                    n_hops=n_hops, adaptive=adaptive
                )
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.stop()

        outputs        = result['outputs']
        retrievers_used = result['retrievers_used']
        switch_log     = result['switch_log']

        idi_series  = compute_idi_series(outputs, gold_answer)
        lam, r2     = fit_decay_model(idi_series)
        hi_series   = [compute_hallucination_rate(o, corpus) for o in outputs]
        n_switches  = sum(
            1 for i in range(1, len(retrievers_used))
            if retrievers_used[i] != retrievers_used[i-1]
        )

        # ── Metric cards ──────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)

        lam_cls = "good" if lam and lam < 0.3 else "warn" if lam and lam < 0.6 else "bad"
        r2_cls  = "good" if r2 and r2 > 0.5 else "warn"
        avg_hi  = sum(hi_series) / len(hi_series)
        hi_cls  = "good" if avg_hi < 0.3 else "warn" if avg_hi < 0.6 else "bad"

        c1.markdown(f"""<div class="metric-card">
            <div class="lbl">Pipeline</div>
            <div class="val" style="font-size:1rem;padding-top:4px">
              {'Adaptive' if adaptive else 'Fixed'}</div></div>""",
            unsafe_allow_html=True)

        c2.markdown(f"""<div class="metric-card">
            <div class="lbl">Decay rate λ</div>
            <div class="val {lam_cls}">{f"{lam:.3f}" if lam else "∞"}</div>
            </div>""", unsafe_allow_html=True)

        c3.markdown(f"""<div class="metric-card">
            <div class="lbl">R²</div>
            <div class="val {r2_cls}">{f"{r2:.3f}" if r2 else "flat"}</div>
            </div>""", unsafe_allow_html=True)

        c4.markdown(f"""<div class="metric-card">
            <div class="lbl">Switches</div>
            <div class="val">{n_switches}</div>
            </div>""", unsafe_allow_html=True)

        st.divider()

        # ── Hop-by-hop trace ──────────────────────────────────────────────────
        st.markdown("**Hop-by-hop trace**")

        for k, (out, idi, hi, ret) in enumerate(
                zip(outputs, idi_series, hi_series, retrievers_used), 1):

            num_cls   = "bad" if idi > 0.7 else "warn" if idi > 0.4 else ""
            idi_badge = "badge-idi-high" if idi > 0.7 else "badge-idi"
            ret_badge = "badge-ret-faiss" if ret == "faiss" else "badge-ret-bm25"

            st.markdown(f"""
            <div class="hop-card">
              <div class="hop-header">
                <div class="hop-num {num_cls}">{k}</div>
                <div>
                  <span class="badge {idi_badge}">IDI: {idi:.3f}</span>
                  <span class="badge {ret_badge}">{ret.upper()}</span>
                  <span class="badge badge-hi">HI: {hi:.3f}</span>
                </div>
              </div>
              <div class="hop-text">{out}</div>
            </div>""", unsafe_allow_html=True)

        # ── Switching log ─────────────────────────────────────────────────────
        if adaptive and switch_log:
            log_html = "".join(
                f'<div class="switch-item">{s}</div>' for s in switch_log
            )
            st.markdown(f"""
            <div class="switch-box">
              <div class="stitle">🔀 Retriever switching log</div>
              {log_html}
            </div>""", unsafe_allow_html=True)

        # ── IDI decay chart ───────────────────────────────────────────────────
        st.divider()
        st.markdown("**IDI decay curve**")

        chart_df = pd.DataFrame({
            "IDI(k) — information lost": idi_series,
            "Hallucination rate":         hi_series,
        }, index=range(1, len(idi_series) + 1))

        st.line_chart(chart_df, color=["#007A6B", "#B83025"])

        # ── Auto-generated insight ────────────────────────────────────────────
        if lam:
            trend = "low" if lam < 0.3 else "moderate" if lam < 0.6 else "high"
            insight = (
                f"<strong>λ = {lam:.3f}</strong> ({trend} decay rate). "
                f"Hallucination rose from <strong>{hi_series[0]:.3f}</strong> "
                f"at hop 1 to <strong>{hi_series[-1]:.3f}</strong> at hop {n_hops}. "
            )
            if r2 and r2 < 0:
                insight += (
                    "Negative R² indicates <strong>non-monotonic IDI</strong> — "
                    "the adaptive switcher interrupted the decay process."
                )
            elif r2:
                insight += f"Exponential model fit: R² = {r2:.3f}."
            st.markdown(
                f'<div class="insight-box">{insight}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="insight-box">IDI series is flat — pipeline '
                'reached maximum decay at hop 1 (retrieval saturation). '
                'Exponential model cannot be fitted to a flat series.</div>',
                unsafe_allow_html=True
            )

    else:
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;color:#9EAFC0">
          <div style="font-size:3rem;margin-bottom:1rem">📉</div>
          <div style="font-size:1rem;font-weight:600;color:#536070">
            Configure the pipeline and click Run
          </div>
          <div style="font-size:0.85rem;margin-top:6px">
            Watch factual decay unfold hop by hop
          </div>
        </div>""", unsafe_allow_html=True)