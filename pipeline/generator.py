import re
from groq import Groq

SYSTEM_PROMPT = """You are a precise factual assistant answering multi-hop questions.
Rules:
1. Answer ONLY using the provided context passages.
2. Be concise — 2-3 sentences maximum.
3. If context does not contain the answer, say: "Not found in context."
4. Do not add information you were not given."""


class GroqGenerator:
    """
    Calls LLMs via Groq API. Free, no local RAM needed.
    Supports: llama-3.3-70b-versatile, qwen/qwen3-32b, openai/gpt-oss-120b
    """

    def __init__(self, api_key, model='llama-3.3-70b-versatile'):
        self.client = Groq(api_key=api_key)
        self.model = model

    def generate(self, prompt, max_tokens=300):
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.0
            )
            raw = resp.choices[0].message.content.strip()
            # Strip <think>...</think> tags from Qwen3 reasoning output
            clean = re.sub(r'<think>.*?</think>', '', raw,
                           flags=re.DOTALL).strip()
            return clean if clean else raw
        except Exception as e:
            return f"[Generation failed: {e}]"