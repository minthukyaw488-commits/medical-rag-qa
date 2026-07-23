"""
rag.py — Retrieval + generation logic. Imported by app.py.
No Streamlit code here — this file is pure logic.
"""

import os

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pickle
from pathlib import Path

import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

STORE_DIR = Path("store")
EMBED_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"  # medical-tuned, 768-dim
LLM_MODEL = "llama-3.3-70b-versatile"
TOP_K = 4

SYSTEM_PROMPT = """You are a careful medical information assistant.
Answer ONLY using the provided context. Rules:
- If the context doesn't contain the answer, say you don't have that information.
- Be clear and concise. Use plain language.
- Never give a diagnosis, prescription, or dosage.
- End every answer with: "This is general information, not medical advice. Consult a doctor."
"""


class MedicalRAG:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.embeddings = np.load(STORE_DIR / "embeddings.npy")
        with open(STORE_DIR / "meta.pkl", "rb") as fh:
            self.metadata = pickle.load(fh)
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])

    def retrieve(self, query: str, k: int = TOP_K) -> list[dict]:
        q = self.embedder.encode([query], normalize_embeddings=True)
        q = np.asarray(q, dtype="float32")[0]
        scores = self.embeddings @ q
        top = np.argsort(-scores)[:k]
        results = []
        for idx in top:
            item = dict(self.metadata[int(idx)])
            item["score"] = float(scores[idx])
            results.append(item)
        return results

    def answer(self, query: str) -> dict:
        hits = self.retrieve(query)
        context = "\n\n---\n\n".join(f"[{h['source']}] {h['text']}" for h in hits)
        user_msg = f"Context:\n{context}\n\nQuestion: {query}"
        resp = self.client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        return {"answer": resp.choices[0].message.content, "sources": hits}


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What is pneumonia?"
    out = MedicalRAG().answer(q)
    print("\n" + out["answer"])
    print("\nSources:", [s["source"] for s in out["sources"]])
