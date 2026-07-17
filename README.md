# 🩺 Medical RAG Q&A Bot

A Retrieval-Augmented Generation bot that answers medical questions **grounded in your own documents** — no hallucinated facts, sources shown for every answer.

**Stack:** sentence-transformers (embeddings) · FAISS (vector search) · Groq LLaMA 3.3 (generation) · Streamlit (UI)

## How it works
```
Docs → chunk → embed → FAISS index        (ingest.py, run once)
Question → embed → search top-k → LLM      (rag.py + app.py, runtime)
```

## Setup

```bash
pip install -r requirements.txt
export GROQ_API_KEY="gsk_..."        # get a free key at console.groq.com
```

## Run

```bash
# 1. Add medical .txt / .md / .pdf files to ./data
#    (e.g. MedlinePlus articles, WHO fact sheets, your notes)

# 2. Build the index (once, or after adding docs)
python ingest.py

# 3. Test from terminal (optional)
python rag.py "what are symptoms of anemia?"

# 4. Launch the app
streamlit run app.py
```

## Deploy (Hugging Face Spaces)
1. Create a new **Streamlit** Space
2. Push these files + your `store/` folder (pre-built index)
3. Add `GROQ_API_KEY` in Space **Settings → Secrets**

## Upgrade ideas
- **Better embeddings:** swap to `pritamdeka/S-PubMedBert-MS-MARCO` in `ingest.py` + `rag.py`
- **Re-ranking:** add a cross-encoder to reorder retrieved chunks
- **Persistent chat:** store conversation history
- **Citations inline:** number sources and reference them in the answer text

## ⚠️ Disclaimer
Educational project only. Not a medical device. Always consult a qualified doctor.
