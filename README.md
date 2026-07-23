## Medical RAG Q&A Bot

A Retrieval-Augmented Generation bot that answers medical questions **grounded in your own documents** — no hallucinated facts, sources shown for every answer, and honest refusals when a question falls outside the docs.

[![Live Demo](https://img.shields.io/badge/Live_Demo-Open_App-brightgreen)](https://medical-rag-app-by6xavamwlximbe6d8xyye.streamlit.app)

**Live demo:** https://medical-rag-app-by6xavamwlximbe6d8xyye.streamlit.app

**Stack:** S-PubMedBERT (medical embeddings) · NumPy (vector search) · Groq LLaMA 3.3 (generation) · Streamlit (UI)

## How it works
```
Docs -> chunk -> embed -> vector store      (ingest.py, run once)
Question -> embed -> search top-k -> LLM     (rag.py + app.py, runtime)
```

- **Grounded, not guessed.** Answers come only from retrieved document chunks. Off-topic questions get an honest "I don't have that information."
- **Medical-tuned retrieval.** S-PubMedBERT understands clinical terms a generic embedder treats as noise.
- **Traceable.** Every answer shows its source documents with similarity scores.
- **Safe by construction.** A system prompt blocks diagnosis and dosage and appends a "consult a doctor" disclaimer.

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

## Deploy (Streamlit Community Cloud)
1. Push this repo to GitHub (public), including the `data/` and `store/` folders.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and click **Create app**.
3. Point it at this repo, branch `main`, main file `app.py`.
4. Under **Advanced settings -> Secrets**, add:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
5. Deploy. First boot downloads the embedder, then the app goes live.

## Upgrade ideas
- **Re-ranking:** add a cross-encoder to reorder retrieved chunks
- **Persistent chat:** store conversation history
- **Inline citations:** number sources and reference them in the answer text
- **Larger corpus:** expand `data/` with vetted sources (MedlinePlus, WHO, NIH)

## Disclaimer
Educational project only. Not a medical device. Always consult a qualified doctor.

## Author

**NOVEM (Min Thu Kyaw)** — Medical AI Student, Konyang University
