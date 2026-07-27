# Medical RAG Q&A Bot

A Retrieval-Augmented Generation bot that answers medical questions **grounded in a curated document set** — no hallucinated facts, sources shown for every answer, and honest refusals when a question falls outside the knowledge base.

[![Live Demo](https://img.shields.io/badge/Live_Demo-Open_App-brightgreen)](https://medical-rag-app-by6xavamwlximbe6d8xyye.streamlit.app)

**Live demo:** https://medical-rag-app-by6xavamwlximbe6d8xyye.streamlit.app

**Stack:** S-PubMedBERT (medical embeddings) · NumPy (vector search) · Groq Llama 3.3 (generation) · Streamlit (UI)

![Architecture](architecture.svg)

## Why this design

- **Grounded, not guessed.** Answers come only from retrieved document chunks. Ask something outside the knowledge base and it says "I don't have that information" instead of hallucinating — the behaviour that matters most for medical content.
- **Medical-tuned retrieval.** S-PubMedBERT is trained on biomedical text, so it understands clinical terms (consolidation, erythrocyte, melanocyte) that a generic embedder treats as noise.
- **Traceable.** Every answer shows which documents it came from, with a similarity score, a confidence label, and a medical-category badge.
- **Safe by construction.** A system prompt blocks diagnosis, prescription, and dosage, and appends a "consult a doctor" disclaimer to every answer.

## Knowledge base

17 documents spanning six medical fields, chunked into ~115 passages:

| Field | Topics |
|---|---|
| Hematology | anemia, blood cells, blood disorders, leukemia |
| Respiratory | pneumonia, asthma, tuberculosis |
| Cardiovascular | heart failure, hypertension |
| Endocrine | diabetes |
| Dermatology | skin lesions, skin cancer, melanoma |
| Imaging and AI | X-rays, CT scans, MRI, medical imaging AI |

All source documents are drawn from **MedlinePlus (U.S. National Library of Medicine)** and **WHO fact sheets**, with a provenance header (source, URL, retrieval date) at the top of each file.

## How it works

```
Docs -> chunk -> embed -> vector store      (ingest.py, run once)
Question -> embed -> search top-k -> LLM     (rag.py + app.py, runtime)
```

1. **ingest.py** splits documents into overlapping chunks, embeds them with S-PubMedBERT, and saves the vectors to `store/`.
2. **rag.py** embeds the question, finds the most similar chunks by cosine similarity, and passes them to Llama 3.3 with a grounded prompt.
3. **app.py** is a Streamlit chat interface with source cards, confidence labels, category badges, and a dark-mode toggle.

## Run locally

```bash
pip install -r requirements.txt

# save your Groq API key (free at console.groq.com)
mkdir -p .streamlit
echo 'GROQ_API_KEY = "gsk_..."' > .streamlit/secrets.toml

python ingest.py                     # build the index (once)
python rag.py "what is anemia?"      # quick CLI test
streamlit run app.py                 # launch the UI
```

## Add your own documents

```bash
# option A: fetch from MedlinePlus / WHO automatically
python fetch_docs.py                 # edit the SOURCES list first
python clean_docs.py                 # strip related-link noise

# option B: drop your own .txt / .pdf files into ./data

python ingest.py                     # rebuild the index (required)
```

## Deploy (Streamlit Community Cloud)

1. Push this repo to GitHub (public), including the `data/` and `store/` folders.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and click **Create app**.
3. Point it at this repo, branch `main`, main file `app.py`.
4. Under **Settings, then Secrets**, add: `GROQ_API_KEY = "gsk_..."`
5. Deploy. First boot downloads the embedder, then the app goes live.

## Project structure

```
medical-rag/
├── app.py              # Streamlit chat UI
├── rag.py              # retrieval + generation
├── ingest.py           # build the vector index
├── fetch_docs.py       # download articles from MedlinePlus / WHO
├── clean_docs.py       # strip related-link noise from fetched docs
├── requirements.txt
├── runtime.txt         # pins Python 3.12 for Streamlit Cloud
├── architecture.svg
├── data/               # source documents (.txt / .pdf)
└── store/              # generated index (embeddings.npy + meta.pkl)
```

## Design notes

- **Why NumPy instead of FAISS.** At this scale (~115 chunks, exact search in microseconds) brute-force cosine similarity is exact and faster to deploy, with one less native dependency. FAISS only pays off past ~100k chunks.
- **Confidence labels.** S-PubMedBERT scores cluster tightly in the high range across all medical text, so the confidence thresholds are tuned to that band and refusals are detected and labelled separately rather than trusting the raw score.
- **Possible next step.** A cross-encoder re-ranker would score query-document pairs directly and give more discriminative confidence than pre-computed vector similarity.

## Disclaimer

This is an educational project, not a medical device. It does not diagnose conditions or recommend treatment. Always consult a qualified healthcare professional.

## Author

**NOVEM (Min Thu Kyaw)** — Medical AI student, Konyang University
