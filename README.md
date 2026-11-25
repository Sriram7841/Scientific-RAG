# 🔬 GraphRAG Scientific Research Assistant

> **Production-ready Django web app** combining SVM/Naive Bayes text classification,
> GraphRAG (FAISS + NetworkX), and Mistral-7B LLM responses on **206,489 arXiv papers across 25 categories**.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Source** | arXiv API (export.arxiv.org) via Kaggle |
| **Total papers** | 206,489 |
| **Categories** | 25 (ML, CV, NLP, Physics, Robotics…) |
| **Columns** | `title`, `abstract`, `category` |
| **File** | `data/arxiv.csv` |

### Top 5 Categories
| Category | Field | Papers |
|----------|-------|--------|
| `cs.LG` | Machine Learning | 21,236 |
| `cs.CV` | Computer Vision | 11,534 |
| `cs.CL` | Natural Language Processing | 10,475 |
| `quant-ph` | Quantum Physics | 9,610 |
| `cs.RO` | Robotics | 9,216 |

---

## 🧠 System Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│ TEXT CLASSIFIER                 │
│ TF-IDF (30K features, bigrams)  │
│ LinearSVC  ←→  ComplementNB     │
│ 80/20 stratified split          │
│ → Predicted category + label    │
└────────────────┬────────────────┘
                 │
    ┌────────────▼─────────────┐
    │  HYBRID RETRIEVAL        │
    │  FAISS (cosine, IndexFlatIP)  │
    │     +                    │
    │  NetworkX Knowledge Graph│
    │  (methods, models, domains) │
    └────────────┬─────────────┘
                 │ top-5 papers + graph context
    ┌────────────▼─────────────────────────┐
    │ LLM  mistralai/Mistral-7B-Instruct   │
    │ HF Inference API · retry · fallback  │
    │ Prompt = query + category + context  │
    └──────────────────────────────────────┘
                 │
         Answer + Sources + Graph Context
```

---

## ⚡ Quick Start (Windows CMD)

```cmd
REM Step 1 — Run automated setup (creates venv, installs packages, migrates DB)
setup_windows.bat

REM Step 2 — Place dataset
REM          data\arxiv.csv  (title, abstract, category)

REM Step 3 — Activate virtual environment
venv\Scripts\activate

REM Step 4 — Run ML pipeline
python manage.py setup_graphrag

REM Step 5 — Start server
python manage.py runserver

REM Step 6 — Open browser
REM          http://127.0.0.1:8000
REM          Admin: admin / admin123
```

---

## ⚡ Quick Start (Linux / macOS / Git Bash)

```bash
bash setup.sh
source venv/bin/activate
# Place data/arxiv.csv
python manage.py setup_graphrag
python manage.py runserver
```

---

## 📋 Manual Setup (All Platforms)

### 1. Create & activate virtual environment

```cmd
:: Windows CMD
python -m venv venv
venv\Scripts\activate
```

```bash
# Linux / macOS / Git Bash / WSL
python3 -m venv venv
source venv/bin/activate
```

### 2. Install packages

```cmd
pip install --upgrade pip
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 3. Configure environment

```cmd
copy .env.example .env
:: Edit .env — add your HF_API_KEY
```

### 4. Create directories & migrate DB

```cmd
mkdir data models media staticfiles
python manage.py makemigrations accounts rag_engine chat
python manage.py migrate
python manage.py create_admin
```

### 5. Place dataset

```
data\arxiv.csv   ← must have columns: title, abstract, category
```

Download: https://www.kaggle.com/datasets/spsayakpaul/arxiv-paper-abstracts

---

## 🚀 Pipeline Commands (CMD)

### Full pipeline (train + index + graph)

```cmd
python manage.py setup_graphrag
```

### With options

```cmd
:: Faster test (20K rows)
python manage.py setup_graphrag --rows 20000

:: Train classifiers only
python manage.py setup_graphrag --skip-index --skip-graph

:: Build FAISS + graph only (skip retraining)
python manage.py setup_graphrag --skip-train

:: All rows, skip graph building
python manage.py setup_graphrag --skip-graph
```

### Train via Admin Panel (GUI)

1. Open http://127.0.0.1:8000/admin-panel/
2. Click **"Train SVM + Naive Bayes"** → runs in background
3. Click **"Build FAISS Index + Knowledge Graph"** → runs in background
4. Watch status dots turn green ✅

---

## 🤖 Classification

| Model | Class | Algorithm | Notes |
|-------|-------|-----------|-------|
| **SVM** | Primary | `LinearSVC(C=1.0)` | Best F1 on multi-class |
| **NB** | Baseline | `ComplementNB(α=0.1)` | Fast, competitive |

**Features:** TF-IDF · max 30,000 features · unigrams + bigrams · sublinear TF  
**Split:** 80% train / 20% test · stratified by category  
**Input:** `title + abstract` (concatenated, cleaned, stopwords removed)

---

## 🔎 GraphRAG Pipeline

| Component | Implementation |
|-----------|---------------|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (~90 MB download) |
| Vector DB | FAISS `IndexFlatIP` (cosine similarity, normalized) |
| Chunking | 400 words, 50-word overlap |
| Knowledge Graph | NetworkX DiGraph |
| Entities | Models (BERT, GPT…), Methods (attention, fine-tuning…), Domains (NLP, CV…) |
| Relations | `improves`, `uses`, `belongs-to`, `related-to` |
| LLM | Mistral-7B-Instruct-v0.2 via HF Inference API |
| Fallback | google/flan-t5-base → extractive answer |
| Cache | In-memory (300s TTL) or Redis |

---

## 🌐 REST API

```bash
# Get JWT token
curl -X POST http://127.0.0.1:8000/api/accounts/token/ \
  -H "Content-Type: application/json" \
  -d "{\"username\":\"admin\",\"password\":\"admin123\"}"

# Chat query (replace TOKEN with access token from above)
curl -X POST http://127.0.0.1:8000/api/chat/query/ \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"Recent advances in few-shot learning\"}"

# Classify a query
curl -X POST http://127.0.0.1:8000/api/rag/predict/ \
  -H "Authorization: Bearer TOKEN" \
  -d "{\"query\":\"Graph neural networks for drug discovery\"}"
```

---

## 📁 Project Structure

```
graphrag_project/
├── data/
│   └── arxiv.csv              ← Place 206K paper dataset here
├── models/                    ← Saved ML models (auto-created after training)
│   ├── svm_model.pkl
│   ├── nb_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── label_encoder.pkl
├── accounts/                  ← Auth (JWT + sessions, custom User)
├── admin_panel/               ← Dashboard, EDA, training, user mgmt
├── rag_engine/                ← Core AI: classifier.py + pipeline.py
│   └── management/commands/setup_graphrag.py
├── chat/                      ← Chat UI + session history + PDF export
├── templates/
│   ├── base/base.html
│   ├── accounts/              ← login, signup, profile
│   ├── admin_panel/           ← dashboard, eda, training_detail, graph_viz, users
│   └── chat/                  ← chat, export_pdf
├── static/css/ + static/js/
├── graphrag_project/settings.py
├── manage.py
├── requirements.txt
├── setup.sh                   ← Linux/macOS/Git Bash setup
├── setup_windows.bat          ← Windows CMD setup
├── .env.example
└── README.md
```

---

## 🔐 Security

- JWT authentication (24h access, 7d refresh)
- Django session auth (web UI)
- PBKDF2 password hashing
- CSRF protection on all POST endpoints
- **HF API key via `.env` only — never hardcoded**
- Input validation on all endpoints

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| `arxiv.csv not found` | Place file at `data/arxiv.csv` |
| `Models not trained yet` | Run `python manage.py setup_graphrag` |
| `FAISS index not found` | Run `python manage.py setup_graphrag --skip-train` |
| Extractive answers only | Add `HF_API_KEY` to `.env` |
| Slow first query | Embedding model downloads ~90 MB on first run |
| `ModuleNotFoundError` | Activate venv: `venv\Scripts\activate` (Windows) |
| Training takes too long | Use `--rows 20000` flag |
| Port 8000 busy | `python manage.py runserver 8080` |
| Windows `activate` fails | Use Git Bash or WSL instead of CMD |

---

## 🗜️ Zip the Project

```cmd
:: Windows
powershell Compress-Archive -Path graphrag_project -DestinationPath graphrag_project.zip -Force

:: Linux / macOS
zip -r graphrag_project.zip graphrag_project/ --exclude "*/venv/*" --exclude "*/__pycache__/*" --exclude "*.pyc" --exclude "*/db.sqlite3" --exclude "*/.env" --exclude "*/models/*.pkl" --exclude "*/data/faiss_index*"
```

---

*Built with Django 4.2 · scikit-learn · FAISS · NetworkX · sentence-transformers · HF API · Tailwind CSS · Plotly*
