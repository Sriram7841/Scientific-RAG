"""
GraphRAG Pipeline
==================
1.  Chunk documents (title + abstract, 400-word windows)
2.  Embed with sentence-transformers/all-MiniLM-L6-v2
3.  FAISS IndexFlatIP (cosine on normalised vectors)  →  save/load to disk
4.  NetworkX knowledge graph  →  entities + relations from abstracts
5.  Hybrid retrieval  =  FAISS top-k  +  graph neighbourhood
6.  LLM answer via HF Inference API  (Mistral-7B → flan-t5 fallback → extractive)
"""

import os
import re
import pickle
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger('rag_engine')

# ── Lazy globals ──────────────────────────────────────────────────
_embed  = None     # SentenceTransformer
_findex = None     # faiss.Index
_fdocs  = None     # list[dict] — parallel to index rows
_graph  = None     # networkx.DiGraph


# ─────────────────────────────────────────────────────────────────
# Embedding helpers
# ─────────────────────────────────────────────────────────────────
def get_embed():
    global _embed
    if _embed is None:
        from sentence_transformers import SentenceTransformer
        from django.conf import settings
        name   = getattr(settings, 'EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        logger.info(f"Loading embedding model: {name}")
        _embed = SentenceTransformer(name)
        logger.info("Embedding model ready.")
    return _embed


def embed_batch(texts: List[str], batch_size: int = 128) -> np.ndarray:
    model = get_embed()
    out   = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        embs  = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        out.append(embs)
        if i > 0 and (i // batch_size) % 20 == 0:
            logger.info(f"    embedded {i}/{len(texts)} …")
    return np.vstack(out).astype('float32')


# ─────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────
def chunk_text(text: str, max_words: int = 400, overlap: int = 50) -> List[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks, start = [], 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(' '.join(words[start:end]))
        start += max_words - overlap
    return chunks


def _clean(text: str) -> str:
    return re.sub(r'\s+', ' ', str(text or '')).strip()


# ─────────────────────────────────────────────────────────────────
# FAISS
# ─────────────────────────────────────────────────────────────────
def build_faiss_index(records: List[Dict]) -> Dict:
    """
    records: list of {'doc_id','title','abstract','category'}
    Chunks each document, embeds, builds FAISS IndexFlatIP, saves to disk.
    """
    import faiss
    from django.conf import settings

    logger.info(f"Building FAISS index for {len(records):,} records …")
    texts, meta = [], []

    for rec in records:
        full   = _clean(rec.get('title', '')) + ' ' + _clean(rec.get('abstract', ''))
        chunks = chunk_text(full)
        for ci, chunk in enumerate(chunks):
            texts.append(chunk)
            meta.append({
                'doc_id':     str(rec.get('doc_id', '')),
                'title':      _clean(rec.get('title', ''))[:200],
                'abstract':   _clean(rec.get('abstract', ''))[:500],
                'category':   str(rec.get('category', '')),
                'chunk_idx':  ci,
                'chunk_text': chunk[:600],
            })

    logger.info(f"  Total chunks: {len(texts):,}")
    embeddings = embed_batch(texts)
    dim        = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    idx_path = Path(settings.FAISS_INDEX_PATH)
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(idx_path.with_suffix('.index')))
    with open(str(idx_path) + '_meta.pkl', 'wb') as f:
        pickle.dump(meta, f, protocol=4)

    global _findex, _fdocs
    _findex = index
    _fdocs  = meta
    logger.info(f"  FAISS saved: {index.ntotal:,} vectors  dim={dim}")
    return {'total_vectors': index.ntotal, 'dimension': dim}


def _load_faiss() -> bool:
    global _findex, _fdocs
    from django.conf import settings
    import faiss

    idx  = Path(settings.FAISS_INDEX_PATH).with_suffix('.index')
    meta = Path(str(settings.FAISS_INDEX_PATH) + '_meta.pkl')
    if not (idx.exists() and meta.exists()):
        return False
    _findex = faiss.read_index(str(idx))
    with open(meta, 'rb') as f:
        _fdocs = pickle.load(f)
    logger.info(f"FAISS loaded: {_findex.ntotal:,} vectors")
    return True


def faiss_search(query: str, k: int = 5) -> List[Dict]:
    if _findex is None:
        if not _load_faiss():
            return []
    model = get_embed()
    q_emb = model.encode([query], normalize_embeddings=True).astype('float32')
    scores, indices = _findex.search(q_emb, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if 0 <= idx < len(_fdocs):
            doc = dict(_fdocs[idx])
            doc['faiss_score'] = round(float(score), 4)
            results.append(doc)
    return results


def faiss_ready() -> bool:
    from django.conf import settings
    return Path(settings.FAISS_INDEX_PATH).with_suffix('.index').exists()


# ─────────────────────────────────────────────────────────────────
# Knowledge Graph  (NetworkX)
# ─────────────────────────────────────────────────────────────────
_ENT = {
    'model': re.compile(
        r'\b(BERT|GPT(?:-\d)?|T5|RoBERTa|LSTM|CNN|ResNet|Transformer|LLaMA|Mistral|'
        r'ViT|CLIP|GAN|VAE|AlphaFold|XGBoost|LightGBM|Word2Vec|GloVe|ALBERT|'
        r'XLNet|EfficientNet|DeiT|DALL-E|Whisper|PaLM|Gemini)\b', re.I),
    'method': re.compile(
        r'\b(attention|fine-tuning|transfer learning|reinforcement learning|'
        r'self-supervised|contrastive learning|few-shot|zero-shot|federated|'
        r'graph neural network|knowledge distillation|data augmentation|'
        r'meta-learning|active learning|prompt tuning|retrieval augmented)\b', re.I),
    'domain': re.compile(
        r'\b(computer vision|natural language processing|NLP|speech recognition|'
        r'object detection|image segmentation|machine translation|'
        r'question answering|information retrieval|robotics|drug discovery|'
        r'medical imaging|autonomous driving|recommendation system|'
        r'quantum computing|time series|graph learning)\b', re.I),
}

_REL = [
    (re.compile(r'outperform|surpass|improv|beat|exceed', re.I),   'improves'),
    (re.compile(r'\buse\b|\bemploy\b|\bleverag\b|\bapply\b', re.I),'uses'),
    (re.compile(r'belong|classify|domain|field|area', re.I),        'belongs-to'),
    (re.compile(r'related|similar|connect|analogue', re.I),         'related-to'),
]


def _extract_entities(text: str) -> Dict[str, List[str]]:
    ents: Dict[str, List[str]] = {}
    for etype, pat in _ENT.items():
        found = list({m.lower() for m in pat.findall(text)})
        if found:
            ents[etype] = found
    return ents


def build_knowledge_graph(records: List[Dict]) -> Dict:
    import networkx as nx
    from django.conf import settings

    logger.info(f"Building knowledge graph for {len(records):,} records …")
    G        = nx.DiGraph()
    nfreq: Dict[str, int] = {}

    for rec in records:
        text = _clean(rec.get('title', '')) + ' ' + _clean(rec.get('abstract', ''))
        cat  = str(rec.get('category', ''))

        # Category node
        cat_key = f'domain:{cat}'
        nfreq[cat_key] = nfreq.get(cat_key, 0) + 1
        if not G.has_node(cat_key):
            G.add_node(cat_key, name=cat, type='domain', frequency=1)
        else:
            G.nodes[cat_key]['frequency'] = nfreq[cat_key]

        ents = _extract_entities(text)
        for etype, names in ents.items():
            for name in names:
                nk = f'{etype}:{name}'
                nfreq[nk] = nfreq.get(nk, 0) + 1
                if not G.has_node(nk):
                    G.add_node(nk, name=name, type=etype, frequency=1)
                else:
                    G.nodes[nk]['frequency'] = nfreq[nk]
                # belongs-to category
                if not G.has_edge(nk, cat_key):
                    G.add_edge(nk, cat_key, relation='belongs-to', weight=1)
                else:
                    G[nk][cat_key]['weight'] = G[nk][cat_key].get('weight', 1) + 1

        # Relation extraction from sentences
        ent_list = [f'{et}:{n}' for et, ns in ents.items() for n in ns]
        if len(ent_list) >= 2:
            for sent in re.split(r'[.!?]', text):
                rel = 'related-to'
                for pat, r in _REL:
                    if pat.search(sent):
                        rel = r
                        break
                src, tgt = ent_list[0], ent_list[1]
                if G.has_node(src) and G.has_node(tgt) and src != tgt:
                    if not G.has_edge(src, tgt):
                        G.add_edge(src, tgt, relation=rel, weight=1)
                    else:
                        G[src][tgt]['weight'] = G[src][tgt].get('weight', 1) + 1

    gpath = Path(settings.GRAPH_PATH)
    gpath.parent.mkdir(parents=True, exist_ok=True)
    with open(gpath, 'wb') as f:
        pickle.dump(G, f, protocol=4)

    global _graph
    _graph = G
    stats  = {'nodes': G.number_of_nodes(), 'edges': G.number_of_edges()}
    logger.info(f"  Graph saved: {stats}")
    return stats


def _load_graph() -> bool:
    global _graph
    from django.conf import settings
    gp = Path(settings.GRAPH_PATH)
    if not gp.exists():
        return False
    with open(gp, 'rb') as f:
        _graph = pickle.load(f)
    logger.info(f"Graph loaded: {_graph.number_of_nodes()} nodes")
    return True


def graph_search(query: str, top_k: int = 5) -> List[Dict]:
    if _graph is None:
        if not _load_graph():
            return []
    ents = _extract_entities(query)
    seen, results = set(), []
    for etype, names in ents.items():
        for name in names:
            nk = f'{etype}:{name}'
            if not _graph.has_node(nk):
                continue
            for nbr in list(_graph.successors(nk)) + list(_graph.predecessors(nk)):
                if nbr in seen:
                    continue
                seen.add(nbr)
                data  = _graph.nodes[nbr]
                edata = _graph.edges.get((nk, nbr), _graph.edges.get((nbr, nk), {}))
                results.append({
                    'node':      nbr,
                    'name':      data.get('name', nbr),
                    'type':      data.get('type', 'unknown'),
                    'frequency': data.get('frequency', 1),
                    'relation':  edata.get('relation', 'related-to'),
                })
    results.sort(key=lambda x: x['frequency'], reverse=True)
    return results[:top_k]


def graph_ready() -> bool:
    from django.conf import settings
    return Path(settings.GRAPH_PATH).exists()


# ─────────────────────────────────────────────────────────────────
# Hybrid retrieval
# ─────────────────────────────────────────────────────────────────
def hybrid_retrieve(query: str, k: int = 5) -> Tuple[List[Dict], List[Dict]]:
    return faiss_search(query, k=k), graph_search(query, top_k=5)


# ─────────────────────────────────────────────────────────────────
# LLM  (HF Inference API)
# ─────────────────────────────────────────────────────────────────
def _build_prompt(query: str, category: str, cat_label: str,
                  docs: List[Dict]) -> str:
    ctx_parts = []
    for i, doc in enumerate(docs[:3], 1):
        snippet = doc.get('chunk_text', doc.get('abstract', ''))[:450]
        ctx_parts.append(f"[Paper {i}] {doc.get('title','')}\n{snippet}")
    ctx = '\n\n'.join(ctx_parts) or 'No context retrieved.'
    return (
        f"User Query: {query}\n"
        f"Predicted Category: {category} ({cat_label})\n\n"
        f"Relevant Context:\n{ctx}\n\n"
        f"Generate a clear scientific answer with key insights based on "
        f"the above context. Be concise and cite the papers."
    )


# ─────────────────────────────────────────────────────────────────
# HUGGING FACE API (OFFICIAL CLIENT)
# ─────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────
# HUGGING FACE INFERENCE API
# ─────────────────────────────────────────────────────────────────

from huggingface_hub import InferenceClient

_hf_client = None


def _call_hf(prompt: str) -> str:

    global _hf_client

    try:

        from django.conf import settings

        api_key = settings.HF_API_KEY

        if not api_key:

            logger.warning(
                "HF_API_KEY missing."
            )

            return ""

        # =====================================================
        # INIT CLIENT
        # =====================================================
        if _hf_client is None:

            logger.info("=" * 60)
            logger.info("Initializing HF Inference Client...")

            _hf_client = InferenceClient(
                token=api_key
            )

            logger.info(
                "HF Inference Client ready."
            )

        # =====================================================
        # FREE PUBLIC MODELS
        # =====================================================
        models = [
            "gpt2",
            "distilgpt2",
        ]

        # =====================================================
        # TRY MODELS
        # =====================================================
        for model in models:

            try:

                logger.info(
                    f"Using HF model: {model}"
                )

                response = _hf_client.text_generation(
                    prompt=prompt,
                    model=model,
                    max_new_tokens=200,
                    temperature=0.3,
                )

                if response and str(response).strip():

                    logger.info(
                        f"HF generation successful: {model}"
                    )

                    return str(response).strip()

            except Exception as model_error:

                logger.warning(
                    f"HF model failed ({model}): "
                    f"{model_error}"
                )

                continue

    except Exception as e:

        logger.warning(
            f"HF generation failed: {e}"
        )

    return ""
def _extractive_answer(query: str, docs: List[Dict]) -> str:
    if not docs:
        return (
            "No relevant documents found for your query.\n\n"
            "**Steps to fix:**\n"
            "1. Ensure the FAISS index is built (Admin Panel → Build Index)\n"
            "2. Verify data/arxiv.csv exists with columns: title, abstract, category\n"
            "3. Add HF_API_KEY to .env for AI-generated answers"
        )
    lines = [f"Based on **{len(docs)}** retrieved arXiv papers:\n"]
    for i, doc in enumerate(docs[:3], 1):
        title = doc.get('title', 'Untitled')
        cat   = doc.get('category', '')
        text  = doc.get('chunk_text', doc.get('abstract', ''))[:350]
        score = doc.get('faiss_score', 0)
        lines.append(
            f"**[{i}] {title}**  `{cat}`  *(similarity: {score:.3f})*\n{text}…"
        )
    lines.append(
        "\n*To get AI-generated answers, set HF_API_KEY in your .env file.*"
    )
    return '\n\n'.join(lines)


def generate_answer(query: str, category: str, cat_label: str,
                    faiss_results: List[Dict],
                    graph_results: List[Dict]) -> Dict:
    from django.core.cache import cache

    cache_key = f'graphrag:{abs(hash(query + category))}'
    cached    = cache.get(cache_key)
    if cached:
        logger.debug("Cache hit for query")
        return cached

    prompt = _build_prompt(query, category, cat_label, faiss_results)
    answer = _call_hf(prompt)
    if not answer:
        answer = _extractive_answer(query, faiss_results)

    seen, sources = set(), []
    for doc in faiss_results[:5]:
        did = doc.get('doc_id', '')
        if did not in seen:
            seen.add(did)
            sources.append({
                'doc_id':   did,
                'title':    doc.get('title', 'Untitled')[:130],
                'category': doc.get('category', ''),
                'score':    doc.get('faiss_score', 0),
            })

    result = {
        'answer':        answer,
        'sources':       sources,
        'graph_context': graph_results[:5],
    }
    cache.set(cache_key, result, timeout=300)
    return result


# ─────────────────────────────────────────────────────────────────
# Graph visualisation data
# ─────────────────────────────────────────────────────────────────
def get_graph_viz_data(max_nodes: int = 120) -> Dict:
    if _graph is None:
        if not _load_graph():
            return {'nodes': [], 'edges': []}

    COLOR = {
        'model':  '#FF6B35',
        'method': '#4FC3F7',
        'domain': '#22c55e',
        'concept':'#f59e0b',
    }
    nodes_sorted = sorted(
        [{'id': n, **d} for n, d in _graph.nodes(data=True)],
        key=lambda x: x.get('frequency', 0), reverse=True,
    )[:max_nodes]
    nids = {n['id'] for n in nodes_sorted}

    nodes = [
        {
            'id':        n['id'],
            'label':     n.get('name', n['id'])[:28],
            'type':      n.get('type', 'unknown'),
            'frequency': n.get('frequency', 1),
            'color':     COLOR.get(n.get('type', ''), '#888'),
        }
        for n in nodes_sorted
    ]
    edges = [
        {
            'source':   u,
            'target':   v,
            'relation': d.get('relation', ''),
            'weight':   d.get('weight', 1),
        }
        for u, v, d in _graph.edges(data=True)
        if u in nids and v in nids
    ][:400]

    return {'nodes': nodes, 'edges': edges}


def reset_pipeline_cache():
    global _findex, _fdocs, _graph
    _findex = None
    _fdocs  = None
    _graph  = None
