"""
Text Classification Module
===========================
Dataset : data/arxiv.csv  — 206,489 papers, 25 categories
Models  : LinearSVC (primary) vs ComplementNB (baseline)
Features: TF-IDF (title + abstract, bigrams, 30k features)
Split   : 80 % train / 20 % test (stratified)
Metrics : Accuracy, Precision, Recall, F1, Confusion Matrix
"""

import re
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger('rag_engine')

# ── Top 5 categories (dataset description) ───────────────────────
TOP_CATEGORIES = {
    'cs.LG': 'Machine Learning',
    'cs.CV': 'Computer Vision',
    'cs.CL': 'Natural Language Processing',
    'quant-ph': 'Quantum Physics',
    'cs.RO': 'Robotics',
}

# ── Lazy globals ──────────────────────────────────────────────────
_svm_model = None
_nb_model  = None
_tfidf     = None
_label_enc = None
_stopwords: set = set()


# ─────────────────────────────────────────────────────────────────
# STOPWORDS
# ─────────────────────────────────────────────────────────────────
def _get_stopwords() -> set:
    global _stopwords
    if _stopwords:
        return _stopwords
    try:
        import nltk
        from nltk.corpus import stopwords
        try:
            _stopwords = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            _stopwords = set(stopwords.words('english'))
    except Exception:
        _stopwords = {
            'a','an','the','and','or','but','in','on','at','to','for','of',
            'with','by','from','is','are','was','were','be','been','has',
            'have','had','this','that','these','those','it','its','we',
            'our','they','their','will','would','could','should','may',
            'which','who','what','how','when','where','not','also','we',
        }
    return _stopwords


# ─────────────────────────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Lowercase → strip special chars → remove stopwords."""
    if not isinstance(text, str) or not text.strip():
        return ''
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stops = _get_stopwords()
    tokens = [w for w in text.split() if len(w) > 2 and w not in stops]
    return ' '.join(tokens)


# ─────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────
def load_arxiv_csv(csv_path: str, max_rows: int = 50000) -> pd.DataFrame:
    """
    Load Kaggle arXiv dataset.

    Supported dataset columns:
    - titles
    - summaries
    - terms

    Converted internally to:
    - title
    - abstract
    - category
    """

    logger.info(
        f"Loading arXiv CSV: {csv_path}  (max_rows={max_rows:,})"
    )

    # =========================================================
    # LOAD CSV
    # =========================================================
    df = pd.read_csv(
        csv_path,
        nrows=max_rows if max_rows > 0 else None
    )

    logger.info(f"Original columns: {list(df.columns)}")

    # =========================================================
    # COLUMN MAPPING
    # =========================================================
    column_mapping = {}

    # Kaggle dataset columns
    if "titles" in df.columns:
        column_mapping["titles"] = "title"

    if "summaries" in df.columns:
        column_mapping["summaries"] = "abstract"

    if "terms" in df.columns:
        column_mapping["terms"] = "category"

    # Rename dataset columns
    df = df.rename(columns=column_mapping)

    logger.info(f"Mapped columns: {list(df.columns)}")

    # =========================================================
    # VALIDATION
    # =========================================================
    required = {"title", "abstract", "category"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset missing columns: {missing}. "
            f"Expected columns: {required}. "
            f"Found columns: {list(df.columns)}"
        )

    # =========================================================
    # CLEANING
    # =========================================================
    before = len(df)

    df = df.dropna(subset=["title", "abstract", "category"])

    df["title"] = df["title"].astype(str)
    df["abstract"] = df["abstract"].astype(str)
    df["category"] = df["category"].astype(str)

    # =========================================================
    # CLEAN CATEGORY VALUES
    # =========================================================
    def clean_category(x):
        x = str(x)

        # Handle stringified lists like:
        # "['cs.LG']"
        x = x.replace("[", "")
        x = x.replace("]", "")
        x = x.replace("'", "")
        x = x.replace('"', "")

        # Take first category if multiple exist
        if "," in x:
            x = x.split(",")[0]

        return x.strip()

    df["category"] = df["category"].apply(clean_category)

    # =========================================================
    # REMOVE INVALID ABSTRACTS
    # =========================================================
    df = df[
        df["abstract"].str.strip().str.len() > 30
    ]

    logger.info(
        f"Dropped {before - len(df)} invalid rows "
        f"→ {len(df):,} valid rows"
    )

    # =========================================================
    # CREATE TEXT COLUMN
    # =========================================================
    df["text"] = (
        df["title"].fillna("").astype(str)
        + " "
        + df["abstract"].fillna("").astype(str)
    ).apply(clean_text)

    # Remove extremely short processed texts
    df = df[
        df["text"].str.len() > 20
    ].reset_index(drop=True)

    # =========================================================
    # FINAL LOGGING
    # =========================================================
    n_cats = df["category"].nunique()

    logger.info(
        f"Dataset ready: {len(df):,} records | "
        f"{n_cats} unique categories"
    )

    logger.info(
        f"Top-5 categories: "
        f"{df['category'].value_counts().head(5).to_dict()}"
    )

    return df
def get_dataset_stats(df: pd.DataFrame) -> Dict:
    """Return EDA statistics for the admin dashboard."""
    top = df['category'].value_counts()
    return {
        'total_records':     int(len(df)),
        'num_categories':    int(df['category'].nunique()),
        'avg_abstract_len':  int(df['abstract'].str.len().mean()),
        'top_categories':    top.head(25).to_dict(),
        'top5':              top.head(5).to_dict(),
        'null_removed':      0,
    }


# ─────────────────────────────────────────────────────────────────
# TF-IDF
# ─────────────────────────────────────────────────────────────────
def build_tfidf(texts: List[str], max_features: int = 30000):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),      # unigrams + bigrams
        sublinear_tf=True,       # apply log(TF)
        min_df=2,
        strip_accents='unicode',
        analyzer='word',
    )
    X = tfidf.fit_transform(texts)
    logger.info(f"TF-IDF matrix: {X.shape}")
    return tfidf, X


# ─────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────
def train_models(df: pd.DataFrame, test_size: float = 0.20) -> Dict:
    """
    Train SVM + Naive Bayes. Return full metrics dict. Save models to disk.
    80 : 20 stratified split on the 25-category arXiv dataset.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.svm import LinearSVC
    from sklearn.naive_bayes import ComplementNB
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, classification_report,
    )
    import joblib
    from django.conf import settings

    t_start = time.time()
    logger.info("▶ Starting training pipeline …")

    # Label encoding
    le = LabelEncoder()
    y  = le.fit_transform(df['category'])
    logger.info(f"Classes: {len(le.classes_)}  → {list(le.classes_[:8])} …")

    # Feature extraction
    tfidf, X = build_tfidf(df['text'].tolist())

    # 80 / 20 stratified split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    logger.info(f"Train: {X_tr.shape[0]:,}   Test: {X_te.shape[0]:,}")

    results = {
        'train_size':  int(X_tr.shape[0]),
        'test_size':   int(X_te.shape[0]),
        'num_classes': int(len(le.classes_)),
        'classes':     list(le.classes_),
        'svm':         {},
        'naive_bayes': {},
        'total_time':  0,
    }

    # ── SVM ──────────────────────────────────────────────────────
    logger.info("  Training LinearSVC …")
    t0  = time.time()
    svm = LinearSVC(C=1.0, max_iter=3000, random_state=42, dual=True)
    svm.fit(X_tr, y_tr)
    y_pred_svm = svm.predict(X_te)
    svm_t = round(time.time() - t0, 2)

    cr_svm = classification_report(
        y_te, y_pred_svm, target_names=le.classes_,
        output_dict=True, zero_division=0,
    )
    results['svm'] = {
        'accuracy':         round(float(accuracy_score(y_te, y_pred_svm)), 4),
        'precision':        round(float(precision_score(y_te, y_pred_svm, average='weighted', zero_division=0)), 4),
        'recall':           round(float(recall_score(y_te, y_pred_svm, average='weighted', zero_division=0)), 4),
        'f1_score':         round(float(f1_score(y_te, y_pred_svm, average='weighted', zero_division=0)), 4),
        'confusion_matrix': confusion_matrix(y_te, y_pred_svm).tolist(),
        'class_report':     cr_svm,
        'train_time_sec':   svm_t,
    }
    logger.info(
        f"  SVM  → Acc={results['svm']['accuracy']}  "
        f"F1={results['svm']['f1_score']}  ({svm_t}s)"
    )

    # ── Naive Bayes ───────────────────────────────────────────────
    logger.info("  Training ComplementNB …")
    t0 = time.time()
    nb = ComplementNB(alpha=0.1)
    nb.fit(X_tr, y_tr)
    y_pred_nb = nb.predict(X_te)
    nb_t = round(time.time() - t0, 2)

    cr_nb = classification_report(
        y_te, y_pred_nb, target_names=le.classes_,
        output_dict=True, zero_division=0,
    )
    results['naive_bayes'] = {
        'accuracy':         round(float(accuracy_score(y_te, y_pred_nb)), 4),
        'precision':        round(float(precision_score(y_te, y_pred_nb, average='weighted', zero_division=0)), 4),
        'recall':           round(float(recall_score(y_te, y_pred_nb, average='weighted', zero_division=0)), 4),
        'f1_score':         round(float(f1_score(y_te, y_pred_nb, average='weighted', zero_division=0)), 4),
        'confusion_matrix': confusion_matrix(y_te, y_pred_nb).tolist(),
        'class_report':     cr_nb,
        'train_time_sec':   nb_t,
    }
    logger.info(
        f"  NB   → Acc={results['naive_bayes']['accuracy']}  "
        f"F1={results['naive_bayes']['f1_score']}  ({nb_t}s)"
    )

    # ── Save to disk ──────────────────────────────────────────────
    models_dir = Path(settings.MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(svm,   settings.SVM_MODEL_PATH,   compress=3)
    joblib.dump(nb,    settings.NB_MODEL_PATH,    compress=3)
    joblib.dump(tfidf, settings.TFIDF_PATH,       compress=3)
    joblib.dump(le,    settings.LABEL_ENC_PATH,   compress=3)
    logger.info(f"  Models saved to {models_dir}")

    # Update lazy globals
    global _svm_model, _nb_model, _tfidf, _label_enc
    _svm_model = svm
    _nb_model  = nb
    _tfidf     = tfidf
    _label_enc = le

    results['total_time'] = round(time.time() - t_start, 2)
    logger.info(f"▶ Training complete in {results['total_time']}s")
    return results


# ─────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────
def _load_models_lazy():
    global _svm_model, _nb_model, _tfidf, _label_enc
    from django.conf import settings
    import joblib

    if _tfidf is None and Path(settings.TFIDF_PATH).exists():
        _tfidf     = joblib.load(settings.TFIDF_PATH)
    if _label_enc is None and Path(settings.LABEL_ENC_PATH).exists():
        _label_enc = joblib.load(settings.LABEL_ENC_PATH)
    if _svm_model is None and Path(settings.SVM_MODEL_PATH).exists():
        _svm_model = joblib.load(settings.SVM_MODEL_PATH)
    if _nb_model is None and Path(settings.NB_MODEL_PATH).exists():
        _nb_model  = joblib.load(settings.NB_MODEL_PATH)


def predict_category(text: str, model: str = 'svm') -> Dict:
    """
    Predict arXiv category for a query.
    Returns {'category': 'cs.LG', 'label': 'Machine Learning', 'confidence': 0.87}
    """
    from django.conf import settings
    _load_models_lazy()

    if _tfidf is None or _label_enc is None:
        return {
            'category': 'Unknown', 'label': 'Unknown',
            'confidence': 0.0, 'error': 'Models not trained yet.'
        }

    cleaned = clean_text(text)
    if not cleaned:
        return {'category': 'Unknown', 'label': 'Unknown', 'confidence': 0.0}

    X   = _tfidf.transform([cleaned])
    clf = _svm_model if model == 'svm' else _nb_model
    if clf is None:
        return {'category': 'Unknown', 'label': 'Unknown',
                'confidence': 0.0, 'error': f'{model} not available.'}

    idx      = int(clf.predict(X)[0])
    category = str(_label_enc.inverse_transform([idx])[0])
    label    = settings.ARXIV_CATEGORIES.get(category, category)

    # Confidence estimate
    confidence = 0.0
    if hasattr(clf, 'predict_proba'):
        proba      = clf.predict_proba(X)[0]
        confidence = float(np.max(proba))
    elif hasattr(clf, 'decision_function'):
        df_vals = clf.decision_function(X)[0]
        if hasattr(df_vals, '__len__') and len(df_vals) > 1:
            shifted    = df_vals - df_vals.min()
            total      = shifted.sum()
            confidence = float(shifted[idx] / total) if total > 0 else 0.5
        else:
            confidence = min(float(abs(df_vals)), 1.0)

    return {
        'category':   category,
        'label':      label,
        'confidence': round(confidence, 3),
    }


def models_ready() -> bool:
    from django.conf import settings
    return (
        Path(settings.SVM_MODEL_PATH).exists()  and
        Path(settings.NB_MODEL_PATH).exists()   and
        Path(settings.TFIDF_PATH).exists()      and
        Path(settings.LABEL_ENC_PATH).exists()
    )


def reset_classifier_cache():
    global _svm_model, _nb_model, _tfidf, _label_enc
    _svm_model = _nb_model = _tfidf = _label_enc = None
