"""
Admin Panel views
──────────────────
Dashboard · EDA · Training · Index Build · User Mgmt · Graph Viz
All heavy jobs run in background threads so the web request returns immediately.
"""
import json
import threading
import logging
from pathlib import Path

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required, user_passes_test
from django.views.decorators.http import require_POST
from django.http import JsonResponse
from django.contrib import messages
from django.contrib.auth import get_user_model
from django.conf import settings
from django.utils import timezone

logger = logging.getLogger('rag_engine')
User   = get_user_model()


def is_admin(user):
    return user.is_active and (user.is_staff or user.is_superuser)


admin_req = user_passes_test(is_admin, login_url='/accounts/login/')


# ── Dashboard ─────────────────────────────────────────────────────
@login_required
@admin_req
def dashboard(request):
    from rag_engine.models import TrainingRun, ModelMetrics, IndexBuildLog
    from rag_engine.classifier import models_ready
    from rag_engine.pipeline import faiss_ready, graph_ready
    from chat.models import ChatMessage, ChatSession

    latest_run = TrainingRun.objects.filter(status='done').last()
    svm_m = nb_m = None
    if latest_run:
        svm_m = latest_run.metrics.filter(model_name='svm').last()
        nb_m  = latest_run.metrics.filter(model_name='naive_bayes').last()

    last_index = IndexBuildLog.objects.filter(status='done').last()
    csv_path   = Path(settings.DATA_DIR) / 'arxiv.csv'

    ctx = {
        'user_count':       User.objects.count(),
        'message_count':    ChatMessage.objects.count(),
        'session_count':    ChatSession.objects.count(),
        'classifier_ready': models_ready(),
        'faiss_ready':      faiss_ready(),
        'graph_ready':      graph_ready(),
        'csv_exists':       csv_path.exists(),
        'csv_size_mb':      round(csv_path.stat().st_size / 1_048_576, 1) if csv_path.exists() else 0,
        'latest_run':       latest_run,
        'svm_m':            svm_m,
        'nb_m':             nb_m,
        'last_index':       last_index,
        'training_runs':    TrainingRun.objects.all()[:6],
    }
    return render(request, 'admin_panel/dashboard.html', ctx)


# ── EDA ───────────────────────────────────────────────────────────
@login_required
@admin_req
def eda_view(request):
    from rag_engine.classifier import load_arxiv_csv, get_dataset_stats

    csv_path = Path(settings.DATA_DIR) / 'arxiv.csv'

    if not csv_path.exists():
        messages.error(
            request,
            'data/arxiv.csv not found. Place dataset file there.'
        )
        return redirect('admin_dashboard')

    try:
        df = load_arxiv_csv(
            str(csv_path),
            max_rows=settings.MAX_DATASET_ROWS
        )

        stats = get_dataset_stats(df)

        top_cats = stats['top_categories']

        cat_labels = list(top_cats.keys())[:25]
        cat_counts = [int(top_cats[c]) for c in cat_labels]

        cat_names = [
            settings.ARXIV_CATEGORIES.get(c, c)
            for c in cat_labels
        ]

        # Build top-5 table data safely
        total_records = stats['total_records']

        top5_data = []

        for category, count in list(top_cats.items())[:5]:

            percentage = round(
                (count / total_records) * 100,
                2
            )

            readable_name = settings.ARXIV_CATEGORIES.get(
                category,
                category
            )

            top5_data.append(
                (
                    category,
                    readable_name,
                    count,
                    percentage
                )
            )

        word_freq = {}

        sample = (
            df['text']
            .dropna()
            .sample(
                min(1000, len(df)),
                random_state=42
            )
        )

        for txt in sample:

            for word in txt.split():

                if len(word) > 4:
                    word_freq[word] = (
                        word_freq.get(word, 0) + 1
                    )

        top_words = sorted(
            word_freq.items(),
            key=lambda x: x[1],
            reverse=True
        )[:100]

        ctx = {

            'total_records': stats['total_records'],
            'num_categories': stats['num_categories'],
            'avg_abstract_len': stats['avg_abstract_len'],

            'cat_labels': json.dumps(cat_labels),
            'cat_names': json.dumps(cat_names),
            'cat_counts': json.dumps(cat_counts),

            'top_words': json.dumps([
                {'text': w, 'size': c}
                for w, c in top_words
            ]),

            'top5_data': top5_data
        }

    except Exception as e:

        messages.error(
            request,
            f'EDA error: {str(e)}'
        )

        ctx = {}

    return render(
        request,
        'admin_panel/eda.html',
        ctx
    )

# ── Train ─────────────────────────────────────────────────────────
@login_required
@admin_req
@require_POST
def train_models_view(request):
    from rag_engine.models import TrainingRun, ModelMetrics
    from rag_engine.classifier import load_arxiv_csv, train_models

    csv_path = Path(settings.DATA_DIR) / 'arxiv.csv'
    if not csv_path.exists():
        messages.error(request, 'data/arxiv.csv not found.')
        return redirect('admin_dashboard')

    run = TrainingRun.objects.create(status='running', model_type='both')

    def task():
        try:
            df = load_arxiv_csv(str(csv_path), max_rows=settings.MAX_DATASET_ROWS)
            results = train_models(df)
            run.status       = 'done'
            run.train_size   = results['train_size']
            run.test_size    = results['test_size']
            run.num_classes  = results['num_classes']
            run.completed_at = timezone.now()
            run.save()
            for mname in ('svm', 'naive_bayes'):
                m = results[mname]
                ModelMetrics.objects.create(
                    training_run=run, model_name=mname,
                    accuracy=m['accuracy'],  precision=m['precision'],
                    recall=m['recall'],       f1_score=m['f1_score'],
                    train_time_sec=m.get('train_time_sec', 0),
                    confusion_matrix_json=json.dumps(m['confusion_matrix']),
                    class_report_json=json.dumps(m['class_report']),
                )
            logger.info(f'Training run {run.id} done.')
        except Exception as e:
            run.status = 'error'; run.error_msg = str(e); run.save()
            logger.error(f'Training failed: {e}')

    threading.Thread(target=task, daemon=True).start()
    messages.info(request,
        '🚀 Training started in background (SVM + Naive Bayes on 25 categories). '
        'Refresh in ~3-8 minutes.')
    return redirect('admin_dashboard')


# ── Build FAISS + Graph ───────────────────────────────────────────
@login_required
@admin_req
@require_POST
def build_index_view(request):
    from rag_engine.models import IndexBuildLog
    from rag_engine.classifier import load_arxiv_csv
    from rag_engine.pipeline import build_faiss_index, build_knowledge_graph, reset_pipeline_cache

    csv_path = Path(settings.DATA_DIR) / 'arxiv.csv'
    if not csv_path.exists():
        messages.error(request, 'data/arxiv.csv not found.')
        return redirect('admin_dashboard')

    log = IndexBuildLog.objects.create(status='running')

    def task():
        try:
            df       = load_arxiv_csv(str(csv_path), max_rows=settings.MAX_DATASET_ROWS)
            recs     = df[['title', 'abstract', 'category']].copy()
            recs['doc_id'] = recs.index.astype(str)
            rec_list = recs.to_dict('records')

            log.total_documents = len(rec_list)
            log.save(update_fields=['total_documents'])

            stats = build_faiss_index(rec_list)
            log.total_vectors = stats['total_vectors']
            log.save(update_fields=['total_vectors'])

            gstats = build_knowledge_graph(rec_list[:15000])
            log.graph_nodes  = gstats['nodes']
            log.graph_edges  = gstats['edges']
            log.status       = 'done'
            log.completed_at = timezone.now()
            log.save()
            reset_pipeline_cache()
            logger.info(f'Index build {log.id} done.')
        except Exception as e:
            log.status = 'error'; log.error_msg = str(e); log.save()
            logger.error(f'Index build failed: {e}')

    threading.Thread(target=task, daemon=True).start()
    messages.info(request,
        '⚙️ FAISS index + Knowledge Graph building started. '
        'This may take 10-20 minutes on the full dataset.')
    return redirect('admin_dashboard')


# ── Training detail page ──────────────────────────────────────────
@login_required
@admin_req
# ── Training detail page ──────────────────────────────────────────
@login_required
@admin_req
def training_detail(request):
    from rag_engine.models import TrainingRun

    run = TrainingRun.objects.filter(status='done').last()

    if not run:
        return render(
            request,
            'admin_panel/training_detail.html',
            {'no_data': True}
        )

    svm_m = run.metrics.filter(
        model_name='svm'
    ).last()

    nb_m = run.metrics.filter(
        model_name='naive_bayes'
    ).last()

    svm_cr = (
        json.loads(svm_m.class_report_json or '{}')
        if svm_m else {}
    )

    nb_cr = (
        json.loads(nb_m.class_report_json or '{}')
        if nb_m else {}
    )

    skip = {
        'accuracy',
        'macro avg',
        'weighted avg'
    }

    classes = [
        k for k in svm_cr
        if k not in skip
    ][:25]

    svm_f1s = [
        round(
            svm_cr.get(c, {}).get(
                'f1-score',
                0
            ),
            3
        )
        for c in classes
    ]

    nb_f1s = [
        round(
            nb_cr.get(c, {}).get(
                'f1-score',
                0
            ),
            3
        )
        for c in classes
    ]

    cat_labels = [
        settings.ARXIV_CATEGORIES.get(
            c,
            c
        )
        for c in classes
    ]

    # NEW: Build iterable rows for template table
    model_rows = []

    for i in range(len(classes)):

        model_rows.append(
            (
                classes[i],      # category code
                cat_labels[i],   # readable name
                svm_f1s[i],      # SVM score
                nb_f1s[i]        # NB score
            )
        )

    return render(
        request,
        'admin_panel/training_detail.html',
        {
            'run': run,
            'svm_m': svm_m,
            'nb_m': nb_m,

            # JS chart data
            'classes': json.dumps(classes),
            'cat_labels': json.dumps(cat_labels),
            'svm_f1s': json.dumps(svm_f1s),
            'nb_f1s': json.dumps(nb_f1s),

            # table data
            'model_rows': model_rows,
        }
    )

# ── JSON APIs ─────────────────────────────────────────────────────
@login_required
@admin_req
def metrics_api(request):
    from rag_engine.models import TrainingRun
    runs = TrainingRun.objects.filter(status='done').order_by('-completed_at')[:10]
    data = []
    for r in runs:
        entry = {'run_id': r.id, 'completed_at': str(r.completed_at), 'metrics': []}
        for m in r.metrics.all():
            entry['metrics'].append({
                'model':     m.model_name,
                'accuracy':  m.accuracy,
                'precision': m.precision,
                'recall':    m.recall,
                'f1_score':  m.f1_score,
            })
        data.append(entry)
    return JsonResponse({'runs': data})


@login_required
@admin_req
def confusion_matrix_api(request):
    from rag_engine.models import ModelMetrics
    model_name = request.GET.get('model', 'svm')
    m = ModelMetrics.objects.filter(model_name=model_name).last()
    if not m:
        return JsonResponse({'error': 'No metrics yet. Train models first.'}, status=404)
    return JsonResponse({
        'model_name':       m.model_name,
        'accuracy':         m.accuracy,
        'f1_score':         m.f1_score,
        'confusion_matrix': json.loads(m.confusion_matrix_json or '[]'),
        'class_report':     json.loads(m.class_report_json or '{}'),
    })


@login_required
@admin_req
def pipeline_status_api(request):
    from rag_engine.models import TrainingRun, IndexBuildLog
    from rag_engine.classifier import models_ready
    from rag_engine.pipeline import faiss_ready, graph_ready
    return JsonResponse({
        'classifier_ready': models_ready(),
        'faiss_ready':      faiss_ready(),
        'graph_ready':      graph_ready(),
        'training_running': TrainingRun.objects.filter(status='running').exists(),
        'indexing_running': IndexBuildLog.objects.filter(status='running').exists(),
    })


# ── User management ───────────────────────────────────────────────
@login_required
@admin_req
def user_management(request):
    users = User.objects.all().order_by('-date_joined')
    return render(request, 'admin_panel/users.html', {'users': users})


@login_required
@admin_req
@require_POST
def delete_user(request, user_id):
    user = get_object_or_404(User, id=user_id)
    if user == request.user:
        return JsonResponse({'error': 'Cannot delete yourself.'}, status=400)
    if user.is_superuser:
        return JsonResponse({'error': 'Cannot delete a superuser.'}, status=403)
    user.delete()
    return JsonResponse({'success': True})


# ── Graph visualisation page ──────────────────────────────────────
@login_required
@admin_req
def graph_viz(request):
    from rag_engine.pipeline import graph_ready
    return render(request, 'admin_panel/graph_viz.html', {'graph_ready': graph_ready()})
