"""
python manage.py setup_graphrag [--rows N] [--skip-train] [--skip-index] [--skip-graph]

Full pipeline on data/arxiv.csv (206,489 papers, 25 categories):
  1. Load & validate CSV
  2. Train LinearSVC + ComplementNB  (80/20 split)
  3. Build FAISS index
  4. Build NetworkX knowledge graph
"""
import time
import json
from pathlib import Path
from django.core.management.base import BaseCommand
from django.conf import settings
from django.utils import timezone


class Command(BaseCommand):
    help = 'Full GraphRAG setup pipeline on data/arxiv.csv'

    def add_arguments(self, parser):
        parser.add_argument('--rows',       type=int, default=0,
                            help='Max CSV rows (0 = settings.MAX_DATASET_ROWS)')
        parser.add_argument('--skip-train', action='store_true')
        parser.add_argument('--skip-index', action='store_true')
        parser.add_argument('--skip-graph', action='store_true')

    def _bar(self):
        self.stdout.write('─' * 60)

    def handle(self, *args, **opts):
        self.stdout.write(self.style.SUCCESS('\n╔══════════════════════════════════════════════════════╗'))
        self.stdout.write(self.style.SUCCESS('║      GraphRAG arXiv Pipeline  (206K papers, 25 cats)  ║'))
        self.stdout.write(self.style.SUCCESS('╚══════════════════════════════════════════════════════╝\n'))

        csv_path = Path(settings.DATA_DIR) / 'arxiv.csv'
        if not csv_path.exists():
            self.stdout.write(self.style.ERROR(
                f'\n❌  data/arxiv.csv not found at {csv_path}\n'
                '    Download from Kaggle:\n'
                '    https://www.kaggle.com/datasets/spsayakpaul/arxiv-paper-abstracts\n'
                '    Place as:  data/arxiv.csv\n'
            ))
            return

        max_rows = opts['rows'] or settings.MAX_DATASET_ROWS
        self._bar()
        self.stdout.write(f'📂  CSV: {csv_path}  (max_rows={max_rows:,})')

        # ── Step 1 : Load CSV ─────────────────────────────────────
        self.stdout.write('\n[1/4] Loading dataset …')
        from rag_engine.classifier import load_arxiv_csv
        df = load_arxiv_csv(str(csv_path), max_rows=max_rows)
        self.stdout.write(self.style.SUCCESS(
            f'      ✓ {len(df):,} records  ·  {df["category"].nunique()} categories'
        ))
        top5 = df['category'].value_counts().head(5)
        for cat, cnt in top5.items():
            self.stdout.write(f'         {cat:<15} {cnt:>7,} papers')
        self._bar()

        # ── Step 2 : Train classifiers ────────────────────────────
        if not opts['skip_train']:
            self.stdout.write('[2/4] Training SVM + Naive Bayes  (80/20 split) …')
            from rag_engine.classifier import train_models
            from rag_engine.models import TrainingRun, ModelMetrics

            run = TrainingRun.objects.create(status='running', model_type='both')
            try:
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
                self.stdout.write(self.style.SUCCESS(
                    f"      ✓ SVM  → Acc={results['svm']['accuracy']}  "
                    f"F1={results['svm']['f1_score']}  "
                    f"({results['svm']['train_time_sec']}s)\n"
                    f"      ✓ NB   → Acc={results['naive_bayes']['accuracy']}  "
                    f"F1={results['naive_bayes']['f1_score']}  "
                    f"({results['naive_bayes']['train_time_sec']}s)"
                ))
            except Exception as e:
                run.status = 'error'; run.error_msg = str(e); run.save()
                self.stdout.write(self.style.ERROR(f'Training error: {e}'))
        else:
            self.stdout.write('[2/4] SKIPPED (--skip-train)')
        self._bar()

        # ── Step 3 : FAISS ────────────────────────────────────────
        if not opts['skip_index']:
            self.stdout.write('[3/4] Building FAISS vector index …')
            from rag_engine.pipeline import build_faiss_index
            from rag_engine.models import IndexBuildLog

            recs = df[['title', 'abstract', 'category']].copy()
            recs['doc_id'] = recs.index.astype(str)
            rec_list = recs.to_dict('records')

            log = IndexBuildLog.objects.create(status='running', total_documents=len(rec_list))
            try:
                stats = build_faiss_index(rec_list)
                log.status        = 'done'
                log.total_vectors = stats['total_vectors']
                log.completed_at  = timezone.now()
                log.save()
                self.stdout.write(self.style.SUCCESS(
                    f"      ✓ {stats['total_vectors']:,} vectors  dim={stats['dimension']}"
                ))
            except Exception as e:
                log.status = 'error'; log.error_msg = str(e); log.save()
                self.stdout.write(self.style.ERROR(f'FAISS error: {e}'))
        else:
            self.stdout.write('[3/4] SKIPPED (--skip-index)')
        self._bar()

        # ── Step 4 : Knowledge Graph ──────────────────────────────
        if not opts['skip_graph']:
            self.stdout.write('[4/4] Building NetworkX knowledge graph …')
            from rag_engine.pipeline import build_knowledge_graph, reset_pipeline_cache
            from rag_engine.classifier import reset_classifier_cache

            g_recs = df[['title', 'abstract', 'category']].head(15000).to_dict('records')
            try:
                gstats = build_knowledge_graph(g_recs)
                self.stdout.write(self.style.SUCCESS(
                    f"      ✓ {gstats['nodes']} nodes  ·  {gstats['edges']} edges"
                ))
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Graph error: {e}'))

            reset_pipeline_cache()
            reset_classifier_cache()
        else:
            self.stdout.write('[4/4] SKIPPED (--skip-graph)')
        self._bar()

        self.stdout.write(self.style.SUCCESS(
            '\n✅  Pipeline complete!\n'
            '   ▶  python manage.py runserver\n'
            '   ▶  http://127.0.0.1:8000\n'
        ))
