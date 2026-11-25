import json
from django.http import JsonResponse
from django.contrib.admin.views.decorators import staff_member_required
from .pipeline import faiss_ready, graph_ready, get_graph_viz_data
from .classifier import models_ready

@staff_member_required
def rag_status(request):
    from .models import TrainingRun, IndexBuildLog
    lt = TrainingRun.objects.filter(status='done').last()
    li = IndexBuildLog.objects.filter(status='done').last()
    return JsonResponse({
        'classifier_ready': models_ready(),
        'faiss_ready':      faiss_ready(),
        'graph_ready':      graph_ready(),
        'last_trained':     str(lt.completed_at) if lt else None,
        'last_indexed':     str(li.completed_at) if li else None,
    })

@staff_member_required
def graph_viz_data(request):
    return JsonResponse(get_graph_viz_data(max_nodes=150))
