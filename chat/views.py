import json
import time
import logging

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST
from django.http import JsonResponse

from .models import ChatSession, ChatMessage

logger = logging.getLogger('rag_engine')


@login_required
def chat_view(request, session_id=None):
    sessions = ChatSession.objects.filter(user=request.user)[:30]
    current  = None
    messages = []

    if session_id:
        current  = get_object_or_404(ChatSession, id=session_id, user=request.user)
        messages = current.messages.all()
    elif sessions.exists():
        current  = sessions.first()
        messages = current.messages.all()

    return render(request, 'chat/chat.html', {
        'sessions':        sessions,
        'current_session': current,
        'messages':        messages,
    })


@login_required
@require_POST
def new_session(request):
    session = ChatSession.objects.create(user=request.user, title='New Session')
    return redirect('chat_session', session_id=session.id)


@login_required
@require_POST
def send_message(request):
    try:
        body  = json.loads(request.body)
        query = body.get('query', '').strip()
        sid   = body.get('session_id')

        if not query:
            return JsonResponse({'error': 'Empty query.'}, status=400)

        if sid:
            session = get_object_or_404(ChatSession, id=sid, user=request.user)
        else:
            session = ChatSession.objects.create(user=request.user, title=query[:80])

        if not session.title or session.title == 'New Session':
            session.title = query[:80]
            session.save(update_fields=['title'])

        ChatMessage.objects.create(session=session, role='user', content=query)

        t0 = time.time()

        # 1. Classify query
        category, cat_label, confidence = 'Unknown', 'Unknown', 0.0
        try:
            from rag_engine.classifier import predict_category
            r          = predict_category(query, model='svm')
            category   = r.get('category', 'Unknown')
            cat_label  = r.get('label', 'Unknown')
            confidence = r.get('confidence', 0.0)
        except Exception as e:
            logger.warning(f'Classifier error: {e}')

        # 2. GraphRAG retrieval + LLM
        faiss_res, graph_res, answer, sources = [], [], '', []
        try:
            from rag_engine.pipeline import hybrid_retrieve, generate_answer
            faiss_res, graph_res = hybrid_retrieve(query, k=5)
            resp    = generate_answer(query, category, cat_label, faiss_res, graph_res)
            answer  = resp.get('answer', '')
            sources = resp.get('sources', [])
        except Exception as e:
            logger.error(f'RAG error: {e}')
            answer = (
                f'⚠️ Pipeline error: {e}\n\n'
                'Make sure models are trained and the FAISS index is built '
                'via Admin Panel → Train Models / Build Index.'
            )

        elapsed = round(time.time() - t0, 3)

        msg = ChatMessage.objects.create(
            session=session, role='assistant', content=answer,
            category=category, cat_label=cat_label, confidence=confidence,
            sources=sources, graph_context=graph_res[:5], response_time=elapsed,
        )
        request.user.increment_query()

        return JsonResponse({
            'session_id':    session.id,
            'session_title': session.title,
            'answer':        answer,
            'category':      category,
            'cat_label':     cat_label,
            'confidence':    confidence,
            'sources':       sources,
            'graph_context': graph_res[:5],
            'response_time': elapsed,
            'message_id':    msg.id,
        })

    except Exception as e:
        logger.exception('Unexpected error in send_message')
        return JsonResponse({'error': str(e)}, status=500)


@login_required
def session_history(request, session_id):
    session  = get_object_or_404(ChatSession, id=session_id, user=request.user)
    messages = [
        {
            'id':           m.id,
            'role':         m.role,
            'content':      m.content,
            'category':     m.category,
            'cat_label':    m.cat_label,
            'confidence':   m.confidence,
            'sources':      m.sources,
            'graph_context':m.graph_context,
            'response_time':m.response_time,
            'created_at':   m.created_at.strftime('%Y-%m-%d %H:%M'),
        }
        for m in session.messages.all()
    ]
    return JsonResponse({'messages': messages, 'session_title': session.title})


@login_required
@require_POST
def delete_session(request, session_id):
    session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    session.delete()
    return JsonResponse({'success': True})


@login_required
def export_pdf(request, session_id):
    session  = get_object_or_404(ChatSession, id=session_id, user=request.user)
    messages = session.messages.all()
    return render(request, 'chat/export_pdf.html',
                  {'session': session, 'messages': messages})
