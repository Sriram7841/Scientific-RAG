import time
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import permissions
from .models import ChatSession, ChatMessage

class ChatQueryAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    def post(self, request):
        query = request.data.get('query','').strip()
        sid   = request.data.get('session_id')
        if not query: return Response({'error':'query required'},status=400)
        session = (ChatSession.objects.get(id=sid, user=request.user)
                   if sid else ChatSession.objects.create(user=request.user, title=query[:80]))
        ChatMessage.objects.create(session=session, role='user', content=query)
        t0 = time.time()
        cat, cat_label, conf = 'Unknown', 'Unknown', 0.0
        faiss_r, graph_r, answer, sources = [], [], '', []
        try:
            from rag_engine.classifier import predict_category
            r = predict_category(query)
            cat = r.get('category','Unknown'); cat_label = r.get('label','Unknown'); conf = r.get('confidence',0.0)
        except: pass
        try:
            from rag_engine.pipeline import hybrid_retrieve, generate_answer
            faiss_r, graph_r = hybrid_retrieve(query, k=5)
            resp = generate_answer(query, cat, cat_label, faiss_r, graph_r)
            answer = resp.get('answer',''); sources = resp.get('sources',[])
        except Exception as e:
            answer = f'Pipeline error: {e}'
        elapsed = round(time.time()-t0, 3)
        ChatMessage.objects.create(session=session, role='assistant', content=answer,
            category=cat, cat_label=cat_label, confidence=conf,
            sources=sources, graph_context=graph_r[:5], response_time=elapsed)
        request.user.increment_query()
        return Response({'answer':answer,'category':cat,'cat_label':cat_label,
            'confidence':conf,'sources':sources,'response_time':elapsed,'session_id':session.id})

class SessionListAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    def get(self, request):
        sessions = ChatSession.objects.filter(user=request.user).values('id','title','updated_at')[:20]
        return Response(list(sessions))
