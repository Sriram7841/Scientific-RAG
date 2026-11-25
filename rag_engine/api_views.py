from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import permissions
from .classifier import predict_category
from .pipeline import hybrid_retrieve

class PredictAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    def post(self, request):
        query = request.data.get('query','').strip()
        if not query: return Response({'error':'query required'},status=400)
        return Response(predict_category(query))

class SearchAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    def post(self, request):
        query = request.data.get('query','').strip()
        k     = int(request.data.get('k', 5))
        if not query: return Response({'error':'query required'},status=400)
        f,g = hybrid_retrieve(query,k=k)
        return Response({'faiss_results':f,'graph_results':g})
