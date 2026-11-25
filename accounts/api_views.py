from rest_framework import generics, permissions
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import User
from .serializers import UserSerializer, RegisterSerializer

class RegisterAPIView(generics.CreateAPIView):
    queryset = User.objects.all()
    serializer_class = RegisterSerializer
    permission_classes = [permissions.AllowAny]

class UserProfileAPIView(generics.RetrieveUpdateAPIView):
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]
    def get_object(self): return self.request.user

class UserStatsAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    def get(self, request):
        return Response({'username': request.user.username, 'query_count': request.user.query_count})
