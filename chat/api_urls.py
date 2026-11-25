from django.urls import path
from .api_views import ChatQueryAPIView, SessionListAPIView
urlpatterns = [
    path('query/',    ChatQueryAPIView.as_view(),   name='api_chat_query'),
    path('sessions/', SessionListAPIView.as_view(), name='api_chat_sessions'),
]
