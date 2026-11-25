from django.urls import path
from .api_views import PredictAPIView, SearchAPIView
urlpatterns = [
    path('predict/', PredictAPIView.as_view(), name='api_predict'),
    path('search/',  SearchAPIView.as_view(),  name='api_search'),
]
