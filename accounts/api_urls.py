from django.urls import path
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from .api_views import RegisterAPIView, UserProfileAPIView, UserStatsAPIView
urlpatterns = [
    path('token/',         TokenObtainPairView.as_view(), name='token_obtain'),
    path('token/refresh/', TokenRefreshView.as_view(),    name='token_refresh'),
    path('register/',      RegisterAPIView.as_view(),     name='api_register'),
    path('profile/',       UserProfileAPIView.as_view(),  name='api_profile'),
    path('stats/',         UserStatsAPIView.as_view(),    name='api_stats'),
]
