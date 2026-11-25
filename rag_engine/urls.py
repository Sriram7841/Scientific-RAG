from django.urls import path
from . import views
urlpatterns = [
    path('status/',    views.rag_status,    name='rag_status'),
    path('graph-viz/', views.graph_viz_data, name='graph_viz_data'),
]
