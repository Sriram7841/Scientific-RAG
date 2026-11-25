from django.urls import path
from . import views
urlpatterns = [
    path('',                            views.dashboard,           name='admin_dashboard'),
    path('eda/',                        views.eda_view,            name='admin_eda'),
    path('train/',                      views.train_models_view,   name='admin_train'),
    path('build-index/',                views.build_index_view,    name='admin_build_index'),
    path('training-detail/',            views.training_detail,     name='admin_training_detail'),
    path('users/',                      views.user_management,     name='admin_users'),
    path('users/<int:user_id>/delete/', views.delete_user,         name='admin_delete_user'),
    path('graph/',                      views.graph_viz,           name='admin_graph_viz'),
    path('api/metrics/',                views.metrics_api,         name='admin_metrics_api'),
    path('api/confusion/',              views.confusion_matrix_api,name='admin_confusion_api'),
    path('api/status/',                 views.pipeline_status_api, name='admin_status_api'),
]
