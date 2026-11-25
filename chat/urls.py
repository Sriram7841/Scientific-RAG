from django.urls import path
from . import views
urlpatterns = [
    path('',                          views.chat_view,       name='chat'),
    path('<int:session_id>/',         views.chat_view,       name='chat_session'),
    path('new/',                      views.new_session,     name='new_session'),
    path('send/',                     views.send_message,    name='send_message'),
    path('history/<int:session_id>/', views.session_history, name='session_history'),
    path('delete/<int:session_id>/',  views.delete_session,  name='delete_session'),
    path('export/<int:session_id>/',  views.export_pdf,      name='export_pdf'),
]
