from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.shortcuts import redirect


def home(request):
    return redirect('/chat/' if request.user.is_authenticated else '/accounts/login/')


urlpatterns = [
    path('',              home,                         name='home'),
    path('django-admin/', admin.site.urls),
    path('accounts/',     include('accounts.urls')),
    path('admin-panel/',  include('admin_panel.urls')),
    path('chat/',         include('chat.urls')),
    path('rag/',          include('rag_engine.urls')),
    path('api/accounts/', include('accounts.api_urls')),
    path('api/chat/',     include('chat.api_urls')),
    path('api/rag/',      include('rag_engine.api_urls')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
