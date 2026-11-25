from django.contrib import admin
from .models import ChatSession, ChatMessage
@admin.register(ChatSession)
class CSAdmin(admin.ModelAdmin):
    list_display = ('user','title','created_at','updated_at')
@admin.register(ChatMessage)
class CMAdmin(admin.ModelAdmin):
    list_display = ('session','role','category','confidence','response_time','created_at')
