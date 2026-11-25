from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import User
@admin.register(User)
class CustomUserAdmin(UserAdmin):
    list_display = ('username','email','query_count','is_staff','created_at')
