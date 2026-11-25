from django.db import models
from django.contrib.auth.models import AbstractUser


class User(AbstractUser):
    bio           = models.TextField(blank=True)
    avatar        = models.ImageField(upload_to='avatars/', blank=True, null=True)
    is_researcher = models.BooleanField(default=True)
    query_count   = models.PositiveIntegerField(default=0)
    created_at    = models.DateTimeField(auto_now_add=True)
    last_active   = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'accounts_user'

    def __str__(self):
        return self.username

    def increment_query(self):
        self.query_count += 1
        self.save(update_fields=['query_count', 'last_active'])
