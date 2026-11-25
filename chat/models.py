from django.db import models
from django.conf import settings


class ChatSession(models.Model):
    user       = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                                   related_name='chat_sessions')
    title      = models.CharField(max_length=300, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']

    def __str__(self):
        return f'{self.user.username} – {self.title or "Session"}'


class ChatMessage(models.Model):
    ROLE = [('user', 'User'), ('assistant', 'Assistant')]

    session       = models.ForeignKey(ChatSession, on_delete=models.CASCADE,
                                       related_name='messages')
    role          = models.CharField(max_length=10, choices=ROLE)
    content       = models.TextField()
    category      = models.CharField(max_length=80, blank=True)
    cat_label     = models.CharField(max_length=120, blank=True)
    confidence    = models.FloatField(default=0.0)
    sources       = models.JSONField(default=list, blank=True)
    graph_context = models.JSONField(default=list, blank=True)
    response_time = models.FloatField(null=True, blank=True)
    created_at    = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f'{self.role}: {self.content[:60]}'
