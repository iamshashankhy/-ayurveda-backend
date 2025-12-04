from __future__ import annotations
from typing import TYPE_CHECKING
from django.conf import settings
from django.db import models

if TYPE_CHECKING:
    from django.contrib.auth.models import User

class Profile(models.Model):
    user: 'User' = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='profile')  # type: ignore[assignment]
    preferences = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Explicitly declare objects manager for type checkers
    objects = models.Manager()

    def __str__(self) -> str:
        return f'Profile for {self.user.username}'