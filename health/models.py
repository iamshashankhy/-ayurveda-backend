from __future__ import annotations
from typing import TYPE_CHECKING
from django.conf import settings
from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator

if TYPE_CHECKING:
    from django.contrib.auth.models import User

class DoshaEvaluation(models.Model):
    # Explicitly declare objects manager for type checkers
    objects = models.Manager()
    
    user: 'User' = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='dosha_evaluations')  # type: ignore[assignment]
    vata = models.PositiveSmallIntegerField()
    pitta = models.PositiveSmallIntegerField()
    kapha = models.PositiveSmallIntegerField()
    advice = models.TextField(blank=True, default='')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'DoshaEvaluation<{self.user.username if self.user else "None"}:{self.created_at:%Y-%m-%d}>'

class CancerRiskResult(models.Model):
    # Explicitly declare objects manager for type checkers
    objects = models.Manager()
    
    RISK_LOW = 'low'
    RISK_MODERATE = 'moderate'
    RISK_HIGH = 'high'
    RISK_CHOICES = [
        (RISK_LOW, 'Low'),
        (RISK_MODERATE, 'Moderate'),
        (RISK_HIGH, 'High'),
    ]
    user: 'User' = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='cancer_risk_results')  # type: ignore[assignment]
    probability = models.FloatField()
    risk = models.CharField(max_length=10, choices=RISK_CHOICES)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'CancerRiskResult<{self.user.username if self.user else "None"}:{self.risk}:{self.probability:.2f}>'

class ActivityLog(models.Model):
    user: 'User' = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True, related_name='activity_logs')  # type: ignore[assignment]
    action = models.CharField(max_length=100)
    meta = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Explicitly declare objects manager for type checkers
    objects = models.Manager()

    def __str__(self):
        user_display = self.user.username if self.user else "Anonymous"
        return f'Activity<{user_display}:{self.action}>'


class Review(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    rating = models.IntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(5)]
    )
    comment = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['-created_at']),
            models.Index(fields=['rating']),
        ]
    
    def __str__(self):
        return f"{self.user.username} - {self.rating} stars"