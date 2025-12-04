from django.conf import settings
from django.db import models
from django.contrib.auth.models import User

class YogaPractice(models.Model):
    DOSHA_CHOICES = [
        ('vata', 'Vata'),
        ('pitta', 'Pitta'),
        ('kapha', 'Kapha'),
        ('balanced', 'Balanced'),
    ]
    
    name = models.CharField(max_length=200)
    description = models.TextField()
    duration = models.PositiveIntegerField(help_text="Duration in minutes")
    dosha_type = models.CharField(max_length=20, choices=DOSHA_CHOICES)
    practice_type = models.CharField(max_length=100, help_text="e.g., Asana, Pranayama, Meditation")
    benefits = models.TextField()
    difficulty = models.CharField(max_length=20, choices=[
        ('beginner', 'Beginner'),
        ('intermediate', 'Intermediate'),
        ('advanced', 'Advanced'),
    ])
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return str(self.name)

class UserYogaPractice(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    practice = models.ForeignKey(YogaPractice, on_delete=models.CASCADE)
    completed = models.BooleanField(default=False)
    completion_date = models.DateTimeField(null=True, blank=True)
    duration_completed = models.PositiveIntegerField(null=True, blank=True, help_text="Minutes completed")
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ('user', 'practice')
    
    def __str__(self):
        return f"{self.user} - {self.practice.name}"

class YogaProgress(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    current_streak = models.PositiveIntegerField(default=0)
    total_sessions = models.PositiveIntegerField(default=0)
    total_minutes = models.PositiveIntegerField(default=0)
    favorite_style = models.CharField(max_length=100, blank=True)
    last_practice_date = models.DateTimeField(null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Progress for {self.user}"