from django.conf import settings
from django.db import models

class UserSettings(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    
    # App Preferences
    dark_mode = models.BooleanField(default=False)  # type: ignore
    language = models.CharField(max_length=20, default='english')
    auto_sync = models.BooleanField(default=True)  # type: ignore
    
    # Notifications
    daily_reminders = models.BooleanField(default=True)  # type: ignore
    assessment_alerts = models.BooleanField(default=True)  # type: ignore
    health_insights = models.BooleanField(default=True)  # type: ignore
    
    # Data & Privacy
    data_backup = models.BooleanField(default=True)  # type: ignore
    analytics = models.BooleanField(default=False)  # type: ignore
    
    # Wellness Preferences
    preferred_practice_time = models.CharField(max_length=20, default='morning', choices=[
        ('morning', 'Morning'),
        ('afternoon', 'Afternoon'),
        ('evening', 'Evening'),
    ])
    
    # Dietary Preferences
    dietary_restrictions = models.TextField(blank=True, help_text="Comma-separated list of dietary restrictions")
    favorite_cuisines = models.TextField(blank=True, help_text="Comma-separated list of favorite cuisines")
    
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Settings for {self.user}"
    
    def get_dietary_restrictions_list(self):
        """Return dietary restrictions as a list"""
        if self.dietary_restrictions:
            restrictions = str(self.dietary_restrictions)
            return [item.strip() for item in restrictions.split(',') if item.strip()]
        return []
    
    def get_favorite_cuisines_list(self):
        """Return favorite cuisines as a list"""
        if self.favorite_cuisines:
            cuisines = str(self.favorite_cuisines)
            return [item.strip() for item in cuisines.split(',') if item.strip()]
        return []