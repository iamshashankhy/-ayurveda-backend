from django.conf import settings
from django.db import models

class MealPlan(models.Model):
    DOSHA_CHOICES = [
        ('vata', 'Vata'),
        ('pitta', 'Pitta'),
        ('kapha', 'Kapha'),
        ('balanced', 'Balanced'),
    ]
    
    name = models.CharField(max_length=200)
    description = models.TextField()
    dosha_type = models.CharField(max_length=20, choices=DOSHA_CHOICES)
    meal_type = models.CharField(max_length=50, choices=[
        ('breakfast', 'Breakfast'),
        ('lunch', 'Lunch'),
        ('dinner', 'Dinner'),
        ('snack', 'Snack'),
    ])
    ingredients = models.TextField(help_text="List of ingredients separated by commas")
    preparation = models.TextField()
    benefits = models.TextField()
    calories = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.name} ({self.dosha_type})"

class UserMealPlan(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    meal_plan = models.ForeignKey(MealPlan, on_delete=models.CASCADE)
    date = models.DateField()
    completed = models.BooleanField(default=False)
    completion_time = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ('user', 'meal_plan', 'date')
    
    def __str__(self):
        return f"{self.user} - {self.meal_plan.name} on {self.date}"

class HydrationLog(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    date = models.DateField()
    water_amount = models.PositiveIntegerField(help_text="Amount of water in ml")
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ('user', 'date')
    
    def __str__(self):
        return f"{self.user} - {self.water_amount}ml on {self.date}"

class DietProgress(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    current_streak = models.PositiveIntegerField(default=0)
    total_meals_completed = models.PositiveIntegerField(default=0)
    total_water_intake = models.PositiveIntegerField(default=0, help_text="Total water intake in ml")
    last_meal_date = models.DateField(null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Diet Progress for {self.user}"