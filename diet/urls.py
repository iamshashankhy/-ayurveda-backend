from django.urls import path
from .views import (
    DietPlanView,
    MarkMealCompletedView,
    HydrationView,
)

urlpatterns = [
    path('plan/', DietPlanView.as_view(), name='diet_plan'),
    path('meals/<int:meal_id>/complete/', MarkMealCompletedView.as_view(), name='mark_meal_completed'),
    path('hydration/', HydrationView.as_view(), name='hydration'),
    path('hydration/add/', HydrationView.as_view(), name='add_hydration'),
]