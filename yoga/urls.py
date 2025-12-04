from django.urls import path
from .views import (
    YogaPracticesView,
    StartYogaPracticeView,
    CompleteYogaPracticeView,
    YogaProgressView,
)

urlpatterns = [
    path('practices/', YogaPracticesView.as_view(), name='yoga_practices'),
    path('practices/<int:practice_id>/start/', StartYogaPracticeView.as_view(), name='start_yoga_practice'),
    path('practices/<int:practice_id>/complete/', CompleteYogaPracticeView.as_view(), name='complete_yoga_practice'),
    path('progress/', YogaProgressView.as_view(), name='yoga_progress'),
]