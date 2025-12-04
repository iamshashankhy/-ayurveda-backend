from django.urls import path
from django.http import JsonResponse
from .views import (
    AnalyzeDoshaView,
    PredictCancerView,
    CombinedAssessmentView,
    AssessmentQuestionsView,
    ProfileView,
    ReviewsView,
)

def api_test(request):
    return JsonResponse({
        'status': 'API is working',
        'endpoints': [
            '/api/analyze_dosha/',
            '/api/predict_cancer/',
            '/api/assessment/combined/',
            '/api/assessment/questions/',
            '/api/profile/',
            '/api/reviews/',
        ]
    })

urlpatterns = [
    path('', api_test, name='api_test'),  # Test endpoint
    path('analyze_dosha/', AnalyzeDoshaView.as_view(), name='analyze_dosha'),
    path('predict_cancer/', PredictCancerView.as_view(), name='predict_cancer'),
    path('assessment/combined/', CombinedAssessmentView.as_view(), name='combined_assessment'),
    path('assessment/questions/', AssessmentQuestionsView.as_view(), name='assessment_questions'),
    path('profile/', ProfileView.as_view(), name='profile'),
    path('reviews/', ReviewsView.as_view(), name='reviews'),
]