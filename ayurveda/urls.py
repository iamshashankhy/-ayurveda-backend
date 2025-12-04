from django.contrib import admin
from django.urls import path, include
from django.http import JsonResponse

def health_check(request):
    return JsonResponse({
        'status': 'healthy',
        'service': 'ayurveda-django-backend',
        'message': 'Backend is running successfully'
    })

def root_view(request):
    return JsonResponse({
        'message': 'Welcome to Ayurveda Backend API',
        'endpoints': {
            'health': '/health/',
            'admin': '/admin/',
            'api': '/api/'
        }
    })

urlpatterns = [
    path('', root_view, name='home'),
    path('health/', health_check, name='health_check'),
    path('admin/', admin.site.urls),
    path('auth/', include('accounts.urls')),
    path('api/', include('health.urls')),
    path('api/yoga/', include('yoga.urls')),
    path('api/diet/', include('diet.urls')),
    path('api/settings/', include('user_settings.urls')),
]