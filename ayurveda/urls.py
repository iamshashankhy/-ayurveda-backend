from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('auth/', include('accounts.urls')),
    path('api/', include('health.urls')),
    path('api/yoga/', include('yoga.urls')),
    path('api/diet/', include('diet.urls')),
    path('api/settings/', include('user_settings.urls')),
]