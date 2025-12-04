from django.urls import path
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)
from .views import RegisterView

urlpatterns = [
    path('register/', RegisterView.as_view(), name='register'),
    path('register', RegisterView.as_view(), name='register-no-slash'),
    path('login/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('login', TokenObtainPairView.as_view(), name='token_obtain_pair-no-slash'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('token/refresh', TokenRefreshView.as_view(), name='token_refresh-no-slash'),
]