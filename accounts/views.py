from django.contrib.auth.models import User
from rest_framework import status
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView
from health.models import ActivityLog
from accounts.models import Profile

class RegisterView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        email = request.data.get('email')
        password = request.data.get('password')
        name = request.data.get('name', '')
        phone = request.data.get('phone', '')
        
        if not email or not password:
            return Response({'error': 'Email and password are required'}, status=status.HTTP_400_BAD_REQUEST)
            
        if User.objects.filter(username=email).exists():
            return Response({'error': 'User with this email already exists'}, status=status.HTTP_409_CONFLICT)
            
        user = User.objects.create_user(username=email, email=email, password=password)
        if name:
            user.first_name = name
        if phone:
            user.last_name = phone  # Using last_name field to store phone number
        user.save()
        
        # Create profile for the user with initial preferences
        profile = Profile.objects.create(user=user)
        # Initialize profile preferences with signup data
        profile.preferences = {
            'fullName': name,
            'phone': phone,
            'email': email
        }
        profile.save()
        ActivityLog.objects.create(user=user, action='user_registered')
        
        return Response({
            'id': user.id, 
            'email': user.email, 
            'name': user.first_name,
            'phone': user.last_name
        }, status=status.HTTP_201_CREATED)