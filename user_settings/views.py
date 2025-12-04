from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from .models import UserSettings

class UserSettingsView(APIView):
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        # Get or create user settings
        user_settings, created = UserSettings.objects.get_or_create(  # type: ignore
            user=request.user
        )
        
        return Response({
            'dark_mode': user_settings.dark_mode,
            'language': user_settings.language,
            'auto_sync': user_settings.auto_sync,
            'daily_reminders': user_settings.daily_reminders,
            'assessment_alerts': user_settings.assessment_alerts,
            'health_insights': user_settings.health_insights,
            'data_backup': user_settings.data_backup,
            'analytics': user_settings.analytics,
            'preferred_practice_time': user_settings.preferred_practice_time,
            'dietary_restrictions': user_settings.get_dietary_restrictions_list(),
            'favorite_cuisines': user_settings.get_favorite_cuisines_list(),
        })
    
    def put(self, request):
        # Get or create user settings
        user_settings, created = UserSettings.objects.get_or_create(  # type: ignore
            user=request.user
        )
        
        # Update settings based on request data
        settings_data = request.data.get('settings', {})
        
        # App Preferences
        if 'dark_mode' in settings_data:
            user_settings.dark_mode = settings_data['dark_mode']
        if 'language' in settings_data:
            user_settings.language = settings_data['language']
        if 'auto_sync' in settings_data:
            user_settings.auto_sync = settings_data['auto_sync']
            
        # Notifications
        if 'daily_reminders' in settings_data:
            user_settings.daily_reminders = settings_data['daily_reminders']
        if 'assessment_alerts' in settings_data:
            user_settings.assessment_alerts = settings_data['assessment_alerts']
        if 'health_insights' in settings_data:
            user_settings.health_insights = settings_data['health_insights']
            
        # Data & Privacy
        if 'data_backup' in settings_data:
            user_settings.data_backup = settings_data['data_backup']
        if 'analytics' in settings_data:
            user_settings.analytics = settings_data['analytics']
            
        # Wellness Preferences
        if 'preferred_practice_time' in settings_data:
            user_settings.preferred_practice_time = settings_data['preferred_practice_time']
            
        # Dietary Preferences
        if 'dietary_restrictions' in settings_data:
            user_settings.dietary_restrictions = ','.join(settings_data['dietary_restrictions']) if isinstance(settings_data['dietary_restrictions'], list) else settings_data['dietary_restrictions']
        if 'favorite_cuisines' in settings_data:
            user_settings.favorite_cuisines = ','.join(settings_data['favorite_cuisines']) if isinstance(settings_data['favorite_cuisines'], list) else settings_data['favorite_cuisines']
        
        user_settings.save()
        
        return Response({
            'message': 'Settings updated successfully',
            'settings': {
                'dark_mode': user_settings.dark_mode,
                'language': user_settings.language,
                'auto_sync': user_settings.auto_sync,
                'daily_reminders': user_settings.daily_reminders,
                'assessment_alerts': user_settings.assessment_alerts,
                'health_insights': user_settings.health_insights,
                'data_backup': user_settings.data_backup,
                'analytics': user_settings.analytics,
                'preferred_practice_time': user_settings.preferred_practice_time,
                'dietary_restrictions': user_settings.get_dietary_restrictions_list(),
                'favorite_cuisines': user_settings.get_favorite_cuisines_list(),
            }
        })