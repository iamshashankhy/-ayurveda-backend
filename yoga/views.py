from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import get_object_or_404
from django.utils import timezone
from .models import YogaPractice, UserYogaPractice, YogaProgress

class YogaPracticesView(APIView):
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        dosha_type = request.GET.get('dosha', None)
        
        if dosha_type:
            practices = YogaPractice.objects.filter(dosha_type=dosha_type)
        else:
            practices = YogaPractice.objects.all()
            
        practice_data = []
        for practice in practices:
            # Check if user has completed this practice
            user_practice = UserYogaPractice.objects.filter(
                user=request.user, 
                practice=practice
            ).first()
            
            practice_data.append({
                'id': practice.id,
                'name': practice.name,
                'description': practice.description,
                'duration': practice.duration,
                'dosha_type': practice.dosha_type,
                'practice_type': practice.practice_type,
                'benefits': practice.benefits,
                'difficulty': practice.difficulty,
                'completed': user_practice.completed if user_practice else False,
                'completion_date': user_practice.completion_date if user_practice and user_practice.completion_date else None,
            })
            
        return Response({'practices': practice_data})

class StartYogaPracticeView(APIView):
    permission_classes = [IsAuthenticated]
    
    def post(self, request, practice_id):
        practice = get_object_or_404(YogaPractice, id=practice_id)
        
        # Create or get user practice record
        user_practice, created = UserYogaPractice.objects.get_or_create(
            user=request.user,
            practice=practice,
            defaults={'completed': False}
        )
        
        return Response({
            'message': f'Started practice: {practice.name}',
            'practice_id': practice.id,
            'duration': practice.duration
        })

class CompleteYogaPracticeView(APIView):
    permission_classes = [IsAuthenticated]
    
    def post(self, request, practice_id):
        practice = get_object_or_404(YogaPractice, id=practice_id)
        
        # Get or create user practice record
        user_practice, created = UserYogaPractice.objects.get_or_create(
            user=request.user,
            practice=practice
        )
        
        # Mark as completed
        user_practice.completed = True
        user_practice.completion_date = timezone.now()
        user_practice.duration_completed = practice.duration
        user_practice.save()
        
        # Update progress
        progress, created = YogaProgress.objects.get_or_create(user=request.user)
        progress.total_sessions += 1
        progress.total_minutes += practice.duration
        
        # Update streak logic
        if progress.last_practice_date:
            # Check if this is a consecutive day
            if (timezone.now().date() - progress.last_practice_date.date()).days == 1:
                progress.current_streak += 1
            elif (timezone.now().date() - progress.last_practice_date.date()).days > 1:
                progress.current_streak = 1
        else:
            progress.current_streak = 1
            
        progress.last_practice_date = timezone.now()
        
        # Update favorite style if this practice type is done most
        if practice.practice_type:
            progress.favorite_style = practice.practice_type
            
        progress.save()
        
        return Response({
            'message': f'Practice completed: {practice.name}',
            'practice_id': practice.id,
            'completed': True
        })

class YogaProgressView(APIView):
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        progress, created = YogaProgress.objects.get_or_create(user=request.user)
        
        return Response({
            'current_streak': progress.current_streak,
            'total_sessions': progress.total_sessions,
            'total_minutes': progress.total_minutes,
            'favorite_style': progress.favorite_style,
            'last_practice_date': progress.last_practice_date
        })