from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import get_object_or_404
from django.utils import timezone
from django.db.models import Sum
from datetime import date, timedelta
from .models import MealPlan, UserMealPlan, HydrationLog, DietProgress

class DietPlanView(APIView):
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        dosha_type = request.GET.get('dosha', None)
        
        if dosha_type:
            meal_plans = MealPlan.objects.filter(dosha_type=dosha_type)
        else:
            meal_plans = MealPlan.objects.all()
            
        # Group meals by type
        breakfast_meals = meal_plans.filter(meal_type='breakfast')
        lunch_meals = meal_plans.filter(meal_type='lunch')
        dinner_meals = meal_plans.filter(meal_type='dinner')
        snack_meals = meal_plans.filter(meal_type='snack')
        
        # Get user's completed meals for today
        today = date.today()
        user_meals = UserMealPlan.objects.filter(
            user=request.user,
            date=today
        )
        
        def serialize_meal(meal):
            user_meal = user_meals.filter(meal_plan=meal).first()
            return {
                'id': meal.id,
                'name': meal.name,
                'description': meal.description,
                'meal_type': meal.meal_type,
                'ingredients': meal.ingredients.split(','),
                'preparation': meal.preparation,
                'benefits': meal.benefits,
                'calories': meal.calories,
                'completed': user_meal.completed if user_meal else False,
                'completion_time': user_meal.completion_time if user_meal else None,
            }
        
        return Response({
            'breakfast': [serialize_meal(meal) for meal in breakfast_meals],
            'lunch': [serialize_meal(meal) for meal in lunch_meals],
            'dinner': [serialize_meal(meal) for meal in dinner_meals],
            'snacks': [serialize_meal(meal) for meal in snack_meals],
        })

class MarkMealCompletedView(APIView):
    permission_classes = [IsAuthenticated]
    
    def post(self, request, meal_id):
        meal_plan = get_object_or_404(MealPlan, id=meal_id)
        today = date.today()
        
        # Get or create user meal plan record
        user_meal, created = UserMealPlan.objects.get_or_create(
            user=request.user,
            meal_plan=meal_plan,
            date=today,
            defaults={'completed': False}
        )
        
        # Mark as completed
        user_meal.completed = True
        user_meal.completion_time = timezone.now()
        user_meal.save()
        
        # Update progress
        progress, created = DietProgress.objects.get_or_create(user=request.user)
        progress.total_meals_completed += 1
        
        # Update streak logic
        if progress.last_meal_date:
            # Check if this is a consecutive day
            if (today - progress.last_meal_date).days == 1:
                progress.current_streak += 1
            elif (today - progress.last_meal_date).days > 1:
                progress.current_streak = 1
        else:
            progress.current_streak = 1
            
        progress.last_meal_date = today
        progress.save()
        
        return Response({
            'message': f'Meal completed: {meal_plan.name}',
            'meal_id': meal_plan.id,
            'completed': True
        })

class HydrationView(APIView):
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        today = date.today()
        hydration_log = HydrationLog.objects.filter(
            user=request.user,
            date=today
        ).first()
        
        # Get weekly hydration data
        week_start = today - timedelta(days=today.weekday())
        weekly_data = HydrationLog.objects.filter(
            user=request.user,
            date__gte=week_start
        ).values('date').annotate(daily_total=Sum('water_amount')).order_by('date')
        
        return Response({
            'today_intake': hydration_log.water_amount if hydration_log else 0,
            'weekly_data': list(weekly_data),
            'goal': 2000  # 2 liters as default goal
        })
    
    def post(self, request):
        amount = request.data.get('amount', 0)
        today = date.today()
        
        # Get or create hydration log for today
        hydration_log, created = HydrationLog.objects.get_or_create(
            user=request.user,
            date=today,
            defaults={'water_amount': 0}
        )
        
        # Add to water amount
        hydration_log.water_amount += int(amount)
        hydration_log.save()
        
        # Update progress
        progress, created = DietProgress.objects.get_or_create(user=request.user)
        progress.total_water_intake += int(amount)
        progress.save()
        
        return Response({
            'message': f'Added {amount}ml to hydration log',
            'total_intake': hydration_log.water_amount
        })