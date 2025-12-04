from django.contrib import admin
from .models import DoshaEvaluation, CancerRiskResult, ActivityLog

@admin.register(DoshaEvaluation)
class DoshaEvaluationAdmin(admin.ModelAdmin):
    list_display = ('user', 'vata', 'pitta', 'kapha', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('user__username', 'user__email')

@admin.register(CancerRiskResult)
class CancerRiskResultAdmin(admin.ModelAdmin):
    list_display = ('user', 'risk', 'probability', 'created_at')
    list_filter = ('risk', 'created_at')
    search_fields = ('user__username', 'user__email')

@admin.register(ActivityLog)
class ActivityLogAdmin(admin.ModelAdmin):
    list_display = ('user', 'action', 'created_at')
    list_filter = ('action', 'created_at')
    search_fields = ('user__username', 'user__email', 'action')




