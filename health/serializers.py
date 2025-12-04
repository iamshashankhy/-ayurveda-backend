from rest_framework import serializers
from .models import Review

class ReviewSerializer(serializers.ModelSerializer):
    username = serializers.ReadOnlyField(source='user.username')
    created_at_formatted = serializers.SerializerMethodField()
    
    class Meta:
        model = Review
        fields = ['id', 'username', 'rating', 'comment', 'created_at', 'created_at_formatted']
        read_only_fields = ['user', 'id', 'created_at', 'updated_at']
        extra_kwargs = {
            'user': {'required': False},
            'rating': {'required': True},
            'comment': {'required': False, 'allow_blank': True}
        }
    
    def create(self, validated_data):
        # Set the user to the current authenticated user
        user = self.context['request'].user
        if not user or not user.is_authenticated:
            raise serializers.ValidationError('Authentication required to submit a review.')
        validated_data['user'] = user
        return super().create(validated_data)
    
    def get_created_at_formatted(self, obj):
        try:
            return obj.created_at.strftime("%B %d, %Y")
        except:
            return "Unknown date"