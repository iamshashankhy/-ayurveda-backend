from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from django.db.models import Avg

from .ml.model_loader import (
    get_dosha_model,
    get_cancer_model,
    get_label_encoder,
    load_feature_names,
    get_feature_encoders,
    get_scaler,
)
from .ml.preprocessing import (
    encode_responses,
    postprocess_dosha_prediction,
    postprocess_cancer_prediction,
)
from .ml.feature_metadata import get_feature_metadata
from .models import DoshaEvaluation, CancerRiskResult, ActivityLog, Review
from accounts.models import Profile
from .serializers import ReviewSerializer

import numpy as np


def _prepare_feature_matrices(payload):
    """Encode responses once and create scaled feature matrices for each model."""
    encoded, normalized = encode_responses(payload)
    dosha_scaler = get_scaler('dosha')
    cancer_scaler = get_scaler('cancer')

    dosha_features = dosha_scaler.transform(encoded) if dosha_scaler else encoded
    cancer_features = cancer_scaler.transform(encoded) if cancer_scaler else encoded

    return encoded, normalized, dosha_features, cancer_features


def _top_features_for_model(model, normalized_inputs, top_n=3):
    """Return the most important features for a given tree-based model."""
    if model is None or not hasattr(model, "feature_importances_"):
        return []
    feature_names = load_feature_names()
    importances = getattr(model, "feature_importances_", None)
    if importances is None or len(importances) != len(feature_names):
        return []

    indices = np.argsort(importances)[::-1][:top_n]
    highlights = []

    for rank, idx in enumerate(indices, start=1):
        feature = feature_names[idx]
        meta = get_feature_metadata(feature)
        highlights.append({
            "rank": rank,
            "feature": feature,
            "question": meta.get("question", feature),
            "description": meta.get("description", ""),
            "examples": meta.get("examples", []),
            "value": normalized_inputs.get(feature),
            "importance": round(float(importances[idx]) * 100, 2),
        })

    return highlights


class AnalyzeDoshaView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        payload = request.data or {}
        try:
            _, normalized, dosha_features, _ = _prepare_feature_matrices(payload)
            model = get_dosha_model('rf') or get_dosha_model('svm') or get_dosha_model('lr')
            label_encoder = get_label_encoder('dosha')
            if model:
                prediction = model.predict(dosha_features)
                probabilities = None
                if hasattr(model, 'predict_proba'):
                    try:
                        probabilities = model.predict_proba(dosha_features)
                    except Exception:
                        probabilities = None

                dosha_result = postprocess_dosha_prediction(
                    prediction,
                    probabilities=probabilities,
                    label_encoder=label_encoder
                )
            else:
                dosha_result = {
                    'vata': 33.33,
                    'pitta': 33.33,
                    'kapha': 33.33,
                    'dominant': 'Balanced',
                    'advice': 'Maintain balanced routines with mindful diet, movement, and rest.',
                    'colors': {'color': '#0EA5E9', 'gradient': ['#BAE6FD', '#0284C7']}
                }

            rf_model = get_dosha_model('rf') or model
            top_features = _top_features_for_model(rf_model, normalized)

            DoshaEvaluation.objects.create(
                user=request.user,
                vata=dosha_result['vata'],
                pitta=dosha_result['pitta'],
                kapha=dosha_result['kapha'],
                advice=dosha_result.get('advice', '')
            )

            ActivityLog.objects.create(
                user=request.user,
                action='analyze_dosha',
                meta={'responses': normalized, 'dosha': dosha_result}
            )

            return Response({
                'dosha': dosha_result,
                'top_features': top_features,
                'inputs': normalized
            })
        except Exception as exc:
            return Response(
                {'error': 'Unable to analyze dosha at this time.', 'details': str(exc)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class PredictCancerView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        payload = request.data or {}
        try:
            _, normalized, _, cancer_features = _prepare_feature_matrices(payload)
            model = get_cancer_model('rf') or get_cancer_model('svm') or get_cancer_model('lr')
            label_encoder = get_label_encoder('cancer')
            if model:
                prediction = model.predict(cancer_features)
                probability = None
                if hasattr(model, 'predict_proba'):
                    try:
                        probability = model.predict_proba(cancer_features)
                    except Exception:
                        probability = None

                cancer_result = postprocess_cancer_prediction(
                    prediction,
                    probability=probability,
                    label_encoder=label_encoder
                )
            else:
                cancer_result = {
                    'risk_level': 'unknown',
                    'label': 'Unknown',
                    'probability': 0.5,
                    'percentage': 50.0,
                    'colors': {'color': '#6B7280', 'gradient': ['#E5E7EB', '#4B5563']},
                    'cancer_status_comment': 'Please consult with a qualified healthcare professional.',
                    'recommendations': 'Please consult with a qualified healthcare professional.'
                }

            rf_model = get_cancer_model('rf') or model
            top_features = _top_features_for_model(rf_model, normalized)

            CancerRiskResult.objects.create(
                user=request.user,
                probability=cancer_result['probability'],
                risk=cancer_result['risk_level']
            )

            ActivityLog.objects.create(
                user=request.user,
                action='predict_cancer',
                meta={'responses': normalized, 'cancer': cancer_result}
            )

            return Response({
                'cancer': cancer_result,
                'top_features': top_features,
                'inputs': normalized
            })
        except Exception as exc:
            return Response(
                {'error': 'Unable to calculate cancer risk at this time.', 'details': str(exc)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class CombinedAssessmentView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        payload = request.data or {}

        try:
            encoded, normalized, dosha_features, cancer_features = _prepare_feature_matrices(payload)

            # Dosha assessment
            dosha_model = get_dosha_model('rf') or get_dosha_model('svm') or get_dosha_model('lr')
            dosha_label_encoder = get_label_encoder('dosha')
            dosha_probabilities = None
            if dosha_model and hasattr(dosha_model, 'predict_proba'):
                try:
                    dosha_probabilities = dosha_model.predict_proba(dosha_features)
                except Exception:
                    dosha_probabilities = None

            if dosha_model:
                dosha_prediction = dosha_model.predict(dosha_features)
                dosha_result = postprocess_dosha_prediction(
                    dosha_prediction,
                    probabilities=dosha_probabilities,
                    label_encoder=dosha_label_encoder
                )
            else:
                dosha_result = {
                    'vata': 33.33,
                    'pitta': 33.33,
                    'kapha': 33.33,
                    'dominant': 'Balanced',
                    'advice': 'Maintain balanced routines with mindful diet, movement, and rest.',
                    'colors': {'color': '#0EA5E9', 'gradient': ['#BAE6FD', '#0284C7']}
                }

            # Cancer risk assessment
            cancer_model = get_cancer_model('rf') or get_cancer_model('svm') or get_cancer_model('lr')
            cancer_label_encoder = get_label_encoder('cancer')
            cancer_probability = None
            if cancer_model and hasattr(cancer_model, 'predict_proba'):
                try:
                    cancer_probability = cancer_model.predict_proba(cancer_features)
                except Exception:
                    cancer_probability = None

            if cancer_model:
                cancer_prediction = cancer_model.predict(cancer_features)
                cancer_result = postprocess_cancer_prediction(
                    cancer_prediction,
                    probability=cancer_probability,
                    label_encoder=cancer_label_encoder
                )
            else:
                cancer_result = {
                    'risk_level': 'unknown',
                    'label': 'Unknown',
                    'probability': 0.5,
                    'percentage': 50.0,
                    'colors': {'color': '#6B7280', 'gradient': ['#E5E7EB', '#4B5563']},
                    'cancer_status_comment': 'Please consult with a qualified healthcare professional.',
                    'recommendations': 'Please consult with a qualified healthcare professional.'
                }

            # Top features
            dosha_rf = get_dosha_model('rf') or dosha_model
            cancer_rf = get_cancer_model('rf') or cancer_model
            dosha_top = _top_features_for_model(dosha_rf, normalized)
            cancer_top = _top_features_for_model(cancer_rf, normalized)

            # Persist results
            DoshaEvaluation.objects.create(
                user=request.user,
                vata=dosha_result['vata'],
                pitta=dosha_result['pitta'],
                kapha=dosha_result['kapha'],
                advice=dosha_result.get('advice', '')
            )
            CancerRiskResult.objects.create(
                user=request.user,
                probability=cancer_result['probability'],
                risk=cancer_result['risk_level']
            )

            ActivityLog.objects.create(
                user=request.user,
                action='combined_assessment',
                meta={
                    'responses': normalized,
                    'dosha': dosha_result,
                    'cancer': cancer_result
                }
            )

            return Response({
                'dosha': {
                    **dosha_result,
                    'top_features': dosha_top
                },
                'cancer': {
                    **cancer_result,
                    'top_features': cancer_top
                },
                'inputs': normalized
            })
        except Exception as exc:
            return Response(
                {'error': 'Unable to complete assessment at this time.', 'details': str(exc)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class AssessmentQuestionsView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        # Get user's language preference from Accept-Language header or profile
        accept_language = request.META.get('HTTP_ACCEPT_LANGUAGE', 'english')
        if accept_language.startswith('kn') or accept_language == 'kannada':
            user_language = 'kannada'
        else:
            user_language = 'english'
        
        # Fallback to profile preference if available
        try:
            profile, _ = Profile.objects.get_or_create(user=request.user)
            profile_language = profile.preferences.get('language', 'english')
            if profile_language in ['kannada', 'english']:
                user_language = profile_language
        except:
            pass
        
        feature_names = load_feature_names()
        encoders = get_feature_encoders() or {}
        questions = []

        for feature in feature_names:
            meta = get_feature_metadata(feature)
            options = []
            
            # Get options from encoder if available
            encoder = encoders.get(feature)
            if encoder:
                options = [str(cls) for cls in encoder.classes_]
            else:
                # Fallback to examples from metadata
                options = meta.get('examples', [])
            
            questions.append({
                'field': feature,
                'question': meta.get('question' if user_language != 'kannada' else 'question_kn', feature),
                'description': meta.get('description' if user_language != 'kannada' else 'description_kn', ''),
                'examples': meta.get('examples' if user_language != 'kannada' else 'examples_kn', []),
                'options': options
            })

        return Response({'questions': questions})


class ReviewsView(APIView):
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        """Get all reviews with average rating"""
        try:
            # Limit to 50 most recent reviews for performance
            reviews = Review.objects.select_related('user').all()[:50]
            serializer = ReviewSerializer(reviews, many=True)
            
            # Calculate average rating
            avg_rating_result = Review.objects.aggregate(Avg('rating'))
            avg_rating = avg_rating_result.get('rating__avg', 0) if avg_rating_result else 0
            
            # Ensure average rating is between 1 and 5
            clamped_avg_rating = max(1, min(5, avg_rating)) if avg_rating is not None else 0
            
            return Response({
                'reviews': serializer.data,
                'average_rating': round(clamped_avg_rating, 1),
                'total_reviews': Review.objects.count()
            })
        except Exception as e:
            # Log the error for debugging
            print(f"Error fetching reviews: {str(e)}")
            # Return empty data instead of failing
            return Response({
                'reviews': [],
                'average_rating': 0,
                'total_reviews': 0
            }, status=status.HTTP_200_OK)
    
    def post(self, request):
        """Create a new review"""
        try:
            rating = request.data.get('rating')
            comment = request.data.get('comment', '')
            
            # Validate rating is between 1 and 5
            if not isinstance(rating, int) or rating < 1 or rating > 5:
                return Response(
                    {'error': 'Rating must be an integer between 1 and 5'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Create a new review for the user
            serializer = ReviewSerializer(data={
                'rating': rating,
                'comment': comment
            }, context={'request': request})
            
            if serializer.is_valid():
                serializer.save()
                # Refresh the data to include computed fields
                serializer = ReviewSerializer(serializer.instance)
                return Response(serializer.data, status=status.HTTP_201_CREATED)
            else:
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            # Log the error for debugging
            print(f"Error creating review: {str(e)}")
            return Response(
                {'error': 'Failed to create review. Please try again.'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ProfileView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        profile, _ = Profile.objects.get_or_create(user=request.user)
        history = {
            'dosha_evaluations': list(
                request.user.dosha_evaluations.order_by('-created_at').values(
                    'vata', 'pitta', 'kapha', 'advice', 'created_at'
                )[:10]
            ),
            'cancer_risk_results': list(
                request.user.cancer_risk_results.order_by('-created_at').values(
                    'probability', 'risk', 'created_at'
                )[:10]
            ),
        }
        return Response({
            'user_id': request.user.id,
            'preferences': profile.preferences,
            'history': history
        })

    def put(self, request):
        profile, _ = Profile.objects.get_or_create(user=request.user)
        prefs = request.data.get('preferences', {})
        if isinstance(prefs, dict):
            profile.preferences = prefs
            profile.save()
            
            # Also update language in localStorage if provided
            language = prefs.get('language')
            if language:
                # This will be handled on the frontend side
                pass
                
        ActivityLog.objects.create(
            user=request.user,
            action='update_profile',
            meta={'preferences': prefs}
        )
        return Response({'preferences': profile.preferences}, status=status.HTTP_200_OK)