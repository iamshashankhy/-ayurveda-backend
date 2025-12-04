"""
Preprocessing utilities for ML model inputs.
Converts user input data into the format expected by trained models.
"""
import numpy as np
from typing import Dict, Tuple, Any

from .model_loader import load_feature_names, get_scaler, get_feature_encoders

DOSHA_THEMES = {
    'vata': {'color': '#60A5FA', 'gradient': ['#DBEAFE', '#3B82F6']},
    'pitta': {'color': '#F97316', 'gradient': ['#FED7AA', '#EA580C']},
    'kapha': {'color': '#22C55E', 'gradient': ['#BBF7D0', '#16A34A']},
    'balanced': {'color': '#0EA5E9', 'gradient': ['#BAE6FD', '#0284C7']}
}

RISK_THEMES = {
    'low': {'color': '#22C55E', 'gradient': ['#BBF7D0', '#16A34A']},
    'moderate': {'color': '#F59E0B', 'gradient': ['#FDE68A', '#D97706']},
    'high': {'color': '#EF4444', 'gradient': ['#FECACA', '#B91C1C']},
    'unknown': {'color': '#6B7280', 'gradient': ['#E5E7EB', '#4B5563']}
}


def _extract_raw_responses(user_data: Any) -> Dict[str, Any]:
    """
    Normalize incoming payload into a flat dictionary keyed by feature name.
    Supports payloads like {"responses": {...}}, {"answers": {...}}, or direct dictionaries.
    """
    if isinstance(user_data, dict):
        if 'responses' in user_data and isinstance(user_data['responses'], dict):
            return dict(user_data['responses'])
        if 'answers' in user_data and isinstance(user_data['answers'], dict):
            return dict(user_data['answers'])
        return dict(user_data)
    return {}


def encode_responses(user_data: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convert raw user responses into the encoded feature matrix expected by the models.
    Returns the encoded numeric array and the normalized (string) responses.
    """
    feature_names = load_feature_names()
    encoders = get_feature_encoders() or {}
    raw = _extract_raw_responses(user_data)

    normalized = {}
    encoded_values = []

    for feature in feature_names:
        value = raw.get(feature)
        encoder = encoders.get(feature)

        # Handle missing values - use a default value
        if value is None or value == '':
            # For categorical features, use the first class as default
            if encoder and len(encoder.classes_) > 0:
                value = encoder.classes_[0]
            else:
                # For numeric features or when no encoder, use 0 as default
                value = 0

        if encoder:
            classes = [str(cls) for cls in encoder.classes_]
            if str(value) not in classes:
                # Use first class as fallback for unseen values
                safe_value = classes[0]
            else:
                safe_value = str(value)
            normalized[feature] = safe_value
            encoded_value = int(encoder.transform([safe_value])[0])
        else:
            try:
                encoded_value = float(value)
                normalized[feature] = float(value)
            except Exception:
                # For non-numeric values without encoder, use 0
                encoded_value = 0.0
                normalized[feature] = str(value)

        encoded_values.append(encoded_value)

    encoded_array = np.array(encoded_values, dtype=float).reshape(1, -1)
    return encoded_array, normalized


def preprocess_dosha_input(user_data: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply scaling for dosha model and return (features, normalized_responses)."""
    encoded, normalized = encode_responses(user_data)
    scaler = get_scaler('dosha')
    features = scaler.transform(encoded) if scaler else encoded
    return features, normalized


def preprocess_cancer_input(user_data: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply scaling for cancer model and return (features, normalized_responses)."""
    encoded, normalized = encode_responses(user_data)
    scaler = get_scaler('cancer')
    features = scaler.transform(encoded) if scaler else encoded
    return features, normalized


def postprocess_dosha_prediction(prediction, probabilities=None, label_encoder=None):
    """
    Convert model prediction to user-friendly dosha results.
    Returns dosha percentages, dominant dosha, advice, and theme colors.
    """
    if label_encoder:
        if isinstance(prediction, (list, np.ndarray)):
            prediction = prediction[0]
        try:
            label = label_encoder.inverse_transform([prediction])[0]
        except Exception:
            label = str(prediction)
    else:
        label = str(prediction)

    dominant_key = label.lower()

    if probabilities is not None and len(probabilities[0]) >= 3:
        vata_pct = float(probabilities[0][0] * 100)
        pitta_pct = float(probabilities[0][1] * 100)
        kapha_pct = float(probabilities[0][2] * 100)
    else:
        # If no probabilities, distribute evenly or use default values
        vata_pct = pitta_pct = kapha_pct = 33.33

    advice_map = {
        'vata': "Favor warm, moist, grounding foods. Maintain regular routines. Practice calming yoga and meditation.",
        'pitta': "Prefer cooling, light foods. Avoid excess heat and spicy meals. Schedule calming breaks throughout the day.",
        'kapha': "Choose light, warm, and stimulating foods. Stay active with daily movement. Limit heavy or oily meals."
    }

    advice = advice_map.get(dominant_key, "Maintain balanced routines with mindful diet, movement, and rest.")
    theme = DOSHA_THEMES.get(dominant_key, DOSHA_THEMES['balanced'])

    return {
        'vata': round(vata_pct, 2),
        'pitta': round(pitta_pct, 2),
        'kapha': round(kapha_pct, 2),
        'dominant': label.capitalize(),
        'advice': advice,
        'colors': theme
    }


def postprocess_cancer_prediction(prediction, probability=None, label_encoder=None):
    """
    Convert model prediction to user-friendly cancer risk results.
    Returns probability (0-1), percentage, risk level, recommendations, and theme colors.
    """
    if label_encoder:
        if isinstance(prediction, (list, np.ndarray)):
            prediction = prediction[0]
        try:
            risk_label = label_encoder.inverse_transform([prediction])[0]
        except Exception:
            risk_label = str(prediction)
    else:
        risk_label = str(prediction)

    if probability is not None:
        if isinstance(probability, (list, np.ndarray)):
            if len(probability[0]) > 1:
                risk_prob = float(probability[0][1])
            else:
                risk_prob = float(probability[0][0])
        else:
            risk_prob = float(probability)
    else:
        # Default probability if not provided
        risk_prob = 0.5

    risk_prob = max(0.0, min(1.0, risk_prob))

    if risk_prob < 0.33:
        risk_level = 'low'
    elif risk_prob < 0.66:
        risk_level = 'moderate'
    else:
        risk_level = 'high'

    if label_encoder is None:
        risk_label = risk_level.capitalize()

    # Cancer status comments based on risk level
    cancer_status_comments = {
        'low': 'Free from cancer risk. Continue maintaining your healthy lifestyle.',
        'moderate': 'Slightly elevated cancer risk detected. Consider preventive measures and regular check-ups.',
        'high': 'Elevated cancer risk detected. Consult with a healthcare professional for further evaluation.'
    }

    recommendations = {
        'low': "Continue your current lifestyle habits and schedule periodic health check-ups.",
        'moderate': "Adopt supportive lifestyle changes, monitor key health indicators, and consult professionals when needed.",
        'high': "Schedule a comprehensive medical evaluation promptly and follow specialist guidance closely."
    }

    theme = RISK_THEMES.get(risk_level, RISK_THEMES['unknown'])

    return {
        'risk_level': risk_level,
        'label': risk_label,
        'probability': round(risk_prob, 4),
        'percentage': round(risk_prob * 100, 2),
        'colors': theme,
        'cancer_status_comment': cancer_status_comments.get(risk_level, "Please consult with a qualified healthcare professional."),
        'recommendations': recommendations.get(risk_level, "Please consult with a qualified healthcare professional.")
    }