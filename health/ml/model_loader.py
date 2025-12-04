import os
import joblib
import json
from functools import lru_cache

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, 'models')
SCALERS_DIR = os.path.join(BASE_DIR, 'scalers')

# Cache for loaded models
_models_cache = None
_scalers_cache = None
_feature_names = None


def load_feature_names():
    """Load feature names from JSON file."""
    global _feature_names
    if _feature_names is None:
        feature_path = os.path.join(MODELS_DIR, 'feature_names.json')
        if os.path.exists(feature_path):
            with open(feature_path, 'r') as f:
                _feature_names = json.load(f)
        else:
            _feature_names = []
    return _feature_names


@lru_cache(maxsize=1)
def get_models():
    """Load all trained ML models from disk."""
    global _models_cache
    
    if _models_cache is not None:
        return _models_cache
    
    _models_cache = {
        'dosha': {},
        'cancer': {},
        'scalers': {},
        'label_encoders': {},
        'feature_encoders': {}
    }
    
    try:
        # Load dosha models
        dosha_rf_path = os.path.join(MODELS_DIR, 'dosha_rf_model.joblib')
        dosha_svm_path = os.path.join(MODELS_DIR, 'dosha_svm_model.joblib')
        dosha_lr_path = os.path.join(MODELS_DIR, 'dosha_lr_model.joblib')
        
        if os.path.exists(dosha_rf_path):
            _models_cache['dosha']['rf'] = joblib.load(dosha_rf_path)
        if os.path.exists(dosha_svm_path):
            _models_cache['dosha']['svm'] = joblib.load(dosha_svm_path)
        if os.path.exists(dosha_lr_path):
            _models_cache['dosha']['lr'] = joblib.load(dosha_lr_path)
        
        # Load cancer models
        cancer_rf_path = os.path.join(MODELS_DIR, 'cancer_rf_model.joblib')
        cancer_svm_path = os.path.join(MODELS_DIR, 'cancer_svm_model.joblib')
        cancer_lr_path = os.path.join(MODELS_DIR, 'cancer_lr_model.joblib')
        
        if os.path.exists(cancer_rf_path):
            _models_cache['cancer']['rf'] = joblib.load(cancer_rf_path)
        if os.path.exists(cancer_svm_path):
            _models_cache['cancer']['svm'] = joblib.load(cancer_svm_path)
        if os.path.exists(cancer_lr_path):
            _models_cache['cancer']['lr'] = joblib.load(cancer_lr_path)
        
        # Load scalers
        dosha_scaler_path = os.path.join(SCALERS_DIR, 'dosha_scaler.joblib')
        cancer_scaler_path = os.path.join(SCALERS_DIR, 'cancer_scaler.joblib')
        
        if os.path.exists(dosha_scaler_path):
            _models_cache['scalers']['dosha'] = joblib.load(dosha_scaler_path)
        if os.path.exists(cancer_scaler_path):
            _models_cache['scalers']['cancer'] = joblib.load(cancer_scaler_path)
        
        # Load label encoders
        dosha_le_path = os.path.join(SCALERS_DIR, 'dosha_label_encoder.joblib')
        cancer_le_path = os.path.join(SCALERS_DIR, 'cancer_label_encoder.joblib')
        
        if os.path.exists(dosha_le_path):
            _models_cache['label_encoders']['dosha'] = joblib.load(dosha_le_path)
        if os.path.exists(cancer_le_path):
            _models_cache['label_encoders']['cancer'] = joblib.load(cancer_le_path)

        # Load feature encoders
        feature_encoders_path = os.path.join(SCALERS_DIR, 'feature_encoders.joblib')
        if os.path.exists(feature_encoders_path):
            _models_cache['feature_encoders'] = joblib.load(feature_encoders_path)
        
    except Exception as e:
        print(f"Error loading models: {e}")
        # Return partial cache if available
        if _models_cache is None:
            _models_cache = {}
    
    return _models_cache


def get_dosha_model(model_type='rf'):
    """Get a specific dosha classification model."""
    models = get_models()
    return models.get('dosha', {}).get(model_type)


def get_cancer_model(model_type='rf'):
    """Get a specific cancer risk prediction model."""
    models = get_models()
    return models.get('cancer', {}).get(model_type)


def get_scaler(model_type='dosha'):
    """Get feature scaler for dosha or cancer models."""
    models = get_models()
    return models.get('scalers', {}).get(model_type)


def get_label_encoder(model_type='dosha'):
    """Get label encoder for dosha or cancer models."""
    models = get_models()
    return models.get('label_encoders', {}).get(model_type)


def get_feature_encoders():
    """Get dictionary of feature label encoders."""
    models = get_models()
    return models.get('feature_encoders', {})


def clear_model_cache():
    """Clear the model cache to force reloading."""
    global _models_cache
    _models_cache = None
    get_models.cache_clear()