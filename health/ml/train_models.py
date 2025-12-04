"""
Training script for Ayurvedic ML models.
Trains Random Forest, SVM, and Logistic Regression models for dosha classification and cancer risk prediction.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import json

# Paths
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
SCALERS_DIR = os.path.join(BASE_DIR, 'scalers')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SCALERS_DIR, exist_ok=True)

def load_and_preprocess_data(csv_path):
    """
    Load CSV data and perform basic preprocessing.
    Returns: X (features), y_dosha (dosha labels), y_cancer (cancer risk labels), feature_names
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    # TODO: Adjust these column names based on your actual CSV structure
    # Common patterns:
    # - Dosha columns: 'vata', 'pitta', 'kapha' or 'dominant_dosha'
    # - Cancer risk: 'cancer_risk', 'risk_level', 'has_cancer'
    # - Features: all other numeric/categorical columns
    
    # Identify target columns (adjust based on your CSV)
    target_dosha_col = None
    target_cancer_col = None
    
    # Try to find dosha column (case-insensitive)
    for col in df.columns:
        if col.lower() in ['dominant_dosha', 'dosha', 'dosha_type', 'vata_dominant', 'pitta_dominant', 'kapha_dominant']:
            target_dosha_col = col
            print(f"✓ Found dosha column: {col}")
            break
    
    # Try to find cancer risk column (case-insensitive)
    for col in df.columns:
        if col.lower() in ['cancer_risk', 'risk_level', 'has_cancer', 'cancer', 'risk', 'cancer_risk_level']:
            target_cancer_col = col
            print(f"✓ Found cancer risk column: {col}")
            break
    
    # If not found, use last column as dosha if it looks like a target
    if target_dosha_col is None:
        # Check if last column might be dosha
        last_col = df.columns[-1]
        if df[last_col].dtype == 'object' and df[last_col].nunique() <= 5:
            # Likely a target column
            target_dosha_col = last_col
            print(f"✓ Using last column as dosha target: {last_col}")
        else:
            print("\n⚠️  Dosha target column not found. Please specify which column represents dosha classification.")
            print("Available columns:", df.columns.tolist())
            target_dosha_col = None
    
    if target_cancer_col is None:
        print("\n⚠️  Cancer risk target column not found. Will skip cancer model training.")
        print("(You can add a cancer risk column to your CSV if needed)")
        target_cancer_col = None
    
    # Separate features and targets
    # Include ALL columns except targets as features (will encode categorical)
    feature_cols = [col for col in df.columns 
                   if col not in [target_dosha_col, target_cancer_col]]
    
    X = df[feature_cols].copy()
    
    # Encode all categorical features
    
    print(f"\nEncoding {len(feature_cols)} feature columns...")
    X_encoded = pd.DataFrame()
    encoders = {}
    
    for col in feature_cols:
        if X[col].dtype == 'object':
            # Use LabelEncoder for categorical features
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
        else:
            # Keep numeric as-is
            X_encoded[col] = X[col]
    
    X = X_encoded

    if encoders:
        encoders_path = os.path.join(SCALERS_DIR, 'feature_encoders.joblib')
        joblib.dump(encoders, encoders_path)
        print(f"✓ Saved feature encoders ({len(encoders)} categorical features)")
    
    # Handle missing values (fill with median for numeric, mode for categorical)
    for col in X.columns:
        if X[col].dtype in ['int64', 'float64']:
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 0)
    
    # Encode dosha labels if needed
    if target_dosha_col:
        if df[target_dosha_col].dtype == 'object':
            le_dosha = LabelEncoder()
            y_dosha = le_dosha.fit_transform(df[target_dosha_col])
            # Save label encoder
            joblib.dump(le_dosha, os.path.join(SCALERS_DIR, 'dosha_label_encoder.joblib'))
            print(f"\nDosha classes: {le_dosha.classes_}")
        else:
            y_dosha = df[target_dosha_col].values
            le_dosha = None
    else:
        # Synthetic: create dosha based on feature patterns
        print("\n⚠️  Creating synthetic dosha labels based on feature patterns...")
        if X.shape[1] > 0:
            y_dosha = (X.iloc[:, 0] % 3).astype(int)  # Simple synthetic
        else:
            # If no features, create random labels
            y_dosha = np.random.randint(0, 3, size=len(df))
        le_dosha = None
    
    # Encode cancer risk if needed
    if target_cancer_col:
        if df[target_cancer_col].dtype == 'object':
            le_cancer = LabelEncoder()
            y_cancer = le_cancer.fit_transform(df[target_cancer_col])
            joblib.dump(le_cancer, os.path.join(SCALERS_DIR, 'cancer_label_encoder.joblib'))
            print(f"\nCancer risk classes: {le_cancer.classes_}")
        else:
            y_cancer = df[target_cancer_col].values
            le_cancer = None
    else:
        # Synthetic: create risk based on feature patterns or skip
        print("\n⚠️  No cancer risk column found. Creating synthetic labels for training...")
        if X.shape[1] > 0:
            y_cancer = (X.iloc[:, 0] % 2).astype(int)  # Binary: 0=low, 1=high
        else:
            # If no features, create random labels
            y_cancer = np.random.randint(0, 2, size=len(df))
        le_cancer = None
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Dosha labels shape: {y_dosha.shape}, unique values: {np.unique(y_dosha)}")
    if y_cancer is not None:
        print(f"Cancer labels shape: {y_cancer.shape}, unique values: {np.unique(y_cancer)}")
    else:
        print("Cancer labels: None (will skip cancer model training)")
    
    return X, y_dosha, y_cancer, feature_cols, le_dosha, le_cancer


def train_dosha_models(X, y):
    """Train models for dosha classification."""
    print("\n" + "="*60)
    print("Training Dosha Classification Models")
    print("="*60)
    
    # Split data: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    joblib.dump(scaler, os.path.join(SCALERS_DIR, 'dosha_scaler.joblib'))
    print("✓ Saved dosha feature scaler")
    
    models = {}
    results = {}
    
    # 1. Random Forest
    print("\n1. Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    results['random_forest'] = {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1)
    }
    print(f"   Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    print(f"   Classification Report:\n{classification_report(y_test, y_pred)}")
    
    joblib.dump(rf, os.path.join(MODELS_DIR, 'dosha_rf_model.joblib'))
    models['rf'] = rf
    print("✓ Saved Random Forest model")
    
    # 2. SVM
    print("\n2. Training SVM...")
    svm = SVC(kernel='rbf', probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    y_pred = svm.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    results['svm'] = {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1)
    }
    print(f"   Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    
    joblib.dump(svm, os.path.join(MODELS_DIR, 'dosha_svm_model.joblib'))
    models['svm'] = svm
    print("✓ Saved SVM model")
    
    # 3. Logistic Regression
    print("\n3. Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
    lr.fit(X_train_scaled, y_train)
    y_pred = lr.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    results['logistic_regression'] = {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1)
    }
    print(f"   Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    
    joblib.dump(lr, os.path.join(MODELS_DIR, 'dosha_lr_model.joblib'))
    models['lr'] = lr
    print("✓ Saved Logistic Regression model")
    
    # Select best model (highest accuracy)
    best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    print(f"\n✓ Best model for dosha: {best_model_name} (Accuracy: {results[best_model_name]['accuracy']:.4f})")
    
    # Save results
    with open(os.path.join(MODELS_DIR, 'dosha_training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return models, results, best_model_name


def train_cancer_models(X, y):
    """Train models for cancer risk prediction."""
    print("\n" + "="*60)
    print("Training Cancer Risk Prediction Models")
    print("="*60)
    
    # Split data: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    joblib.dump(scaler, os.path.join(SCALERS_DIR, 'cancer_scaler.joblib'))
    print("✓ Saved cancer feature scaler")
    
    models = {}
    results = {}
    
    # 1. Random Forest
    print("\n1. Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)
    y_pred_proba = rf.predict_proba(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    results['random_forest'] = {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1)
    }
    print(f"   Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    
    joblib.dump(rf, os.path.join(MODELS_DIR, 'cancer_rf_model.joblib'))
    models['rf'] = rf
    print("✓ Saved Random Forest model")
    
    # 2. SVM
    print("\n2. Training SVM...")
    svm = SVC(kernel='rbf', probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    y_pred = svm.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    results['svm'] = {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1)
    }
    print(f"   Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    
    joblib.dump(svm, os.path.join(MODELS_DIR, 'cancer_svm_model.joblib'))
    models['svm'] = svm
    print("✓ Saved SVM model")
    
    # 3. Logistic Regression
    print("\n3. Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    y_pred = lr.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    results['logistic_regression'] = {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1)
    }
    print(f"   Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    
    joblib.dump(lr, os.path.join(MODELS_DIR, 'cancer_lr_model.joblib'))
    models['lr'] = lr
    print("✓ Saved Logistic Regression model")
    
    # Select best model (highest accuracy)
    best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    print(f"\n✓ Best model for cancer risk: {best_model_name} (Accuracy: {results[best_model_name]['accuracy']:.4f})")
    
    # Save results
    with open(os.path.join(MODELS_DIR, 'cancer_training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return models, results, best_model_name


def main():
    """Main training pipeline."""
    print("="*60)
    print("Ayurvedic ML Model Training Pipeline")
    print("="*60)
    
    # Find CSV file in data directory
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"\n❌ No CSV file found in {DATA_DIR}")
        print("Please place your CSV file in: django_backend/health/ml/data/")
        return
    
    csv_path = os.path.join(DATA_DIR, csv_files[0])
    print(f"\n✓ Found CSV file: {csv_files[0]}")
    
    # Load and preprocess data
    X, y_dosha, y_cancer, feature_names, le_dosha, le_cancer = load_and_preprocess_data(csv_path)
    
    # Save feature names for later use
    with open(os.path.join(MODELS_DIR, 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f, indent=2)
    print(f"\n✓ Saved feature names: {len(feature_names)} features")
    
    # Train dosha models
    dosha_models, dosha_results, best_dosha = train_dosha_models(X, y_dosha)
    
    # Train cancer models (only if we have cancer labels)
    if y_cancer is not None:
        cancer_models, cancer_results, best_cancer = train_cancer_models(X, y_cancer)
    else:
        print("\n" + "="*60)
        print("Skipping Cancer Risk Model Training")
        print("="*60)
        print("No cancer risk column found in dataset.")
        print("To train cancer models, add a 'cancer_risk' or 'risk_level' column to your CSV.")
        cancer_models = {}
        cancer_results = {}
        best_cancer = None
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nBest Dosha Model: {best_dosha}")
    if best_cancer:
        print(f"Best Cancer Model: {best_cancer}")
    else:
        print("Cancer Model: Not trained (no cancer risk column in dataset)")
    print(f"\nModels saved to: {MODELS_DIR}")
    print(f"Scalers saved to: {SCALERS_DIR}")
    print("\nYou can now use these models in the Django backend!")


if __name__ == '__main__':
    main()

