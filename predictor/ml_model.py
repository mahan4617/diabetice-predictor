import os
import json
import numpy as np
import pandas as pd
from django.conf import settings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib


def get_base_dir():
    try:
        return settings.BASE_DIR
    except Exception:
        from pathlib import Path
        return Path(__file__).resolve().parent.parent


BASE_DIR = get_base_dir()
ARTIFACT_DIR = BASE_DIR / 'artifacts'
MODEL_PATH = ARTIFACT_DIR / 'model.joblib'
SCALER_PATH = ARTIFACT_DIR / 'scaler.joblib'
FEATURES_PATH = ARTIFACT_DIR / 'feature_names.json'


def load_and_clean_data():
    data_path = BASE_DIR / 'data.csv'
    df = pd.read_csv(data_path)
    df_clean = df.copy()
    zero_as_missing_cols = [
        'Glucose',
        'BloodPressure',
        'SkinThickness',
        'BMI',
    ]
    for col in zero_as_missing_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].replace(0, np.nan)
            median_value = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_value)
    return df_clean


def train_model():
    df = load_and_clean_data()
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(
        solver='liblinear',
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)
    feature_names = X.columns.tolist()
    return model, scaler, feature_names


def save_artifacts(model, scaler, feature_names):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(FEATURES_PATH, 'w', encoding='utf-8') as f:
        json.dump(feature_names, f)


def load_artifacts():
    if MODEL_PATH.exists() and SCALER_PATH.exists() and FEATURES_PATH.exists():
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        with open(FEATURES_PATH, 'r', encoding='utf-8') as f:
            feature_names = json.load(f)
        return model, scaler, feature_names
    return None


def get_model():
    loaded = load_artifacts()
    if loaded is not None:
        return loaded
    model, scaler, feature_names = train_model()
    save_artifacts(model, scaler, feature_names)
    return model, scaler, feature_names


model, scaler, feature_names = get_model()


if __name__ == '__main__':
    m, s, fn = get_model()
    print('Artifacts saved to:', ARTIFACT_DIR)
