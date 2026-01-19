import numpy as np
import pandas as pd
from django.conf import settings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def load_and_clean_data():
    data_path = settings.BASE_DIR / 'data.csv'
    df = pd.read_csv(data_path)
    df_clean = df.copy()
    zero_as_missing_cols = [
        'Glucose',
        'BloodPressure',
        'SkinThickness',
        'Insulin',
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


model, scaler, feature_names = train_model()

