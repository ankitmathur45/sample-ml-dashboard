import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def load_and_profile(df):
    """Returns basic profile of the dataframe"""
    profile = {
        "rows" : df.shape[0],
        "columns" : df.shape[1],
        "missing" : df.isnull().sum().sum(),
        "dtypes" : df.dtypes.astype(str).to_dict(),
        "numeric_cols" : df.select_dtypes(include=np.number).columns.tolist(),
        "categorical_cols" : df.select_dtypes(include="object").columns.tolist(),
    }
    return profile

def preprocess(df, target_col):
    """Encodes categoricals and splits into X, y"""
    df = df.copy()
    df = df.dropna()

    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop(columns=[target_col])
    y = df[target_col]

    return train_test_split(X,y, test_size=0.2, random_state=42)

def train_models(X_train, X_test, y_train, y_test):
    """Train models and return results"""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = {
            "model" : model,
            "accuracy" : round(accuracy_score(y_test,preds) *100,2),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
            "predictions": preds,
        }

    return results

def get_feature_importance(model, feature_names):
    """Returns feature importance if available."""
    if hasattr(model, "feature_importances_"):
        return pd.Series(
            model.feature_importances_, index=feature_names
        ).sort_values(ascending=False)
    elif hasattr(model, "coef_"):
        return pd.Series(
            np.abs(model.coef_[0]), index=feature_names
        ).sort_values(ascending=False)
    return None 