# Model Module
# -----------

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from .preprocessing import get_preprocessor, transform_features
from .config import (
    TARGET_COLUMN,
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    DROP_COLUMNS,
    TEST_SIZE,
    RANDOM_STATE,
)


def get_models():
    """Return a dictionary of model pipelines.
    
    Each pipeline consists of:
    - Preprocessor (ColumnTransformer with StandardScaler + OneHotEncoder)
    - Classifier
    
    Returns:
        Dictionary mapping model names to sklearn Pipelines.
    """
    preprocessor = get_preprocessor()
    
    models = {
        "Logistic Regression": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "KNN": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    DecisionTreeClassifier(
                        random_state=RANDOM_STATE,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    XGBClassifier(
                        random_state=RANDOM_STATE,
                        eval_metric="logloss",
                        n_estimators=400,
                        max_depth=8,
                        learning_rate=0.08,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }
    
    return models


def split_data(X, y, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    """Split data into train and test sets with stratification.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        test_size: Proportion of test data
        random_state: Random seed
        
    Returns:
        (X_train, X_test, y_train, y_test) tuple
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    
    return X_train, X_test, y_train, y_test


def train_models(models, X_train, y_train):
    """Train all models.
    
    Args:
        models: Dictionary of model pipelines
        X_train: Training features
        y_train: Training target
        
    Returns:
        Dictionary of trained models
    """
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
    
    return trained


def evaluate_model(model, X_test, y_test, target_names=None):
    """Evaluate a model on test data.
    
    Args:
        model: Trained pipeline
        X_test: Test features
        y_test: Test target
        target_names: List of class name labels
        
    Returns:
        Dictionary with accuracy, precision, recall, f1-score
    """
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    
    report = classification_report(
        y_test,
        y_pred,
        target_names=target_names or ["Not Canceled", "Canceled"],
        output_dict=True,
    )
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "classification_report": classification_report(
            y_test, y_pred, target_names=target_names or ["Not Canceled", "Canceled"]
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def cross_validate_models(models, X_train, y_train, n_splits=5):
    """Perform cross-validation on all models.
    
    Args:
        models: Dictionary of model pipelines
        X_train: Training features
        y_train: Training target
        n_splits: Number of folds for CV
        
    Returns:
        Dictionary mapping model names to mean F1 scores
    """
    cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
    )
    
    cv_results = {}
    for name, model in models.items():
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="f1",
            n_jobs=-1,
        )
        cv_results[name] = {
            "mean_f1": scores.mean(),
            "scores": scores,
        }
    
    return cv_results


def train_full_pipeline(X, y, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    """Complete training workflow: split, train, evaluate, and cross-validate.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        test_size: Proportion of test data
        random_state: Random seed
        
    Returns:
        Dictionary with all training results
    """
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)
    
    # Get and train models
    models = get_models()
    trained_models = train_models(models, X_train, y_train)
    
    # Evaluate each model
    results = {}
    for name, model in trained_models.items():
        results[name] = evaluate_model(model, X_test, y_test)
    
    # Cross-validate
    cv_results = cross_validate_models(models, X_train, y_train)
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "models": trained_models,
        "results": results,
        "cv_results": cv_results,
    }