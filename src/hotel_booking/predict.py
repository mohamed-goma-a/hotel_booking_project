# Prediction Module
# ----------------

import joblib
import numpy as np
import pandas as pd

from .preprocessing import transform_features
from .config import MODEL_PATH, CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS


def load_model(path: str = None):
    """Load the trained model from disk.
    
    Args:
        path: Path to model file. Uses config MODEL_PATH if None.
        
    Returns:
        Loaded pipeline object (preprocessor + classifier).
    """
    if path is None:
        path = MODEL_PATH
    
    model = joblib.load(path)
    return model


def prepare_input(data: dict) -> pd.DataFrame:
    """Prepare input data for prediction.
    
    Converts a dictionary of feature values into a DataFrame
    with the correct column structure for the model.
    Includes both raw features and engineered features.
    
    Args:
        data: Dictionary with feature names as keys and values.
              Expected keys include numerical features, categorical features,
              and engineered features (total_guests, total_stay_nights,
              total_previous_bookings, is_family, adr_per_guest, has_parking,
              has_previous_cancellations).
              
    Returns:
        DataFrame ready for model transformation.
    """
    # All columns the preprocessor expects (from the training pipeline)
    all_expected_cols = (
        NUMERICAL_COLUMNS  # lead_time, adr, etc. + engineered numerical
        + CATEGORICAL_COLUMNS  # hotel, arrival_date_month, etc.
    )
    
    df = pd.DataFrame([data])
    
    # Ensure all expected columns are present, with defaults if missing
    for col in all_expected_cols:
        if col not in df.columns:
            # Set sensible defaults based on column type
            if col in ["total_guests", "total_stay_nights", "total_previous_bookings",
                       "is_family", "has_parking", "has_previous_cancellations"]:
                df[col] = 0
            elif col == "adr_per_guest":
                df[col] = 0.0
            elif col == "reserved_room_type":
                df[col] = "Resort Hotel"  # default category
            else:
                df[col] = "Unknown" if col in CATEGORICAL_COLUMNS else 0.0
    
    # Select only the columns the preprocessor expects (in the right order)
    X = df[NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS]
    
    return X


def predict(model, data: dict) -> dict:
    """Make a prediction using the loaded model.
    
    Args:
        model: Trained pipeline (preprocessor + classifier)
        data: Dictionary of feature values (see prepare_input for expected keys)
        
    Returns:
        Dictionary with prediction result:
        - "prediction": 0 (Not Canceled) or 1 (Canceled)
        - "probability": Probability of the predicted class
        - "probabilities": Full probability array [P(Not Canceled), P(Canceled)]
        - "raw_prediction": Raw model output (0 or 1)
    """
    # Prepare input
    X = prepare_input(data)
    
    # Transform features using the model's preprocessor
    X_transformed = transform_features(model.named_steps["preprocessor"], X)
    
    # Make prediction
    raw_prediction = model.named_steps["classifier"].predict(X_transformed)[0]
    
    # Get probabilities
    probabilities = model.named_steps["classifier"].predict_proba(X_transformed)[0]
    
    # Map to class labels
    # The classes are typically [0, 1] where 0 = Not Canceled, 1 = Canceled
    prediction_label = int(raw_prediction)
    
    # Return structured result
    return {
        "prediction": prediction_label,
        "prediction_label": "Canceled" if prediction_label == 1 else "Not Canceled",
        "probability": float(probabilities[prediction_label]),
        "probabilities": {
            "Not Canceled": float(probabilities[0]),
            "Canceled": float(probabilities[1]),
        },
        "raw_prediction": raw_prediction,
    }


def predict_single(data: dict, model_path: str = None) -> dict:
    """Convenience function: load model and make a single prediction.
    
    Args:
        data: Dictionary of feature values
        model_path: Path to model file. Uses config MODEL_PATH if None.
        
    Returns:
        Prediction dictionary (see predict() for format)
    """
    model = load_model(model_path)
    return predict(model, data)