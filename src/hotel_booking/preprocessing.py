# Preprocessing Pipeline
# ---------------------

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS


def get_preprocessor():
    """Create and return the ColumnTransformer preprocessing pipeline.
    
    The preprocessing includes:
    - Numerical features: StandardScaler (zero mean, unit variance)
    - Categorical features: OneHotEncoder with handle_unknown='ignore'
      (handles unseen categories in production data)
    
    Returns:
        Configured ColumnTransformer ready for fit/transform.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_COLUMNS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLUMNS,
            ),
        ]
    )
    
    return preprocessor


def transform_features(preprocessor, X: pd.DataFrame) -> np.ndarray:
    """Transform features using the fitted preprocessor.
    
    Args:
        preprocessor: Fitted ColumnTransformer
        X: DataFrame with feature columns (should NOT include target)
        
    Returns:
        Transformed numpy array ready for model prediction.
    """
    return preprocessor.transform(X)