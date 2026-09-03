# Data Loading and Basic Cleaning
# -------------------------------

import pandas as pd
import numpy as np

from .config import (
    DATA_PATH,
    CATEGORICAL_COLUMNS_RAW,
    NUMERICAL_COLUMNS_RAW,
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    DROP_COLUMNS,
    NATIONALITY_TOP_COUNTRIES,
)


def load_data(path: str = None) -> pd.DataFrame:
    """Load the hotel bookings dataset.
    
    Args:
        path: Path to CSV file. Uses config DATA_PATH if None.
        
    Returns:
        Loaded DataFrame with raw data.
    """
    if path is None:
        path = DATA_PATH
    
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic cleaning steps to the dataset.
    
    Steps from the original notebook:
    - Remove duplicates
    - Fill missing values
    - Remove bookings with no guests
    - Remove negative ADR values
    - Group rare countries
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Cleaned DataFrame.
    """
    df = df.copy()
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Fill missing values
    df["country"] = df["country"].fillna("Unknown")
    df["children"] = df["children"].fillna(0)
    
    # Remove bookings with no guests (adults + children + babies == 0)
    no_guests = (df["adults"] + df["children"] + df["babies"]) == 0
    df = df[~no_guests]
    
    # Remove negative ADR values
    df = df[df["adr"] >= 0]
    
    # Group rare countries to 'Other'
    top_countries = df["country"].value_counts().head(NATIONALITY_TOP_COUNTRIES).index
    
    def group_rare_countries(country):
        if country in top_countries:
            return country
        else:
            return "Other"
    
    df["country"] = df["country"].apply(group_rare_countries)
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer new features from existing columns.
    
    Features from the original notebook:
    - total_guests: sum of adults + children + babies
    - total_stay_nights: sum of weekend + week nights
    - total_previous_bookings: sum of previous_cancellations + previous_bookings_not_canceled
    - is_family: 1 if children or babies > 0, else 0
    - adr_per_guest: adr / total_guests (with 0 total_guests replaced by 1)
    - has_parking: 1 if required_car_parking_spaces > 0, else 0
    - has_previous_cancellations: 1 if previous_cancellations > 0, else 0
    
    Args:
        df: DataFrame with raw columns
        
    Returns:
        DataFrame with engineered features added.
    """
    df = df.copy()
    
    # Engineered features
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df["total_stay_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_previous_bookings"] = (
        df["previous_cancellations"] + df["previous_bookings_not_canceled"]
    )
    df["is_family"] = ((df["children"] + df["babies"]) > 0).astype(int)
    df["adr_per_guest"] = df["adr"] / df["total_guests"].replace(0, 1)
    df["has_parking"] = (df["required_car_parking_spaces"] > 0).astype(int)
    df["has_previous_cancellations"] = (df["previous_cancellations"] > 0).astype(int)
    
    return df