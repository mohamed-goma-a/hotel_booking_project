# FastAPI Application
# -------------------

import sys
import os
from pathlib import Path

# Add project root to path so we can import src.hotel_booking
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

from src.hotel_booking.predict import load_model, predict_single
from src.hotel_booking.config import MODEL_PATH

# Load model at startup
try:
    model = load_model(MODEL_PATH)
    model_loaded = True
except Exception as e:
    model = None
    model_loaded = False
    print(f"Warning: Could not load model at startup: {e}")

# Initialize FastAPI app
app = FastAPI(
    title="Hotel Booking Cancellation Predictor",
    description="API for predicting hotel booking cancellations based on booking features",
    version="1.0.0",
)


# Pydantic models for request/response validation

class NumericalFeatures(BaseModel):
    """Numerical feature inputs from the hotel booking dataset."""
    lead_time: float = Field(..., ge=0, description="Lead time in days")
    arrival_date_year: int = Field(..., ge=2015, le=2018, description="Arrival year")
    arrival_date_week_number: int = Field(..., ge=1, le=53, description="Week number")
    arrival_date_day_of_month: int = Field(..., ge=1, le=31, description="Day of month")
    stays_in_weekend_nights: float = Field(..., ge=0, description="Weekend night stays")
    stays_in_week_nights: float = Field(..., ge=0, description="Week night stays")
    adults: int = Field(..., ge=0, description="Number of adults")
    children: float = Field(default=0, ge=0, description="Number of children")
    babies: float = Field(ge=0, description="Number of babies")
    is_repeated_guest: int = Field(..., ge=0, le=1, description="Repeated guest flag")
    previous_cancellations: int = Field(..., ge=0, description="Previous cancellations")
    previous_bookings_not_canceled: int = Field(..., ge=0, description="Not canceled bookings")
    booking_changes: int = Field(..., ge=0, description="Booking changes count")
    days_in_waiting_list: int = Field(ge=0, description="Waiting list days")
    adr: float = Field(..., ge=0, description="Average daily rate")
    required_car_parking_spaces: int = Field(ge=0, description="Parking spaces required")
    total_of_special_requests: int = Field(ge=0, description="Special requests count")
    total_guests: float = Field(..., ge=0, description="Total number of guests")
    total_stay_nights: float = Field(..., ge=0, description="Total stay nights")
    total_previous_bookings: int = Field(ge=0, description="Total previous bookings")
    is_family: int = Field(..., ge=0, le=1, description="Is family flag")
    adr_per_guest: float = Field(..., ge=0, description="ADR per guest")


class CategoricalFeatures(BaseModel):
    """Categorical feature inputs from the hotel booking dataset."""
    hotel: str = Field(..., description="Hotel type (City Hotel or Resort Hotel)")
    arrival_date_month: str = Field(..., description="Arrival month")
    meal: str = Field(..., description="Meal plan")
    country: str = Field(..., description="Customer country")
    market_segment: str = Field(..., description="Market segment")
    distribution_channel: str = Field(..., description="Distribution channel")
    deposit_type: str = Field(..., description="Deposit type")
    customer_type: str = Field(..., description="Customer type")


class BookingInput(BaseModel):
    """Complete input schema for hotel booking prediction."""
    numerical: NumericalFeatures = Field(..., description="Numerical booking features")
    categorical: CategoricalFeatures = Field(..., description="Categorical booking features")


class PredictionResult(BaseModel):
    """Response schema for prediction endpoint."""
    prediction: int = Field(..., description="Raw prediction: 0 or 1")
    prediction_label: str = Field(..., description="Human-readable label")
    probability: float = Field(..., ge=0, le=1, description="Probability of predicted class")
    probabilities: Dict[str, float] = Field(
        ..., description="Probabilities for both classes"
    )
    raw_prediction: int = Field(..., description="Raw model output")


class HealthCheck(BaseModel):
    """Response schema for health check endpoint."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model loaded successfully")


# API Endpoints

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint returning API info."""
    return {
        "message": "Hotel Booking Cancellation Predictor API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    return HealthCheck(
        status="ok" if model_loaded else "model_not_loaded",
        model_loaded=model_loaded,
    )


@app.post("/predict", response_model=PredictionResult)
async def predict_endpoint(input_data: BookingInput):
    """Predict hotel booking cancellation.
    
    Accepts booking features and returns cancellation prediction
    with probabilities.
    
    Args:
        input_data: Booking features including numerical and categorical
        
    Returns:
        Prediction result with label and probabilities
    """
    if not model_loaded or model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please retrain the model."
        )
    
    # Combine numerical and categorical features into a single dictionary
    data = {
        # Numerical features
        "lead_time": input_data.numerical.lead_time,
        "arrival_date_year": input_data.numerical.arrival_date_year,
        "arrival_date_week_number": input_data.numerical.arrival_date_week_number,
        "arrival_date_day_of_month": input_data.numerical.arrival_date_day_of_month,
        "stays_in_weekend_nights": input_data.numerical.stays_in_weekend_nights,
        "stays_in_week_nights": input_data.numerical.stays_in_week_nights,
        "adults": input_data.numerical.adults,
        "children": input_data.numerical.children,
        "babies": input_data.numerical.babies,
        "is_repeated_guest": input_data.numerical.is_repeated_guest,
        "previous_cancellations": input_data.numerical.previous_cancellations,
        "previous_bookings_not_canceled": input_data.numerical.previous_bookings_not_canceled,
        "booking_changes": input_data.numerical.booking_changes,
        "days_in_waiting_list": input_data.numerical.days_in_waiting_list,
        "adr": input_data.numerical.adr,
        "required_car_parking_spaces": input_data.numerical.required_car_parking_spaces,
        "total_of_special_requests": input_data.numerical.total_of_special_requests,
        "total_guests": input_data.numerical.total_guests,
        "total_stay_nights": input_data.numerical.total_stay_nights,
        "total_previous_bookings": input_data.numerical.total_previous_bookings,
        "is_family": input_data.numerical.is_family,
        "adr_per_guest": input_data.numerical.adr_per_guest,
        # Categorical features
        "hotel": input_data.categorical.hotel,
        "arrival_date_month": input_data.categorical.arrival_date_month,
        "meal": input_data.categorical.meal,
        "country": input_data.categorical.country,
        "market_segment": input_data.categorical.market_segment,
        "distribution_channel": input_data.categorical.distribution_channel,
        "deposit_type": input_data.categorical.deposit_type,
        "customer_type": input_data.categorical.customer_type,
    }
    
    # Make prediction
    result = predict_single(data, MODEL_PATH)
    
    return result


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)