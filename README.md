# Hotel Booking Cancellation Predictor

A production-quality machine learning application for predicting hotel booking cancellations.

## Project Overview

This application predicts whether a hotel booking will be canceled based on booking features. Built from an exploratory Machine Learning notebook, this package provides a modular Python implementation, a FastAPI backend, and a Streamlit frontend for interactive predictions.

## Problem Statement

Hotels face significant revenue loss from unexpected booking cancellations. Predicting cancellation likelihood enables better overbooking strategies and resource planning. This model predicts the probability of cancellation using 30 features from historical booking data.

## Dataset

- **Source**: `hotel_bookings.csv`
- **Entries**: 119,390 bookings (after preprocessing)
- **Target Variable**: `is_canceled` (0 = Not Canceled, 1 = Canceled)
- **Cancellation Rate**: ~27.5%
- **Features**: 32 original columns including categorical (hotel type, month, country, etc.) and numerical (lead time, ADR, stay duration, etc.) features

### Preprocessing Highlights

- Removed duplicates and invalid entries (bookings with no guests, negative ADR values)
- Handled missing values: `country` filled with 'Unknown', `children` filled with 0
- Engineered 7 new features: `total_guests`, `total_stay_nights`, `total_previous_bookings`, `is_family`, `adr_per_guest`, `has_parking`, `has_previous_cancellations`
- Grouped rare countries to 'Other' (top 20 countries kept)
- Removed data leakage features: `reservation_status`, `reservation_status_date`, `assigned_room_type`
- Selected final feature set: 8 categorical + 20 numerical features

## Model

### Best Model: XGBoost

- **Cross-Validation Mean F1-Score**: 0.6979 (best among 5 models)
- **Test Set Performance**:
  - Accuracy: 0.8318
  - Precision (Canceled): 0.6905
  - Recall (Canceled): 0.7051
  - F1-Score (Canceled): 0.6977

### Model Details

- **Algorithm**: XGBoost Classifier
- **Pipeline**: ColumnTransformer (StandardScaler + OneHotEncoder) → XGBoost
- **Class Handling**: Balanced class weights (no SMOTE - caused decimal issues with OHE)
- **Hyperparameters**: n_estimators=400, max_depth=8, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8
- **Serialization**: Saved via `joblib` as `hotel_booking_model.pkl`

### Model Comparison

| Model | Test F1-Score | CV Mean F1 |
|-------|--------------|------------|
| XGBoost | 0.7090 | 0.6979 |
| Random Forest | 0.7004 | 0.6826 |
| Decision Tree | 0.6311 | 0.6126 |
| Logistic Regression | 0.6437 | 0.6360 |
| KNN | 0.6026 | 0.5932 |

## Repository Structure

```
hotel-booking-prediction/
├── api/                    # FastAPI backend
│   ├── main.py            # API endpoints
│   └── model.pkl          # Trained model (at project root)
│
├── src/                    # Python package
│   ├── hotel_booking/     # Main package
│   │   ├── __init__.py
│   │   ├── config.py      # Configuration paths/hyperparameters
│   │   ├── data.py        # Data loading and cleaning
│   │   ├── preprocessing.py # ColumnTransformer pipeline
│   │   ├── model.py       # Model training/evaluation functions
│   │   └── predict.py     # Prediction interface
│   └── tests/             # Unit tests
│
├── app.py                 # Streamlit frontend
├── hotel_booking_model.pkl # Trained model binary
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages

See `requirements.txt` for the complete list, including:
- `fastapi` + `uvicorn` for the backend API
- `streamlit` for the interactive frontend
- `scikit-learn`, `xgboost` for ML functionality
- `joblib`, `pydantic` for modeling and validation
- `pandas`, `numpy` for data manipulation

### Model File

Ensure the trained model file exists at the project root:
- `hotel_booking_model.pkl`

If the model file is missing, retrain the model using the notebook or train script.

## Usage

### FastAPI Backend

Start the API server:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

#### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint with API info |
| `/health` | GET | Health check - verifies model is loaded |
| `/predict` | POST | Predict cancellation from booking features |

#### Prediction Request Format

```json
{
  "numerical": {
    "lead_time": 342,
    "arrival_date_year": 2015,
    "arrival_date_week_number": 27,
    "arrival_date_day_of_month": 1,
    "stays_in_weekend_nights": 0,
    "stays_in_week_nights": 0,
    "adults": 2,
    "children": 0,
    "babies": 0,
    "is_repeated_guest": 0,
    "previous_cancellations": 0,
    "previous_bookings_not_canceled": 0,
    "booking_changes": 0,
    "days_in_waiting_list": 0,
    "adr": 0.0,
    "required_car_parking_spaces": 0,
    "total_of_special_requests": 0,
    "total_guests": 2,
    "total_stay_nights": 0,
    "total_previous_bookings": 0,
    "is_family": 0,
    "adr_per_guest": 0.0
  },
  "categorical": {
    "hotel": "Resort Hotel",
    "arrival_date_month": "July",
    "meal": "Undefined",
    "country": "USA",
    "market_segment": "Direct",
    "distribution_channel": "Direct",
    "deposit_type": "No Deposit",
    "customer_type": "Transient"
  }
}
```

#### Prediction Response

```json
{
  "prediction": 0,
  "prediction_label": "Not Canceled",
  "probability": 0.9281,
  "probabilities": {
    "Not Canceled": 0.9281,
    "Canceled": 0.0719
  },
  "raw_prediction": 0
}
```

#### Health Check Response

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### Streamlit Frontend

Start the Streamlit app:

```bash
streamlit run app.py
```

The frontend provides:
- Interactive form for all booking features
- Real-time prediction with probability visualization
- Risk assessment (Low/Medium/High)
- Input summary and detailed probability breakdown

### Programmatic Usage (Python)

```python
from src.hotel_booking.predict import load_model, predict_single

# Load model
model = load_model("hotel_booking_model.pkl")

# Make prediction
input_data = {
    "lead_time": 342,
    "arrival_date_year": 2015,
    "arrival_date_month": "July",
    "adults": 2,
    "children": 0,
    "babies": 0,
    "is_repeated_guest": 0,
    "previous_cancellations": 0,
    "previous_bookings_not_canceled": 0,
    "booking_changes": 0,
    "days_in_waiting_list": 0,
    "adr": 0.0,
    "required_car_parking_spaces": 0,
    "total_of_special_requests": 0,
    "total_guests": 2,
    "total_stay_nights": 0,
    "total_previous_bookings": 0,
    "is_family": 0,
    "adr_per_guest": 0.0,
    "hotel": "Resort Hotel",
    "arrival_date_month": "July",
    "meal": "Undefined",
    "country": "USA",
    "market_segment": "Direct",
    "distribution_channel": "Direct",
    "deposit_type": "No Deposit",
    "customer_type": "Transient",
}

result = predict_single(input_data, "hotel_booking_model.pkl")
print(f"Prediction: {result['prediction_label']}")
print(f"Probability: {result['probability']:.2%}")
```

## API Usage Example (cURL)

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "numerical": {
      "lead_time": 30,
      "arrival_date_year": 2017,
      "arrival_date_week_number": 35,
      "arrival_date_day_of_month": 15,
      "stays_in_weekend_nights": 2,
      "stays_in_week_nights": 5,
      "adults": 2,
      "children": 0,
      "babies": 0,
      "is_repeated_guest": 0,
      "previous_cancellations": 0,
      "previous_bookings_not_canceled": 0,
      "booking_changes": 0,
      "days_in_waiting_list": 0,
      "adr": 150.5,
      "required_car_parking_spaces": 0,
      "total_of_special_requests": 0,
      "total_guests": 2,
      "total_stay_nights": 7,
      "total_previous_bookings": 0,
      "is_family": 0,
      "adr_per_guest": 75.25
    },
    "categorical": {
      "hotel": "City Hotel",
      "arrival_date_month": "August",
      "meal": "Undefined",
      "country": "USA",
      "market_segment": "Direct",
      "distribution_channel": "Direct",
      "deposit_type": "No Deposit",
      "customer_type": "Transient"
    }
  }'
```

## Development

### Project Structure

The project is organized as a Python package (`src/hotel_booking/`) with separate modules:

- **config.py**: Central configuration for paths, hyperparameters, and column names
- **data.py**: Data loading, cleaning, and feature engineering
- **preprocessing.py**: scikit-learn ColumnTransformer pipeline
- **model.py**: Model definitions, training, cross-validation, and evaluation
- **predict.py**: Prediction interface for loading models and making inferences

### Adding New Models

To add a new model to the pipeline:

1. Add a new Pipeline entry in `src/hotel_booking/model.py::get_models()`
2. Ensure the model pipeline includes the preprocessor
3. Retrain and evaluate using `src/hotel_booking/model.py::train_full_pipeline()`

### Testing

Run the existing test suite:

```bash
pytest tests/
```

### Code Quality

- Follows Python naming conventions (snake_case for functions/variables)
- Pydantic models for request/response validation in the API
- Type hints throughout
- Docstrings on all public functions
- Modular design for easy maintenance and extension

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Future Improvements

- **Batch Prediction**: Support for predicting multiple bookings at once
- **Model Monitoring**: Track prediction drift and performance over time
- **Feature Importance**: Visualize which features most influence predictions
- **A/B Testing**: Framework for comparing new model versions
- **Deployment**: Docker containerization for easy deployment
- **Explainability**: integrate SHAP values for prediction explanations
- **Ensemble Methods**: Combine multiple models for improved robustness