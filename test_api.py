#!/usr/bin/env python
"""Test script for FastAPI."""

from api.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

# Health check
response = client.get('/health')
print(f'Health check: {response.status_code} - {response.json()}')

# Test prediction with sample data
sample_input = {
    'numerical': {
        'lead_time': 342,
        'arrival_date_year': 2015,
        'arrival_date_week_number': 27,
        'arrival_date_day_of_month': 1,
        'stays_in_weekend_nights': 0,
        'stays_in_week_nights': 0,
        'adults': 2,
        'children': 0,
        'babies': 0,
        'is_repeated_guest': 0,
        'previous_cancellations': 0,
        'previous_bookings_not_canceled': 0,
        'booking_changes': 0,
        'days_in_waiting_list': 0,
        'adr': 0.0,
        'required_car_parking_spaces': 0,
        'total_of_special_requests': 0,
        'total_guests': 2,
        'total_stay_nights': 0,
        'total_previous_bookings': 0,
        'is_family': 0,
        'adr_per_guest': 0.0,
    },
    'categorical': {
        'hotel': 'Resort Hotel',
        'arrival_date_month': 'July',
        'meal': 'Undefined',
        'country': 'USA',
        'market_segment': 'Direct',
        'distribution_channel': 'Direct',
        'deposit_type': 'No Deposit',
        'customer_type': 'Transient',
    }
}

response = client.post('/predict', json=sample_input)
print(f'Predict status: {response.status_code}')
print(f'Predict response: {response.json()}')