#!/usr/bin/env python
"""Test script for the prediction pipeline."""

from src.hotel_booking.predict import load_model, predict_single
from src.hotel_booking.data import load_data, clean_data, engineer_features
import pandas as pd

# Load and preprocess data
df = load_data()
df = clean_data(df)
df = engineer_features(df)

# Take a sample for testing
sample = df.iloc[0]
print('Sample features:')
print(f'Hotel: {sample["hotel"]}')
print(f'ADR: {sample["adr"]}')
print(f'Lead time: {sample["lead_time"]}')

# Make prediction
model = load_model('hotel_booking_model.pkl')
input_data = {
    'lead_time': float(sample['lead_time']),
    'arrival_date_year': int(sample['arrival_date_year']),
    'arrival_date_week_number': int(sample['arrival_date_week_number']),
    'arrival_date_day_of_month': int(sample['arrival_date_day_of_month']),
    'stays_in_weekend_nights': float(sample['stays_in_weekend_nights']),
    'stays_in_week_nights': float(sample['stays_in_week_nights']),
    'adults': int(sample['adults']),
    'children': float(sample['children']),
    'babies': float(sample['babies']),
    'is_repeated_guest': int(sample['is_repeated_guest']),
    'previous_cancellations': int(sample['previous_cancellations']),
    'previous_bookings_not_canceled': int(sample['previous_bookings_not_canceled']),
    'booking_changes': int(sample['booking_changes']),
    'days_in_waiting_list': int(sample['days_in_waiting_list']),
    'adr': float(sample['adr']),
    'required_car_parking_spaces': int(sample['required_car_parking_spaces']),
    'total_of_special_requests': int(sample['total_of_special_requests']),
    'total_guests': float(sample['total_guests']),
    'total_stay_nights': float(sample['total_stay_nights']),
    'total_previous_bookings': int(sample['total_previous_bookings']),
    'is_family': int(sample['is_family']),
    'adr_per_guest': float(sample['adr_per_guest']),
    'hotel': str(sample['hotel']),
    'arrival_date_month': str(sample['arrival_date_month']),
    'meal': str(sample['meal']),
    'country': str(sample['country']),
    'market_segment': str(sample['market_segment']),
    'distribution_channel': str(sample['distribution_channel']),
    'deposit_type': str(sample['deposit_type']),
    'customer_type': str(sample['customer_type']),
}

result = predict_single(input_data, 'hotel_booking_model.pkl')
print('\nPrediction result:')
print(f'Prediction: {result["prediction"]} ({result["prediction_label"]})')
print(f'Probability: {result["probability"]:.4f}')
print(f'Probabilities: {result["probabilities"]}')