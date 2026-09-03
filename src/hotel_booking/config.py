# Project Configuration
# -------------------

# File paths
DATA_PATH = "hotel_bookings.csv"
MODEL_PATH = "hotel_booking_model.pkl"

# Target variable
TARGET_COLUMN = "is_canceled"

# Raw column names from the dataset
CATEGORICAL_COLUMNS_RAW = [
    "hotel",
    "arrival_date_month",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "assigned_room_type",
    "deposit_type",
    "customer_type",
]

NUMERICAL_COLUMNS_RAW = [
    "lead_time",
    "arrival_date_year",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
]

# Engineered features (created during data preprocessing)
ENGINEERED_COLUMNS = [
    "total_guests",
    "total_stay_nights",
    "total_previous_bookings",
    "is_family",
    "adr_per_guest",
    "has_parking",
    "has_previous_cancellations",
]

# Complete numerical columns (raw + engineered features)
NUMERICAL_COLUMNS = [
    "lead_time",
    "arrival_date_year",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "total_guests",
    "total_stay_nights",
    "total_previous_bookings",
    "adr_per_guest",
    "is_family",
    "has_parking",
    "has_previous_cancellations",
]

# Complete categorical columns (including reserved_room_type which was kept in X)
CATEGORICAL_COLUMNS = [
    "hotel",
    "arrival_date_month",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
]

# Features to drop (data leakage, identifier, etc.)
DROP_COLUMNS = [
    "is_canceled",
    "reservation_status",
    "reservation_status_date",
    "assigned_room_type",
    "lead_time_group",
    "agent",
    "company",
]

# Modeling configuration
MODEL_NAME = "XGBoost"
TEST_SIZE = 0.2
RANDOM_STATE = 42
NATIONALITY_TOP_COUNTRIES = 20

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_WORKERS = 1