"""
Data Preprocessing and Feature Engineering Functions

This module provides functions for feature engineering and data preprocessing
for the hotel booking dataset.
"""

import pandas as pd
import numpy as np
from datetime import datetime


def create_total_nights(df):
    """
    Create total_nights feature from weekend and weekday nights.
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        pd.DataFrame: DataFrame with total_nights column
    """
    df = df.copy()
    df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    return df


def create_total_guests(df):
    """
    Create total_guests feature from adults, children, and babies.
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        pd.DataFrame: DataFrame with total_guests column
    """
    df = df.copy()
    df['total_guests'] = df['adults'] + df['children'].fillna(0) + df['babies'].fillna(0)
    return df


def extract_date_features(df):
    """
    Extract date-related features from arrival_date columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        pd.DataFrame: DataFrame with extracted date features
    """
    df = df.copy()
    
    # Create date string and parse
    df['arrival_date'] = pd.to_datetime(
        df['arrival_date_year'].astype(str) + '-' +
        df['arrival_date_month'].astype(str) + '-' +
        df['arrival_date_day_of_month'].astype(str),
        format='%Y-%B-%d',
        errors='coerce'
    )
    
    # Extract date features
    df['arrival_day_of_week'] = df['arrival_date'].dt.dayofweek
    df['arrival_week_of_year'] = df['arrival_date'].dt.isocalendar().week
    df['arrival_is_weekend'] = (df['arrival_day_of_week'] >= 5).astype(int)
    df['arrival_month'] = df['arrival_date'].dt.month
    
    # Season feature
    df['arrival_season'] = df['arrival_month'].apply(lambda x: 
        'Spring' if x in [3, 4, 5] else
        'Summer' if x in [6, 7, 8] else
        'Fall' if x in [9, 10, 11] else 'Winter'
    )
    
    return df


def create_booking_value_features(df):
    """
    Create booking value-related features.
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        pd.DataFrame: DataFrame with booking value features
    """
    df = df.copy()
    
    # Total booking value
    df['total_booking_value'] = df['adr'] * df['total_nights']
    
    # Revenue per guest
    df['revenue_per_guest'] = df['total_booking_value'] / df['total_guests'].replace(0, np.nan)
    
    # Booking value category
    df['booking_value_category'] = pd.cut(
        df['total_booking_value'],
        bins=[0, 100, 300, 600, np.inf],
        labels=['Low', 'Medium', 'High', 'Very High']
    )
    
    return df


def create_cancellation_risk_features(df):
    """
    Create features related to cancellation risk.
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        pd.DataFrame: DataFrame with cancellation risk features
    """
    df = df.copy()
    
    # Previous cancellation ratio
    df['previous_cancellation_ratio'] = df['previous_cancellations'] / (
        df['previous_cancellations'] + df['previous_bookings_not_canceled'] + 1
    )
    
    # Lead time categories
    df['lead_time_category'] = pd.cut(
        df['lead_time'],
        bins=[0, 30, 90, 180, 365, np.inf],
        labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long']
    )
    
    # Deposit type risk (Non Refund is higher risk)
    df['deposit_risk'] = df['deposit_type'].apply(lambda x: 
        2 if x == 'Non Refund' else
        1 if x == 'Refundable' else 0
    )
    
    return df


def handle_missing_values(df):
    """
    Handle missing values in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        pd.DataFrame: DataFrame with handled missing values
    """
    df = df.copy()
    
    # Fill missing values in children with 0
    df['children'] = df['children'].fillna(0)
    
    # Fill missing values in country with 'Unknown'
    df['country'] = df['country'].fillna('Unknown')
    
    # Fill missing values in agent with 0
    df['agent'] = df['agent'].fillna(0)
    
    # Fill missing values in company with 0
    df['company'] = df['company'].fillna(0)
    
    return df


def encode_categorical_features(df, categorical_cols=None):
    """
    Encode categorical features using one-hot encoding.
    
    Args:
        df (pd.DataFrame): Input dataframe
        categorical_cols (list): List of categorical columns to encode
    
    Returns:
        pd.DataFrame: DataFrame with encoded features
    """
    df = df.copy()
    
    if categorical_cols is None:
        categorical_cols = [
            'hotel', 'meal', 'country', 'market_segment',
            'distribution_channel', 'reserved_room_type',
            'assigned_room_type', 'deposit_type', 'customer_type',
            'arrival_season', 'lead_time_category', 'booking_value_category'
        ]
    
    # Only encode columns that exist in dataframe
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    
    # One-hot encode
    df_encoded = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)
    
    return df_encoded


def prepare_features_for_ml(df, target_col='is_canceled'):
    """
    Complete feature engineering pipeline.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Name of target column
    
    Returns:
        tuple: (X, y) where X is features and y is target
    """
    df = df.copy()
    
    # Apply all preprocessing steps
    df = create_total_nights(df)
    df = create_total_guests(df)
    df = extract_date_features(df)
    df = create_booking_value_features(df)
    df = create_cancellation_risk_features(df)
    df = handle_missing_values(df)
    
    # Separate target
    if target_col in df.columns:
        y = df[target_col]
        X = df.drop(target_col, axis=1)
    else:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    
    # Drop non-feature columns
    columns_to_drop = [
        'arrival_date', 'reservation_status', 'reservation_status_date',
        'arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month'
    ]
    columns_to_drop = [col for col in columns_to_drop if col in X.columns]
    X = X.drop(columns_to_drop, axis=1)
    
    # Encode categorical features
    X = encode_categorical_features(X)
    
    return X, y

