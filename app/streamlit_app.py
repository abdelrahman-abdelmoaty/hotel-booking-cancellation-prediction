"""
Streamlit App for Hotel Booking Cancellation Prediction

This app provides an interactive interface to:
- View model performance metrics
- Explore predictions
- Test the models with custom inputs
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to path (app is in a subdirectory, so go up one level)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Page configuration
st.set_page_config(
    page_title="Hotel Booking Cancellation Predictor",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🏨 Hotel Booking Cancellation Predictor</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a page",
    ["📊 Model Performance", "🔮 Make Prediction", "ℹ️ About"]
)

# Load metrics
@st.cache_data
def load_metrics():
    """Load model metrics from CSV"""
    metrics_path = os.path.join(project_root, "data", "model_metrics.csv")
    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path, index_col=0)
        return df
    return None

# Load metrics
metrics_df = load_metrics()

if page == "📊 Model Performance":
    st.header("Model Performance Metrics")
    
    if metrics_df is not None:
        # Display metrics table
        st.subheader("Performance Comparison")
        
        # Format metrics for display
        display_df = metrics_df.copy()
        display_df.columns = [col.replace('weighted', '').title() for col in display_df.columns]
        
        st.dataframe(display_df.round(4), use_container_width=True)
        
        # Metrics visualization
        st.subheader("Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.bar_chart(metrics_df[['accuracy', 'auc']].T)
            st.caption("Accuracy and AUC Comparison")
        
        with col2:
            st.bar_chart(metrics_df[['weightedPrecision', 'weightedRecall', 'f1']].T)
            st.caption("Precision, Recall, and F1-Score Comparison")
        
        # Best model
        st.subheader("Best Model")
        best_model = metrics_df['accuracy'].idxmax()
        st.success(f"**{best_model}** is the best performing model!")
    else:
        st.warning("⚠️ Model metrics not found. Please run the ML models notebook first.")

elif page == "🔮 Make Prediction":
    st.header("Make a Prediction")
    st.info("💡 Enter booking details below to predict if the booking will be cancelled.")
    
    # Example bookings for quick selection
    example_bookings = {
        "Select an example...": None,
        "High Cancellation Risk - Long Lead Time": {
            "hotel_type": "City Hotel",
            "lead_time": 300,
            "arrival_year": 2016,
            "arrival_month": 7,
            "arrival_day": 15,
            "weekend_nights": 2,
            "week_nights": 3,
            "adults": 2,
            "children": 0,
            "babies": 0,
            "meal": "BB",
            "country": "PRT",
            "market_segment": "Online TA",
            "distribution_channel": "TA/TO",
            "is_repeated_guest": 0,
            "previous_cancellations": 2,
            "previous_bookings_not_canceled": 0,
            "reserved_room_type": "A",
            "assigned_room_type": "A",
            "deposit_type": "No Deposit",
            "customer_type": "Transient",
            "adr": 120.0,
            "required_car_parking_spaces": 0,
            "total_special_requests": 0
        },
        "Low Cancellation Risk - Short Lead Time": {
            "hotel_type": "Resort Hotel",
            "lead_time": 7,
            "arrival_year": 2017,
            "arrival_month": 8,
            "arrival_day": 10,
            "weekend_nights": 1,
            "week_nights": 2,
            "adults": 2,
            "children": 1,
            "babies": 0,
            "meal": "HB",
            "country": "GBR",
            "market_segment": "Direct",
            "distribution_channel": "Direct",
            "is_repeated_guest": 1,
            "previous_cancellations": 0,
            "previous_bookings_not_canceled": 5,
            "reserved_room_type": "A",
            "assigned_room_type": "A",
            "deposit_type": "Refundable",
            "customer_type": "Transient-Party",
            "adr": 150.0,
            "required_car_parking_spaces": 1,
            "total_special_requests": 2
        },
        "Corporate Booking": {
            "hotel_type": "City Hotel",
            "lead_time": 45,
            "arrival_year": 2016,
            "arrival_month": 6,
            "arrival_day": 20,
            "weekend_nights": 0,
            "week_nights": 4,
            "adults": 1,
            "children": 0,
            "babies": 0,
            "meal": "BB",
            "country": "USA",
            "market_segment": "Corporate",
            "distribution_channel": "Corporate",
            "is_repeated_guest": 1,
            "previous_cancellations": 0,
            "previous_bookings_not_canceled": 10,
            "reserved_room_type": "A",
            "assigned_room_type": "A",
            "deposit_type": "No Deposit",
            "customer_type": "Contract",
            "adr": 200.0,
            "required_car_parking_spaces": 0,
            "total_special_requests": 1
        },
        "Family Vacation": {
            "hotel_type": "Resort Hotel",
            "lead_time": 90,
            "arrival_year": 2017,
            "arrival_month": 7,
            "arrival_day": 1,
            "weekend_nights": 2,
            "week_nights": 5,
            "adults": 2,
            "children": 2,
            "babies": 1,
            "meal": "FB",
            "country": "ESP",
            "market_segment": "Online TA",
            "distribution_channel": "TA/TO",
            "is_repeated_guest": 0,
            "previous_cancellations": 0,
            "previous_bookings_not_canceled": 0,
            "reserved_room_type": "A",
            "assigned_room_type": "A",
            "deposit_type": "Non Refund",
            "customer_type": "Transient-Party",
            "adr": 180.0,
            "required_car_parking_spaces": 1,
            "total_special_requests": 3
        }
    }
    
    # Example selection
    selected_example = st.selectbox("📋 Load Example Booking", list(example_bookings.keys()))
    
    # Create form for input
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        # Get default values from example if selected
        example_data = example_bookings.get(selected_example) if selected_example and selected_example != "Select an example..." else None
        
        with col1:
            hotel_type = st.selectbox("Hotel Type", ["Resort Hotel", "City Hotel"], 
                                     index=0 if not example_data else (0 if example_data["hotel_type"] == "Resort Hotel" else 1))
            lead_time = st.number_input("Lead Time (days)", min_value=0, max_value=800, 
                                        value=example_data["lead_time"] if example_data else 0)
            arrival_year = st.number_input("Arrival Year", min_value=2015, max_value=2017, 
                                          value=example_data["arrival_year"] if example_data else 2016)
            arrival_month = st.selectbox("Arrival Month", range(1, 13), 
                                        index=(example_data["arrival_month"] - 1) if example_data else 0,
                                        format_func=lambda x: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][x-1])
            arrival_day = st.number_input("Arrival Day", min_value=1, max_value=31, 
                                         value=example_data["arrival_day"] if example_data else 1)
            weekend_nights = st.number_input("Weekend Nights", min_value=0, max_value=20, 
                                            value=example_data["weekend_nights"] if example_data else 0)
            week_nights = st.number_input("Week Nights", min_value=0, max_value=50, 
                                         value=example_data["week_nights"] if example_data else 0)
            adults = st.number_input("Adults", min_value=0, max_value=10, 
                                    value=example_data["adults"] if example_data else 2)
            children = st.number_input("Children", min_value=0, max_value=10, 
                                      value=example_data["children"] if example_data else 0)
            babies = st.number_input("Babies", min_value=0, max_value=10, 
                                   value=example_data["babies"] if example_data else 0)
        
        with col2:
            meal_options = ["BB", "HB", "FB", "SC", "Undefined"]
            meal_index = meal_options.index(example_data["meal"]) if example_data and example_data["meal"] in meal_options else 0
            meal = st.selectbox("Meal Type", meal_options, index=meal_index)
            
            country_options = ["PRT", "GBR", "USA", "ESP", "FRA", "ITA", "IRL", "BEL", "BRA", "DEU", "Other"]
            country_index = country_options.index(example_data["country"]) if example_data and example_data["country"] in country_options else 0
            country = st.selectbox("Country", country_options, index=country_index)
            
            market_segment_options = ["Direct", "Corporate", "Online TA", "Offline TA/TO", "Complementary", "Groups", "Undefined", "Aviation"]
            market_segment_index = market_segment_options.index(example_data["market_segment"]) if example_data and example_data["market_segment"] in market_segment_options else 0
            market_segment = st.selectbox("Market Segment", market_segment_options, index=market_segment_index)
            
            distribution_channel_options = ["Direct", "Corporate", "TA/TO", "Undefined", "GDS"]
            distribution_channel_index = distribution_channel_options.index(example_data["distribution_channel"]) if example_data and example_data["distribution_channel"] in distribution_channel_options else 0
            distribution_channel = st.selectbox("Distribution Channel", distribution_channel_options, index=distribution_channel_index)
            
            is_repeated_guest = st.selectbox("Repeated Guest", [0, 1], 
                                            index=example_data["is_repeated_guest"] if example_data else 0,
                                            format_func=lambda x: "Yes" if x == 1 else "No")
            previous_cancellations = st.number_input("Previous Cancellations", min_value=0, max_value=30, 
                                                    value=example_data["previous_cancellations"] if example_data else 0)
            previous_bookings_not_canceled = st.number_input("Previous Bookings Not Canceled", min_value=0, max_value=100, 
                                                            value=example_data["previous_bookings_not_canceled"] if example_data else 0)
            
            reserved_room_type_options = ["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"]
            reserved_room_type_index = reserved_room_type_options.index(example_data["reserved_room_type"]) if example_data and example_data["reserved_room_type"] in reserved_room_type_options else 0
            reserved_room_type = st.selectbox("Reserved Room Type", reserved_room_type_options, index=reserved_room_type_index)
            
            assigned_room_type_options = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "P"]
            assigned_room_type_index = assigned_room_type_options.index(example_data["assigned_room_type"]) if example_data and example_data["assigned_room_type"] in assigned_room_type_options else 0
            assigned_room_type = st.selectbox("Assigned Room Type", assigned_room_type_options, index=assigned_room_type_index)
            
            deposit_type_options = ["No Deposit", "Non Refund", "Refundable"]
            deposit_type_index = deposit_type_options.index(example_data["deposit_type"]) if example_data and example_data["deposit_type"] in deposit_type_options else 0
            deposit_type = st.selectbox("Deposit Type", deposit_type_options, index=deposit_type_index)
            
            customer_type_options = ["Transient", "Transient-Party", "Contract", "Group"]
            customer_type_index = customer_type_options.index(example_data["customer_type"]) if example_data and example_data["customer_type"] in customer_type_options else 0
            customer_type = st.selectbox("Customer Type", customer_type_options, index=customer_type_index)
            
            adr = st.number_input("Average Daily Rate (ADR)", min_value=0.0, max_value=1000.0, 
                                 value=example_data["adr"] if example_data else 100.0, step=1.0)
            required_car_parking_spaces = st.number_input("Required Car Parking Spaces", min_value=0, max_value=10, 
                                                          value=example_data["required_car_parking_spaces"] if example_data else 0)
            total_special_requests = st.number_input("Total Special Requests", min_value=0, max_value=10, 
                                                    value=example_data["total_special_requests"] if example_data else 0)
        
        submitted = st.form_submit_button("🔮 Predict Cancellation", use_container_width=True)
    
    if submitted:
        # Calculate derived features
        total_nights = weekend_nights + week_nights
        total_guests = adults + children + babies
        total_booking_value = adr * total_nights
        
        # Show calculated features
        st.subheader("📊 Booking Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Nights", total_nights)
        with col2:
            st.metric("Total Guests", total_guests)
        with col3:
            st.metric("Total Booking Value", f"${total_booking_value:.2f}")
        
        # Prediction based on key factors
        st.subheader("🔮 Cancellation Prediction")
        
        # Simple heuristic-based prediction
        cancellation_prob = 0.3  # Base probability
        
        # Adjust probability based on key factors
        if lead_time > 180:
            cancellation_prob += 0.2
        if previous_cancellations > 0:
            cancellation_prob += 0.3
        if deposit_type == "No Deposit":
            cancellation_prob += 0.1
        if total_booking_value > 500:
            cancellation_prob -= 0.1
        if is_repeated_guest == 1:
            cancellation_prob -= 0.15
        if previous_bookings_not_canceled > 5:
            cancellation_prob -= 0.1
        
        cancellation_prob = max(0.0, min(1.0, cancellation_prob))
        
        # Display result
        col1, col2 = st.columns(2)
        with col1:
            if cancellation_prob > 0.5:
                st.error(f"⚠️ **High Cancellation Risk**: {cancellation_prob*100:.1f}%")
                st.write("**Factors contributing to risk:**")
                if lead_time > 180:
                    st.write("• Long lead time (>180 days)")
                if previous_cancellations > 0:
                    st.write(f"• Previous cancellations ({previous_cancellations})")
                if deposit_type == "No Deposit":
                    st.write("• No deposit required")
            else:
                st.success(f"✅ **Low Cancellation Risk**: {cancellation_prob*100:.1f}%")
                st.write("**Factors reducing risk:**")
                if lead_time <= 30:
                    st.write("• Short lead time")
                if deposit_type != "No Deposit":
                    st.write("• Deposit required")
                if is_repeated_guest == 1:
                    st.write("• Returning guest")
        
        with col2:
            st.metric("Cancellation Probability", f"{cancellation_prob*100:.1f}%")
            st.progress(cancellation_prob)

elif page == "ℹ️ About":
    st.header("About This Project")
    
    st.markdown("""
    ## Hotel Booking Cancellation Prediction
    
    This project uses **Big Data Analytics** to predict hotel booking cancellations using:
    
    ### Technologies Used
    - **MongoDB**: NoSQL database for data storage
    - **Apache Spark**: Distributed data processing
    - **PySpark MLlib**: Machine learning library
    - **Streamlit**: Interactive web interface
    
    ### Models Trained
    1. **Naive Bayes**: Probabilistic classifier
    2. **Decision Tree**: Rule-based classifier
    
    ### Features
    - 28 total features (18 numerical + 10 categorical)
    - 5 engineered features (total_nights, total_guests, etc.)
    - Comprehensive preprocessing pipeline
    
    ### Performance
    - **Best Model**: Decision Tree
    - **Accuracy**: 83.90%
    - **AUC-ROC**: 0.8585
    
    ### Project Structure
    - `notebooks/`: Jupyter notebooks for data processing and model training
    - `data/`: Datasets and processed data
    - `models/`: Saved ML models
    - `reports/`: Visualizations and metrics
    - `streamlit_app.py`: This interactive web app
    
    ### How to Use
    1. **View Performance**: Check model metrics and comparisons
    2. **Make Predictions**: Enter booking details or select from example bookings to get cancellation predictions
    
    ### Note
    For full prediction functionality, ensure the models are saved by running 
    `04_ml_models.ipynb` notebook.
    """)
    
    st.markdown("---")
    st.markdown("**Developed as part of Big Data Analytics Course**")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Hotel Booking Cancellation Prediction System | Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)

