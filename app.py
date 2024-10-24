# src/app.py

import streamlit as st
from forecast import DataProcessor

# Streamlit App Title
st.title("Sales Forecasting Data Upload")

# Instantiate DataProcessor object
forecast = DataProcessor()

# Step 1: Upload data
forecast.upload_data()



# Step 2: Preview data
forecast.preview_data()

# Step 3: Display data info
forecast.data_info()

# Step 4: Display data statistics
forecast.data_statistics()

forecast.select_columns()

forecast.time_series_analysis()

# forecast.select_influencing_columns()

# forecast.handle_null_values()

forecast.feature_scaling()

forecast.category_analysis()

forecast.forecasting1()

forecast.weather_data_integration()
