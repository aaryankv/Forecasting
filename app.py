
import streamlit as st
from forecast import forecast

# Streamlit App Title
st.title("Sales Forecasting Data Upload")

# Instantiate DataProcessor object
forecast = forecast()

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

print("Which Model Do You Want To Use")
model=st.radio("Pick The Model",["LSTM1","LSTM2","GRU"])
if(model==("LSTM1")):
    forecast.forecasting1()
elif(model==("LSTM2")):
    forecast.forecasting2()
else:
    forecast.forecasting3()
    
forecast.weather_data_integration()
