import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

# Title
st.title('General Dataset Analysis')

# File upload
uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])
if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file, engine='openpyxl')

    # Display the dataframe
    st.write("Uploaded DataFrame:")
    st.write(df)

    # Display column names and data types
    st.write("Column Names and Data Types:")
    st.write(df.dtypes)

    # Display missing values
    st.write("Missing Values:")
    missing_values = df.isnull().sum()
    st.write(missing_values)

    # Plot missing values
    st.write("Missing Values Heatmap:")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title('Missing Values Heatmap')
    st.pyplot(fig)

    # Ask user how to handle missing values
    if missing_values.sum() > 0:
        st.write("Handling Missing Values:")
        method = st.selectbox("Choose method to handle missing values", ['Drop rows', 'Fill with mean', 'Fill with median', 'Fill with mode', 'Interpolate'])
        
        if method == 'Drop rows':
            df = df.dropna()
        elif method == 'Fill with mean':
            df = df.fillna(df.mean())
        elif method == 'Fill with median':
            df = df.fillna(df.median())
        elif method == 'Fill with mode':
            df = df.fillna(df.mode().iloc[0])
        elif method == 'Interpolate':
            df = df.interpolate()

        # Display the updated dataframe
        st.write("Updated DataFrame after handling missing values:")
        st.write(df)
    
    # Ask user to select sales column
    sales_column = st.selectbox("Select sales column for analysis", df.columns)
    
    # Optional Category Analysis
    if st.checkbox("Perform category analysis?"):
        category_column = st.selectbox("Select category column", df.columns)
        st.write("Category Analysis:")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=category_column, y=sales_column, data=df, estimator=sum, ci=None, palette='viridis')
        plt.title(f'Sales by {category_column}')
        st.pyplot(fig)
    
    # Ensure the Date column is in datetime format
    date_column = st.selectbox("Select date column", df.select_dtypes(include=['object', 'datetime']).columns)
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    df.set_index(date_column, inplace=True)
    
    # Resample the data to monthly
    df_resampled = df[sales_column].resample('M').sum()
    
    # Perform Seasonal Decomposition
    result = seasonal_decompose(df_resampled, model='multiplicative', period=12)
    
    # Plotting
    fig, axs = plt.subplots(4, 1, figsize=(14, 10))
    result.observed.plot(ax=axs[0], title='Observed')
    result.trend.plot(ax=axs[1], title='Trend')
    result.seasonal.plot(ax=axs[2], title='Seasonal')
    result.resid.plot(ax=axs[3], title='Residual')
    st.pyplot(fig)
    
    # Ask user if they have a weather dataset
    if st.checkbox("Do you have a weather dataset?"):
        weather_file = st.file_uploader("Upload your weather dataset", type=['csv', 'xlsx'])
        if weather_file is not None:
            if weather_file.name.endswith('.csv'):
                weather_df = pd.read_csv(weather_file)
            elif weather_file.name.endswith('.xlsx'):
                weather_df = pd.read_excel(weather_file, engine='openpyxl')
            
            # Display the weather dataframe
            st.write("Weather DataFrame:")
            st.write(weather_df)
            
            # Ensure weather dates are in datetime format
            weather_date_column = st.selectbox("Select date column in weather dataset", weather_df.select_dtypes(include=['object', 'datetime']).columns)
            weather_df[weather_date_column] = pd.to_datetime(weather_df[weather_date_column], errors='coerce')
            
            # Reset index for merging
            df.reset_index(inplace=True)
            
            # Merge with sales data
            combined_df = pd.merge(df, weather_df, left_on=date_column, right_on=weather_date_column)
            st.write(weather_df.dtypes)
            
            # Display the combined dataframe
            st.write("Combined DataFrame:")
            st.write(combined_df)
            
            # Perform analysis on combined data (for example, plotting trends with weather)
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.lineplot(data=combined_df, x=combined_df[date_column], y=sales_column, label='Sales')
            sns.lineplot(data=combined_df, x=combined_df[date_column], y='Temperature (C)', label='Temperature', ax=ax)
            plt.title(f'{sales_column} and Temperature Trends')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.legend()
            st.pyplot(fig)

    # LSTM for sales forecasting
    if st.button("Train LSTM and Forecast Sales"):
        # Group sales by month
        df_monthly = df.resample('MS')[sales_column].sum()
        
        # Prepare data for LSTM
        data = df_monthly.values.reshape(-1, 1)
        
        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        # Prepare training and testing datasets
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]

        def create_dataset(dataset, time_step=1):
            dataX, dataY = [], []
            for i in range(len(dataset) - time_step):
                dataX.append(dataset[i:(i + time_step), 0])
                dataY.append(dataset[i + time_step, 0])
            return np.array(dataX), np.array(dataY)

        time_step = 5  # Adjusted to ensure compatibility with the dataset length
        X_train, y_train = create_dataset(train_data, time_step)
        X_test, y_test = create_dataset(test_data, time_step)

        # Ensure there's enough data to reshape
        if X_train.shape[0] == 0 or X_test.shape[0] == 0:
            raise ValueError("Insufficient data for the given time_step. Try reducing the time_step value even further.")

        # Reshape input to be [samples, time steps, features] which is required for LSTM
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        # Create and train LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, batch_size=1, epochs=1)
        
        # Make predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)
        
        # Transform back to original form
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)
        
        # Plot the results
        plt.figure(figsize=(14, 8))
        # Corrected plotting
        train_index = df_monthly.index[time_step:time_step + len(train_predict)]
        test_index = df_monthly.index[time_step + len(train_predict):time_step + len(train_predict) + len(test_predict)]
        plt.plot(df_monthly.index, data, label='Original Sales')
        plt.plot(train_index, train_predict, label='Train Prediction')
        plt.plot(test_index, test_predict, label='Test Prediction')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.title('Sales Forecasting using LSTM')
        plt.legend()
        st.pyplot(plt)
