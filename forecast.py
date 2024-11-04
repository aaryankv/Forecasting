
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import dateutil.parser
import shlex
import io
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, GRU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np

class forecast:
    def __init__(self):
        self.data = None
        self.date_column = None
        self.target_column = None

    def upload_data(self):
        """Uploads a CSV or Excel file via Streamlit file uploader"""
        uploaded_file = st.file_uploader("Upload a CSV or Excel", type=["csv", "xls", "xlsx"])
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                self.data = pd.read_csv(uploaded_file, encoding='latin1')  # or 'utf-16' depending on the file's encoding
            elif uploaded_file.name.endswith('.xlsx'):
                self.data = pd.read_excel(uploaded_file, engine='openpyxl')
            st.success("File uploaded successfully!")
        else:
            st.warning("Please upload a CSV or Excel file.")


    def preview_data(self):
        """
        Displays the first few rows of the uploaded dataset
        """
        if self.data is not None:
            st.write("### Dataset Preview")
            st.dataframe(self.data.head())
            # st.write(self.data.columns)
        else:
            st.error("No data available to preview.")
           
    def data_info(self):
        """
        Displays the size and shape of the dataset and a sample of the data
        """
        if self.data is not None:
            st.write("### Dataset Info")
            
            # Display the shape of the dataset
            num_rows, num_cols = self.data.shape
            st.write(f"**Number of rows:** {num_rows}")
            st.write(f"**Number of columns:** {num_cols}")

            # Display null value information using the info() method
            buf = io.StringIO()
            self.data.info(buf=buf)
            s = buf.getvalue()
            lines = [shlex.split(line) for line in s.splitlines()[3:-2]]
            print(f"Lines[0]: {lines[0]}")  # Print the column names
            print(f"Lines[1]: {lines[1]}")  # Print the first row of data
            
            # Find the maximum number of columns in the data
            max_cols = max(len(line) for line in lines)
            
            # Pad the column names with None to match the maximum number of columns
            column_names = lines[0] + [None] * (max_cols - len(lines[0]))
            
            info_df = pd.DataFrame(lines[1:], columns=column_names)  # Select all columns
            st.table(info_df)
                
            # Create a bar chart to visualize null value distribution
            # null_counts = self.data.isnull().sum()
            # st.write("### Null Value Distribution")
            # st.bar_chart(null_counts)
            
            null_counts = self.data.isnull().sum()
            total_counts = self.data.shape[0]

            # Calculate the percentage of null values for each column
            null_percentages = (null_counts / total_counts) * 100

            # Create a bar chart using Plotly
            # fig = px.bar(x=null_percentages.index, y=null_percentages.values, text_auto='.2f')
            fig = px.bar(x=null_percentages.index, y=null_percentages.values, 
             color=[ 'high' if val > 25 else 'low' for val in null_percentages.values], 
             color_discrete_map={'high': '#ee4a28', 'low': '#324bde'},
             text_auto='.2f')

            fig.update_layout(coloraxis_showscale=False)
            
            fig.update_layout(title=" Null Value Distribution in Percentage", 
                            xaxis_title="Columns", 
                            yaxis_title="Count of Null Values")

            # Show the plot using Streamlit
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("*Note: If the null values are above 25%, they should be removed.")
        else:
            st.error("No data available for info.")
        
    def data_statistics(self):
        """
        Displays basic statistical summary of the dataset
        """
        if self.data is not None:
            st.write("### Dataset Statistics")
            st.write(self.data.describe())
        else:
            st.error("No data available for statistics.")
    

    def select_columns(self) -> None:
        """
        Selects the date column and target column
        """
        if self.data is None:
            st.error("No data available to select columns.")
            return

        st.write("### Select Columns")
        columns = self.data.columns.tolist()

        # Select date column
        self.date_column = st.selectbox("Select Date Column", columns)
        if self.date_column:
            try:
                self.data[self.date_column] = self.data[self.date_column].apply(dateutil.parser.parse)
            except ValueError:
                st.error("Invalid date format for date column. Please select a column with a valid date format.")
                self.date_column = None
            else:
                if not self.data[self.date_column].notnull().all():
                    st.error("Date column contains null values. Please select a column with no null values.")

        # Select target column
        if self.date_column:
            self.target_column = st.selectbox("Select Target Column", [col for col in columns if col != self.date_column])
            if self.target_column:
                try:
                    self.data[self.target_column] = self.data[self.target_column].astype(float)
                except ValueError:
                    st.error("Invalid value for target column. Please select a column with numeric values.")
                    self.target_column = None
                else:
                    if not self.data[self.target_column].notnull().all():
                        st.error("Target column contains null values. Please select a column with no null values.")


    def time_series_analysis(self):
        """
        Performs time series analysis on the selected column
        """
        if self.data is not None and self.date_column is not None and self.target_column is not None:
            st.write("### Time Series Analysis")
            
            # Check if the date column already contains datetime objects
            if self.data[self.date_column].dtype == 'datetime64[ns]':
                self.data.set_index(self.date_column, inplace=True)
            else:
                # Parse the date column using dateutil.parser.parse
                self.data[self.date_column] = self.data[self.date_column].apply(dateutil.parser.parse)
                self.data.set_index(self.date_column, inplace=True)
            
            # # Display the original DataFrame
            # st.write("### Original DataFrame")
            # st.write(self.data[[self.target_column]])

            # # Create a line chart for the original time series
            # st.write("### Original Time Series")
            # st.line_chart(self.data[[self.target_column]])

            # Add a selectbox for sampling frequency
            frequency_options = ['Weekly', 'Monthly', 'Quarterly', 'Yearly']
            frequency = st.selectbox('Select sampling frequency', frequency_options)

            if frequency == 'Weekly':
                freq_code = 'w'
            elif frequency == 'Monthly':
                freq_code = 'M'
            elif frequency == 'Quarterly':
                freq_code = 'Q'
            elif frequency == 'Yearly':
                freq_code = 'Y'

            # Resample the data to the selected frequency, starting from the first date
            resampled_data = self.data[[self.target_column]].resample(freq_code, origin=self.data.index[0]).mean()


            # Display the resampled DataFrame
            st.write("### Resampled DataFrame")
            st.write(resampled_data)

            # Create a line chart for the resampled time series
            st.write("### Resampled Time Series")
            st.line_chart(resampled_data)

            # Interpolate missing values
            # resampled_data.interpolate(method='linear', inplace=True)

            # Perform seasonal decomposition
            decomposition = seasonal_decompose(resampled_data, model='additive')

            # Create a line chart for the seasonal component
            st.write("### Seasonal Component")
            st.line_chart(decomposition.seasonal)

            # Create a line chart for the trend component
            st.write("### Trend Component")
            st.line_chart(decomposition.trend)

    def select_influencing_columns(self):
        """
        Selects the columns that influence the target column
        """
        if self.data is not None and self.target_column is not None and self.date_column is not None:
            # Create a new DataFrame with the date as the index and the target column as the first column
            final_df = self.data[[self.target_column]].copy()

            # Get the columns to analyze, excluding the target column and date column
            columns_to_analyze = [col for col in self.data.columns if col not in [self.target_column, self.date_column]]

            # Allow the user to add  columns to the influencing_df
            add_columns = st.multiselect("Select columns to include:", columns_to_analyze, key="add_columns_unique")

            final_df = pd.concat([final_df, self.data[add_columns]], axis=1)

            st.write("### Selected Columns DataFrame")
            st.dataframe(final_df)

            return final_df
        else:
            st.error("Please select a date column and a target column.")

    def handle_null_values(self):
        """
        Handles null values in the final DataFrame.
        """
        final_df = self.select_influencing_columns()
        if final_df is not None:
            st.write("### Handling Null Values")

            # Display the percentage of null values in each column
            null_percentages = (final_df.isnull().sum() / len(final_df)) * 100
            st.write("#### Null Value Distribution")
            st.write(null_percentages)

            # Allow the user to choose how to handle null values
            st.write("#### Choose Null Value Handling Method")
            null_handling_method = st.selectbox("Select method", ["Mean Imputation", "Interpolation"], key="null_handling_method_unique")

            if null_handling_method == "Mean Imputation":
                st.write("Handling null values using mean imputation")
                final_df = final_df.fillna(final_df.mean())
            elif null_handling_method == "Interpolation":
                st.write("Handling null values using interpolation")
                final_df = final_df.interpolate(method='linear')

            # Display the updated null value distribution
            st.write("#### Updated Null Value Distribution")
            null_percentages = (final_df.isnull().sum() / len(final_df)) * 100
            st.write(null_percentages)

            return final_df
        else:
            st.error("No data available to handle null values.")

    def feature_scaling(self):
        """
        Scales the features in the final DataFrame.
        """
        final_df = self.handle_null_values()
        if final_df is not None and not final_df.empty:
            st.write("### Feature Scaling")

            # Check if there are any numeric columns to scale
            numeric_columns = final_df.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_columns) == 0:
                st.error("No numeric columns available for scaling.")
                return final_df

            # Allow the user to choose which numeric columns to scale
            columns_to_scale = st.multiselect("Select columns to scale:", numeric_columns, default=list(numeric_columns), key="columns_to_scale")

            if not columns_to_scale:
                st.warning("No columns selected for scaling. Returning original DataFrame.")
                return final_df

            # Allow the user to choose a scaling method
            st.write("#### Choose Scaling Method")
            scaling_method = st.selectbox("Select method", ["Standard Scaler", "Min-Max Scaler"], key="scaling_method_unique")

            # Create a copy of the DataFrame to avoid modifying the original
            scaled_df = final_df.copy()
            
            try:
                if scaling_method == "Standard Scaler":
                    scaler = StandardScaler()
                    scaled_df[columns_to_scale] = scaler.fit_transform(scaled_df[columns_to_scale])
                elif scaling_method == "Min-Max Scaler":
                    scaler = MinMaxScaler()
                    scaled_df[columns_to_scale] = scaler.fit_transform(scaled_df[columns_to_scale])

                st.write("#### Scaled DataFrame Preview")
                st.dataframe(scaled_df.head())

                st.write("#### Scaling Statistics")
                st.write(scaled_df.describe())

                return scaled_df
            except Exception as e:
                st.error(f"An error occurred during scaling: {str(e)}")
                return final_df
        else:
            st.error("No data available for feature scaling.")
            return None
        

    def category_analysis(self):
        """Performs category analysis on the uploaded data"""
        if st.checkbox("Perform category analysis?"):
            st.balloons()
            category_column = st.selectbox("Select category column", self.data.columns)
            st.write("Category Analysis:")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x=category_column, y=self.target_column, data=self.data, estimator=sum, ci=None, palette='viridis')
            plt.title(f'Sales by {category_column}')
            st.pyplot(fig)

    def weather_data_integration(self):
        """Integrates weather data with the existing data and performs analysis"""
        if st.checkbox("Do you have a weather dataset?"):
            st.balloons()
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
                self.data.reset_index(inplace=True)

                # Merge with sales data
                combined_df = pd.merge(self.data, weather_df, left_on=self.date_column, right_on=weather_date_column)
                st.write(weather_df.dtypes)

                # Display the combined dataframe
                st.write("Combined DataFrame:")
                st.write(combined_df)

                # Perform analysis on combined data (for example, plotting trends with weather)
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.lineplot(data=combined_df, x=combined_df[self.date_column], y=self.target_column, label='Sales')
                sns.lineplot(data=combined_df, x=combined_df[self.date_column], y='Temperature (C)', label='Temperature', ax=ax)
                plt.title(f'{self.target_column} and Temperature Trends')
                plt.xlabel('Date')
                plt.ylabel('Value')
                plt.legend()
                st.pyplot(fig)
                
    def forecasting(self):
        """Performs bidirectional LSTM-based forecasting on the uploaded data."""
        if self.data is not None:
            target_column = self.target_column
            # Resample data to weekly frequency
            df_resampled = self.data[target_column].resample('W').sum()
            # Prepare data for LSTM
            data = df_resampled.values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)
            # Define a time step
            time_step = 10  # Using 10 weeks as the time step for better context

            def create_dataset(dataset, time_step=1):
                X, Y = [], []
                for i in range(len(dataset) - time_step):
                    X.append(dataset[i:i + time_step, 0])
                    Y.append(dataset[i + time_step, 0])
                return np.array(X), np.array(Y)

            # Split data into training and testing sets
            train_size = int(len(scaled_data) * 0.8)
            train, test = scaled_data[:train_size], scaled_data[train_size:]
            # Create dataset using the training data
            X_train, y_train = create_dataset(train, time_step)
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            # Create dataset using the test data
            X_test, y_test = create_dataset(test, time_step)
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

            # Build and train the bidirectional LSTM model
            model = Sequential()
            model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=(time_step, 1)))
            model.add(Dropout(0.2))  # Prevent overfitting
            model.add(Bidirectional(LSTM(100, return_sequences=False)))
            model.add(Dropout(0.2))
            model.add(Dense(25, activation='relu'))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, batch_size=16, epochs=100, verbose=1)

            # Make predictions on the training data
            train_predictions = model.predict(X_train)
            train_predictions = scaler.inverse_transform(train_predictions)
            # Make predictions on the test data
            test_predictions = model.predict(X_test)
            test_predictions = scaler.inverse_transform(test_predictions)
            
            # Inverse transform predictions to original scale
            # Adjust the predictions index for plotting
            predicted_index_train = df_resampled.index[time_step:train_size]
            predicted_index_test = df_resampled.index[train_size + time_step:]

            # Prepare for future predictions
            future_weeks = 32  # Define how many weeks to predict
            future_predictions = []
            # Use the full history of scaled data for future predictions
            last_known_data = scaled_data[-time_step:].tolist()  # Convert to list for appending
            for _ in range(future_weeks):
                # Prepare the current batch for prediction
                current_batch = np.array(last_known_data[-time_step:]).reshape((1, time_step, 1))
                future_pred = model.predict(current_batch)[0]
                future_predictions.append(future_pred[0])  # Append the prediction
                last_known_data.append(future_pred)  # Add the prediction to the history for the next step
            future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

            # Generate future dates
            last_date = df_resampled.index[-1]
            future_dates = pd.date_range(last_date, periods=future_weeks + 1, freq='W')[1:]

            # Prepare future predictions DataFrame
            future_df = pd.DataFrame(future_predictions, index=future_dates, columns=['Forecast'])

            # Prepare data for Plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_resampled.index, y=data.flatten(), mode='lines', name='Original Sales', line=dict(color='black')))
            fig.add_trace(go.Scatter(x=predicted_index_train, y=train_predictions.flatten(), mode='lines', name='Train Predictions', line=dict(color='#FF5733')))  # Train predictions
            fig.add_trace(go.Scatter(x=predicted_index_test, y=test_predictions.flatten(), mode='lines', name='Test Predictions', line=dict(color='#3357FF')))  # Test predictions
            fig.add_trace(go.Scatter(x=future_df.index, y=future_df['Forecast'], mode='lines', name='Future Predictions', line=dict(color='#75FF33')))  # Future predictions

            # Update layout
            fig.update_layout(title='Sales Forecasting using Bidirectional LSTM',
                            xaxis_title='Date',
                            yaxis_title='Sales',
                            legend_title='Legend',
                            template='plotly_white')

            # Display plot
            st.plotly_chart(fig)

    def forecasting1(self):
        """Performs bidirectional LSTM-based forecasting on the uploaded data."""
        if self.data is not None:
            target_column = self.target_column
            # Resample data to weekly frequency
            df_resampled = self.data[target_column].resample('W').sum()
            # Prepare data for LSTM
            data = df_resampled.values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)

            # Ask user for the time step
            time_step = st.number_input("Enter the time step (number of weeks) for LSTM input:", min_value=1, value=10)

            def create_dataset(dataset, time_step=1):
                X, Y = [], []
                for i in range(len(dataset) - time_step):
                    X.append(dataset[i:i + time_step, 0])
                    Y.append(dataset[i + time_step, 0])
                return np.array(X), np.array(Y)

            # Split data into training and testing sets
            train_size = int(len(scaled_data) * 0.8)
            train, test = scaled_data[:train_size], scaled_data[train_size:]
            # Create dataset using the training data
            X_train, y_train = create_dataset(train, time_step)
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            # Create dataset using the test data
            X_test, y_test = create_dataset(test, time_step)
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

            # Ask user for future weeks
            future_weeks = st.number_input("How many weeks into the future would you like to predict?", min_value=1, value=32)

            # Ask user for colors
            original_color = st.color_picker("Pick a color for Original Data", "#FF7")
            train_color = st.color_picker("Pick a color for Train Predictions", "#FF5733")
            test_color = st.color_picker("Pick a color for Test Predictions", "#3357FF")
            future_color = st.color_picker("Pick a color for Future Predictions", "#75FF33")

            if st.button("Train LSTM and Forecast Sales"):
                # Build and train the bidirectional LSTM model
                model = Sequential()
                model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=(time_step, 1)))
                model.add(Dropout(0.2))
                model.add(Bidirectional(LSTM(100, return_sequences=False)))
                model.add(Dropout(0.2))
                model.add(Dense(25, activation='relu'))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(X_train, y_train, batch_size=16, epochs=100, verbose=1)

                # Make predictions on the training data
                train_predictions = model.predict(X_train)
                train_predictions = scaler.inverse_transform(train_predictions)
                # Make predictions on the test data
                test_predictions = model.predict(X_test)
                test_predictions = scaler.inverse_transform(test_predictions)

                # Inverse transform predictions to original scale
                predicted_index_train = df_resampled.index[time_step:train_size]
                predicted_index_test = df_resampled.index[train_size + time_step:]

                # Prepare for future predictions
                future_predictions = []
                last_known_data = scaled_data[-time_step:].tolist()
                for _ in range(future_weeks):
                    current_batch = np.array(last_known_data[-time_step:]).reshape((1, time_step, 1))
                    future_pred = model.predict(current_batch)[0]
                    future_predictions.append(future_pred[0])
                    last_known_data.append(future_pred)
                future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

                # Generate future dates
                last_date = df_resampled.index[-1]
                future_dates = pd.date_range(last_date, periods=future_weeks + 1, freq='W')[1:]

                # Prepare future predictions DataFrame
                future_df = pd.DataFrame(future_predictions, index=future_dates, columns=['Forecast'])

                # Prepare data for Plotly
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_resampled.index, y=data.flatten(), mode='lines', name='Original Sales', line=dict(color=original_color)))
                fig.add_trace(go.Scatter(x=predicted_index_train, y=train_predictions.flatten(), mode='lines', name='Train Predictions', line=dict(color=train_color)))
                fig.add_trace(go.Scatter(x=predicted_index_test, y=test_predictions.flatten(), mode='lines', name='Test Predictions', line=dict(color=test_color)))
                fig.add_trace(go.Scatter(x=future_df.index, y=future_df['Forecast'], mode='lines', name='Future Predictions', line=dict(color=future_color)))

                # Update layout
                fig.update_layout(
                    title='Sales Forecasting using Bidirectional LSTM',
                    xaxis_title='Date',
                    yaxis_title='Sales',
                    legend_title='Legend',
                    template='plotly_white'
                )

                # Display plot
                st.plotly_chart(fig)

    def forecasting2(self):
        """Improved Bidirectional LSTM forecasting with optimized structure and parameters."""
        if self.data is not None:
            target_column = self.target_column
            # Resample data to weekly frequency
            df_resampled = self.data[target_column].resample('W').sum()
            
            # Prepare data for LSTM
            data = df_resampled.values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)

            # Set time step
            time_step = st.number_input("Enter the time step (number of weeks) for LSTM input:", min_value=1, value=10)

            def create_dataset(dataset, time_step=1):
                X, Y = [], []
                for i in range(len(dataset) - time_step):
                    X.append(dataset[i:i + time_step, 0])
                    Y.append(dataset[i + time_step, 0])
                return np.array(X), np.array(Y)

            # Split data into training and testing sets
            train_size = int(len(scaled_data) * 0.85)  # Using 85% of data for training
            train, test = scaled_data[:train_size], scaled_data[train_size:]
            
            # Create dataset using the training data
            X_train, y_train = create_dataset(train, time_step)
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            
            # Create dataset using the test data
            X_test, y_test = create_dataset(test, time_step)
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

            # Number of future weeks for prediction
            future_weeks = st.number_input("How many weeks into the future would you like to predict?", min_value=1, value=32)

            # Colors for plotting
            original_color = st.color_picker("Pick a color for Original Data", "#FF7")
            train_color = st.color_picker("Pick a color for Train Predictions", "#FF5733")
            test_color = st.color_picker("Pick a color for Test Predictions", "#3357FF")
            future_color = st.color_picker("Pick a color for Future Predictions", "#75FF33")

            if st.button("Train Enhanced LSTM and Forecast Sales"):
                # Model architecture with optimized parameters
                model = Sequential([
                    Bidirectional(LSTM(128, return_sequences=True), input_shape=(time_step, 1)),
                    Dropout(0.3),
                    Bidirectional(LSTM(64, return_sequences=True)),
                    Dropout(0.3),
                    Bidirectional(LSTM(32)),
                    Dense(1)
                ])

                model.compile(optimizer='adam', loss='mean_squared_error')
                
                # Early stopping callback
                early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

                # Train model with validation data
                model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test), 
                        callbacks=[early_stopping], verbose=1)

                # Make predictions on training and test data
                train_predictions = model.predict(X_train)
                train_predictions = scaler.inverse_transform(train_predictions)
                test_predictions = model.predict(X_test)
                test_predictions = scaler.inverse_transform(test_predictions)

                # Inverse transform for original scale
                predicted_index_train = df_resampled.index[time_step:train_size]
                predicted_index_test = df_resampled.index[train_size + time_step:]

                # Prepare future predictions
                future_predictions = []
                last_known_data = scaled_data[-time_step:].tolist()
                for _ in range(future_weeks):
                    current_batch = np.array(last_known_data[-time_step:]).reshape((1, time_step, 1))
                    future_pred = model.predict(current_batch)[0]
                    future_predictions.append(future_pred[0])
                    last_known_data.append(future_pred)
                future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

                # Generate future dates
                last_date = df_resampled.index[-1]
                future_dates = pd.date_range(last_date, periods=future_weeks + 1, freq='W')[1:]
                future_df = pd.DataFrame(future_predictions, index=future_dates, columns=['Forecast'])

                # Plot results using Plotly
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_resampled.index, y=data.flatten(), mode='lines', name='Original Sales', line=dict(color=original_color)))
                fig.add_trace(go.Scatter(x=predicted_index_train, y=train_predictions.flatten(), mode='lines', name='Train Predictions', line=dict(color=train_color)))
                fig.add_trace(go.Scatter(x=predicted_index_test, y=test_predictions.flatten(), mode='lines', name='Test Predictions', line=dict(color=test_color)))
                fig.add_trace(go.Scatter(x=future_df.index, y=future_df['Forecast'], mode='lines', name='Future Predictions', line=dict(color=future_color)))

                fig.update_layout(
                    title='Enhanced Sales Forecasting with Bidirectional LSTM',
                    xaxis_title='Date',
                    yaxis_title='Sales',
                    legend_title='Legend',
                    template='plotly_white'
                )

                st.plotly_chart(fig)
                
    def forecasting3(self):
        """Improved Bidirectional GRU forecasting with optimized structure and parameters."""
        if self.data is not None:
            target_column = self.target_column
            # Resample data to weekly frequency
            df_resampled = self.data[target_column].resample('W').sum()
            
            # Prepare data for GRU
            data = df_resampled.values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)

            # Set time step
            time_step = st.number_input("Enter the time step (number of weeks) for GRU input:", min_value=1, value=10)

            def create_dataset(dataset, time_step=1):
                X, Y = [], []
                for i in range(len(dataset) - time_step):
                    X.append(dataset[i:i + time_step, 0])
                    Y.append(dataset[i + time_step, 0])
                return np.array(X), np.array(Y)

            # Split data into training and testing sets
            train_size = int(len(scaled_data) * 0.85)  # Using 85% of data for training
            train, test = scaled_data[:train_size], scaled_data[train_size:]
            
            # Create dataset using the training data
            X_train, y_train = create_dataset(train, time_step)
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            
            # Create dataset using the test data
            X_test, y_test = create_dataset(test, time_step)
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

            # Number of future weeks for prediction
            future_weeks = st.number_input("How many weeks into the future would you like to predict?", min_value=1, value=32)

            # Colors for plotting
            original_color = st.color_picker("Pick a color for Original Data", "#FF7")
            train_color = st.color_picker("Pick a color for Train Predictions", "#FF5733")
            test_color = st.color_picker("Pick a color for Test Predictions", "#3357FF")
            future_color = st.color_picker("Pick a color for Future Predictions", "#75FF33")

            if st.button("Train Enhanced GRU and Forecast Sales"):
                # Model architecture with optimized parameters
                model = Sequential([
                    Bidirectional(GRU(128, return_sequences=True), input_shape=(time_step, 1)),
                    Dropout(0.3),
                    Bidirectional(GRU(64, return_sequences=True)),
                    Dropout(0.3),
                    Bidirectional(GRU(32)),
                    Dense(1)
                ])

                model.compile(optimizer='adam', loss='mean_squared_error')
                
                # Early stopping callback
                early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

                # Train model with validation data
                model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test), 
                        callbacks=[early_stopping], verbose=1)

                # Make predictions on training and test data
                train_predictions = model.predict(X_train)
                train_predictions = scaler.inverse_transform(train_predictions)
                test_predictions = model.predict(X_test)
                test_predictions = scaler.inverse_transform(test_predictions)

                # Inverse transform for original scale
                predicted_index_train = df_resampled.index[time_step:train_size]
                predicted_index_test = df_resampled.index[train_size + time_step:]

                # Prepare future predictions
                future_predictions = []
                last_known_data = scaled_data[-time_step:].tolist()
                for _ in range(future_weeks):
                    current_batch = np.array(last_known_data[-time_step:]).reshape((1, time_step, 1))
                    future_pred = model.predict(current_batch)[0]
                    future_predictions.append(future_pred[0])
                    last_known_data.append(future_pred)
                future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

                # Generate future dates
                last_date = df_resampled.index[-1]
                future_dates = pd.date_range(last_date, periods=future_weeks + 1, freq='W')[1:]
                future_df = pd.DataFrame(future_predictions, index=future_dates, columns=['Forecast'])

                # Plot results using Plotly
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_resampled.index, y=data.flatten(), mode='lines', name='Original Sales', line=dict(color=original_color)))
                fig.add_trace(go.Scatter(x=predicted_index_train, y=train_predictions.flatten(), mode='lines', name='Train Predictions', line=dict(color=train_color)))
                fig.add_trace(go.Scatter(x=predicted_index_test, y=test_predictions.flatten(), mode='lines', name='Test Predictions', line=dict(color=test_color)))
                fig.add_trace(go.Scatter(x=future_df.index, y=future_df['Forecast'], mode='lines', name='Future Predictions', line=dict(color=future_color)))

                fig.update_layout(
                    title='Enhanced Sales Forecasting with Bidirectional GRU',
                    xaxis_title='Date',
                    yaxis_title='Sales',
                    legend_title='Legend',
                    template='plotly_white'
                )

                st.plotly_chart(fig)
