import streamlit as st
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

# Load the data
@st.cache  # Add caching for improved performance
def load_data():
    merged_df = pd.read_csv("https://raw.githubusercontent.com/bahau88/G2Elab-Energy-Building-/main/dataset/combined_data_green-er_2020_2023.csv")
    return merged_df

merged_df = load_data()

# Convert the date and hour columns to datetime format
# merged_df['Datetime'] = pd.to_datetime(merged_df['Datetime'])

# Split the data into input (X) and output (Y) variables
X = merged_df[['Number of Room', 'Dayindex', 'Occupants', 'Temperature', 'Cloudcover', 'Visibility']].values
Y = merged_df['Consumption'].values

# Normalize the input data
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

# Define the model
model = Sequential()
model.add(Dense(12, input_dim=X.shape[1], activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
train_X = X[:-24]
train_Y = Y[:-24]
val_X = X[-24:]
val_Y = Y[-24:]
model.fit(train_X, train_Y, epochs=2, batch_size=10, verbose=2, validation_data=(val_X, val_Y))

# Define a function to make predictions
def make_predictions(input_data):
    input_data = (input_data - X_mean) / X_std
    predictions = model.predict(input_data)
    return predictions

# Streamlit app
st.title('Energy Consumption Prediction')

# Display user inputs and predictions
num_hours = st.slider('How many hours ahead would you like to predict?', 1, 48, 24)

# Generate input data for predictions
numberofroom_arr = st.text_input('Number of rooms (comma-separated values)', '0,0,0,0,0,0,0,0,21,21,21,21,5,25,25,21,19,11,2,2,0,0,0,0')
dayindex_arr = st.text_input('Day index (comma-separated values)', '0.635,0.635,0.635,0.635,0.635,0.635,0.635,0.635,0.635,0.635,0.635,0.635,0.635,0.635,0.635,0.635,0.635,0.635,0.635,0.635,0.635,0.635,0.635,0.635')
occupants_arr = st.text_input('Number of occupants (comma-separated values)', '0,0,0,0,0,0,0,0,923,923,923,923,633,1068,1068,964,908,791,371,371,0,0,0,0')
temperature_arr = st.text_input('Temperature (comma-separated values)', '7.6,6.8,5.9,4.6,4.4,4.2,3.7,3.1,5.2,9.2,11.6,13.1,14.9,16.9,18,19.4,20.8,21.1,21,18.5,17.5,15.6,14,12.8')
cloudcover_arr = st.text_input('Cloud cover (comma-separated values)', '60,40,0.7,0,80,80,60,80,50,50,60,50,60,60,80,80,90,100,94.3,95.7,90,96.3,98.9,96.3')
visibility_arr = st.text_input('Visibility (comma-separated values)', '33.2,25.2,24.4,19.7,16.5,20,16.2,14.9,15.3,23.9,23.9,24.5,17.6,29.9,33.1,19.2,33.2,30.7,34.8,38.2,28.8,26.3,38.2,34.9')

input_data = np.column_stack((
    np.fromstring(numberofroom_arr, dtype=int, sep=','),
    np.fromstring(dayindex_arr, dtype=float, sep=','),
    np.fromstring(occupants_arr, dtype=int, sep=','),
    np.fromstring(temperature_arr, dtype=float, sep=','),
    np.fromstring(cloudcover_arr, dtype=float, sep=','),
    np.fromstring(visibility_arr, dtype=float, sep=',')
))

if st.button('Predict'):
    predictions = make_predictions(input_data)
    selected_datetimes = pd.date_range(merged_df['Datetime'].max(), periods=num_hours+1, freq='H')[1:]
    
    for i, dt in enumerate(selected_datetimes):
        st.write('Predicted consumption for {}: {:.2f}'.format(dt, predictions[i][0]))
