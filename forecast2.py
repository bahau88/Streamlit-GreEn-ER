import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

st.title("Energy Consumption Forecasting")

@st.cache
def load_data():
    merged_df = pd.read_csv("https://raw.githubusercontent.com/bahau88/G2Elab-Energy-Building-/main/dataset/combined_data_green-er_2020_2023.csv")
    merged_df['Date'] = pd.to_datetime(merged_df['Date'])
    merged_df.set_index('Date', inplace=True)
    return merged_df

merged_df = load_data()

merged_df['Datetime'] = merged_df.index
data = merged_df.copy()

data['Number of Room'] = data['Number of Room'].astype(float)
data['Dayindex'] = data['Dayindex'].astype(float)
data['Occupants'] = data['Occupants'].astype(float)
data['Temperature'] = data['Temperature'].astype(float)
data['Cloudcover'] = data['Cloudcover'].astype(float)
data['Visibility'] = data['Visibility'].astype(float)
data['Consumption'] = data['Consumption'].astype(float)

# Normalize the input data
X_mean = data[['Number of Room', 'Dayindex', 'Occupants', 'Temperature', 'Cloudcover', 'Visibility']].mean()
X_std = data[['Number of Room', 'Dayindex', 'Occupants', 'Temperature', 'Cloudcover', 'Visibility']].std()

data[['Number of Room', 'Dayindex', 'Occupants', 'Temperature', 'Cloudcover', 'Visibility']] = (
    data[['Number of Room', 'Dayindex', 'Occupants', 'Temperature', 'Cloudcover', 'Visibility']] - X_mean) / X_std

# Split the data into input (X) and output (Y) variables
X = data[['Number of Room', 'Dayindex', 'Occupants', 'Temperature', 'Cloudcover', 'Visibility']].values
Y = data['Consumption'].values

# Define the model
model = Sequential()
model.add(Dense(12, input_dim=X.shape[1], activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
def train_model(train_X, train_Y, val_X, val_Y, epochs, batch_size):
    model.fit(train_X, train_Y, epochs=epochs, batch_size=batch_size, verbose=0, validation_data=(val_X, val_Y))

# Initialize default values
num_hours = 24
epochs = 2
batch_size = 10

# Number of Hours selection
num_hours = st.slider("Number of Hours (1-24)", 1, 24, 24)

# Number of Epochs selection
epochs = st.slider("Number of Epochs (1-100)", 1, 100, 2)

# Batch Size selection
batch_size = st.selectbox("Batch Size", [10, 15, 20])

# Train the model
train_X = X[:-num_hours]
train_Y = Y[:-num_hours]
val_X = X[-num_hours:]
val_Y = Y[-num_hours:]
train_model(train_X, train_Y, val_X, val_Y, epochs, batch_size)

# Generate the list of dates and hours to predict
last_datetime = data['Datetime'].max()
next_day = last_datetime + pd.DateOffset(hours=1)
datetime_range = pd.date_range(next_day, periods=num_hours, freq='H')
selected_datetimes = [str(d) for d in datetime_range]

# Make predictions for the selected dates and hours
input_data = np.zeros((num_hours, X.shape[1]))
numberofroom_arr = [0, 0, 0, 0, 0, 0, 0, 0, 21, 21, 21, 21, 5, 25, 25, 21, 19, 11, 2, 2, 0, 0, 0, 0]
dayindex_arr = [0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635,  0.635,  0.635, 0.635]
occupants_arr = [0, 0, 0, 0, 0, 0, 0, 0, 923, 923, 923, 923, 633, 1068, 1068, 964, 908, 791, 371, 371, 0, 0, 0, 0]
temperature_arr = [7.6, 6.8, 5.9, 4.6, 4.4, 4.2, 3.7, 3.1, 5.2, 9.2, 11.6, 13.1, 14.9, 16.9, 18, 19.4, 20.8, 21.1, 21, 18.5, 17.5, 15.6, 14, 12.8]
cloudcover_arr = [60, 40, 0.7, 0, 80, 80, 60, 80, 50, 50, 60 , 50, 60, 60, 80, 80, 90, 100, 94.3, 95.7, 90, 96.3, 98.9, 96.3]
visibility_arr = [33.2, 25.2, 24.4, 19.7, 16.5, 20, 16.2, 14.9, 15.3, 23.9, 23.9, 24.5, 17.6, 29.9, 33.1, 19.2, 33.2, 30.7, 34.8, 38.2, 28.8, 26.3, 38.2, 34.9]

for i in range(num_hours):
    numberofroom = numberofroom_arr[i]
    dayindex = dayindex_arr[i]
    occupants = occupants_arr[i]
    temperature = temperature_arr[i]
    cloudcover = cloudcover_arr[i]
    visibility = visibility_arr[i]
    input_data[i] = [numberofroom, dayindex, occupants, temperature, cloudcover, visibility]

input_data = (input_data - X_mean) / X_std
predictions = model.predict(input_data)

# Display the predictions
st.header("Energy Consumption Predictions")

for i in range(num_hours):
    st.write("Predicted consumption for {}: {:.2f}".format(selected_datetimes[i], predictions[i][0]))
