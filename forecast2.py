import streamlit as st
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import plotly.graph_objects as go

# Load the data
@st.cache
def load_data():
    merged_df = pd.read_csv("https://raw.githubusercontent.com/bahau88/G2Elab-Energy-Building-/main/dataset/combined_data_green-er_2020_2023.csv")
    merged_df['Date'] = pd.to_datetime(merged_df['Date'])
    merged_df.set_index('Date', inplace=True)
    return merged_df

merged_df = load_data()

# Define the prediction function
def predict_consumption(num_hours, num_epochs, batch_size, variables):
    data = merged_df.copy()
    data.index.names = ['Datetime']

    # Prepare selected variables
    selected_variables = ['Number of Room', 'Dayindex', 'Occupants', 'Temperature', 'Cloudcover', 'Visibility']
    selected_variables = [var for var in selected_variables if var in variables]

    # Split the data into input (X) and output (Y) variables
    X = data[selected_variables].values
    Y = data['Consumption'].values

    # Normalize the input data
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X = (X - X_mean) / X_std

    # Define the model
    model = Sequential()
    model.add(Dense(12, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Split the data into training and validation sets
    train_X = X[:-24]
    train_Y = Y[:-24]
    val_X = X[-24:]
    val_Y = Y[-24:]

    # Train the model
    model.fit(train_X, train_Y, epochs=num_epochs, batch_size=batch_size, verbose=2, validation_data=(val_X, val_Y))

    # Generate the list of dates and hours to predict
    last_datetime = data.index.max()
    next_day = last_datetime + pd.DateOffset(hours=1)
    datetime_range = pd.date_range(next_day, periods=num_hours, freq='H')
    selected_datetimes = [str(d) for d in datetime_range]

    # Prepare input data for prediction
    input_data = np.zeros((num_hours, X.shape[1]))
    numberofroom_arr = [0, 0, 0, 0, 0, 0, 0, 0, 21, 21, 21, 21, 5, 25, 25, 21, 19, 11, 2, 2, 0, 0, 0, 0]
    dayindex_arr = [0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635,  0.635,  0.635,  0.635, 0.635]
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

    # Make predictions
    predictions = model.predict(input_data)

    # Print the predictions
    for i in range(num_hours):
        st.write('Predicted consumption for {}: {:.2f}'.format(selected_datetimes[i], predictions[i][0]))

    # Plot the true consumption values and the corresponding predicted values
    train_predictions = model.predict(X)
    fig = plot_predictions(data, Y, train_predictions)
    st.plotly_chart(fig)


def plot_predictions(data, Y, train_predictions):
    fig_tp = go.Figure()
    fig_tp.add_trace(go.Scatter(x=data.index, y=Y, name='True Consumption', line_color='orange'))
    fig_tp.add_trace(go.Scatter(x=data.index, y=train_predictions.flatten(), name='Predicted Consumption', line_color='red'))
    fig_tp.update_layout(
        title='True vs. Predicted Consumption for Training Data',
        plot_bgcolor='white',
        xaxis_title='Date and Time',
        yaxis_title='Consumption'
    )
    return fig_tp


# Create a Streamlit web app
st.title('Energy Consumption Prediction')
num_hours = st.slider('Select the number of hours ahead to predict', 1, 48, 24)
num_epochs = st.slider('Select the number of epochs', 1, 10, 2)
batch_size = st.slider('Select the batch size', 1, 128, 32)
variables = st.multiselect('Select the variables to use for prediction', ['Number of Room', 'Dayindex', 'Occupants', 'Temperature', 'Cloudcover', 'Visibility'])

if st.button('Predict'):
    predict_consumption(num_hours, num_epochs, batch_size, variables)
