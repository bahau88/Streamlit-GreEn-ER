import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from sklearn.model_selection import train_test_split

# Load the data
@st.cache
def load_data():
    url = "https://raw.githubusercontent.com/bahau88/G2Elab-Energy-Building-/main/dataset/combined_data_green-er_2020_2023.csv"
    df = pd.read_csv(url)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

# Preprocess the data
def preprocess_data(df, variables):
    X = df[variables].values
    Y = df['Consumption'].values
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X = (X - X_mean) / X_std
    return X, Y, X_mean, X_std

# Build the model
def build_model(input_dim):
    model = Sequential()
    model.add(Dense(12, input_dim=input_dim, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Predict consumption
def predict_consumption(num_hours, num_epochs, batch_size, variables):
    # Load the data
    data = load_data()
    X, Y, X_mean, X_std = preprocess_data(data, variables)

    # Split the data into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, shuffle=False)

    # Build the model
    model = build_model(X_train.shape[1])

    # Train the model
    history = model.fit(X_train, Y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_val, Y_val), verbose=0)

    # Generate the list of dates and hours to predict
    last_datetime = data.index.max()
    next_day = last_datetime + pd.DateOffset(hours=1)
    datetime_range = pd.date_range(next_day, periods=num_hours, freq='H')
    selected_datetimes = [str(d) for d in datetime_range]

    # Make predictions for the selected dates and hours
    input_data = np.zeros((num_hours, X.shape[1]))
    for i in range(num_hours):
        input_data[i] = [(variables[j] - X_mean[j]) / X_std[j] for j in range(len(variables))]

    predictions = model.predict(input_data)

    # Print the predictions
    for i in range(num_hours):
        st.write('Electricity consumption forecast for {}: {:.2f}'.format(selected_datetimes[i], predictions[i][0]))

    # Plot the true consumption values and the corresponding predicted values
    fig_tp = go.Figure()
    fig_tp.add_trace(go.Scatter(x=data.index, y=Y, name='True Consumption', line_color='orange'))
    fig_tp.add_trace(go.Scatter(x=datetime_range, y=predictions.flatten(), name='Predicted Consumption', line_color='red'))
    fig_tp.update_layout(title='True vs. Predicted Consumption',
                      plot_bgcolor='white',
                      xaxis_title='Datetime', yaxis_title='Consumption')
    st.plotly_chart(fig_tp)

    # Evaluate the model on the training set
    train_predictions = model.predict(X)
    rmse = sqrt(mean_squared_error(Y, train_predictions))
    mse = mean_squared_error(Y, train_predictions)
    mae = mean_absolute_error(Y, train_predictions)
    r2 = r2_score(Y, train_predictions)
    st.write('RMSE: {:.2f}'.format(rmse))
    st.write('MSE: {:.2f}'.format(mse))
    st.write('MAE: {:.2f}'.format(mae))
    st.write('R2 score: {:.2f}'.format(r2))

    # Plot the training and validation loss
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    fig_tvl = go.Figure()
    fig_tvl.add_trace(go.Scatter(x=list(range(1, len(train_loss) + 1)), y=train_loss, mode='lines', name='Training Loss'))
    fig_tvl.add_trace(go.Scatter(x=list(range(1, len(val_loss) + 1)), y=val_loss, mode='lines', name='Validation Loss'))
    fig_tvl.update_layout(title='Training and Validation Loss', plot_bgcolor='white', xaxis_title='Epoch', yaxis_title='Loss')
    st.plotly_chart(fig_tvl)

# Streamlit App
st.title('Electricity Consumption Forecast')

# Variable selection
variables = ['Number of Room', 'Dayindex', 'Occupants', 'Temperature', 'Cloudcover', 'Visibility']
selected_variables = st.multiselect('Select variables', variables, default=variables)

# Model parameters
num_epochs = st.slider('Number of Epochs', min_value=1, max_value=10, value=2)
batch_size = st.slider('Batch Size', min_value=1, max_value=32, value=10)

# Prediction
num_hours = 24
if st.button('Predict'):
    predict_consumption(num_hours, num_epochs, batch_size, selected_variables)
