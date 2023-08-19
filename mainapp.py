import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import plotly.graph_objects as go
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots
from scipy import stats
import streamlit as st
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from sklearn.model_selection import train_test_split



# DATA VISUALIZATION #--------------------------------------------------------------------------------------------------------------------
# Load the data
merged_df = pd.read_csv('https://raw.githubusercontent.com/bahau88/G2Elab-Energy-Building-/main/dataset/combined_data_green-er_2020_2023.csv') 

# Convert the date column to a datetime object
merged_df['Date'] = pd.to_datetime(merged_df['Date'])

# Create the figure and traces
fig = go.Figure()

fig.add_trace(go.Scatter(x=merged_df['Date'], y=merged_df['Consumption'], name='Consumption',
                         line=dict(color='red', width=2), visible=True))
fig.add_trace(go.Scatter(x=merged_df['Date'], y=merged_df['Other'], name='Other',
                        line=dict(color='blue', width=2), yaxis='y2', visible=True))
fig.add_trace(go.Scatter(x=merged_df['Date'], y=merged_df['Heating'], name='Heating',
                        line=dict(color='orange', width=2), yaxis='y3', visible=True))
fig.add_trace(go.Scatter(x=merged_df['Date'], y=merged_df['Lighting'], name='Lighting',
                        line=dict(color='green', width=2), yaxis='y3', visible=True))

# Set the axis titles
fig.update_layout(
    xaxis=dict(title='Date'),
    yaxis=dict(title='Consumption', titlefont=dict(color='red')),
    yaxis2=dict(title='Other', titlefont=dict(color='blue'), overlaying='y', side='right'),
    yaxis3=dict(title='Heating', titlefont=dict(color='orange'), overlaying='y', side='right'),
    yaxis4=dict(title='Lighting', titlefont=dict(color='green'), overlaying='y', side='right'),
    plot_bgcolor='white'
)

# Add hover information
fig.update_traces(hovertemplate='%{y:.2f}')

#--------------------------------------------------------------------------------------------------------------------
# DATA ANALYSIS ------------------------------------------------------------------------------------------------------------------------

# Load data
merged_df = pd.read_csv('https://raw.githubusercontent.com/bahau88/G2Elab-Energy-Building-/main/dataset/combined_data_green-er_2020_2023.csv') 
df_boxplot = merged_df.copy()

df_boxplot = df_boxplot.set_index('Date')
df_boxplot.index = pd.to_datetime(df_boxplot.index)

# Create dataframes for each season (adjust the date ranges as needed)
# Create dataframes for each season (adjust the date ranges as needed)
df_spring = pd.concat([df_boxplot[(df_boxplot.index >= '2021-03-01') & (df_boxplot.index <= '2021-05-31')],
                       df_boxplot[(df_boxplot.index >= '2022-03-01') & (df_boxplot.index <= '2022-05-31')],
                       df_boxplot[(df_boxplot.index >= '2023-03-01') & (df_boxplot.index <= '2023-04-26')]])
df_summer = pd.concat([df_boxplot[(df_boxplot.index >= '2020-08-24') & (df_boxplot.index <= '2020-08-31')],
                       df_boxplot[(df_boxplot.index >= '2021-06-01') & (df_boxplot.index <= '2021-08-31')],
                       df_boxplot[(df_boxplot.index >= '2022-06-01') & (df_boxplot.index <= '2022-08-31')]])

df_autumn = pd.concat([df_boxplot[(df_boxplot.index >= '2020-09-01') & (df_boxplot.index <= '2020-11-30')],
                       df_boxplot[(df_boxplot.index >= '2021-09-01') & (df_boxplot.index <= '2021-11-30')],
                       df_boxplot[(df_boxplot.index >= '2022-09-01') & (df_boxplot.index <= '2022-11-30')]])

df_winter = pd.concat([df_boxplot[(df_boxplot.index >= '2020-12-01') & (df_boxplot.index <= '2021-02-28')],
                       df_boxplot[(df_boxplot.index >= '2021-12-01') & (df_boxplot.index <= '2022-02-28')],
                       df_boxplot[(df_boxplot.index >= '2022-12-01') & (df_boxplot.index <= '2023-02-28')]])

# Create a list of day names to use for X axis labels
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Create a function to generate a Plotly figure for each season
def create_season_boxplot(df, season_name, marker_color):
    fig_season = go.Figure()
    fig_season.add_trace(go.Box(x=df.index.day_name(), y=df['Consumption'], name=season_name,
                         boxmean='sd', marker_color=marker_color))
    fig_season.update_layout(
        xaxis=dict(title='Day of the Week'),
        yaxis=dict(title='Consumption'),
        title_text=f"Consumption in {season_name}",
        boxmode='group',
        showlegend=False
    )
    return fig_season
#---------------------------------------------------------------------------------------------------------------------------------------------------

# FEATURE IMPORTANCE #--------------------------------------------------------------------------------------------------------------------
# Load the data
merged_df = pd.read_csv("https://raw.githubusercontent.com/bahau88/G2Elab-Energy-Building-/main/dataset/combined_data_green-er_2020_2023.csv")

# Define features and target
features = ['Number of Room', 'Dayindex', 'Occupants', 'Temperature', 'Cloudcover', 'Visibility']
target = 'Consumption'

# Function to train and evaluate the model
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return model, y_pred, r2

# Function to plot feature importances
def plot_feature_importances(features, importances):
    fig = go.Figure([go.Bar(x=features, y=importances, text=importances, textposition='auto', textfont=dict(size=12))])
    fig.update_layout(
        title='Feature Importances',
        xaxis_title='Feature',
        yaxis_title='Importance',
        font=dict(size=12),
        plot_bgcolor='white',
        xaxis=dict(tickangle=45)
    )
    st.plotly_chart(fig)
    
#--------------------------------------------------------------------------------------------------------------------
# FFORECAST FNN #--------------------------------------------------------------------------------------------------------------------
# Load the data
@st.cache
def load_data():
    merged_df2 = pd.read_csv("https://raw.githubusercontent.com/bahau88/G2Elab-Energy-Building-/main/dataset/combined_data_green-er_2020_2023.csv")
    merged_df2['Date'] = pd.to_datetime(merged_df2['Date'])
    merged_df2.set_index('Date', inplace=True)
    return merged_df2

merged_df2 = load_data()

# Define the prediction function
def predict_consumption(num_hours, num_epochs, batch_size, variables):
    data = merged_df2.copy()
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
    history = model.fit(train_X, train_Y, epochs=num_epochs, batch_size=batch_size, verbose=2, validation_data=(val_X, val_Y))

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
        
    # Evaluate the model on the training set
    rmse = sqrt(mean_squared_error(Y, model.predict(X)))
    mse = mean_squared_error(Y, model.predict(X))
    mae = mean_absolute_error(Y, model.predict(X))
    r2 = r2_score(Y, model.predict(X))
    st.write('RMSE: {:.2f}'.format(rmse))
    st.write('MSE: {:.2f}'.format(mse))
    st.write('MAE: {:.2f}'.format(mae))
    st.write('R2 score: {:.2f}'.format(r2))

    # Plot the true consumption values and the corresponding predicted values
    train_predictions = model.predict(X)
    fig = plot_predictions(data, Y, train_predictions)
    st.plotly_chart(fig)
    
    # Show the chart of the last three days and the predicted days
    last_three_days = data.iloc[-24:]
    predicted_days = pd.DataFrame(predictions, columns=['Consumption'], index=datetime_range)

    fig_prediction = go.Figure()
    fig_prediction.add_trace(go.Bar(x=last_three_days.index, y=last_three_days['Consumption'], name='Previous days'))
    fig_prediction.add_trace(go.Bar(x=predicted_days.index, y=predicted_days['Consumption'], name='Predicted days'))
    fig_prediction.update_layout(title='Electricity consumption forecast', plot_bgcolor='white', xaxis_title='Date', yaxis_title='Electricity consumption')
    st.plotly_chart(fig_prediction)

    # Plot the training and validation loss
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    fig_tvl = go.Figure()
    fig_tvl.add_trace(go.Scatter(
        x=list(range(1, len(train_loss) + 1)),
        y=train_loss,
        mode='lines',
        name='Training Loss'
    ))
    fig_tvl.add_trace(go.Scatter(
        x=list(range(1, len(val_loss) + 1)),
        y=val_loss,
        mode='lines',
        name='Validation Loss'
    ))
    fig_tvl.update_layout(
        title='Training and Validation Loss',
        plot_bgcolor='white',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    st.plotly_chart(fig_tvl)


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

#--------------------------------------------------------------------------------------------------------------------

# FFORECAST LSTM #--------------------------------------------------------------------------------------------------------------------
# Load the data
@st.cache
def load_data():
    merged_df2 = pd.read_csv("https://raw.githubusercontent.com/bahau88/G2Elab-Energy-Building-/main/dataset/combined_data_green-er_2020_2023.csv")
    merged_df2['Date'] = pd.to_datetime(merged_df2['Date'])
    merged_df2.set_index('Date', inplace=True)
    return merged_df2

merged_df2 = load_data()

# Define the prediction function
def predict_consumption2(num_hours, num_epochs, batch_size, variables):
    data = merged_df2.copy()
    data.index.names = ['Datetime']
    
    # Split the data into input (X) and output (Y) variables
    X = data[['Number of Room', 'Occupants', 'Temperature', 'Dayindex', 'Cloudcover', 'Visibility']].values
    #X = data[['Temperature', 'Dayindex' , 'Cloudcover', 'Visibility', 'Solarradiation']].values
    Y = data['Consumption'].values
    
    # Normalize the input data
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X = (X - X_mean) / X_std
    
    # Reshape input data for LSTM
    X = X.reshape(X.shape[0], 1, X.shape[1])
    
    # Split the data into training, validation, and testing sets
    train_X, temp_X, train_Y, temp_Y = train_test_split(X, Y, test_size=0.2, random_state=42)
    val_X, test_X, val_Y, test_Y = train_test_split(temp_X, temp_Y, test_size=0.5, random_state=42)
    
    # Define the model
    model = Sequential()
    model.add(LSTM(12, input_shape=(1, X.shape[2]), activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='linear'))
    
    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    # Train the model and store the history object
    history = model.fit(train_X, train_Y, epochs=50, batch_size=10, verbose=2, validation_data=(val_X, val_Y), shuffle=False)
    
    # Ask the user how many hours ahead to predict
    num_hours = 24
    
    # Generate the list of dates and hours to predict
    last_datetime = data.index.max()
    next_day = last_datetime + pd.DateOffset(hours=1)
    datetime_range = pd.date_range(next_day, periods=num_hours, freq='H')
    selected_datetimes = [str(d) for d in datetime_range]
    
    # Make predictions for the selected dates and hours
    input_data = np.zeros((num_hours, X.shape[2]))
    numberofroom_arr = [0, 0, 0, 0, 0, 0, 0, 0, 21, 21, 21, 21, 5, 25, 25, 21, 19, 11, 2, 2, 0, 0, 0, 0]  # input values for number of rooms
    events_arr =       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
    dayindex_arr = [0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635,  0.635,  0.635,  0.635, 0.635]  # input values for day index
    occupants_arr = [0, 0, 0, 0, 0, 0, 0, 0, 923, 923, 923, 923, 633, 1068, 1068, 964, 908, 791, 371, 371, 0, 0, 0, 0]  # input values for number of occupants
    temperature_arr = [7.6, 6.8, 5.9, 4.6, 4.4, 4.2, 3.7, 3.1, 5.2, 9.2, 11.6, 13.1, 14.9, 16.9, 18, 19.4, 20.8, 21.1, 21, 18.5, 17.5, 15.6, 14, 12.8]  # input values for number of temperature
    cloudcover_arr = [60, 40, 0.7, 0, 80, 80, 60, 80, 50, 50, 60 , 50, 60, 60, 80, 80, 90, 100, 94.3, 95.7, 90, 96.3, 98.9, 96.3]  # input values for number of cloudcover
    visibility_arr = [33.2, 25.2, 24.4, 19.7, 16.5, 20, 16.2, 14.9, 15.3, 23.9, 23.9, 24.5, 17.6, 29.9, 33.1, 19.2, 33.2, 30.7, 34.8, 38.2, 28.8, 26.3, 38.2, 34.9]  # input values for number of visibility
    solarradiation_arr = [0, 0, 0, 0, 0, 0, 0, 19, 190, 373, 589, 744, 856, 896, 930, 803, 780, 587, 153, 113, 25, 14, 0, 0]  # input values for number of solarradiation
    
    
    
    for i in range(num_hours):
        numberofroom = numberofroom_arr[i]
        events= events_arr[i]
        dayindex = dayindex_arr[i]
        occupants = occupants_arr[i]
        temperature =temperature_arr[i]
        cloudcover = cloudcover_arr[i]
        visibility= visibility_arr[i]
        solarradiation = solarradiation_arr[i]
        input_data[i] = [numberofroom, occupants, temperature,  dayindex, cloudcover, visibility,]
        #input_data[i] = [temperature, dayindex, cloudcover, visibility, solarradiation]
    
    input_data = (input_data - X_mean) / X_std
    input_data = input_data.reshape(input_data.shape[0], 1, input_data.shape[1])
    predictions = model.predict(input_data)

    # Print the predictions
    for i in range(num_hours):
        st.write('Predicted consumption for {}: {:.2f}'.format(selected_datetimes[i], predictions[i][0]))
        
    # Evaluate the model on the training set
    rmse = sqrt(mean_squared_error(Y, model.predict(X)))
    mse = mean_squared_error(Y, model.predict(X))
    mae = mean_absolute_error(Y, model.predict(X))
    r2 = r2_score(Y, model.predict(X))
    st.write('RMSE: {:.2f}'.format(rmse))
    st.write('MSE: {:.2f}'.format(mse))
    st.write('MAE: {:.2f}'.format(mae))
    st.write('R2 score: {:.2f}'.format(r2))

    # Plot the true consumption values and the corresponding predicted values
    train_predictions = model.predict(X)
    fig = plot_predictions2(data, Y, train_predictions)
    st.plotly_chart(fig)
    
    # Show the chart of the last three days and the predicted days
    last_three_days = data.iloc[-24:]
    predicted_days = pd.DataFrame(predictions, columns=['Consumption'], index=datetime_range)

    fig_prediction = go.Figure()
    fig_prediction.add_trace(go.Bar(x=last_three_days.index, y=last_three_days['Consumption'], name='Previous days'))
    fig_prediction.add_trace(go.Bar(x=predicted_days.index, y=predicted_days['Consumption'], name='Predicted days'))
    fig_prediction.update_layout(title='Electricity consumption forecast', plot_bgcolor='white', xaxis_title='Date', yaxis_title='Electricity consumption')
    st.plotly_chart(fig_prediction)

    # Plot the training and validation loss
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    fig_tvl = go.Figure()
    fig_tvl.add_trace(go.Scatter(
        x=list(range(1, len(train_loss) + 1)),
        y=train_loss,
        mode='lines',
        name='Training Loss'
    ))
    fig_tvl.add_trace(go.Scatter(
        x=list(range(1, len(val_loss) + 1)),
        y=val_loss,
        mode='lines',
        name='Validation Loss'
    ))
    fig_tvl.update_layout(
        title='Training and Validation Loss',
        plot_bgcolor='white',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    st.plotly_chart(fig_tvl)


def plot_predictions2(data, Y, train_predictions):
    fig_tp = go.Figure()
    fig_tp.add_trace(go.Scatter(x=data.index, y=Y, name='True Consumption', line_color='orange'))
    fig_tp.add_trace(go.Scatter(x=data.index, y=train_predictions.flatten(), name='Predicted Consumption', line_color='green'))
    fig_tp.update_layout(
        title='True vs. Predicted Consumption for Training Data',
        plot_bgcolor='white',
        xaxis_title='Date and Time',
        yaxis_title='Consumption'
    )
    return fig_tp

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Page 1 - Visualization page
def visualization_page():
    st.title('Data Visualization')
    st.subheader("📊 Time Series Data")
    st.write("Below is the visualization data of the GreEn-ER dataset, combining the dataset of class schedule, weather, and electricity usage")
    # Checkbox to select the data to show
    selected_data = st.multiselect('Select data to show', ['Consumption', 'Other', 'Heating', 'Lighting', 'All'],
                                   default=['All'])
    # Filter the data based on the user's selection
    if 'All' in selected_data:
        selected_data = ['Consumption', 'Other', 'Heating', 'Lighting']
    selected_traces = []
    for data in selected_data:
        if data == 'Consumption':
            selected_traces.append(fig['data'][0])
        elif data == 'Other':
            selected_traces.append(fig['data'][1])
        elif data == 'Heating':
            selected_traces.append(fig['data'][2])
        elif data == 'Lighting':
            selected_traces.append(fig['data'][3])
    # Update the figure with the selected traces
    fig.update(data=selected_traces)
    # Display the figure
    st.plotly_chart(fig)
    # Create the subplots
    fig_exogeneous = make_subplots(rows=2, cols=3, subplot_titles=('Number of Room', 'Dayindex', 'Occupants', 'Temperature', 'Cloudcover', 'Visibility'))
    # Add the traces to the subplots
    fig_exogeneous.add_trace(go.Scatter(x=merged_df['Date'], y=merged_df['Number of Room'], name='Number of Room',
                             line=dict(color='Coral', width=2), fill='tozeroy'), row=1, col=1)
    
    fig_exogeneous.add_trace(go.Scatter(x=merged_df['Date'], y=merged_df['Dayindex'], name='Dayindex',
                            line=dict(color='orange', width=2), fill='tozeroy'), row=1, col=2)

    fig_exogeneous.add_trace(go.Scatter(x=merged_df['Date'], y=merged_df['Occupants'], name='Occupants',
                             line=dict(color='Crimson', width=2), fill='tozeroy'), row=1, col=3)

    fig_exogeneous.add_trace(go.Scatter(x=merged_df['Date'], y=merged_df['Temperature'], name='Temperature',
                             line=dict(color='blue', width=2), fill='tozeroy'), row=2, col=1)

    fig_exogeneous.add_trace(go.Scatter(x=merged_df['Date'], y=merged_df['Cloudcover'], name='Cloudcover',
                             line=dict(color='DarkCyan', width=2), fill='tozeroy'), row=2, col=2)

    fig_exogeneous.add_trace(go.Scatter(x=merged_df['Date'], y=merged_df['Visibility'], name='Visibility',
                             line=dict(color='purple', width=2), fill='tozeroy'), row=2, col=3)

    # Set the axis titles
    fig_exogeneous.update_xaxes(title_text='Date', row=1, col=1)
    fig_exogeneous.update_yaxes(title_text='Number of Room', title_font=dict(color='Coral'), row=1, col=1)
    fig_exogeneous.update_yaxes(title_text='Dayindex', title_font=dict(color='orange'), row=1, col=2)
    fig_exogeneous.update_yaxes(title_text='Occupants', title_font=dict(color='Crimson'), row=1, col=3)
    fig_exogeneous.update_yaxes(title_text='Temperature', title_font=dict(color='blue'), row=2, col=1)
    fig_exogeneous.update_yaxes(title_text='Cloudcover', title_font=dict(color='DarkCyan'), row=2, col=2)
    fig_exogeneous.update_yaxes(title_text='Visibility', title_font=dict(color='purple'), row=2, col=3)

    # Add hover information
    fig_exogeneous.update_traces(hovertemplate='%{y:.2f}')

    # Update the layout
    fig_exogeneous.update_layout(plot_bgcolor='white', showlegend=False)

    # Show the figure
    st.plotly_chart(fig_exogeneous)

# Page 2 - Consumption Analysis
def analysis_page():
  # Create Streamlit app
  st.title('Energy Consumption Analysis')
  st.subheader("📑 Consumption Distribution by Season")
  st.write("Random Forest, Gradient Boosting, and Decision Tree are all supervised machine learning algorithms commonly used for classification and regression tasks.")
  st.plotly_chart(create_season_boxplot(df_spring, 'Spring', 'red'))
  st.plotly_chart(create_season_boxplot(df_summer, 'Summer', 'blue'))
  st.plotly_chart(create_season_boxplot(df_autumn, 'Autumn', 'green'))
  st.plotly_chart(create_season_boxplot(df_winter, 'Winter', 'orange'))
  

# Page 3 - Feature importance page
def importance_page():
    st.title("Features Importance")
    st.subheader("📑 ML Algorithms")
    st.write("Random Forest, Gradient Boosting, and Decision Tree are all supervised machine learning algorithms commonly used for classification and regression tasks.")
    method = st.selectbox("Select Method", ["Random Forest", "Gradient Boosting", "Decision Tree"])
    test_size = st.slider("Select Test Size", 0.1, 0.4, step=0.1)
    
    if st.button("Train and Evaluate"):
        X = merged_df[features]
        y = merged_df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        if method == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif method == "Gradient Boosting":
            model = GradientBoostingRegressor()
        else:
            model = DecisionTreeRegressor()
        
        trained_model, y_pred, r2 = train_and_evaluate_model(model, X_train, X_test, y_train, y_test)
        
        st.write("R2 score:", r2)
        plot_feature_importances(features, trained_model.feature_importances_)
        
# Page 4 - Forecast page
def forecast_page_fnn():
    st.title('Energy Consumption Prediction')
    st.subheader("📈 Neural Network")
    st.write("Neural network is flexible in terms of input features, since it allows to include a wide range of variables and handle large amounts of data efficiently. It is also capable of capturing complex nonlinear relationships between input variables and electricity consumption.")
    st.write("⚠️ MAE score shows the maximum error of the predicted value. An R2 score is close or eaqual to 1 means that the model perfectly explains all the variance in the dependent variable.")
    st.write("📝 Note: Try to decrease the number of epochs for fast computing (for example : 2, 5, or 10)")
    num_hours = st.slider('Select the number of hours ahead to predict', 1, 24, 12)
    num_epochs = st.slider('Select the number of epochs', 1, 50, 5)
    batch_size = st.slider('Select the batch size', 5, 15, 10)
    variables = st.multiselect('Select the variables to use for prediction', ['Number of Room', 'Dayindex', 'Occupants', 'Temperature', 'Cloudcover', 'Visibility'])

    if st.button('Predict'):
        predict_consumption(num_hours, num_epochs, batch_size, variables)

# Page 4 - Forecast page
def forecast_page_lstm():
    st.title('Energy Consumption Prediction')
    st.subheader("📈 Neural Network")
    st.write("Neural network is flexible in terms of input features, since it allows to include a wide range of variables and handle large amounts of data efficiently. It is also capable of capturing complex nonlinear relationships between input variables and electricity consumption.")
    st.write("⚠️ MAE score shows the maximum error of the predicted value. An R2 score is close or eaqual to 1 means that the model perfectly explains all the variance in the dependent variable.")
    st.write("📝 Note: Try to decrease the number of epochs for fast computing (for example : 2, 5, or 10)")
    num_hours = st.slider('Select the number of hours ahead to predict', 1, 24, 12)
    num_epochs = st.slider('Select the number of epochs', 1, 50, 5)
    batch_size = st.slider('Select the batch size', 5, 15, 10)
    variables = st.multiselect('Select the variables to use for prediction', ['Number of Room', 'Dayindex', 'Occupants', 'Temperature', 'Cloudcover', 'Visibility'])

    if st.button('Predict'):
        predict_consumption2(num_hours, num_epochs, batch_size, variables)


      
        
# Page 5 - About page
def about_page():
    st.title("About")
    st.subheader("🏛 MIAI Grenoble")
    st.write("MIAI Grenoble Alpes (Multidisciplinary Institute in Artificial Intelligence) aims to conduct research in artificial intelligence at the highest level, to offer attractive courses for students and professionals of all levels, to support innovation in large companies, SMEs and startups and to inform and interact with citizens on all aspects of AI.")
    st.subheader("🏛 Grenoble INP Ense3")
    st.write("The École nationale supérieure de l'énergie, l'eau et l'environnement or Grenoble INP-Ense3 (pronounced enne-sé-cube) is one of the 204 French engineering schools that combines technical and scientific skills in the domains of electrical, mechanical, hydraulic, civil and environmental engineering to be able to handle the full energy chain (production, distribution, usages, trading) as well as the full water cycle (harnessing, storage, supply, treatment).")
    st.subheader("🏛 G2Elab")
    st.write("The G2Elab is a laboratory recognized nationally and internationally as a major player in the field of Electrical Engineering. Its activity covers a wide spectrum of research from electrical engineering materials, to component design and the study and management of complex systems such as power grids.")

    
# Page 6 - Contact page
def contact_page():
    st.title("Contact")
    st.subheader("👨‍🎓 Student")
    st.write("Bahauddin Habibullah")
    st.write("bahauddin-habibullah@grenoble-inp.org")
    st.subheader("👨‍🏫 Supervisor")
    st.write("Benoit Delinchant")
    st.write("benoit.delinchant@grenoble-inp.fr")


# Main app
def main():
    # Set page title
    st.sidebar.image("https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEixIt_aFT2aA2Y_FUL6N1uAAX8CW-NdlNxP6BG-ggzuhCgMdzBzeQpYb5Wb6HqtlkSentBuKjIzIY-TtlR1TPnkFyh1jSrmwyKXgUzlw0aljCT-m1O44MFo8is_tIlg59JVf4biACzqIICfONNqicCIMvA1TQzl0QlVmzkgylnfiyNVf3As0Er8jMHK0w/s1600/download__1_-removebg-preview.png", 
                 use_column_width=False, 
                 width=160)
    st.sidebar.title("GreEn-ER Electricity")
    selected_page = st.sidebar.radio(
        "Go to",
        [
            ("Data Visualization", "📊 "),
            ("Consumption Analysis", "📑"),
            ("Regression and Corelation", "📑"),
            ("Features Importance", "📑"),
            ("Electricity Forecast FNN", "📈"),
            ("Electricity Forecast LSTM", "📈"),
            ("Data Sources", "📈"),
            ("About", "🚀"),
            ("Contact", "📫")
        ],
        index=0,
        format_func=lambda x: x[1] + " " + x[0]
    )

    if selected_page[0] == "Data Visualization":
        visualization_page()
    elif selected_page[0] == "Consumption Analysis":
        analysis_page() 
    elif selected_page[0] == "Regression and Corelation":
        regression_page() 
    elif selected_page[0] == "Features Importance":
        importance_page()
    elif selected_page[0] == "Electricity Forecast FNN":
        forecast_page_fnn()
    elif selected_page[0] == "Electricity Forecast LSTM":
        forecast_page_lstm()
    elif selected_page[0] == "About":
        about_page()
    elif selected_page[0] == "Contact":
        contact_page()

if __name__ == "__main__":
    main()

