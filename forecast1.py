import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import plotly.graph_objects as go



# Streamlit app
st.title("Energy Consumption Prediction")

    
    
    
    
    
    
    
    
    
    
    
    
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt


# Load the data
merged_df = pd.read_csv("https://raw.githubusercontent.com/bahau88/G2Elab-Energy-Building-/main/dataset/combined_data_green-er_2020_2023.csv")
merged_df['Date'] = pd.to_datetime(merged_df['Date'])
merged_df.set_index('Date', inplace=True)



# First page - Energy Consumption Prediction
def energy_consumption_prediction():
    st.title('Energy Consumption Prediction')

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


# Second page - Neural Network
def neural_network():
    st.title('Neural Network')

    # User inputs for the neural network
    epochs = st.slider('Number of Epochs', 1, 100, 2)
    batch_size = st.selectbox('Batch Size', [5, 10, 15])
    num_hours = st.slider('Number of Hours Ahead to Predict', 1, 24, 24)

    import numpy as np
import pandas as pd
import plotly.graph_objects as go
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

# Convert the date and hour columns to datetime format
data = merged_df.copy()

# Rename the index level "Date" to "Datetime"
data.index.names = ['Datetime']


# Split the data into input (X) and output (Y) variables
X = data[['Number of Room', 'Dayindex', 'Occupants', 'Temperature', 'Cloudcover', 'Visibility']].values
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

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Split the data into training and validation sets
train_X = X[:-24]
train_Y = Y[:-24]
val_X = X[-24:]
val_Y = Y[-24:]

# Train the model and store the history object
history = model.fit(train_X, train_Y, epochs=2, batch_size=10, verbose=2, validation_data=(val_X, val_Y))

# Ask the user how many hours ahead to predict
#num_hours = int(input('How many hours ahead would you like to predict? '))
num_hours = 24
# Generate the list of dates and hours to predict
#last_datetime = pd.to_datetime('2023-04-26 00:00')
last_datetime = data.index.max()
next_day = last_datetime + pd.DateOffset(hours=1)
datetime_range = pd.date_range(next_day, periods=num_hours, freq='H')
selected_datetimes = [str(d) for d in datetime_range]

# Make predictions for the selected dates and hours
input_data = np.zeros((num_hours, X.shape[1]))
numberofroom_arr = [0, 0, 0, 0, 0, 0, 0, 0, 21, 21, 21, 21, 5, 25, 25, 21, 19, 11, 2, 2, 0, 0, 0, 0]  # input values for number of rooms
dayindex_arr = [0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635, 0.635,  0.635,  0.635,  0.635, 0.635]  # input values for day index
occupants_arr = [0, 0, 0, 0, 0, 0, 0, 0, 923, 923, 923, 923, 633, 1068, 1068, 964, 908, 791, 371, 371, 0, 0, 0, 0]  # input values for number of occupants
temperature_arr = [7.6, 6.8, 5.9, 4.6, 4.4, 4.2, 3.7, 3.1, 5.2, 9.2, 11.6, 13.1, 14.9, 16.9, 18, 19.4, 20.8, 21.1, 21, 18.5, 17.5, 15.6, 14, 12.8]  # input values for number of temperature
cloudcover_arr = [60, 40, 0.7, 0, 80, 80, 60, 80, 50, 50, 60 , 50, 60, 60, 80, 80, 90, 100, 94.3, 95.7, 90, 96.3, 98.9, 96.3]  # input values for number of cloudcover
visibility_arr = [33.2, 25.2, 24.4, 19.7, 16.5, 20, 16.2, 14.9, 15.3, 23.9, 23.9, 24.5, 17.6, 29.9, 33.1, 19.2, 33.2, 30.7, 34.8, 38.2, 28.8, 26.3, 38.2, 34.9]  # input values for number of visibility

for i in range(num_hours):
    numberofroom = numberofroom_arr[i]
    dayindex = dayindex_arr[i]
    occupants = occupants_arr[i]
    temperature =temperature_arr[i]
    cloudcover = cloudcover_arr[i]
    visibility= visibility_arr[i]
    input_data[i] = [numberofroom, dayindex, occupants, temperature, cloudcover, visibility]

input_data = (input_data - X_mean) / X_std
predictions = model.predict(input_data)

# Print the predictions
for i in range(num_hours):
    print('Predicted consumption for {}: {:.2f}'.format(selected_datetimes[i], predictions[i][0]))



# Main app
def main():
    # Sidebar navigation
    pages = {
        'Energy Consumption Prediction': energy_consumption_prediction,
        'Neural Network': neural_network
    }
    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Go to', tuple(pages.keys()))

    # Display the selected page
    pages[page]()


if __name__ == '__main__':
    main()

