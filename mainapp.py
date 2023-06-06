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
# Page 1 - Visualization page
def visualization_page():
    st.title('Data Visualization')
    st.subheader("📊 Timeseries Data")
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
        
# Page 2 - Feature importance page
def importance_page():
    st.title("Energy Consumption Prediction")
    st.subheader("📊 Home")
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

# Page 2 - About page
def about_page():
    st.title("Energy Consumption Prediction")
    st.subheader("📊 About")
    st.write("This is the about page.")
    # Add any additional content or functionality for this page
    
# Page 3 - Contact page
def contact_page():
    st.title("Energy Consumption Prediction")
    st.subheader("📊 Contact")
    st.write("This is the contact page.")
    # Add any additional content or functionality for this page

# Main app
def main():
    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio(
        "Go to",
        [
            ("Data Visualization", "🏠"),
            ("Features Importance", "🏠"),
            ("About", "ℹ️"),
            ("Contact", "📞")
        ],
        index=0,
        format_func=lambda x: x[1] + " " + x[0]
    )

    if selected_page[0] == "Data Visualization":
        visualization_page()
    elif selected_page[0] == "Features Importance":
        importance_page()
    elif selected_page[0] == "About":
        about_page()
    elif selected_page[0] == "Contact":
        contact_page()

if __name__ == "__main__":
    main()

