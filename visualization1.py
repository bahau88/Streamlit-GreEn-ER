import streamlit as st
import plotly.graph_objects as go
import pandas as pd

# Load the data
merged_df = pd.read_csv('https://raw.githubusercontent.com/bahau88/G2Elab-Energy-Building-/main/dataset/combined_data_green-er_2020_2023.csv')  # Replace 'your_data.csv' with your actual data file

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

# Create the Streamlit app
st.title('Data Visualization')

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
