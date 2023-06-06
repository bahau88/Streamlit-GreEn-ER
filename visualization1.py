import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots

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
