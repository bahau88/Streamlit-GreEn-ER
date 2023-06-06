import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import plotly.graph_objects as go

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

# Define the page names and corresponding URLs
pages = {
    "Home": "/",
    "About": "/about",
    "Contact": "/contact"
}

# Define the default page and current page
default_page = "Home"
current_page = st.experimental_get_query_params().get("page", [default_page])[0]

# Page 1 - Home page
if current_page == "Home":
    st.title("Energy Consumption Prediction")
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Pages")
    
    # Create links for each page
    for page_name, page_url in pages.items():
        if page_name == current_page:
            st.sidebar.markdown(f"**{page_name}**")
        else:
            st.sidebar.markdown(f"[{page_name}]({page_url})")
    
    st.sidebar.markdown("---")
    method = st.sidebar.selectbox("Select Method", ["Random Forest", "Gradient Boosting", "Decision Tree"])
    test_size = st.sidebar.slider("Select Test Size", 0.1, 0.4, step=0.1)
    
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
elif current_page == "About":
    st.title("Energy Consumption Prediction - About")
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Pages")
    
    # Create links for each page
    for page_name, page_url in pages.items():
        if page_name == current_page:
            st.sidebar.markdown(f"**{page_name}**")
        else:
            st.sidebar.markdown(f"[{page_name}]({page_url})")
    
    st.sidebar.markdown("---")
    st.write("This is the About page.")
    # Add any additional content or functionality for this page

# Page 3 - Contact page
elif current_page == "Contact":
    st.title("Energy Consumption Prediction - Contact")
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Pages")
    
    # Create links for each page
    for page_name, page_url in pages.items():
        if page_name == current_page:
            st.sidebar.markdown(f"**{page_name}**")
        else:
            st.sidebar.markdown(f"[{page_name}]({page_url})")
    
    st.sidebar.markdown("---")
    st.write("This is the Contact page.")
    # Add any additional content or functionality for this page

# Handle page navigation
if current_page not in pages.keys():
    st.experimental_set_query_params(page=default_page)
else:
    st.experimental_set_query_params(page=current_page)
