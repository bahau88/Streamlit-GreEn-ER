import streamlit as st
from Streamlit_GreEn_ER.forecast1 import run_forecast1
from Streamlit_GreEn_ER.forecast2 import run_forecast2

PAGES = {
    "Forecast 1": run_forecast1,
    "Forecast 2": run_forecast2,
}

def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    page = PAGES[selection]
    page()

if __name__ == "__main__":
    main()
