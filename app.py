import streamlit as st
import streamlit.components.v1 as stc
from ml_app import run_ml_app

html_temp = """
            <div style="background-color:#2C7865;padding:15px;border-radius:15px">
            <h1 style="color:#D9EDBF;text-align:center;font-size:2em;">Employee Promotion Prediction App</h1>
            <h3 style="color:#D9EDBF;text-align:center;">HR Team<h3>
            </div>
            """

desc_temp = """
            ### Employee Promotion Prediction App
            This app will used by the HR Team to Predict whether the employee promoted or not
            #### Data Source
            -
            #### App Content
            - Exploratory Data Analysis
            - Machine Learning
            """


def main():
    # st.title("Main app")
    stc.html(html_temp)

    menu = ["Menu", "Machine Learning"]

    choice = st.sidebar.selectbox("Home", menu)

    if choice == "Home":
        st.subheader("Home")
        st.markdown(desc_temp, unsafe_allow_html=True)
    elif choice == "Machine Learning":
        run_ml_app()


if __name__ == "__main__":
    main()
