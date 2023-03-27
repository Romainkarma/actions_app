import streamlit as st
import requests

url = "https://boiling-brook-69195.herokuapp.com/predict"

def get_prediction(SK_ID_CURR):
    params = {'SK_ID_CURR': SK_ID_CURR}
    response = requests.get(url, params=params)
    prediction = response.json()['prediction']
    return prediction


# Create a Streamlit app
def app():
    st.title("Loan Prediction App")

    # Add an input field for the user to enter SK_ID_CURR
    SK_ID_CURR = st.text_input("Enter SK_ID_CURR")

    # Add a button to submit the form
    if st.button("Predict"):
        if SK_ID_CURR:
            # Call the get_prediction function to make a request to the API
            prediction = get_prediction(int(SK_ID_CURR))

            # Display the prediction
            st.success(f"The probability of default is {prediction:.2%}")

if __name__ == "__main__":
    app()
