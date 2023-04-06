import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap
import pickle
import streamlit.components.v1 as components

url = "https://boiling-brook-69195.herokuapp.com/predict"
col_train = pd.read_pickle("app_train_col.pkl")
shap_values_test = np.load('data.npy')
good_test = pd.read_pickle("good_app_test-Copy1.pkl")
good_test_sans_sk=good_test.drop(columns=["SK_ID_CURR"])

with open('explainer.pkl', 'rb') as f:
    explainer = pickle.load(f)


def get_prediction(SK_ID_CURR):
    params = {'SK_ID_CURR': SK_ID_CURR}
    response = requests.get(url, params=params)
    prediction = response.json()['prediction']
    days_birth = response.json()['DAYS_BIRTH']
    ext2 = response.json()['EXT_SOURCE_2']
    ext3 = response.json()['EXT_SOURCE_3']
    if prediction > 0.2:
        reponse = 1
    else:
        reponse = 0
    return prediction, days_birth, reponse, ext2, ext3

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# Create a Streamlit app
def app():
    st.title("Loan Prediction App")

    # Add an input field for the user to enter SK_ID_CURR
    SK_ID_CURR = st.text_input("Enter SK_ID_CURR")

    # Add a button to submit the form
    if st.button("Predict"):
        if SK_ID_CURR:
            # Call the get_prediction function to make a request to the API
            prediction, days_birth, reponse, ext2, ext3= get_prediction(int(SK_ID_CURR))
            # Display the prediction and days_birth
            st.success(f"La probabilité de faire defaut est de {prediction:.2%}")
            st.success(f"Sur les graphiques ci dessous votre situation est représentée par le point rouge, si il est dans 0 le prêt peut " 
                       f"etre accepté, si c'est dans 1 il est refusé, avec une représentation par les 3 facteurs principaux qui ont fait "
                      f"défaut ou non par le passé")
            # Use the SHAP values to create a force plot for a specific instance
            #instance_index = good_test.index[good_test['SK_ID_CURR'] == SK_ID_CURR].tolist()[0]
            instance_index = good_test[good_test['SK_ID_CURR'] == int(SK_ID_CURR)].index.tolist()
            if not instance_index:
                print("Invalid INDEX")
            else:
                instance_index = instance_index[0]
                print("instance_index:", instance_index)
                # Generate the SHAP force plot
                feature_names = good_test_sans_sk.columns
                #force_plot = shap.plots._force.waterfall_legacy(explainer.expected_value[0], shap_values_test[instance_index], features=good_test_sans_sk.iloc[[instance_index]], feature_names=feature_names, show=False)
                force_plot = shap.force_plot(explainer.expected_value[0], shap_values_test[instance_index], good_test_sans_sk.iloc[[instance_index]])
            # Display the force plot in your Streamlit app
            st.write('## SHAP Force Plot')
            st.write('This plot shows the features that contributed to the predicted outcome for a specific instance.')
            #st.pyplot(force_plot)
            st_shap(force_plot)
            
            #boxplot
            fig = plt.figure(figsize=(10, 4))
            sns.boxplot(x=col_train['TARGET'], y=col_train['DAYS_BIRTH'])
            plt.scatter(x=reponse, y=days_birth, color='red', s=50, zorder=10)
            st.pyplot(fig)
            fig = plt.figure(figsize=(10, 4))
            sns.boxplot(x=col_train['TARGET'], y=col_train['EXT_SOURCE_2'])
            plt.scatter(x=reponse, y=ext2, color='red', s=50, zorder=10)
            st.pyplot(fig)
            fig = plt.figure(figsize=(10, 4))
            sns.boxplot(x=col_train['TARGET'], y=col_train['EXT_SOURCE_3'])
            plt.scatter(x=reponse, y=ext3, color='red', s=50, zorder=10)
            st.pyplot(fig)
            

if __name__ == "__main__":
    app()
