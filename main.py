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
shap_values_test = np.load('data_sample.npy')
good_test = pd.read_pickle("sample_test.pkl")
good_test_sans_sk=good_test.drop(columns=["SK_ID_CURR"])
col_train = pd.read_pickle("graph.pkl")
col_test = pd.read_pickle("graph_test.pkl")

with open('explainer.pkl', 'rb') as f:
    explainer = pickle.load(f)


def get_prediction(SK_ID_CURR):
    params = {'SK_ID_CURR': SK_ID_CURR}
    response = requests.get(url, params=params)
    prediction = response.json()['prediction']
    #days_birth = response.json()['DAYS_BIRTH']
    #ext2 = response.json()['EXT_SOURCE_2']
    #ext3 = response.json()['EXT_SOURCE_3']
    if prediction > 0.2:
        reponse = 1
    else:
        reponse = 0
    days_birthk = col_test.loc[col_test['SK_ID_CURR'] == SK_ID_CURR, 'DAYS_BIRTH'].values[0]
    ext2k = col_test.loc[col_test['SK_ID_CURR'] == SK_ID_CURR, 'EXT_SOURCE_2'].values[0]
    ext3k = col_test.loc[col_test['SK_ID_CURR'] == SK_ID_CURR, 'EXT_SOURCE_3'].values[0]
    return prediction, reponse, days_birthk, ext2k, ext3k

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

def app():
    st.title("Application de classification prêt")

    SK_ID_CURR = st.text_input("Entrez le SK_ID_CURR")

    if st.button("Predict"):
        if SK_ID_CURR:
            prediction, reponse, days_birthk, ext2k, ext3k= get_prediction(int(SK_ID_CURR))
            # Display the prediction and days_birth
            st.write(f"La probabilité de faire defaut est de {prediction:.2%}")
            st.write(f"Ci dessous s'affiche les facteurs propres à votre situation qui explique la prise de décision de l'entreprise " 
                     f"sur le premier graphique les facteurs qui s'oppose en votre faveur ou défaveur et les trois graphiques suivant "
                     f"vous situe par rapport aux clients ayant fait défaut ou non par le passé")
            instance_index = good_test[good_test['SK_ID_CURR'] == int(SK_ID_CURR)].index.tolist()
            if not instance_index:
                print("Invalid INDEX")
            else:
                instance_index = instance_index[0]
                print("instance_index:", instance_index)
                feature_names = good_test_sans_sk.columns
                force_plot = shap.force_plot(explainer.expected_value[0], shap_values_test[instance_index], good_test_sans_sk.iloc[[instance_index]])
                st_shap(force_plot)
                
            fig = plt.figure(figsize=(10, 4))
            sns.boxplot(x=col_train['TARGET'], y=col_train['DAYS_BIRTH'])
            plt.scatter(x=reponse, y=days_birthk, color='red', s=50, zorder=10)
            plt.title("Votre age par rapport à l'ensemble")
            plt.xlabel("Faire défaut")
            plt.ylabel("Age(jours)")
            st.pyplot(fig)
            fig = plt.figure(figsize=(10, 4))
            sns.boxplot(x=col_train['TARGET'], y=col_train['EXT_SOURCE_2'])
            plt.scatter(x=reponse, y=ext2k, color='red', s=50, zorder=10)
            plt.title("Votre EXT_SOURCE_2 par rapport à l'ensemble")
            plt.xlabel("Faire défaut")
            plt.ylabel("EXT_SOURCE_2")
            st.pyplot(fig)
            fig = plt.figure(figsize=(10, 4))
            sns.boxplot(x=col_train['TARGET'], y=col_train['EXT_SOURCE_3'])
            plt.scatter(x=reponse, y=ext3k, color='red', s=50, zorder=10)
            plt.title("Votre EXT_SOURCE_3 par rapport à l'ensemble")
            plt.xlabel("Faire défaut")
            plt.ylabel("EXT_SOURCE_3")
            st.pyplot(fig)
            

if __name__ == "__main__":
    app()
