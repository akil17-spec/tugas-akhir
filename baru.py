import streamlit as st
import pandas as pd
import requests
import pickle

st.title('Klasifikasi Cyberbullying')

metode = st.selectbox(
    'Pilih Metode',
    ('Naive Bayes','K-Nearest Neighbors')
)
st.write('Metode :', metode)

input = st.text_input('Input Kata')
st.write(input)

def index():
    msg = None
    if(requests.method == "POST"):
        if requests.method == 'POST':
            text = requests.form['text']
            print(text)
            pickle_in = open("multinomial_naivebayes", "rb")
            model = pickle.load(pickle_in)
            vectorize = pickle.load(open("vectorize", "rb"))
            print(vectorize)
            testing_news = {"text":[text]}
            new_def_test = pd.DataFrame(testing_news)
            new_x_test = new_def_test["text"]
            print(new_x_test)
            new_xv_test = vectorize.transform([text])
            print(new_xv_test)
            preds = model.predict(new_xv_test)
            if preds[0] == 0:
                result = "Bully"
            elif preds[0] == 1:
                result = "Not A Bully"
            print(preds)
            return (result) 
        else:
            msg = "Username is not available"
st.write(index)