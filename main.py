# Import Streamlit.
from turtle import Turtle
import streamlit as st
import pickle

# Import pandas to read CSV files.
import pandas as pd

# Import natural language toolkit.
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

import matplotlib.pyplot as plt

# Import Naive Bayes Module
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
# Import module to display accuracy
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def predict():
    df = pd.read_csv(r"C:\Users\ASUS\aplikasi\stemming.csv")
    df = df.drop(['Unnamed: 0','waktu','user','text','STOP_REMOVAL'], axis=1)
    df.rename(columns = {'STEMMING': 'message', 'labeling':'labels'}, inplace= True)
    df.drop_duplicates(inplace = True)
    df['labels'] = df['labels'].map({'bully':0,'not bully':1})
    print(df.head())

    x = df['message'].values.astype('U')
    y = df['labels']

    cv = CountVectorizer()

    x = cv.fit_transform(x,y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

    model = MultinomialNB().fit(x_train, y_train)
    predictions = model.predict(x_test)

    akurasi = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    cr = classification_report(y_test, predictions)
    hasil = (akurasi,cm,cr)
    return hasil

def knn():
    with open("KNeighborsClassifier.sav", 'rb') as pickle_file:
        knn_ = pickle.load(pickle_file)
    return knn_
test = predict()
test_knn = knn()

st.header("Naive Bayes")
st.write("akurasi",test[0])
st.write("confussion matrix",test[1])
string_hasil = test[2].split("\n")
for i in string_hasil:
    if len(i) < 1:
        pass
    else:
        st.code(i)


st.header("KNN")
st.subheader("KNN 1")
st.write("akurasi",test_knn[0][0])
st.write("klasifikasi report")
string_hasil = test_knn[0][1].split("\n")
for i in string_hasil:
    if len(i) < 1:
        pass
    else:
        st.code(i)

bully = 0
not_bully = 0
for i in test_knn[0][2]:
    if i == 'bully':
        bully+=1
    else:
        not_bully+=1

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Bully', 'Not Bully'
sizes = [bully,not_bully]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

st.pyplot(fig1)

st.subheader("KNN 5")
st.write("akurasi",test_knn[1][0])
st.write("klasifikasi report")
string_hasil = test_knn[1][1].split("\n")
for i in string_hasil:
    if len(i) < 1:
        pass
    else:
        st.code(i)

bully = 0
not_bully = 0
for i in test_knn[1][2]:
    if i == 'bully':
        bully+=1
    else:
        not_bully+=1

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Bully', 'Not Bully'
sizes = [bully,not_bully]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

st.pyplot(fig1)

st.subheader("KNN 7")
st.write("akurasi",test_knn[2][0])
st.write("klasifikasi report")
string_hasil = test_knn[2][1].split("\n")
for i in string_hasil:
    if len(i) < 1:
        pass
    else:
        st.code(i)

bully = 0
not_bully = 0
for i in test_knn[2][2]:
    if i == 'bully':
        bully+=1
    else:
        not_bully+=1

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Bully', 'Not Bully'
sizes = [bully,not_bully]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

st.pyplot(fig1)

st.subheader("KNN 9")
st.write("akurasi",test_knn[3][0])
st.write("klasifikasi report")
string_hasil = test_knn[3][1].split("\n")
for i in string_hasil:
    if len(i) < 1:
        pass
    else:
        st.code(i)

bully = 0
not_bully = 0
for i in test_knn[3][2]:
    if i == 'bully':
        bully+=1
    else:
        not_bully+=1

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Bully', 'Not Bully'
sizes = [bully,not_bully]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

st.pyplot(fig1)