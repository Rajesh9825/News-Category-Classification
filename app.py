import streamlit as st
import pandas
import pickle
import nltk


# loading the model
count_vectorizer = pickle.load(open('pickle_files/count_vectorizer.pkl','rb'))
nb_classifier = pickle.load(open("pickle_files/nb_classifier_for count_vectorizer.pkl",'rb'))


def Classification(input):
    # transformation and Prediction of user headline
    headline_counts = count_vectorizer.transform([input])
    predicted_category = nb_classifier.predict(headline_counts)

    return predicted_category

# Values encoded by LabelEncoder
encoded = {0:"Business",1:"Entertainment",2:"Health",3:"Science and Technology"}


st.title("News Category Classification")

input_title = st.text_input("News Headline",)

if st.button('Predict Category'):
    category = Classification(input_title)
    st.write(encoded[category[0]])

