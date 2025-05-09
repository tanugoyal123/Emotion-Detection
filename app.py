import streamlit as st
import joblib
import numpy as np
import neattext as nt
from neattext.functions import clean_text
import nltk
from nltk.tokenize import  word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


stop_words = set(stopwords.words("english"))


lemmatizer = WordNetLemmatizer()
vectorization_model=joblib.load('vectorizer_model.pkl')
classification_model = joblib.load('emotion_model.pkl')



def testing_script(input,vectorizer,classification_model):
    cleaned_text = clean_text(input, stopwords=False)
    words_list = word_tokenize(cleaned_text)
    filtered_list = []

    for word in words_list:
        if word.casefold() not in stop_words:
            filtered_list.append(word)
    lemma_text = [lemmatizer.lemmatize(word) for word in filtered_list]
    p_text= ' '.join(lemma_text)
    x_test=vectorizer.transform([p_text])
    prediction= classification_model.predict(x_test)
    label={0:"Sadness",1:"joy",2:"love",3:"Anger",4:"Fear",5:"Surprise"}
    return label[prediction[0]]


#Stramlit UI
st.title("Emotion Detection from Text")
st.write("Enter a sentence and let the model detect the emotion (e.g., happy, sad, angry, fear, surprise, joy ).")

# User input
user_input = st.text_area("Enter text here:")

# Predict button
if st.button("Detect Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        prediction=testing_script(user_input,vectorization_model,classification_model)
        st.success(f"**Predicted Emotion:** {prediction}")