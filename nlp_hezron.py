import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from tensorflow import keras
from tensorflow.keras.models import load_model

image_path = "comment.png"

st.set_page_config(layout="wide", page_title="Cyberbullying Detector", page_icon = image_path)

padding = 20

st.title('Cyberbullying Detection')

port_stem = PorterStemmer()

def remove_special_characters(txt):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', txt)
    return text

def remove_numbers(txt):
    # Using regex to substitute all numeric characters with an empty string
    text = re.sub(r'\d', '', txt)
    return text

import demoji

def remove_emojis(txt):
    return demoji.replace(txt, '')

from langdetect import detect, LangDetectException
def remove_nonEnglish_text(txt):
    try:
        lang = detect(txt)
    except LangDetectException:
        lang = "unknown"
    return txt if lang == "en" else ""

def remove_short_words(txt, min_len=3):
    words = txt.split()
    long_words = [word for word in words if len(word) >= min_len]
    return ' '.join(long_words)

def remove_short_tweets(txt, min_words=4):
    words = txt.split()
    return txt if len(words) >= min_words else ""

def remove_long_tweets(txt, max_words=100):
    words = txt.split()
    return txt if len(words) < max_words else ""

def remove_url(txt):
    return re.sub(r'(?:http[s]?://)?(?:www\.)?(?:bit\.ly|goo\.gl|t\.co|tinyurl\.com|tr\.im|is\.gd|cli\.gs|u\.nu|url\.ie|tiny\.cc|alturl\.com|ow\.ly|bit\.do|adoro\.to)\S+', '', txt)

import string
# string.punctuation

def remove_punct(txt): 
    txt_nopunct = "".join([c for c in txt if c not in string.punctuation])
    return txt_nopunct

import re

def tokenize(txt): 
    tokens = re.split('\W+', txt)
    return tokens 

import nltk
from nltk.corpus import stopwords
",".join(stopwords.words('english'))
stopwords = nltk.corpus.stopwords.words('english')

#function to remove stopwords
def remove_stopwords(x): 
    return " ".join([word for word in str(x).split() if word not in stopwords])

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def lemmatization(txt):
    lemmatized_words = [lemmatizer.lemmatize(word) for word in txt]
    return ' '.join(lemmatized_words)

def finalClean_text(txt):
    txt = remove_special_characters(txt)
    txt = remove_numbers(txt)
    txt = remove_emojis(txt)
    txt = remove_nonEnglish_text(txt)
    txt = remove_short_words(txt)
    txt = remove_short_tweets(txt)
    txt = remove_long_tweets(txt)
    txt = remove_url(txt)
    txt = remove_punct(txt)
    txt = remove_stopwords(txt)
    txt = tokenize(txt)
    txt = lemmatization(txt)
    txt = ' '.join(txt.split())
    return txt

# loading the saved model
loaded_model = joblib.load(open('cyberbullying_SVM_model.sav', 'rb')) #rb means read as binary
vectorizer = joblib.load(open('tfidf_fit.sav', 'rb'))

def hezron(text):
        text_f = finalClean_text(text)
        text_f = vectorizer.transform([text])

        prediction = loaded_model.predict(text_f)
        prob = loaded_model.predict_proba(text_f)

        prob_nonBully = round((prob[0][0])*100)
        prob_gender = round((prob[0][1])*100)
        prob_religion = round((prob[0][2])*100)
        prob_age = round((prob[0][3])*100)
        prob_ethnicity = round((prob[0][4])*100)
        prob_otherBully = round((prob[0][5])*100)

        if prediction[0] == 0 : 
                return("HEZRON SAD, this is not cyberbullying " + str(prob_nonBully) + "%")

        elif prediction[0] == 1:
                return("Yayy, this is most likely gender bullying with " + str(prob_gender) + "%")
        
        elif prediction[0] == 2: 
                return("Good, Hezron likey. This is most likely religion bullying with " + str(prob_religion) + "%")
        
        elif prediction[0] == 3: 
                return("Good boy, this is most likely age bullying with " + str(prob_age) + "%")
        
        elif prediction[0] == 4: 
                return("Nice job kid, this is most likely ethinicity bullying with " + str(prob_ethnicity) + "%")
        
        elif prediction[0] == 5: 
                return('Hezron approve, this is most likely other cyberbullying with ' + str(prob_otherBully) + '%')
                


if "visibility" not in st.session_state:
        st.session_state.visibility = "visible"
        st.session_state.disabled = False

col1, col2, col3 = st.columns([3,1,3])

with col1:
        # st.write("Wr")
        st.subheader("Put a comment here")
        text_input = st.text_area(
        "👇",
        # label_visibility=st.session_state.visibility,
        # disabled=st.session_state.disabled,
        placeholder="Enter your comment here 💬",
        )
        proceed = st.button("Predict")
with col3:
        # hezron(text_input)
        if(text_input):
            result = hezron(text_input)
            st.header(result)
        
# def main():
# if __name__ == "__main__":
        # main()
