"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os
import base64
import re
import os
import json
import collections
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import rcParams
from wordcloud import WordCloud, STOPWORDS

# Data dependencies
from pandas import DataFrame
import pandas as pd
from PIL import Image
import base64

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
        <style>
        body {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
        }
        </style>
        ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return


# Vectorizer
news_vectorizer = open("resources/tfidf.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file
result = {2 : 'News: the text links to factual news about climate change',
    1 :'Pro: the text supports the belief of man-made climate change',
    0 : 'Neutral: the text neither supports nor refutes the belief of man-made climate change',
    -1 :'Anti: the text does not believe in man-made climate change'}
# Load your raw data
raw = pd.read_csv("resources/train.csv")
data = pd.DataFrame(raw)
# The main function where we will build the actual app
def main():
    """Tweet Classifier App with Streamlit """
    # Creates a main title and subheader on your page -
    # these are static across all pages
    st.title("Tweet Classifer")
    st.subheader("Climate change tweet classification")
    set_png_as_page_bg('background.png')
    # Creating sidebar with selection box you can create multiple pages this way
    options = ["Prediction", "Information"]
    selection = st.sidebar.selectbox("Choose Option", options)
    # Building out the "Information" page
    if selection == "Information":
        st.info("General Information")
        # You can read a markdown file from supporting resources folder
        # st.markdown("Some information here")
        #st.video('twitter.mp4')
        st.subheader("Raw Twitter data and label")
        if st.checkbox('Show raw data'):
            st.table(data)
        MAX_WORDS = 500
        #if res > MAX_WORDS:
            #st.warning(
            #    "⚠️ Only the first 500 words will be reviewed😊"
            #)

        doc = tweet_text[:MAX_WORDS]
            # st.write(raw[['sentiment', 'message']])

    # Building out the predication page
    if selection == "Prediction":
        #st.info("Prediction with ML Models")
        option = [ "Logistic Regression Classifier", "Naive Bayes Classifier",  "Support Vector Classifier"]
        select = st.sidebar.selectbox("Choose Classifier", option)
        if select == "Logistic Regression Classifier":
                            image = Image.open('lrm.jpg')
                            st.image(image)
                            predictor = joblib.load(open(os.path.join("resources/lrm.pkl"),"rb"))
        if select == "Naive Bayes Classifier":
                            image = Image.open('nb.jpg')
                            st.image(image)
                            predictor = joblib.load(open(os.path.join("resources/nb.pkl"),"rb"))
        if select == "Support Vector Classifier":
                            image = Image.open('svc.jpg')
                            st.image(image)
                            predictor = joblib.load(open(os.path.join("resources/svc.pkl"),"rb"))
        # Creating a text box for user input
        tweet_text = st.text_area("Enter Text","Type Here")
        if st.button("Classify"):
            MAX_WORDS = 500
            doc = tweet_text[:MAX_WORDS]
            stopwords = STOPWORDS
            wordcloud = WordCloud(stopwords=stopwords, background_color="white", max_words=500).generate(doc)

            # Transforming user input with vectorizer
            vect_text = tweet_cv.transform([tweet_text]).toarray()
            prediction = predictor.predict(vect_text)
            if prediction == 1:
                prediction = 'The text supports the belief of man-made climate change'
            if prediction == 2:
                prediction = 'The text links to factual news about climate change'
            if prediction == 0:
                prediction = 'The text neither supports nor refutes the belief of man-made climate change'
            if prediction == -1:
                prediction = 'The text does not support the belief of man-made climate change'
            # When model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.
            st.success(prediction)
            fig, ax = plt.subplots(figsize = (12, 8))
            ax.imshow(wordcloud)
            plt.axis("off")
            st.pyplot(fig)
# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()
