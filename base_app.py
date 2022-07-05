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
from wordcloud import WordCloud
from functionforDownloadButtons import download_button
from nltk.corpus import stopwords
from collections import Counter

# Data dependencies
from pandas import DataFrame
import pandas as pd
from PIL import Image
import base64


# Vectorizer
news_vectorizer = open("resources/tfidf.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file
# Load your raw data
raw = pd.read_csv("resources/train.csv")
data = pd.DataFrame(raw)
def clean_title(title):

    pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+' #Pattern to remove all hyperlinks
    title = re.sub(pattern_url,'',title)   #Removing all hyperlinks from statement
    #title = re.sub('RT @.*:','',title)    #Removing all preceding RT @ from every post
    title = re.sub('@\w+','',title)  #Removing all  ''@'' twitter handles
    title = re.sub(r'(\s)#\w+','',title)  ##Removing  all hashtags
    title = re.sub("[^a-zA-Z0-9 ]", "", title) #Extracting Remaining text from
    title = re.sub(r'[,!?;-]+', '.', title) #Removing any remaining non-characters
    title = title.replace('RT', '')     #Removing RT{retweet} from posts
    title = title.strip().capitalize()   #Removing all remaining whitespaces and Capitalizing all Posts
    return title
data['message'] = data['message'].apply(clean_title)
plt.rcParams['font.size'] = '12'
# The main function where we will build the actual app
def main():
    """Tweet Classifier App with Streamlit """
    # Creates a main title and subheader on your page -
    # these are static across all pages
    st.title("Tweet Classifer")
    st.subheader("Climate change tweet classification")
    # Creating sidebar with selection box you can create multiple pages this way
    options = ["Single Prediction", "Explore", "Multiple Predictions"]
    selection = st.sidebar.selectbox("Choose Page", options)
    # Building out the "Information" page
    if selection == "Explore":
        st.info("Explore the data and the different sentiments")
        image = Image.open('resources/background.png')
        st.image(image, width = 400)
        option = [ "All", "Pro - climate change", "Anti - climate change",  "Neutral", "Factual news"]
        selector = st.sidebar.selectbox("Choose category to explore", option)
        if selector == "All":
            data1 = data
            num = len(data1)
            st.info(f"There are {num} tweets in the dataset")
            st.info("We begin the exploration by looking at the distribution of tweets in the dataset")
            labels = 'Pro', 'News', 'Neutral','Anti'
            sizes = list(data1['sentiment'].value_counts())
            explode = (0, 0, 0, 0)
            fig1, ax1 = plt.subplots()
            ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=False, startangle=90)
            ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            ax1.set_title("Distribution of all the tweets" )
            plt.show()
            st.pyplot(fig1)
            st.info("We can then look at the most common words and their frequency")
            def remove_stopword(x):
                return [y for y in x if y not in stopwords.words('english')]
            data1['temp_list'] = data1['message'].apply(lambda x:str(x).split())
            data1['temp_list'] = data1['temp_list'].apply(lambda x:remove_stopword(x))
            top = Counter([item for sublist in data1['temp_list'] for item in sublist])
            temp = pd.DataFrame(top.most_common(15))
            temp = temp.iloc[1:,:]
            temp.columns = ['Common_words','count']
            st.table(temp)
            st.subheader("Raw Twitter data and label")
            if st.checkbox('Show raw data'):
                st.table(data1)
        if selector == "Pro - climate change":
            filter = 1
            data1 = data[data['sentiment'] == filter]
            num = len(data1)
            st.info(f"There are {num} tweets in the dataset")
            st.info("We begin the exploration by looking at the distribution of tweets in the dataset")
            labels = 'Pro', 'News', 'Neutral','Anti'
            sizes = list(data['sentiment'].value_counts())
            explode = (0.1, 0, 0, 0)
            fig1, ax1 = plt.subplots()
            ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=False, startangle=90)
            ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            ax1.set_title("Pro - climate change tweets" )
            plt.show()
            st.pyplot(fig1)
            st.info("We can now look at the most common words and their frequency")
            def remove_stopword(x):
                return [y for y in x if y not in stopwords.words('english')]
            data1['temp_list'] = data1['message'].apply(lambda x:str(x).split())
            data1['temp_list'] = data1['temp_list'].apply(lambda x:remove_stopword(x))
            top = Counter([item for sublist in data1['temp_list'] for item in sublist])
            temp = pd.DataFrame(top.most_common(15))
            temp = temp.iloc[1:,:]
            temp.columns = ['Common_words','count']
            st.table(temp)
            st.subheader("Raw Twitter data and label")
            if st.checkbox('Show raw data'):
                st.table(data1)
        if selector == "Anti - climate change":
            filter = -1
            data1 = data[data['sentiment'] == filter]
            num = len(data1)
            st.info(f"There are {num} tweets in the dataset")
            st.info("We begin the exploration by looking at the distribution of tweets in the dataset")
            labels = 'Pro', 'News', 'Neutral','Anti'
            sizes = list(data['sentiment'].value_counts())
            explode = (0, 0, 0, 0.1)
            fig1, ax1 = plt.subplots()
            ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=False, startangle=90)
            ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            ax1.set_title("Anti - climate tweets" )
            plt.show()
            st.pyplot(fig1)
            st.info("We can now look at the most common words and their frequency")
            def remove_stopword(x):
                return [y for y in x if y not in stopwords.words('english')]
            data1['temp_list'] = data1['message'].apply(lambda x:str(x).split())
            data1['temp_list'] = data1['temp_list'].apply(lambda x:remove_stopword(x))
            top = Counter([item for sublist in data1['temp_list'] for item in sublist])
            temp = pd.DataFrame(top.most_common(15))
            temp = temp.iloc[1:,:]
            temp.columns = ['Common_words','count']
            st.table(temp)
            st.subheader("Raw Twitter data and label")
            if st.checkbox('Show raw data'):
                st.table(data1)
        if selector == "Neutral":
            filter = 0
            data1 = data[data['sentiment'] == filter]
            num = len(data1)
            st.info(f"There are {num} tweets in the dataset")
            st.info("We begin the exploration by looking at the distribution of tweets in the dataset")
            labels = 'Pro', 'News', 'Neutral','Anti'
            sizes = list(data['sentiment'].value_counts())
            explode = (0, 0, 0.1, 0)
            fig1, ax1 = plt.subplots()
            ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=False, startangle=90)
            ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            ax1.set_title("Neutral tweets" )
            plt.show()
            st.info("We can now look at the most common words and their frequency")
            def remove_stopword(x):
                return [y for y in x if y not in stopwords.words('english')]
            data1['temp_list'] = data1['message'].apply(lambda x:str(x).split())
            data1['temp_list'] = data1['temp_list'].apply(lambda x:remove_stopword(x))
            top = Counter([item for sublist in data1['temp_list'] for item in sublist])
            temp = pd.DataFrame(top.most_common(15))
            temp = temp.iloc[1:,:]
            temp.columns = ['Common_words','count']
            st.table(temp)
            st.pyplot(fig1)
            st.subheader("Raw Twitter data and label")
            if st.checkbox('Show raw data'):
                st.table(data1)
        if selector == "Factual news":
            filter = 2
            data1 = data[data['sentiment'] == filter]
            num = len(data1)
            st.info(f"There are {num} tweets in the dataset")
            st.info("We begin the exploration by looking at the distribution of tweets in the dataset")
            labels = 'Pro', 'News', 'Neutral','Anti'
            sizes = list(data['sentiment'].value_counts())
            explode = (0, 0.1, 0, 0)
            fig1, ax1 = plt.subplots()
            ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=False, startangle=90)
            ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            ax1.set_title("Factual news tweets" )
            plt.show()
            st.info("We can now look at the most common words and their frequency")
            def remove_stopword(x):
                return [y for y in x if y not in stopwords.words('english')]
            data1['temp_list'] = data1['message'].apply(lambda x:str(x).split())
            data1['temp_list'] = data1['temp_list'].apply(lambda x:remove_stopword(x))
            top = Counter([item for sublist in data1['temp_list'] for item in sublist])
            temp = pd.DataFrame(top.most_common(15))
            temp = temp.iloc[1:,:]
            temp.columns = ['Common_words','count']
            st.table(temp)
            st.pyplot(fig1)
            st.subheader("Raw Twitter data and label")
            if st.checkbox('Show raw data'):
                st.table(data1)

    # Building out the predication page
    if selection == "Single Prediction":
        st.info("Predict a single tweet")
        predictor = joblib.load(open(os.path.join("resources/lmr.pkl"),"rb"))
        tweet_text = st.text_area("Enter Text","Type Here")
        vect_text = tweet_cv.transform([tweet_text]).toarray()
        prediction = predictor.predict(vect_text)
        if st.button("Classify"):
            MAX_WORDS = 500
            wordcloud = WordCloud(stopwords=stopwords.words('english'), background_color="white", max_words=500).generate(vect_text)
            if prediction == 1:
                prediction = 'The tweet supports the belief of man-made climate change'
            if prediction == 2:
                prediction = 'The tweet links to factual news about climate change'
            if prediction == 0:
                prediction = 'The tweet neither supports nor refutes the belief of man-made climate change'
            if prediction == -1:
                prediction = 'The tweet does not support the belief of man-made climate change'
            st.success(prediction)
            fig, ax = plt.subplots(figsize = (12, 8))
            ax.imshow(wordcloud)
            plt.axis("off")
            st.pyplot(fig)
    if selection == "Multiple Predictions":
        image = Image.open('resources/beta.png')
        st.image(image, width = 300)
        st.info("Upload a file containing cleaned tweets to predict multiple tweets and select a specific group of potential customers")
        st.info("This is a beta test feature, feel free to give it a try!!  ;)")
        predictor = joblib.load(open(os.path.join("resources/lmr.pkl"),"rb"))
        option = [ "Predict all tweets", "Pro - climate change", "Anti - climate change",  "Neutral", "Factual news"]
        select = st.sidebar.selectbox("Choose Target Customer", option)
        if select == "Predict all tweets":
            uploaded_file = st.file_uploader(
                "",
                key="1",
                help="To activate 'wide mode', go to the hamburger menu > Settings > turn on 'wide mode'",
                )
            if uploaded_file is not None:
                file_container = st.expander("Check your uploaded .csv")
                shows = pd.read_csv(uploaded_file)
                shows['message'] = shows['message'].apply(clean_title)
                uploaded_file.seek(0)
                file_container.write(shows)
            else:
                st.info(
                    f"""
                    👆 Upload a .csv file first.
                    """)

            if st.button("Classify"):
                predictions = []
                tweet_text = shows['message']
                for text in tweet_text:
                    tweet_text1 = text
                    vect_text = tweet_cv.transform([tweet_text1]).toarray()
                    MAX_WORDS = 500
                    prediction = predictor.predict(vect_text)
                    if prediction == 1:
                        prediction = 'The tweet supports the belief of man-made climate change'
                    if prediction == 2:
                        prediction = 'The tweet links to factual news about climate change'
                    if prediction == 0:
                        prediction = 'The tweet neither supports nor refutes the belief of man-made climate change'
                    if prediction == -1:
                        prediction = 'The tweet does not support the belief of man-made climate change'
                    predictions.append(prediction)
                shows['predictions'] = predictions
                result_tweets = shows[['tweetid', 'predictions']]
                df = pd.DataFrame(result_tweets)
                st.success('Your predictions are ready!!')
                st.subheader("A sample of your predictions will appear below 👇 ")
                st.text("")

                st.table(df.head())

                st.text("")
                c29, c30, c31 = st.columns([1, 1, 2])
                with c29:
                    CSVButton = download_button(
                    df,
                    "File.csv",
                    "Download to CSV",
                    )
                with c30:
                    CSVButton = download_button(
                    df,
                    "File.csv",
                    "Download to TXT",
                    )
        else:
            if select == "Pro - climate change":
                filter = 1
            if select == "Anti - climate change":
                filter = -1
            if select == "Neutral":
                filter = 0
            if select == "Factual news":
                filter = 2
            uploaded_file = st.file_uploader(
                "",
                key="1",
                help="To activate 'wide mode', go to the hamburger menu > Settings > turn on 'wide mode'",
                )
            if uploaded_file is not None:
                file_container = st.expander("Check your uploaded .csv")
                shows = pd.read_csv(uploaded_file)
                shows['message'] = shows['message'].apply(clean_title)
                uploaded_file.seek(0)
                file_container.write(shows)
            else:
                st.info(
                f"""
                    👆 Upload a .csv file first.
                    """
                    )
            if st.button("Classify"):
                predictions = []
                tweet_text = shows['message']
                for text in tweet_text:
                    tweet_text1 = text
                    vect_text = tweet_cv.transform([tweet_text1]).toarray()
                    prediction = predictor.predict(vect_text)
                    predictions.append(prediction)
                shows['predictions'] = predictions
                result_df = shows[shows['predictions'] == filter]
                result_tweets = result_df['tweetid']
                df = pd.DataFrame(result_tweets)
                st.success('Your predictions are ready!!')
                st.subheader("A sample of your potential customers will appear below 👇 ")
                st.text("")

                st.table(df.head())

                st.text("")
                c29, c30, c31 = st.columns([1, 1, 2])
                with c29:
                    CSVButton = download_button(
                    df,
                    "File.csv",
                    "Download to CSV",
                    )
                with c30:
                    CSVButton = download_button(
                    df,
                    "File.csv",
                    "Download to TXT",
                    )

# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()
