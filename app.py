import streamlit as st

from utils.predict import predict_sentiment

st.set_page_config(page_title="Sentiment Analysis")

st.title("😊 Sentiment Analysis")

text = st.text_area("Enter Review")

if st.button("Analyze"):

    if text:

        sentiment = predict_sentiment(text)

        if sentiment=="positive":
            st.success("Positive Review 😊")

        else:
            st.error("Negative Review 😞")

    else:
        st.warning("Please enter text.")