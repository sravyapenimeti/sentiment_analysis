import joblib

from utils.preprocess import clean_text

model = joblib.load("models/sentiment_model.pkl")

vectorizer = joblib.load("models/tfidf.pkl")


def predict_sentiment(text):

    text = clean_text(text)

    vector = vectorizer.transform([text])

    prediction = model.predict(vector)[0]

    return prediction