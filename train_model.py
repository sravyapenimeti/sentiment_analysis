import pandas as pd
import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from utils.preprocess import clean_text

df = pd.read_csv("dataset/IMDB Dataset.csv")

df["clean_review"] = df["review"].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(df["clean_review"])

y = df["sentiment"]

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression()

model.fit(X_train,y_train)

pred = model.predict(X_test)

accuracy = accuracy_score(y_test,pred)

print("Accuracy:",accuracy)

os.makedirs("models",exist_ok=True)

joblib.dump(model,"models/sentiment_model.pkl")

joblib.dump(vectorizer,"models/tfidf.pkl")

print("Model Saved Successfully")