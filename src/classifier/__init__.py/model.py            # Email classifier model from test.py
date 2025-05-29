import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

class EmailClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('classifier', MultinomialNB()),
        ])
        self.is_trained = False

    def train(self, X, y):
        self.pipeline.fit(X, y)
        self.is_trained = True

    def predict(self, email_text):
        if not self.is_trained:
            raise Exception("Model is not trained yet.")
        return self.pipeline.predict([email_text])[0]

    def predict_proba(self, email_text):
        if not self.is_trained:
            raise Exception("Model is not trained yet.")
        return self.pipeline.predict_proba([email_text])[0]

    def save_model(self, file_path):
        joblib.dump(self.pipeline, file_path)

    def load_model(self, file_path):
        self.pipeline = joblib.load(file_path)
        self.is_trained = True