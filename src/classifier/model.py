import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class EmailClassifier:
    def __init__(self):
        """Initialize the email classifier with improved spam detection."""
        self.pipeline = Pipeline([
            ('vectorizer', CountVectorizer(
                ngram_range=(1, 2),           # Use both single words and word pairs
                min_df=2,                      # Ignore very rare words
                max_df=0.95,                   # Ignore very common words
                stop_words='english'           # Remove common English words
            )),
            ('tfidf', TfidfTransformer()),
            ('classifier', MultinomialNB(alpha=0.1))  # Smaller alpha makes model more sensitive to features
        ])
        self.is_trained = False
        
    def train(self, X, y, test_size=0.2):
        """
        Train the classifier with email text and labels.
        
        Args:
            X: Series or list containing email text
            y: Series or list containing labels ('spam' or 'ham')
            test_size: Fraction of data to use for testing
        
        Returns:
            Dictionary containing accuracy and classification report
        """
        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Train the model
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate the model
        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)
        
        return {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    def classify_email(self, email_text):
        """
        Classify an email as spam or ham.
        
        Args:
            email_text: The text content of the email
            
        Returns:
            Dictionary with classification result and confidence score
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet. Call train() first.")
            
        prediction = self.pipeline.predict([email_text])[0]
        probability = self.pipeline.predict_proba([email_text])[0]
        
        return {
            'classification': prediction, 
            'confidence': float(max(probability))
        }
    
    def explain_classification(self, email_text):
        """Explain why an email is classified as spam or ham"""
        if not self.is_trained:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        # Get the vectorizer and classifier from the pipeline
        vectorizer = self.pipeline.named_steps['vectorizer']
        
        # Transform the email into a vector
        email_vector = vectorizer.transform([email_text])
        
        # Get the feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Get the coefficients for the spam class from the classifier
        classifier = self.pipeline.named_steps['classifier']
        
        # Create a dictionary of word -> importance score
        word_importance = {}
        for i, word in enumerate(feature_names):
            # Check if word appears in the email
            word_idx = vectorizer.vocabulary_.get(word, -1)
            if word_idx >= 0 and email_vector[0, word_idx] > 0:
                # Get the log probability for this word
                word_importance[word] = classifier.feature_log_prob_[1, i] - classifier.feature_log_prob_[0, i]
        
        # Sort words by importance
        sorted_words = sorted(word_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Keep the top 5 spam indicators
        spam_indicators = sorted_words[:5]
        
        return {
            'spam_indicators': spam_indicators,
            'word_count': len(word_importance)
        }