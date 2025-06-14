import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from src.utils.red_flags import RED_FLAGS

# Email classification model
# This model uses a Naive Bayes classifier with TF-IDF features for improved spam detection.
# It includes n-grams for better context understanding and removes common English stop words.
class EmailClassifier:
    def __init__(self):
        """Initialize the email classifier with improved spam detection."""
        # Create a machine learning pipeline with three main steps:
        # 1. Convert text to numerical features (CountVectorizer)
        # 2. Apply TF-IDF weighting (TfidfTransformer)
        # 3. Train a Naive Bayes classifier (MultinomialNB)
        self.pipeline = Pipeline([
            # Step 1: Convert text to a matrix of token counts
            ('vectorizer', CountVectorizer(
                ngram_range=(1, 3),         # Use single words, pairs and triplets for context
                min_df=1,                   # Include words that appear in at least 1 document
                max_df=0.9,                 # Ignore words that appear in more than 90% of documents
                stop_words='english'        # Remove common English words that don't carry much meaning
            )),
            # Step 2: Transform a count matrix to a normalized TF-IDF representation
            ('tfidf', TfidfTransformer()),
            # Step 3: Apply Naive Bayes classifier optimized for text classification
            ('classifier', MultinomialNB(alpha=0.01))  # Lower alpha = more sensitive model
        ])
        # Track if the model has been trained
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
        """
        Explains why an email is classified as spam or ham
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        # Get pipeline components
        vectorizer = self.pipeline.named_steps['vectorizer']
        classifier = self.pipeline.named_steps['classifier']
        
        # Transform email to vector
        email_vector = vectorizer.transform([email_text])
        
        # Get all words/features
        feature_names = vectorizer.get_feature_names_out()
        
        # Actual classification result
        prediction = self.pipeline.predict([email_text])[0]
        is_spam = prediction == 'spam'
        
        # Word importance scores for both classes
        spam_importance = {}
        ham_importance = {}
        
        # For each word in the email
        for i, word in enumerate(feature_names):
            word_idx = vectorizer.vocabulary_.get(word, -1)
            if word_idx >= 0 and email_vector[0, word_idx] > 0:
                # Calculate word importance for spam vs ham
                spam_score = classifier.feature_log_prob_[1, i]  # Spam class (usually index 1)
                ham_score = classifier.feature_log_prob_[0, i]   # Ham class (usually index 0)
                
                # If word is more indicative of spam
                if spam_score > ham_score:
                    spam_importance[word] = spam_score - ham_score
                # If word is more indicative of ham
                else:
                    ham_importance[word] = ham_score - spam_score
        
        # Sort indicators by importance
        spam_indicators = sorted(spam_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        ham_indicators = sorted(ham_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'classification': prediction,
            'spam_indicators': spam_indicators,
            'ham_indicators': ham_indicators,
            'word_count': len(spam_importance) + len(ham_importance)
        }
    
    def classify_email_with_rules(self, email_text):
        """Classifies an email using ML and additional rules"""
        # First check for known red flags
        email_lower = email_text.lower()
        
        # Count found red flags
        found_flags = []
        for flag in RED_FLAGS:
            if flag.lower() in email_lower:
                found_flags.append(flag)
        
        # Perform standard ML classification
        ml_result = self.classify_email(email_text)
        
        # If at least 2 red flags found, mark as spam
        if len(found_flags) >= 2:
            return {
                'classification': 'spam', 
                'confidence': max(0.95, ml_result['confidence']),
                'rule_based': True,
                'matched_flags': found_flags
            }
        
        # Otherwise return ML result, but with found flags
        return {
            **ml_result,
            'rule_based': False,
            'matched_flags': found_flags
        }