from flask import Flask, render_template, request
import os
import sys
import pandas as pd
import glob
import email

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import using the full path
from src.classifier.model import EmailClassifier

app = Flask(__name__)
classifier = EmailClassifier()

def load_emails_from_directory(directory, label):
    """Load emails from a directory and label them with better HTML handling"""
    data = []
    
    # Process all files in the directory
    for file_path in glob.glob(os.path.join(directory, "*")):
        try:
            with open(file_path, 'r', errors='ignore') as file:
                try:
                    msg = email.message_from_file(file)
                    # Extract email body (both text and HTML)
                    body = ""
                    
                    if msg.is_multipart():
                        for part in msg.walk():
                            content_type = part.get_content_type()
                            if content_type == "text/plain":
                                body += part.get_payload(decode=True).decode('utf-8', errors='ignore')
                            elif content_type == "text/html":
                                # Simple handling for HTML content - just add it
                                html_content = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                                body += html_content
                    else:
                        body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
                    
                    # Add email subject to the body for better classification
                    if msg['subject']:
                        body = f"Subject: {msg['subject']}\n\n{body}"
                    
                    # If we got here, it's a valid email
                    if body.strip():  # Only add if body is not empty
                        data.append({'text': body, 'label': label})
                        print(f"Loaded {file_path} as email")
                except:
                    # If email parsing fails, try as plain text
                    file.seek(0)
                    body = file.read()
                    if body.strip():
                        data.append({'text': body, 'label': label})
                        print(f"Loaded {file_path} as plain text")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
                
    return data

# Example usage:
# spam_emails = load_emails_from_directory("path/to/spam/emails", "spam")
# ham_emails = load_emails_from_directory("path/to/ham/emails", "ham")
# all_emails = spam_emails + ham_emails
# df = pd.DataFrame(all_emails)

# Try to load emails from directories first
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spam_dir = os.path.join(base_dir, "data", "spam")
ham_dir = os.path.join(base_dir, "data", "ham")

print(f"Looking for spam emails in: {spam_dir}")
print(f"Looking for ham emails in: {ham_dir}")

spam_emails = load_emails_from_directory(spam_dir, "spam")
ham_emails = load_emails_from_directory(ham_dir, "ham")
all_emails = spam_emails + ham_emails

print(f"Found {len(spam_emails)} spam emails and {len(ham_emails)} ham emails")

if all_emails:
    print(f"Training with {len(all_emails)} emails from directories")
    df = pd.DataFrame(all_emails)
    
    # Print some sample spam words to verify content is loaded correctly
    spam_df = df[df['label'] == 'spam']
    if not spam_df.empty:
        sample_spam = spam_df.iloc[0]['text']
        print(f"Sample spam words: {sample_spam[:200]}...")
    
    # Train the model with the emails from directories
    classifier.train(df['text'], df['label'])
else:
    # Fall back to CSV if no emails in directories
    try:
        data_path = os.path.join(os.path.dirname(__file__), "utils", "email_samples.csv")
        if os.path.exists(data_path):
            print(f"Training with data from {data_path}")
            df = pd.read_csv(data_path)
            classifier.train(df['text'], df['label'])
        else:
            # Last resort: use hardcoded examples
            print("No training data found. Using sample examples.")
            emails = [
                {'text': 'Get rich quick! Buy now!', 'label': 'spam'},
                {'text': 'Meeting scheduled for tomorrow', 'label': 'ham'},
                {'text': 'Free discount coupons inside', 'label': 'spam'},
                {'text': 'Project status update needed', 'label': 'ham'},
                # Add more examples here
            ]
            df = pd.DataFrame(emails)
            classifier.train(df['text'], df['label'])
    except Exception as e:
        print(f"Error training model: {e}")

@app.route('/', methods=['GET', 'POST'])
def index():
    spam_message = None
    email_text = None
    is_spam = False
    explanation = None

    if request.method == 'POST':
        email_text = request.form['email_text']
            
        result = classifier.classify_email(email_text)
        is_spam = result['classification'] == 'spam'
        confidence = result['confidence']
        
        # Get explanation if it's classified as spam
        if is_spam:
            explanation = classifier.explain_classification(email_text)
            spam_indicators = explanation['spam_indicators']
            indicator_text = ", ".join([f"'{word}'" for word, score in spam_indicators])
            spam_message = f"This email is classified as SPAM with {confidence:.2f} confidence. Suspicious words: {indicator_text}"
        else:
            spam_message = f"This email is classified as NOT SPAM with {confidence:.2f} confidence."

    return render_template('index.html', spam_message=spam_message, email_text=email_text, 
                          is_spam=is_spam, explanation=explanation)

# Add this to your app.py file to see what's happening during classification
@app.route('/debug', methods=['GET', 'POST'])
def debug():
    """Debug route to test and understand the classification"""
    result = None
    
    if request.method == 'POST':
        email_text = request.form['email_text']
        
        # Get the vectorizer and classifier from the pipeline
        vectorizer = classifier.pipeline.named_steps['vectorizer']
        
        # Transform the email into a vector
        email_vector = vectorizer.transform([email_text])
        
        # Get the feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Create a list of (word, count) pairs
        word_counts = [(feature_names[idx], email_vector[0, idx]) 
                      for idx in email_vector.nonzero()[1]]
        
        # Sort words by count
        sorted_words = sorted(word_counts, key=lambda x: x[1], reverse=True)
        
        # Get classification and probability
        prediction = classifier.classify_email(email_text)
        
        result = {
            'classification': prediction,
            'top_words': sorted_words[:20]  # Show top 20 words
        }
        
    return render_template('debug.html', result=result)

if __name__ == '__main__':
    print("Starting Email Spam Detector...")
    app.run(debug=True)