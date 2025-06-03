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
    """Load emails from a directory and label them"""
    data = []
    for file_path in glob.glob(os.path.join(directory, "*.eml")):
        with open(file_path, 'r', errors='ignore') as file:
            msg = email.message_from_file(file)
            # Extract email body
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body += part.get_payload(decode=True).decode('utf-8', errors='ignore')
            else:
                body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
            
            data.append({'text': body, 'label': label})
    return data

# Example usage:
# spam_emails = load_emails_from_directory("path/to/spam/emails", "spam")
# ham_emails = load_emails_from_directory("path/to/ham/emails", "ham")
# all_emails = spam_emails + ham_emails
# df = pd.DataFrame(all_emails)

# Try to load emails from directories first
spam_emails = load_emails_from_directory("data/spam", "spam")
ham_emails = load_emails_from_directory("data/ham", "ham")
all_emails = spam_emails + ham_emails

if all_emails:
    print(f"Training with {len(all_emails)} emails from directories ({len(spam_emails)} spam, {len(ham_emails)} ham)")
    df = pd.DataFrame(all_emails)
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

if __name__ == '__main__':
    print("Starting Email Spam Detector...")
    app.run(debug=True)