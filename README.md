# Email Spam Detector

This project is an email spam detection application that allows users to classify emails as spam or not spam. The application uses a machine learning model trained on a dataset of emails to make predictions based on user input.

There are two types of mails: SPAM and HAM (to wanted, good ones)

Use as trainingdataset: https://spamassassin.apache.org/old/publiccorpus/


## Project Structure

```
email-spam-detector
├── src
│   ├── app.py                  # Main application entry point
│   ├── classifier
│   │   ├── __init__.py
│   │   ├── model.py            # Email classifier model
│   │   └── training_data.py    # Training data preparation
│   ├── ui
│   │   ├── __init__.py
│   │   ├── main_window.py      # Main UI window
│   │   ├── components.py       # Reusable UI components
│   │   └── styles.py           # UI styling and colors
│   └── utils
│   │   ├── __init__.py
│   │   └── text_processing.py  # Text preprocessing utilities
│   └── templates               #
├── data
│   └── spam                    # Training data spam (https://spamassassin.apache.org/old/publiccorpus/)
│   └── ham                     # Training data ham 
├── requirements.txt            # Project dependencies
├── setup.py                    # Packaging information
└── README.md                   # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd email-spam-detector
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```
   python src/app.py
   ```

2. Paste the text of an email into the input field.

3. The application will classify the email as spam or not spam. If classified as spam, the window will turn red and display a spam message.

