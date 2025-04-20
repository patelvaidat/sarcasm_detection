import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import xgboost as xgb
import numpy as np

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load XGBoost model and vectorizer
xgb_model = xgb.XGBClassifier()
xgb_model.load_model("./models/xgboost.json")

with open("./models/vectorizer_xgboost.pkl", "rb") as f:
    xgb_vectorizer = pickle.load(f)

# Load Random Forest model and vectorizer
with open("./models/randomforest.pkl", "rb") as file:
    rf_classifier = pickle.load(file)

with open("./models/vectorizer_randomforest.pkl", "rb") as file:
    rf_vectorizer = pickle.load(file)

# Load Naive Bayes model and vectorizer
with open("./models/naivebayes.pkl", "rb") as file:
    nb_classifier = pickle.load(file)

with open("./models/vectorizer_naivebayes.pkl", "rb") as file:
    nb_vectorizer = pickle.load(file)

# Text Preprocessing Function (for Random Forest)
def preprocess_text(text):
    text = re.sub(r"#\w+", "", text)  # Remove hashtags
    text = re.sub(r"<[^>]+>", "", text)  # Remove HTML tags
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    return ' '.join(tokens)

# Function to predict sarcasm using XGBoost
def predict_sarcasm_xgboost(text):
    features = xgb_vectorizer.transform([text])
    prediction = xgb_model.predict(features)
    return "Sarcastic" if prediction[0] == 1 else "Not Sarcastic"

# Function to predict sarcasm using Random Forest
def predict_sarcasm_random_forest(text):
    processed_text = preprocess_text(text)
    text_tfidf = rf_vectorizer.transform([processed_text])
    prediction = rf_classifier.predict(text_tfidf)[0]
    return "Sarcastic" if prediction == 1 else "Not Sarcastic"

# Function to predict sarcasm percentage using Naive Bayes
def predict_sarcasm_naive_bayes(text):
    text_tfidf = nb_vectorizer.transform([text])
    prediction = nb_classifier.predict_proba(text_tfidf)[0]
    return prediction[1] * 100  # Probability of sarcasm in percentage
