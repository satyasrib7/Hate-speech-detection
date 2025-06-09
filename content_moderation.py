import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import joblib
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

# Text preprocessing
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    return ' '.join(text.split())

# Load and preprocess data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df.dropna(subset=['Content', 'Label'], inplace=True)
    df['Content'] = df['Content'].apply(preprocess_text)
    return df['Content'].tolist(), df['Label'].tolist()

# Train the model
def train_model():
    try:
        texts, labels = load_and_preprocess_data('HateSpeechDataset.csv')
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )

        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        train_vectors = vectorizer.fit_transform(train_texts)
        val_vectors = vectorizer.transform(val_texts)

        class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
        class_weight_dict = dict(zip(np.unique(train_labels), class_weights))

        model = LogisticRegression(random_state=42, class_weight=class_weight_dict, max_iter=1000)
        model.fit(train_vectors, train_labels)

        val_predictions = model.predict(val_vectors)
        print('Validation Performance:')
        print(classification_report(val_labels, val_predictions))

        joblib.dump(model, 'content_moderation_model.joblib')
        joblib.dump(vectorizer, 'vectorizer.joblib')

        return model, vectorizer

    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        return None, None

# Function to predict and mitigate hate speech
def predict_and_mitigate(text, model, vectorizer):
    try:
        preprocessed_text = preprocess_text(text)
        vector = vectorizer.transform([preprocessed_text])
        probabilities = model.predict_proba(vector)[0]

        prob_hate_speech = probabilities[1]
        confidence_threshold = 0.5
        prediction = "Hate speech" if prob_hate_speech > confidence_threshold else "Not hate speech"
        confidence = prob_hate_speech if prediction == "Hate speech" else 1 - prob_hate_speech

        offensive_words = ["retard", "nazi", "vagina"]  # Add more words as needed
        if prediction == "Hate speech":
            for word in offensive_words:
                text = re.sub(r'\b{}\b'.format(word), '[redacted]', text, flags=re.IGNORECASE)
            warning = "Warning: This content has been flagged for containing potentially harmful language."
            text = warning + "\n" + text

        return prediction, confidence, prob_hate_speech, text
    except Exception as e:
        print(f"An error occurred during prediction: {str(e)}")
        return "Error", 0.0, 0.0, text

# Flask app
app = Flask(__name__)
CORS(app, resources={r"/analyze": {"origins": "http://localhost:3000"}})

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.json['text']
    prediction, confidence, prob_hate_speech, mitigated_text = predict_and_mitigate(text, model, vectorizer)

    return jsonify({
        'classification': prediction,
        'confidence': float(confidence),
        'probability_hate_speech': float(prob_hate_speech),
        'mitigated_text': mitigated_text
    })

if __name__ == '__main__':
    # Load the model and vectorizer if they exist, otherwise train them
    if os.path.exists('content_moderation_model.joblib') and os.path.exists('vectorizer.joblib'):
        model = joblib.load('content_moderation_model.joblib')
        vectorizer = joblib.load('vectorizer.joblib')
    else:
        model, vectorizer = train_model()
    
    if model and vectorizer:
        app.run(debug=True, port=5000)
    else:
        print("Failed to train or load the model. Please check your dataset and try again.")
