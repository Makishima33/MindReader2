from flask import Flask, render_template, request, jsonify
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

app = Flask(__name__)

# 载入模型
sentiment_model = AutoModelForSequenceClassification.from_pretrained('./models/sentiment_model')
sentiment_tokenizer = AutoTokenizer.from_pretrained('./models/sentiment_model')
sentiment_model.eval()
label_encoder = joblib.load('./models/sentiment_model/sentiment_label_encoder.pkl')

# Load other models
emotion_model = joblib.load('./models/emotion_model/linear_svm_tfidf_model.joblib')

mbti_models = {
    'I/E': joblib.load("./models/mbti_model/mbti_I_E_SVC.joblib"),
    'N/S': joblib.load("./models/mbti_model/mbti_N_S_SVC.joblib"),
    'T/F': joblib.load("./models/mbti_model/mbti_T_F_SVC.joblib"),
    'J/P': joblib.load("./models/mbti_model/mbti_J_P_SVC.joblib")
}
mbti_dimension_mapping = {
    'I/E': {1: 'I', 0: 'E'},
    'N/S': {1: 'N', 0: 'S'},
    'T/F': {1: 'T', 0: 'F'},
    'J/P': {1: 'J', 0: 'P'}
}

country_model = AutoModelForSequenceClassification.from_pretrained('./models/country_model')
country_tokenizer = AutoTokenizer.from_pretrained('./models/country_model')
with open("./models/country_model/label_mapping.json", "r") as f:
    country_label_mapping = json.load(f)
country_id2label = {v: k for k, v in country_label_mapping.items()}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json.get('text')

    # Sentiment
    sentiment_inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        sentiment_outputs = sentiment_model(**sentiment_inputs)
    sentiment_pred = torch.argmax(sentiment_outputs.logits, dim=1).item()
    sentiment = label_encoder.inverse_transform([sentiment_pred])[0]

    # Emotion
    emotion = emotion_model.predict([text])[0]

    # MBTI
    def predict_mbti(text):
        import re
        def clean_text(t):
            t = t.lower()
            t = re.sub(r'http[s]?://\S+', 'url', t)
            t = re.sub(r'[^a-zA-Z\s]', ' ', t)
            t = re.sub(r'\s+', ' ', t).strip()
            return t

        cleaned = clean_text(text)
        mbti = []
        for dim, model in mbti_models.items():
            pred = model.predict([cleaned])[0]
            mbti.append(mbti_dimension_mapping[dim][pred])
        return ''.join(mbti)

    mbti = predict_mbti(text)

    # Country
    country_inputs = country_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        country_outputs = country_model(**country_inputs)
    country_pred = torch.argmax(country_outputs.logits, dim=1).item()
    country = country_id2label[country_pred]

    return jsonify({
        'sentiment': sentiment,
        'emotion': emotion,
        'mbti': mbti,
        'country': country
    })

if __name__ == '__main__':
    app.run(debug=True)