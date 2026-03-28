from flask import Flask, request, jsonify, render_template_string
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

app = Flask(__name__)

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'fake_news_model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'models', 'vectorizer.pkl')

# Load model and vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# NLTK setup
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_input(text):
    """Preprocess input text."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

@app.route('/')
def home():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fake News Detection</title>
    </head>
    <body>
        <h1>Fake News Detection System</h1>
        <form action="/predict" method="post">
            <textarea name="text" rows="10" cols="50" placeholder="Enter news article text here..."></textarea><br>
            <input type="submit" value="Check">
        </form>
    </body>
    </html>
    ''')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        processed_text = preprocess_input(text)
        vectorized = vectorizer.transform([processed_text])
        prediction = model.predict(vectorized)[0]
        probability = model.decision_function(vectorized)[0]

        result = 'Fake' if prediction == 1 else 'Real'
        confidence = abs(probability)

        return jsonify({
            'prediction': result,
            'confidence': confidence
        })

if __name__ == '__main__':
    app.run(debug=True)