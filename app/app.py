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
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Fake News Detection System</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            
            .container {
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                max-width: 900px;
                width: 100%;
                overflow: hidden;
            }
            
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 40px 30px;
                text-align: center;
            }
            
            .header h1 {
                font-size: 2.5em;
                margin-bottom: 10px;
                font-weight: 700;
            }
            
            .header p {
                font-size: 1.1em;
                opacity: 0.9;
            }
            
            .content {
                padding: 40px;
            }
            
            .form-group {
                margin-bottom: 25px;
            }
            
            label {
                display: block;
                font-size: 1.1em;
                font-weight: 600;
                margin-bottom: 10px;
                color: #333;
            }
            
            textarea {
                width: 100%;
                padding: 15px;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                font-size: 1em;
                font-family: inherit;
                resize: vertical;
                min-height: 150px;
                transition: border-color 0.3s;
            }
            
            textarea:focus {
                outline: none;
                border-color: #667eea;
                box-shadow: 0 0 10px rgba(102, 126, 234, 0.1);
            }
            
            .button-group {
                display: flex;
                gap: 10px;
                justify-content: center;
            }
            
            button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 15px 40px;
                border-radius: 10px;
                font-size: 1.1em;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s;
            }
            
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
            }
            
            button:active {
                transform: translateY(0);
            }
            
            .clear-btn {
                background: #e0e0e0;
                color: #333;
            }
            
            .clear-btn:hover {
                background: #d0d0d0;
            }
            
            .info-box {
                background: #f5f7fa;
                border-left: 5px solid #667eea;
                padding: 15px;
                border-radius: 5px;
                margin-top: 30px;
                font-size: 0.95em;
                color: #555;
            }
            
            .info-box strong {
                color: #667eea;
            }
            
            @media (max-width: 600px) {
                .header h1 {
                    font-size: 1.8em;
                }
                
                .content {
                    padding: 20px;
                }
                
                button {
                    padding: 12px 30px;
                    font-size: 1em;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔍 Fake News Detection</h1>
                <p>Advanced AI-Powered Misinformation Detection</p>
            </div>
            
            <div class="content">
                <form action="/predict" method="post">
                    <div class="form-group">
                        <label for="text">📰 Enter News Article Text:</label>
                        <textarea name="text" id="text" placeholder="Paste your news article here..." required></textarea>
                    </div>
                    
                    <div class="button-group">
                        <button type="submit">🚀 Check Article</button>
                        <button type="reset" class="clear-btn">Clear Text</button>
                    </div>
                </form>
                
                <div class="info-box">
                    <strong>📌 How It Works:</strong> Our machine learning model analyzes the text using advanced NLP techniques to predict if the article is likely fake or real. Results are based on patterns learned from the training dataset.
                </div>
            </div>
        </div>
    </body>
    </html>
    ''')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form.get('text', '')
        if not text:
            return render_template_string('''
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Error - Fake News Detection</title>
                <style>
                    * { margin: 0; padding: 0; box-sizing: border-box; }
                    body {
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        min-height: 100vh;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        padding: 20px;
                    }
                    .container {
                        background: white;
                        border-radius: 20px;
                        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                        max-width: 600px;
                        width: 100%;
                        padding: 40px;
                        text-align: center;
                    }
                    h1 { color: #e74c3c; margin-bottom: 20px; }
                    p { color: #555; margin-bottom: 30px; font-size: 1.1em; }
                    a {
                        display: inline-block;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 12px 30px;
                        border-radius: 10px;
                        text-decoration: none;
                        font-weight: 600;
                        transition: transform 0.2s;
                    }
                    a:hover { transform: translateY(-2px); }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>⚠️ Error</h1>
                    <p>Please enter some text to analyze.</p>
                    <a href="/">← Go Back</a>
                </div>
            </body>
            </html>
            ''')

        processed_text = preprocess_input(text)
        vectorized = vectorizer.transform([processed_text])
        prediction = model.predict(vectorized)[0]
        probability = model.decision_function(vectorized)[0]

        result = 'Fake' if prediction == 1 else 'Real'
        confidence = min(abs(probability) * 100, 100)
        result_color = '#e74c3c' if result == 'Fake' else '#27ae60'
        result_icon = '❌' if result == 'Fake' else '✅'
        
        return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Result - Fake News Detection</title>
            <style>
                * {
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }
                
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    padding: 20px;
                }
                
                .container {
                    background: white;
                    border-radius: 20px;
                    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                    max-width: 800px;
                    width: 100%;
                    overflow: hidden;
                }
                
                .result-section {
                    background: {{ result_color }};
                    color: white;
                    padding: 40px;
                    text-align: center;
                }
                
                .result-icon {
                    font-size: 4em;
                    margin-bottom: 15px;
                }
                
                .result-title {
                    font-size: 2.5em;
                    font-weight: 700;
                    margin-bottom: 10px;
                }
                
                .confidence-section {
                    padding: 30px 40px;
                }
                
                .confidence-box {
                    background: #f5f7fa;
                    border-left: 5px solid {{ result_color }};
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                }
                
                .confidence-label {
                    font-size: 0.9em;
                    color: #888;
                    margin-bottom: 10px;
                }
                
                .confidence-value {
                    font-size: 2.5em;
                    font-weight: 700;
                    color: {{ result_color }};
                }
                
                .progress-bar {
                    background: #e0e0e0;
                    height: 10px;
                    border-radius: 5px;
                    overflow: hidden;
                    margin-top: 10px;
                }
                
                .progress-fill {
                    background: {{ result_color }};
                    height: 100%;
                    width: {{ confidence }}%;
                    transition: width 0.5s ease;
                }
                
                .text-preview {
                    background: #f9f9f9;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                }
                
                .text-preview h3 {
                    color: #333;
                    margin-bottom: 10px;
                    font-size: 0.95em;
                }
                
                .text-preview p {
                    color: #666;
                    font-size: 0.9em;
                    max-height: 100px;
                    overflow-y: auto;
                    white-space: normal;
                    word-wrap: break-word;
                }
                
                .button-group {
                    display: flex;
                    gap: 10px;
                    justify-content: center;
                }
                
                a, button {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border: none;
                    padding: 12px 30px;
                    border-radius: 10px;
                    text-decoration: none;
                    font-weight: 600;
                    cursor: pointer;
                    transition: transform 0.2s;
                }
                
                a:hover, button:hover {
                    transform: translateY(-2px);
                }
                
                .info-text {
                    font-size: 0.85em;
                    color: #888;
                    margin-top: 20px;
                    font-style: italic;
                }
                
                @media (max-width: 600px) {
                    .result-title { font-size: 1.8em; }
                    .confidence-value { font-size: 1.8em; }
                    .button-group { flex-direction: column; }
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="result-section">
                    <div class="result-icon">{{ result_icon }}</div>
                    <div class="result-title">{{ result }} News</div>
                </div>
                
                <div class="confidence-section">
                    <div class="confidence-box">
                        <div class="confidence-label">Confidence Score</div>
                        <div class="confidence-value">{{ "%.1f"|format(confidence) }}%</div>
                        <div class="progress-bar">
                            <div class="progress-fill"></div>
                        </div>
                    </div>
                    
                    <div class="text-preview">
                        <h3>📰 Article Preview:</h3>
                        <p>{{ text[:300] }}{{ "..." if text|length > 300 else "" }}</p>
                    </div>
                    
                    <div class="button-group">
                        <a href="/">← Check Another Article</a>
                    </div>
                    
                    <div class="info-text">
                        ℹ️ This prediction is based on machine learning analysis and should be verified with trusted sources.
                    </div>
                </div>
            </div>
        </body>
        </html>
        ''', result_color=result_color, result_icon=result_icon, result=result, 
             confidence=confidence, text=text)

if __name__ == '__main__':
    app.run(debug=True)