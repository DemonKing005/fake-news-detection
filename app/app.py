from flask import Flask, request, jsonify, render_template_string
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# News API Configuration (Get free API key from newsapi.org)
NEWS_API_KEY = os.environ.get('NEWS_API_KEY', '')  # Set as environment variable
NEWS_API_URL = 'https://newsapi.org/v2/top-headlines'
CACHE_TIMEOUT = 3600  # Cache for 1 hour
last_fetch_time = None
cached_articles = None

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

# Sample articles for display - Global News from various domains and regions
SAMPLE_ARTICLES = [
    {
        'title': 'India Launches Revolutionary Space Mission to Mars',
        'excerpt': 'Indian Space Research Organization successfully launched its latest mission to explore potential water deposits on Mars...',
        'source': 'ISRO News',
        'date': 'Mar 28, 2026',
        'category': 'Science',
        'region': '🇮🇳 India'
    },
    {
        'title': 'European Union Reaches Net-Zero Carbon Target',
        'excerpt': 'EU member states announce significant progress toward 2030 carbon neutrality goals with renewable energy investments...',
        'source': 'European Commission',
        'date': 'Mar 27, 2026',
        'category': 'Environment',
        'region': '🇪🇺 Europe'
    },
    {
        'title': 'Japan Introduces AI-Powered Healthcare System',
        'excerpt': 'Tokyo Medical Center deploys artificial intelligence to diagnose rare diseases with 99.2% accuracy rate...',
        'source': 'Japan Times',
        'date': 'Mar 26, 2026',
        'category': 'Technology',
        'region': '🇯🇵 Japan'
    },
    {
        'title': 'Brazil Discovers New Renewable Energy Source',
        'excerpt': 'Brazilian researchers develop efficient green hydrogen technology that could replace fossil fuels within a decade...',
        'source': 'Folha de S.Paulo',
        'date': 'Mar 25, 2026',
        'category': 'Energy',
        'region': '🇧🇷 Brazil'
    },
    {
        'title': 'African Nations Form Tech Innovation Hub',
        'excerpt': 'Kenya, Nigeria, and South Africa collaborate to establish a continental technology development center...',
        'source': 'African Union',
        'date': 'Mar 24, 2026',
        'category': 'Technology',
        'region': '🌍 Africa'
    },
    {
        'title': 'Australia Expands Quantum Computing Research',
        'excerpt': 'Australian universities receive record funding for quantum computing projects with global applications...',
        'source': 'Australian Tech Review',
        'date': 'Mar 23, 2026',
        'category': 'Science',
        'region': '🇦🇺 Australia'
    },
    {
        'title': 'Canada Launches Universal Basic Income Pilot',
        'excerpt': 'Canadian province initiates groundbreaking universal basic income experiment affecting 50,000 residents...',
        'source': 'CBC News',
        'date': 'Mar 22, 2026',
        'category': 'Policy',
        'region': '🇨🇦 Canada'
    },
    {
        'title': 'Singapore Becomes First Smart City Nation',
        'excerpt': 'Singapore completes its transformation into a fully integrated smart city with IoT and AI infrastructure...',
        'source': 'Singapore News',
        'date': 'Mar 21, 2026',
        'category': 'Technology',
        'region': '🇸🇬 Singapore'
    },
    {
        'title': 'Global Stock Markets Reach All-Time High',
        'excerpt': 'International financial markets celebrate unprecedented gains driven by tech innovation and economic recovery...',
        'source': 'Bloomberg',
        'date': 'Mar 20, 2026',
        'category': 'Business',
        'region': '🌐 Global'
    },
    {
        'title': 'New Medical Breakthrough in Cancer Treatment',
        'excerpt': 'International research team announces 85% success rate in new immunotherapy for multiple cancer types...',
        'source': 'Medical Times',
        'date': 'Mar 19, 2026',
        'category': 'Health',
        'region': '🌐 Global'
    },
    {
        'title': 'UAE Completes Largest Solar Farm Project',
        'excerpt': 'United Arab Emirates inaugurates massive solar facility producing enough clean energy for 2 million homes...',
        'source': 'Gulf Today',
        'date': 'Mar 18, 2026',
        'category': 'Energy',
        'region': '🇦🇪 UAE'
    },
    {
        'title': 'UK Reaches Milestone in Green Transport',
        'excerpt': 'Britain announces 10 million electric vehicles on roads, surpassing all European peers in EV adoption...',
        'source': 'BBC News',
        'date': 'Mar 17, 2026',
        'category': 'Environment',
        'region': '🇬🇧 UK'
    }
]

def fetch_latest_news():
    """Fetch latest news from NewsAPI with fallback to sample articles"""
    global last_fetch_time, cached_articles
    
    # Return cached articles if still valid
    if cached_articles and last_fetch_time:
        if datetime.now() - last_fetch_time < timedelta(seconds=CACHE_TIMEOUT):
            return cached_articles
    
    # Try to fetch from NewsAPI
    if NEWS_API_KEY:
        try:
            params = {
                'apiKey': NEWS_API_KEY,
                'language': 'en',
                'pageSize': 12,
                'sortBy': 'publishedAt'
            }
            response = requests.get(NEWS_API_URL, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            if data.get('status') == 'ok' and data.get('articles'):
                articles = []
                category_map = {
                    'technology': 'Technology',
                    'science': 'Science',
                    'health': 'Health',
                    'business': 'Business',
                    'entertainment': 'Entertainment',
                    'sports': 'Sports',
                    'general': 'News'
                }
                
                region_map = {
                    'cnn': '🇺🇸',
                    'bbc': '🇬🇧',
                    'reuters': '🌐',
                    'deutsche': '🇩🇪',
                    'france': '🇫🇷',
                    'nikkei': '🇯🇵',
                    'hindustantimes': '🇮🇳',
                    'chinadaily': '🇨🇳',
                }
                
                for article in data['articles'][:12]:
                    try:
                        source_name = article.get('source', {}).get('name', 'News API')
                        region = '🌐 Global'
                        for key, val in region_map.items():
                            if key.lower() in source_name.lower():
                                region = f'{val} {source_name}'
                                break
                        
                        articles.append({
                            'title': article.get('title', 'Untitled')[:80],
                            'excerpt': article.get('description', article.get('content', 'No description'))[:150],
                            'source': source_name,
                            'date': article.get('publishedAt', '').split('T')[0] if article.get('publishedAt') else 'Today',
                            'category': 'News',
                            'region': region,
                            'url': article.get('url', '#')
                        })
                    except:
                        continue
                
                if articles:
                    cached_articles = articles
                    last_fetch_time = datetime.now()
                    return articles
        except Exception as e:
            print(f"Error fetching news from API: {e}")
    
    # Return sample articles as fallback
    return SAMPLE_ARTICLES

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
    # Fetch latest news from API or use sample articles
    articles = fetch_latest_news()
    articles_html = ''.join([f'''
    <div class="article-card">
        <div class="article-header">
            <span class="article-region">{article['region']}</span>
            <span class="article-category">{article['category']}</span>
        </div>
        <h4>{article['title']}</h4>
        <p>{article['excerpt']}</p>
        <div class="article-footer">
            <span class="article-source">📰 {article['source']}</span>
            <span class="article-date">{article['date']}</span>
        </div>
    </div>
    ''' for article in SAMPLE_ARTICLES])
    
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
                padding: 20px;
            }
            
            .main-container {
                max-width: 1400px;
                margin: 0 auto;
                display: grid;
                grid-template-columns: 2fr 1fr;
                gap: 20px;
            }
            
            .left-section {
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                overflow: hidden;
            }
            
            .right-section {
                display: flex;
                flex-direction: column;
                gap: 20px;
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
                min-height: 180px;
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
            
            .sidebar-header {
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                padding: 25px;
                text-align: center;
            }
            
            .sidebar-header h2 {
                font-size: 1.5em;
                color: #333;
                margin-bottom: 5px;
            }
            
            .sidebar-header p {
                color: #888;
                font-size: 0.9em;
            }
            
            .articles-container {
                display: flex;
                flex-direction: column;
                gap: 12px;
                max-height: calc(100vh - 280px);
                overflow-y: auto;
                padding: 0 5px 5px 0;
            }
            
            .articles-container::-webkit-scrollbar {
                width: 7px;
            }
            
            .articles-container::-webkit-scrollbar-track {
                background: rgba(255,255,255,0.1);
                border-radius: 10px;
            }
            
            .articles-container::-webkit-scrollbar-thumb {
                background: #667eea;
                border-radius: 10px;
            }
            
            .article-card {
                background: white;
                border-radius: 15px;
                padding: 20px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                transition: transform 0.3s, box-shadow 0.3s;
                cursor: pointer;
                border-left: 5px solid #667eea;
            }
            
            .article-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 15px 40px rgba(0,0,0,0.3);
            }
            
            .article-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 12px;
                gap: 10px;
            }
            
            .article-region {
                font-size: 0.75em;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 4px 10px;
                border-radius: 5px;
                font-weight: 600;
                white-space: nowrap;
            }
            
            .article-category {
                font-size: 0.75em;
                background: #e8f0ff;
                color: #667eea;
                padding: 4px 10px;
                border-radius: 5px;
                font-weight: 600;
                white-space: nowrap;
            }
            
            .article-card h4 {
                font-size: 0.95em;
                color: #333;
                margin-bottom: 10px;
                line-height: 1.3;
                font-weight: 600;
            }
            
            .article-card p {
                font-size: 0.85em;
                color: #666;
                margin-bottom: 12px;
                line-height: 1.5;
                min-height: 40px;
            }
            
            .article-footer {
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-size: 0.8em;
                color: #999;
                border-top: 1px solid #f0f0f0;
                padding-top: 10px;
            }
            
            .article-source {
                color: #667eea;
                font-weight: 600;
            }
            
            .article-date {
                color: #999;
            }
            
            .refresh-btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 8px;
                font-size: 0.9em;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s;
                margin-top: 12px;
                display: inline-block;
            }
            
            .refresh-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
            }
            
            .refresh-btn:active {
                transform: translateY(0);
            }
            
            .refresh-btn.spinning {
                animation: spin 0.6s ease-in-out;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            @media (max-width: 1024px) {
                .main-container {
                    grid-template-columns: 1fr;
                }
                
                .articles-container {
                    max-height: 300px;
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 15px;
                }
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
                
                .articles-container {
                    grid-template-columns: 1fr;
                }
                
                .button-group {
                    flex-direction: column;
                }
            }
        </style>
    </head>
    <body>
        <div class="main-container">
            <!-- Left Section: Input Form -->
            <div class="left-section">
                <div class="header">
                    <h1>🔍 Fake News Detector</h1>
                    <p>Advanced AI-Powered Misinformation Detection</p>
                </div>
                
                <div class="content">
                    <form action="/predict" method="post">
                        <div class="form-group">
                            <label for="text">📰 Paste News Article:</label>
                            <textarea name="text" id="text" placeholder="Enter or paste your news article here to check if it's fake or real..." required></textarea>
                        </div>
                        
                        <div class="button-group">
                            <button type="submit">🚀 Analyze Article</button>
                            <button type="reset" class="clear-btn">🗑️ Clear</button>
                        </div>
                    </form>
                    
                    <div class="info-box">
                        <strong>📌 How It Works:</strong> Our machine learning model analyzes news articles using advanced NLP techniques to detect potential misinformation. Results show a confidence score indicating how likely the article is fake.
                    </div>
                </div>
            </div>
            
            <!-- Right Section: Featured Articles -->
            <div class="right-section">
                <div class="sidebar-header">
                    <h2>📰 Featured News</h2>
                    <p>Recent articles you can check</p>
                    <button class="refresh-btn" onclick="refreshArticles()">🔄 Refresh News</button>
                </div>
                
                <div class="articles-container" id="articlesContainer">
                    ''' + articles_html + '''
                </div>
            </div>
        </div>
        
        <script>
            const allArticles = ''' + repr(articles) + ''';
            
            function refreshArticles() {
                const btn = document.querySelector('.refresh-btn');
                btn.classList.add('spinning');
                
                // Shuffle and update articles
                setTimeout(() => {
                    const shuffled = [...allArticles].sort(() => Math.random() - 0.5);
                    const container = document.getElementById('articlesContainer');
                    container.innerHTML = shuffled.map(article => `
                        <div class="article-card">
                            <div class="article-header">
                                <span class="article-region">${article.region}</span>
                                <span class="article-category">${article.category}</span>
                            </div>
                            <h4>${article.title}</h4>
                            <p>${article.excerpt}</p>
                            <div class="article-footer">
                                <span class="article-source">📰 ${article.source}</span>
                                <span class="article-date">${article.date}</span>
                            </div>
                        </div>
                    `).join('');
                    
                    btn.classList.remove('spinning');
                }, 600);
            }
        </script>
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