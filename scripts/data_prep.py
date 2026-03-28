import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def preprocess_text(text):
    """Preprocess the text: lowercase, remove punctuation, stopwords, lemmatize."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

def load_and_preprocess_data(filepath):
    """Load data, preprocess, and split."""
    df = pd.read_csv(filepath)
    df['text'] = df['text'].fillna('')
    df['title'] = df['title'].fillna('')
    df['content'] = df['title'] + ' ' + df['text']  # Combine title and text
    df['processed_text'] = df['content'].apply(preprocess_text)
    df = df[['processed_text', 'label']]  # Assuming 'label' is 0 for real, 1 for fake
    return df

def vectorize_data(df):
    """Vectorize the text data."""
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['processed_text'])
    y = df['label']
    return X, y, vectorizer

if __name__ == "__main__":
    # Assume data is in data/train.csv
    df = load_and_preprocess_data('data/train.csv')
    X, y, vectorizer = vectorize_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save processed data
    joblib.dump((X_train, X_test, y_train, y_test), 'data/processed_data.pkl')
    joblib.dump(vectorizer, 'models/vectorizer.pkl')
    print("Data preprocessing completed.")