# Fake News Detection System

This project implements a machine learning-based fake news detection system using natural language processing techniques.

## Features

- Data collection from news APIs
- Text preprocessing and feature extraction
- Machine learning model training (Passive Aggressive Classifier)
- Web application for real-time prediction using Flask
- REST API for integration

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fake-news-detection.git
   cd fake-news-detection
   ```

2. Set up Python environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

1. Place your dataset in `data/train.csv` with columns: id, title, text, label (0 for real, 1 for fake).
2. Run the data preprocessing:
   ```bash
   python scripts/data_prep.py
   ```
3. Train the model:
   ```bash
   python scripts/train_model.py
   ```

### Running the Web App

Start the Flask application:
```bash
python app/app.py
```

Open your browser to `http://localhost:5000` and enter a news article to check if it's fake or real.

### API Usage

Send a POST request to `/predict` with JSON payload:
```json
{
  "text": "Your news article text here"
}
```

Response:
```json
{
  "prediction": "Fake",
  "confidence": 0.85
}
```

## Dataset

The system uses a CSV dataset with news articles. For a real dataset, use the Fake News Dataset from Kaggle (https://www.kaggle.com/c/fake-news/data).

## Model

Trained using TF-IDF vectorization and Passive Aggressive Classifier.

## Project Structure

```
fake-news-detection/
├── data/
│   ├── train.csv
│   └── processed_data.pkl
├── models/
│   ├── fake_news_model.pkl
│   ├── vectorizer.pkl
│   └── confusion_matrix.png
├── scripts/
│   ├── data_prep.py
│   └── train_model.py
├── app/
│   └── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

MIT License