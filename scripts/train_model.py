import joblib
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def train_model():
    """Train the fake news detection model."""
    # Load processed data
    X_train, X_test, y_train, y_test = joblib.load('data/processed_data.pkl')

    # Train model
    model = PassiveAggressiveClassifier(max_iter=50)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('models/confusion_matrix.png')
    # plt.show()  # Remove show for headless

    # Save model
    joblib.dump(model, 'models/fake_news_model.pkl')
    print("Model trained and saved.")

if __name__ == "__main__":
    train_model()