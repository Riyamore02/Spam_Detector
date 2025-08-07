# Spam_Detector

# 📧 Spam Classifier - Machine Learning Project

A simple machine learning model to classify SMS messages as **Spam** or **Not Spam (Ham)** using **Natural Language Processing (NLP)** techniques and the **logistic Regression** algorithm.

## 📌 Overview

This project uses a dataset of SMS messages labeled as spam or ham. The text data is preprocessed, vectorized using **TF-IDF**, and then fed into a **logistic Regression** to detect spam messages with high accuracy.

---

## 🛠️ Features

- Text preprocessing 
- Vectorization with **TF-IDF**
- Model training using **Logistic Regression**
- Evaluation metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- Model persistence using `pickle`
- Predictive interface (optional CLI or notebook input)

---

## 📂 Project Structure

```bash
spam_classifier/
│
├── spam_classifier.ipynb       # Main Jupyter notebook with code
├── model.pkl                   # Trained model (pickled)
├── vectorizer.pkl              # TF-IDF vectorizer (pickled)
├── README.md                   # This file
├── requirements.txt            # List of dependencies
└── dataset/
    └── spam.csv                # SMS Spam Collection Dataset
📊 Dataset
Source: UCI SMS Spam Collection

Format: CSV

Columns:

label: spam or ham

message: text content of the SMS

⚙️ How to Run
1. Clone the Repository
git clone https://github.com/yourusername/spam_classifier.git
cd spam_classifier

2. Install Dependencies
pip install -r requirements.txt

3. Run the Notebook
Open and run the Jupyter Notebook:
jupyter notebook spam_classifier.ipynb

4. Predict
You can input a new message at the bottom of the notebook or use the model in a separate Python file:

def predict_message(msg):
    msg_clean = preprocess(msg)
    msg_vec = tfidf.transform([msg_clean])
    prediction = model.predict(msg_vec)[0]
    return 'Spam' if prediction == 1 else 'Ham'

test_messages = [
    "URGENT! You have won a lottery worth 2 crores. Call this number now!",
]

for msg in test_messages:
    print(f" Message: {msg}")
    print(f" Prediction: {predict_message(msg)}\n")

📦 Requirements
pandas
numpy
scikit-learn
matplotlib
seaborn
nltk
jupyter
joblib
flask

📈 Model Performance
Accuracy: ~98%
Precision: ~97%
F1-Score: ~97%


🧑‍💻 Author
Riya More

📝 License
This project is licensed under the MIT License.
