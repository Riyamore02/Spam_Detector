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
bash
Copy
Edit
git clone https://github.com/yourusername/spam_classifier.git
cd spam_classifier
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Run the Notebook
Open and run the Jupyter Notebook:

jupyter notebook spam_classifier.ipynb
4. Predict
You can input a new message at the bottom of the notebook or use the model in a separate Python file:

import pickle

model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

message = ["Congratulations! You've won a prize!"]
vector = vectorizer.transform(message)
print(model.predict(vector))

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
