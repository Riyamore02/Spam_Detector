from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load your pre-trained pipeline
pipeline = joblib.load("spam_classifier_pipeline.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    prediction = pipeline.predict([message])
    result = 'Spam' if prediction[0] == 1 else 'Not Spam'
    return render_template('index.html', message=message, result=result)

if __name__ == '__main__':
    app.run(debug=True)
