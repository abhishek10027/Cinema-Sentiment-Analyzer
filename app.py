from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the CountVectorizer and Logistic Regression model
with open('count_vectorizer.pkl', 'rb') as cv_file:
    save_cv = pickle.load(cv_file)
with open('logistic_model.pkl', 'rb') as model_file:
    model_s = pickle.load(model_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    sentence = request.form['sentence']
    transformed_input = save_cv.transform([sentence]).toarray()

    # Make prediction
    prediction = model_s.predict(transformed_input)[0]
    output = 'Positive review' if prediction == 1 else 'Negative review'

    return render_template('index.html', prediction_text='Prediction: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)

