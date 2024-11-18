from flask import Flask, request, render_template, redirect, url_for
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load CountVectorizer and Logistic Regression model
save_cv = pickle.load(open('count_vectorizer.pkl', 'rb'))
model_s = pickle.load(open('logistic_model.pkl', 'rb'))

@app.route('/')
def home():
    # Render the home page
    return render_template('index.html', warning=None, prediction_text=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the user input from the form
        sentence = request.form.get('sentence', '')
        if not sentence.strip():
            warning = "Please enter a review to analyze."
            return render_template('index.html', warning=warning, prediction_text=None)
        
        # Transform the input using CountVectorizer
        transformed_input = save_cv.transform([sentence]).toarray()

        # Predict sentiment
        prediction = model_s.predict(transformed_input)[0]
        prediction_text = 'Positive review' if prediction == 1 else 'Negative review'

        return render_template('index.html', warning=None, prediction_text=prediction_text)
    except Exception as e:
        return render_template('index.html', warning=f"Error: {str(e)}", prediction_text=None)

@app.route('/reset')
def reset():
    # Redirect to the home page, clearing any predictions or warnings
    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=True)
