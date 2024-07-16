from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load the CountVectorizer and model (example)
save_cv = pickle.load(open('count_vectorizer.pkl', 'rb'))  # Load your CountVectorizer model
model_s = pickle.load(open('logistic_model.pkl', 'rb'))  # Load your Logistic Regression model

# Serve the index.html file
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for handling POST requests to /predict
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the sentence from the POST request data
        sentence = request.form['sentence']

        # Transform the sentence using the CountVectorizer
        sen = save_cv.transform([sentence]).toarray()

        # Predict using the loaded model
        prediction = model_s.predict(sen)[0]

        # Determine prediction text based on the model's prediction
        prediction_text = 'Positive review' if prediction == 1 else 'Negative review'

        # Return prediction result as JSON response
        return jsonify({'prediction_text': prediction_text})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
