from flask import Flask, request, jsonify, send_file
import pickle

app = Flask(__name__)

# Load the CountVectorizer and model
save_cv = pickle.load(open('count_vectorizer.pkl', 'rb'))
model_s = pickle.load(open('logistic_model.pkl', 'rb'))

# Define route to serve your HTML file
@app.route('/')
def home():
    return send_file('index.html')


# Define the route for processing sentiment analysis
@app.route('/predict', methods=['POST'])
def predict():
    sentence = request.form['sentence']
    sen = save_cv.transform([sentence]).toarray()
    res = model_s.predict(sen)[0]
    if res == 1:
        prediction_text = 'Positive review'
    else:
        prediction_text = 'Negative review'
    return jsonify({'prediction_text': prediction_text})

if __name__ == '__main__':
    app.run(debug=True)
