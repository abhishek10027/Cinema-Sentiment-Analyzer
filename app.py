from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load your model and CountVectorizer
save_cv = pickle.load(open('count_vectorizer.pkl', 'rb'))
model_s = pickle.load(open('logistic_model.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    sentence = data['sentence']
    sen = save_cv.transform([sentence]).toarray()
    prediction = model_s.predict(sen)[0]
    prediction_text = 'Positive review' if prediction == 1 else 'Negative review'
    return jsonify({'prediction_text': prediction_text})

if __name__ == '__main__':
    app.run(debug=True)
