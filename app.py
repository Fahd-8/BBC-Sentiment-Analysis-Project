from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)
    return jsonify({'sentiment': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True , port=5001)
