from flask import Flask, request, jsonify, render_template
import joblib

# Load the trained model
# Load the trained model
model = joblib.load('sentiment_svm_model.pkl')  # Use a relative path

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']  # Get the input text from the form
    sentiment = model.predict([text])[0]  # Predict sentiment
    return jsonify({'sentiment': sentiment})  # Return the sentiment as JSON

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

