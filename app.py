from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained machine learning model
model = pickle.load(open("marks_predictor_model.pkl", "rb"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    # Convert input marks to a numpy array
    marks = np.array([[
        float(data["subject1"]),
        float(data["subject2"]),
        float(data["subject3"]),
        float(data["subject4"])
    ]])
    # Predict next semester marks
    predicted_marks = model.predict(marks)
    return jsonify({"predictedMarks": round(predicted_marks[0], 2)})

if __name__ == "__main__":
    app.run(debug=True)
