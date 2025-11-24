from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("grade_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = [
        float(request.form["hours"]),
        float(request.form["attendance"]),
        float(request.form["previous"]),
        float(request.form["sleep"]),
        float(request.form["participation"])
    ]
    output = model.predict([data])[0]
    return render_template("index.html", prediction=round(output, 2))

if __name__ == "__main__":
    app.run(debug=True)
