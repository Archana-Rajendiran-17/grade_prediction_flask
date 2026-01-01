from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
with open("model/grade_model.pkl", "rb") as file:
    model = pickle.load(file)
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        hours = float(request.form["hours"])
        prediction = model.predict(np.array([[hours]]))[0]
        prediction = round(prediction, 2)
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
