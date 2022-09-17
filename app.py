import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("wine.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    alcohol = request.form.get("alcohol")
    flavanoids = request.form.get("flavanoids")
    color_intensity = request.form.get("color_intensity")
    hue = request.form.get("hue")
    od_diluted_wines = request.form.get("od_diluted_wines")
    proline=request.form.get("proline")

 
    real_values = np.array([[alcohol,flavanoids,color_intensity,hue,od_diluted_wines,proline]],dtype=float)

    prediction = model.predict(real_values)

    if prediction==[0]:
        prediction='bad'
    elif prediction==[1]:
        prediction='good'
    else:
        prediction='best'
    

              

    return render_template("index.html",prediction=prediction)

if __name__ == "__main__":
    flask_app.run()