import flask
from flask import Flask, request, render_template, Response
from flask_cors import CORS

from main_logic import service
import pandas as pd

app = flask.Flask(__name__)
CORS(app)


@app.route("/predict", methods=['POST'])
def predict():
    data = {"success": False}
    input_json = request.json
    try:
        prob_of_disease = service(input_json)

        if prob_of_disease>=0.5:
            # return render_template('form.html', pred=f'Your liver is in Danger.\nProbability of Liver disease is {prob_of_disease:.2f}')
            return f'Your liver is in Danger.\nProbability of Liver disease is {prob_of_disease:.2f}'
        else:
            # return render_template('form.html',pred=f'Your liver is safe.\n Probability of Liver disease is {prob_of_disease:.2f}')
            return f'Your liver is safe.\n Probability of Liver disease is {prob_of_disease:.2f}'
        return 
    except Exception as exc:
        print("exception")
        data["prediction_Exception"] = str(exc)
        return data
    

# if this is the main thread of execution first load the model and
@app.route("/")
def homepage():
    return "Hey Welcome to ML Liver Disease Predictor!"


if __name__ == "__main__":
    app.run()