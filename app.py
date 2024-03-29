import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

import preprocessing_functions

app = Flask(__name__)

## Load the model
model=pickle.load(open('catboost_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    new_data=preprocessing_functions.preprocess_data(pd.DataFrame([data]))
    print(f'NEW DATA SHAPE: {new_data.shape}')
    output=model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    # Get field names and values from the form
    form_data = request.form.items()

    print(form_data)

    # Extract field names and values separately
    field_names = [item[0] for item in form_data]
    field_values = [x for x in request.form.values()]

    print(field_names)
    print(field_values)

    # Create a DataFrame with field names as columns
    df = pd.DataFrame([field_values], columns=field_names)

    print(df)


    final_input=preprocessing_functions.preprocess_data(df)
    print(final_input)
    output=model.predict(final_input)[0]
    return render_template("home.html",prediction_text="The Prediction for Price of the Listing is {} GBP".format(output))



if __name__=="__main__":
    app.run(debug=True)