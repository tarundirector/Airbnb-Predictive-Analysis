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
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=preprocessing_functions.preprocess_data(pd.DataFrame([data]))
    print(f'NEW DATA SHAPE: {new_data.shape}')
    output=model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=preprocessing_functions.preprocess_data(np.array(data).reshape(1,-1))
    print(final_input)
    output=model.predict(final_input)[0]
    return render_template("home.html",prediction_text="The House price prediction is {}".format(output))



if __name__=="__main__":
    app.run(debug=True)