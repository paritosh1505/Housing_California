import pandas as pd
from flask import Flask,render_template,request,jsonify
import numpy as np
import pickle

app = Flask(__name__)

model_load = pickle.load(open('housing_california.pkl','rb'))
scaled_data= pickle.load(open('scaled_data.pkl','rb'))

@app.route("/")
def home_page():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(-1,1))
    new_entry = scaled_data.transform(np.array(list(data.values())).reshape(1,-1))
    print("***********************")
    pred_val =model_load.predict(new_entry)
    print("pred value",pred_val[0])
    return jsonify(pred_val[0])

@app.route('/UserPredict',methods=['POST'])
def UserPredict():
    data = [float(x) for x in request.form.values()]
    new_entry = scaled_data.transform(np.array(data).reshape(1,-1))
    pred_val =model_load.predict(new_entry)[0]
    return render_template("home.html",prediction_val = "The prediction value is {}".format(pred_val))

if __name__=="__main__":
    app.run(debug=True)

