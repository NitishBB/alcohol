import json
import pickle
from flask import Flask, Response,jsonify,request,app,url_for,render_template

import numpy as np
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


@app.route("/predict_api", methods=['POST'])
def predict_api():
    try:
        if request.json['data'] is not None:
            data = request.json['data']
            print('data is:     ', data)
            res = predict_log(data)
            print('result is        ',res)
            return Response(res)
    except ValueError:
        return Response("Value not found")
    except Exception as e:
        print('exception is   ',e)
        return Response(e)

def predict_log(dict_pred):
    with open("scaler.pkl", 'rb') as f:
        scalar = pickle.load(f)

    with open("mymodel.pkl", 'rb') as f:
        model = pickle.load(f)

    data_df = pd.DataFrame(dict_pred,index=[1,])
    scaled_data = scalar.transform(data_df)
    predict = model.predict(scaled_data)
    if predict[0] == 3:
        result = 'Bad'
    elif predict[0] == 4 :
        result = 'Below Average'
    elif predict[0]==5:
        result = 'Average'
    elif predict[0] == 6:
        result = 'Good'
    elif predict[0] == 7:
        result = 'Very Good'
    else :
        result = 'Excellent'

    return result
if __name__ == "__main__":
    app.run(debug=True)