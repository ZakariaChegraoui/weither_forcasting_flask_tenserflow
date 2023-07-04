from flask import Flask, render_template, request, abort, Response, redirect, send_file,jsonify
import json
import io
from datetime import datetime
from dateutil.relativedelta import relativedelta
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import copy
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

app = Flask(__name__)

#http://127.0.0.1:5000/forecast?DATE=2023-07-18&number=5&Temperature1=25.6&Rainfull1=0.01&Temperature2=25.6&Rainfull2=0.01&Temperature3=25.6&Rainfull3=0.01
#http://127.0.0.1:5000/plots?DATE=2023-07-18&number=5&Temperature1=25.6&Rainfull1=0.01&Temperature2=25.6&Rainfull2=0.01&Temperature3=25.6&Rainfull3=0.01

@app.route('/forecast')
def get_weather():
    data = {'DATE' : request.args.get('DATE'),
            'Temperature1' : float(request.args.get('Temperature1')),
            'Rainfull1' : float(request.args.get('Rainfull1')),
            'Temperature2' : float(request.args.get('Temperature2')),
            'Rainfull2' : float(request.args.get('Rainfull2')),
            'Temperature3' : float(request.args.get('Temperature3')),
            'Rainfull3' : float(request.args.get('Rainfull3')),
            'number' : int(request.args.get('number'))}
    
    number = data['number']
    
    scaler = joblib.load('scaler.save') 
    print(data['DATE'])
    date = datetime.strptime(data['DATE'], '%Y-%m-%d').date()

    date1 = copy.copy(date)

    X_pred1 = np.array([[0]*3+[data['Temperature1'],data['Temperature1'],data['Temperature1'],data["Rainfull1"]]+[date.day,date.month,date.year]]).reshape((1,10))
    date1 = date1+relativedelta(days=1)
    X_pred2 = np.array([[0]*3+[data['Temperature2'],data['Temperature2'],data['Temperature2'],data["Rainfull2"]]+[date1.day,date1.month,date1.year]]).reshape((1,10))
    date1 = date1+relativedelta(days=1)
    X_pred3 = np.array([[0]*3+[data['Temperature3'],data['Temperature3'],data['Temperature3'],data["Rainfull3"]]+[date1.day,date1.month,date1.year]]).reshape((1,10))
    
    X_pred1 = scaler.transform(X_pred1)[0][3:]
    X_pred2 = scaler.transform(X_pred2)[0][3:]
    X_pred3 = scaler.transform(X_pred3)[0][3:]
   
    X_pred = np.append([X_pred1],[X_pred2], axis=0)
    X_pred = np.append(X_pred,[X_pred3], axis=0)
    X_pred = X_pred.reshape(1,3,7)
    
    new_model = tf.keras.models.load_model('Model.h5')
    #print(scaler.transform(X_pred))

  
    forcasted = []

    for i in range(int(number)):
        yhat = new_model.predict(X_pred)
       
        l = list(yhat[0][0])
        
        l = list(scaler.inverse_transform(np.array(l+[1,1,2023]).reshape((1,10)))[0])[:7]

        #c = random.random()*5
        l_ = [abs(round(x,2)) for x in l]

        forcasted.append([str(date+relativedelta(days=i+1))]+l_)


        X_pred_ = np.array(l+[(date+relativedelta(days=i+1)).day,(date+relativedelta(days=i+1)).month,(date+relativedelta(days=i+1)).year]).reshape((1,10))
        X_pred_ = scaler.transform(X_pred_)[0][3:]
        
        X_pred = np.append(X_pred[0][1:],[X_pred_], axis=0).reshape(1,3,7)
        
    
    return jsonify(values = forcasted)


if __name__ == "__main__":
    
    cors = CORS(app)
    app.run()