# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 20:52:09 2020

@author: admin
"""
import numpy as np
import pickle
from sklearn.svm import SVC

from flask import Flask,render_template,request,url_for


model=pickle.load(open('iris.pkl','rb'))

app=Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])

def home():
    data1=int(request.form['a'])
    data2=int(request.form['b'])
    data3=int(request.form['c'])
    data4=int(request.form['d'])
    
    arr=np.array([[data1,data2,data3,data4]])
    pred=model.predict(arr)
    pred=pred[0]
    
    return render_template('after.html',data=pred)
if __name__=='__main__':
    app.run(debug=True)