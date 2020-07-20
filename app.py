# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 15:12:58 2020

@author: zainu
"""


import numpy as np
from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

data = pd.read_csv("Amazon - Movies and TV Ratings-Revised.csv")
data1 = pd.read_csv("Amazon - Movies and TV Ratings.csv")

data = data.fillna(0)

user_ids = list(data1['user_id'])
movie_ids= []
for col in data.columns:
    if col != 'user_id':
        movie_ids.append(col)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    user_i = [str(x) for x in request.form.values()]
    print(user_i)
    
    user = user_ids.index(user_i[0])
    distances, indices = model.kneighbors(data.iloc[user,:].values.reshape(1, -1), n_neighbors = 6)
    a  = data.iloc[indices.flatten()[1]].tolist()
    a = a[1:]
    ind = a.index(max(a))
    
    return render_template('index.html', prediction_text='The recommended movie is $ {}'.format(movie_ids[ind]))

if __name__ == "__main__":
    app.run(debug=True)