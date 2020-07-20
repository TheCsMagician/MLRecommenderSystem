# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 14:17:58 2020

@author: zainu
"""


import pandas as pd
import numpy as np
import pickle


data = pd.read_csv("Amazon - Movies and TV Ratings-Revised.csv")
data1 = pd.read_csv("Amazon - Movies and TV Ratings.csv")

user_ids = data1['user_id']
movie_ids = []

for col in data.columns:
    if col != 'user_id':
        movie_ids.append(col)
lst = []
data = data.fillna(0)




from scipy.sparse import csr_matrix

movie_features_df_matrix = csr_matrix(data.values)

from sklearn.neighbors import NearestNeighbors



#Recommends Movie to a particular user according to dataset
model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(movie_features_df_matrix)




pickle.dump(model_knn,open('model.pkl','wb'))





