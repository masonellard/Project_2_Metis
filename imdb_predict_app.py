#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 14:47:44 2020

@author: mason
"""

import streamlit as st
import pickle
import xgboost
import pandas as pd
import numpy as np

with open('imdb_xgb.pickle','rb') as read_file:
    xgb = pickle.load(read_file)
    
with open('df_test.pickle','rb') as read_file:
    df_test = pickle.load(read_file)

with open('df_train.pickle','rb') as read_file:
    df_train = pickle.load(read_file)
    
with open('y_test.pickle','rb') as read_file:
    y_test = pickle.load(read_file)

with open('dir_means.pickle','rb') as read_file:
    dir_means = pickle.load(read_file)

with open('wri_means.pickle','rb') as read_file:
    wri_means = pickle.load(read_file)

with open('cast_means.pickle','rb') as read_file:
    cast_means = pickle.load(read_file)


st.title('Domestic Gross Revenue Predictor')

st.markdown('## Enter Movie Parameters:')

budget = st.number_input('Budget:',min_value=0, value=0, step=1)

runtime = st.number_input('Runtime (in minutes):', min_value=0, value=0, step=1)

days = st.number_input('Number of Days After Release Date:', min_value=0, value=360, step=1)

Ratings = ['R', 'PG-13', 'PG']
rating = st.selectbox('Select a Rating:', Ratings)

genres = ['Horror', 'Crime', 'Thriller', 'Sci-Fi', 'Romance', 'Action',
          'Fantasy', 'Adventure', 'Drama', 'Animation', 'Family']
gen = st.multiselect('Genre(s):', genres)

directors = st.multiselect('Directors:', dir_means['directors'])
writers = st.multiselect('Writers:', wri_means['writers'])
cast = st.multiselect('Cast:', cast_means['cast'])


try:
    director_ranks = []
    for director in directors:
        director_ranks.append(int(dir_means[dir_means['directors']==director].dir_rank))
        dir_rank = max(director_ranks)
except:
    None

try:
    writer_ranks = []
    for writer in writers:
        writer_ranks.append(int(wri_means[wri_means['writers']==writer].wri_rank))        
        wri_rank = max(writer_ranks)
except:
    None

try:
    cast_ranks = []
    for actor in cast:
        cast_ranks.append(int(cast_means[cast_means['cast']==actor].cast_rank))        
        cast_rank = max(cast_ranks)
except:
    None

st.markdown('## Domestic Gross Revenue Prediction:')
try:
    df = pd.DataFrame(columns=df_train.columns)
    df.loc[0] = [np.uint8(bool('Horror' in gen)),
                 np.int64(wri_rank),
                 np.int64(dir_rank),
                 np.float64(budget),
                 np.uint8(bool('Crime' in gen)),
                 np.uint8(bool('Thriller' in gen)),
                 np.uint8(bool('Sci-Fi' in gen)),
                 np.float64(runtime),
                 np.uint8(bool(rating == 'PG-13')),
                 np.uint8(bool('Romance' in gen)),
                 np.int64(cast_rank),
                 np.uint8(bool(rating == 'PG')),
                 np.uint8(bool('Action' in gen)),
                 np.uint8(bool('Fantasy' in gen)),
                 np.int64(days),
                 np.uint8(bool(rating == 'R')),
                 np.uint8(bool('Adventure' in gen)),
                 np.uint8(bool('Drama' in gen)),
                 np.uint8(bool('Animation' in gen)),
                 np.uint8(bool('Family' in gen))]

    df_merged = pd.concat([df_test, df], axis=0)
    
    predict = xgb.predict(df_merged)
    
    gross = (np.exp(predict[-1])*100000).round(2)
    st.header(str(gross))
except:
    st.write('')