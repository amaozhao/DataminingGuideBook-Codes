# coding: utf-8
import pandas as pd
import numpy as np
from pandas import Series, DataFrame

data = pd.read_csv('Movie_Ratings.csv')


def pearson(s1, s2):
    return s1.corr(s2)
    
def get_corr_users(username, data=data):
    return data.columns.values[data.columns.values != username]

def get_corr_series(username, data=data):
    p_d = dict()
    for user in get_corr_users(username):
        p_d[user] = pearson(data[username], data[user])
    return Series(p_d, index=p_d.keys())
    

def get_recommend(username, data=data):
    corr_s = get_corr_series(username, data)
    result = dict()
    r = data * corr_s
    for c in r.columns.values:
        for i in r[c].index.values:
            if result.get(i, 0) < r[c][i]:
                result[i] = r[c][i]
    return Series(result, index=result.keys()).sort_values(ascending=False)
