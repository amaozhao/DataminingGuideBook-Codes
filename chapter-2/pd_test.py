# coding: utf-8
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import os

path = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), 'Movie_Ratings.csv')
data = pd.read_csv(path)


pearson_result = DataFrame(
    np.ones(len(data.columns) ** 2).reshape(
        len(data.columns),
        len(data.columns)),
    index=data.columns.values,
    columns=data.columns.values
)

euclidean_result = DataFrame(
    np.ones(len(data.columns) ** 2).reshape(
        len(data.columns),
        len(data.columns)),
    index=data.columns.values,
    columns=data.columns.values
)


def set_pearson(data=data, result=pearson_result):
    users = data.columns
    for user in users:
        for u in users.drop(user):
            result[user][u] = data[user].corr(data[u])


def pearson(s1, s2):
    return s1.corr(s2)


def get_corr_users(username, data=data):
    return data.columns.values[data.columns.values != username]


def get_corr_series(username, data=data):
    p_d = dict()
    for user in get_corr_users(username):
        p_d[user] = pearson(data[username], data[user])
    return Series(p_d, index=p_d.keys())


def get_recommend(username, data=data, pearson_result=pearson_result):
    pearson_user = pearson_result[username].drop(username).index
    current_user_index = data[username].dropna().index
    result = Series()
    for user in pearson_user:
        no_index = data[user].dropna().index.difference(
            current_user_index)
        for i in no_index:
            rating = data[user][i] * pearson_result[username][user]
            if result.get(i, None) and (result[i] > rating):
                result[i] = rating
            else:
                result[i] = rating
    return result.sort_values(ascending=False)


if __name__ == '__main__':
    set_pearson()
    username = 'Thomas'
    print (get_recommend(username))
