# coding: utf-8

import pandas as pd
import numpy as np
from pandas import DataFrame

data = pd.read_csv('../chapter-2/Movie_Ratings.csv')


def get_slope(m1, m2, dataframe):
    # 计算m1和m2的差异值
    slope_df = dataframe.loc[[m1, m2]].T.dropna()
    return (slope_df[m2] - slope_df[m1]).sum() / len(slope_df)


def get_slope_df(dataframe):
    df = DataFrame(
        np.zeros(len(data.index) ** 2).reshape(
            len(data.index), -1),
        index=data.index,
        columns=data.index
    )
    for m in dataframe.index:
        for _m in dataframe.index.drop(m):
            df[m][_m] = get_slope(m, _m, dataframe)
    return df


def get_numerator(user, movie, slope_df, dataframe):
    user_series = dataframe[user]
    if movie in user_series:
        user_series = user_series.drop(movie)
    user_series = user_series.dropna()
    movies = dataframe[user].dropna()
    if movie in movies:
        movies = movies.drop(movie)
    result = 0.0
    for m in movies.index:
        _m = slope_df[movie][m] + dataframe[user][m]
        _m *= len(dataframe.loc[[movie, m]].T.dropna())
        result += _m
    return result


def get_denominator(user, movie, slope_df, dataframe):
    movies = dataframe[user].dropna()
    if movie in movies:
        movies = movies.drop(movie)
    result = 0
    for m in movies.index:
        result += len(dataframe.loc[[movie, m]].T.dropna())
    return result


def get_slope_value(user, movie, slope_df, dataframe):
    numerator = get_numerator(user, movie, slope_df, dataframe)
    denominator = get_denominator(user, movie, slope_df, dataframe)
    return numerator / denominator


if __name__ == '__main__':
    user = 'Heather'
    movie = 'Blade Runner'
    data = pd.read_csv('../chapter-2/Movie_Ratings.csv')
    slope_df = get_slope_df(data)
    print (get_slope_value(user, movie, slope_df, data))
