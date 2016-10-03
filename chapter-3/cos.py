# coding: utf-8
import pandas as pd
import numpy as np


def get_users_intersection(book1, book2, data):
    """返回评价果"""
    book1_users = data[data['book'] == book1]['userId'].values
    book2_users = data[data['book'] == book2]['userId'].values
    return np.intersect1d(book1_users, book2_users, assume_unique=True)


def get_user_mean(user, data):
    """返回均值"""
    return data[data['userId'] == user]['rating'].mean()


def get_numerator(book1, book2, data):
    """返回分子"""
    users = get_users_intersection(book1, book2, data)
    if users:
        result = 0
        for u in users:
            u_mean = get_user_mean(u, data)
            books_ratings = data.loc[
                data['userId'] == u][
                data['book'].isin([book1, book2])]['rating']
            result += np.cumprod(books_ratings.values - u_mean)[-1]
        return result
    return 0


def get_denominator(book1, book2, data):
    """返回分母"""
    users = get_users_intersection(book1, book2, data)
    if users:
        result = 1
        for u in users:
            u_mean = get_user_mean(u, data)
            books_ratings = data.loc[
                data['userId'] == u][
                data['book'].isin([book1, book2])]['rating']
            books_ratings -= u_mean
            books_ratings = books_ratings.pow(2) ** 0.5
            result *= np.cumprod(books_ratings.values)[-1]
        return result
    return 0


def get_cos_rating(book1, book2, data):
    denominator = get_denominator(book1, book2, data)
    if denominator:
        return get_numerator(book1, book2, data) / denominator
    return -1


def get_data(path):
    """返回数据"""
    data = pd.read_csv(path, sep=';', header=None)
    data.columns = ['userId', 'book', 'rating']
    return data


if __name__ == '__main__':
    path = '../chapter-2/BX-Dump/BX-Book-Ratings.csv'
    data = get_data(path)
    data = data[data['rating'] > 0]
    books = data.drop_duplicates('book')['book']
    book1, book2 = books.sample(n=2).values
    print (get_cos_rating(book1, book2, data))
