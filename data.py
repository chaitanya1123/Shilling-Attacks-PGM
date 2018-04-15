from __future__ import print_function
import sys
import numpy as np
import pandas as pd

np.set_printoptions(threshold=np.inf)

def build_movies_dict(movies_file):
    movie_id_dict = {}
    with open(movies_file, 'r') as f:
        next(f)
        for i, line in enumerate(f):
            movie_id, title, genres = line.split(',')
            movie_id_dict[int(movie_id)] = i
    return movie_id_dict

def generate_matrix(input_file, movies_dict):
    users = 671
    movies = len(movies_dict)
    
    X = np.zeros(shape=(users,movies))
    with open(input_file, 'r') as f:
        next(f)
        for i,line in enumerate(f):
            user, movie_id, rating, timestamp = line.split(',')
            id = movies_dict[int(movie_id)]
            X[int(user)-1, id] = float(rating)
    return X

if __name__ == '__main__':
    
    movies_file = 'Data/movies.csv'
    ratings_file = 'Data/ratings.csv'

    movies_dict = build_movies_dict(movies_file)
    R = generate_matrix(ratings_file, movies_dict)