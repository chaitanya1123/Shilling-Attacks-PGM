from __future__ import print_function
import sys
import numpy as np
import pandas as pd

np.set_printoptions(threshold=np.inf)

def build_movies_dict(movies_file):
    i=0
    movie_id_dict = {}
    with open(movies_file, 'r') as f:
        for line in f:
            if i==0:
                i+=1
            else:
                #print(len(line.split(',')))
                print(i)
                movie_id, title, genres = line.split(',')
                movie_id_dict[int(movie_id)] = i-1
                i+=1
    return movie_id_dict

def read_data(input_file, movies_dict):
    users = 671
    movies = 9125
    
    X = np.zeros(shape=(users,movies))
    i = 0
    with open(input_file, 'r') as f:
        for line in f:
            if i==0:
                i+=1
            else:
                #print(line)
                user, movie_id, rating, timestamp = line.split(',')
                id = movies_dict[int(movie_id)]
                X[int(user)-1, id] = float(rating)
                i+=1
    return X

if __name__ == '__main__':
    
    movies_file = 'Data/movies.csv'
    ratings_file = 'Data/ratings.csv'
    movies_dict = build_movies_dict(movies_file)#movies.csv
    R = read_data(ratings_file, movies_dict)
    print(R[0,1029])
