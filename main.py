from __future__ import print_function
import sys
import numpy as np
import pandas as pd
import pgmpy
from pgmpy.models import FactorGraph
from pgmpy.factors.discrete import DiscreteFactor
from data import build_movies_dict, generate_matrix


# Set paths
movies_data = 'Data/movies.csv'
ratings_data = 'Data/ratings.csv'

#Generate user-item rating matrix
movies_dict = build_movies_dict(movies_data)
R = generate_matrix(ratings_data, movies_data)

print('Building Fac Graph...')
G = FactorGraph()


# Data Statistics

num_users = 700
num_targets = 9000

# Create nodes
user_nodes = ['u1', 'u2', 'u3']
item_nodes = ['t1', 't2', 't3']
#node_list = ['u1', 'u2', 'u3', 't1', 't2', 't3']
G.add_nodes_from(user_nodes)
G.add_nodes_from(item_nodes)

#Create Factors
g = []
h = []
f = []

for node in user_nodes:
    g.append(DiscreteFactor([node], [2], np.ones(2)))

for node in item_nodes:
    h.append(DiscreteFactor([node], [2], np.ones(2)))

for user in user_nodes:
    for item in item_nodes:
        f.append(DiscreteFactor([user, item], [2, 2], np.ones([2,2])))

#Add factors to graph
for factor in g:
    G.add_factors(factor)
for factor in h:
    G.add_factors(factor)
for factor in f:
    G.add_factors(factor)

#Add edges
for idx in range(len(user_nodes)):
    G.add_edge(user_nodes[idx], g[idx])

for idx in range(len(item_nodes)):
    G.add_edge(item_nodes[idx], h[idx])
    G.add_edge(item_nodes[idx], f[idx])

for i in range(len(f)):
    for j in range(len(user_nodes)):
        G.add_edge(f[i], user_nodes[j])

print('Done')

#todo: 1) Change discrete factors to continuous factors! 2) Code rating bias, f, g, h