from __future__ import print_function
from __future__ import division
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
R = generate_matrix(ratings_data, movies_dict)
print('Building Fac Graph...')
G = FactorGraph()



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


num_users = np.shape(R)[0]
num_items = np.shape(R)[1]

##rui (user u's rating on item i)
def r_u_i(u,i):
    return R[u,i]


##ri (average rating of item i)
def r_i_dash(i):
    avg_rating = np.sum(R,axis=0)/num_users
    return avg_rating[i]


def meetsCondition(u,element):
    return bool(R[u,element] !=0 and R[u,element] != 5)

##feature (MeanVar) phi_u for every user u


phi_u_all = []
print("Calculating phi_u for all the users")
for u in range(num_users):
    avg_rating = np.sum(R, axis=0) / num_users

    #Iu\Iu_bar
    u_ratings=[R[u,items] for items in range(num_items) if meetsCondition(u,items)]

    avg_subset_u = [avg_rating[element] for element in range(num_items) if meetsCondition(u,element)]
    #print(u_ratings)
    # #print(avg_subset_u)
    phi_u_arr = [(a_i - b_i)**2 for a_i, b_i in zip(u_ratings,avg_subset_u)]
    phi_u = sum(phi_u_arr)/len(phi_u_arr)
    phi_u_all.append(phi_u)

print(phi_u_all)
print(len(phi_u_all))
print("Finished calculating phi for all the users")
#todo: 1) Change discrete factors to continuous factors!

