from __future__ import print_function
from __future__ import division


import numpy as np
import pandas as pd
import pgmpy
from pgmpy.models import FactorGraph
from pgmpy.factors.discrete import DiscreteFactor

# Read data in
print('Reading data...')
data = pd.read_csv('Data/ratings.csv')
print('Done!!')

print(data.values)
#print(data[1])
#print(data[2])

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
        f.append(DiscreteFactor([user,item], [2,2], np.ones([2,2])))

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

# Rating matrix
R = np.matrix([[3, 4, 5],
              [4, 3, 5],
              [2, 5, 4]])


num_users = R.shape[0]
num_items = R.shape[1]


##rui (user u's rating on item i)
def r_u_i(u,i):
    return R[u,i]


##ri (average rating of item i)
def r_i_dash(i):
    avg_rating = np.sum(R,axis=0)/num_users
    return avg_rating[0,i]

##feature (MeanVar) phi_u for every user u
def phi_u(u):
    sum =0
    count=0
    for items in range(num_items):
        if(R[u,items]!=5):
            count = count+1
            sum = sum+((R[u,items] - r_i_dash(items))**2)

    sum=sum/count
    return sum

#print feature MeanVar of 0th user
print(phi_u(0))





