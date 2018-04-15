from __future__ import print_function

import numpy as np
import pgmpy
from pgmpy.models import FactorGraph
from pgmpy.factors.discrete import DiscreteFactor

#Example User-Item Matrix
data = pd.read_csv('Data/ratings.csv')


# Initialize Factor Graph
G = FactorGraph()

# Data Statistics
num_users = 700
num_items = 9000


# Create nodes : node_list = ['m1', 'm2', 'm3', 't1', 't2', 't3']
user_nodes = []
for i in range(1,num_users+1):
    user_nodes.append('m' + str(i))

item_nodes = []
for i in range(1,num_items+1):
    item_nodes.append('t' + str(i))


# Spam Users and Target Items Initializations
m = np.random.rand(num_users)
m = [1  if i>0.5  else 0 for i in m]

t = np.random.rand(num_items)
t = [1  if i>0.5  else 0 for i in t]


# Add Nodes to Factor Graph
G.add_nodes_from(user_nodes)
G.add_nodes_from(item_nodes)


# Item Rating Bias
min_rating = 1;
max_rating = 5;

r_item_bias = []

for item in range(0,num_items):
    item_ratings = R[:,item]
    m_i = (item_ratings == max_rating)
    r_i = ((sum(item_ratings)/num_users) - (sum(item_ratings) - max_rating*sum(m_i)))/(num_users - sum(m_i))
    r_item_bias.append(((sum(item_ratings)/num_users) - (sum(item_ratings) - max_rating*sum(m_i)))/(num_users - sum(m_i)))



# Factor (f) over each Item
alpha_t = -0.1
del_r = 0.1

f = []

for item in range(0,num_items):
    f_i = 1/(1 + np.exp(np.power(-1,1-t[item])*alpha_t*(r_item_bias[item] - del_r)))
    f.append(f_i)


# Factor (h) over each Item
phi = np.var(R,axis = 0)
beta_2 = 0.1
tau_2 = 0.1

h = []

for item in range(0,num_items):
    h_i = 1/(1 + np.exp(np.power(-1,1 - t[item])*beta_2*(phi[item]-tau_2)))
    h.append(h_i)



# Phi for Every User
phi = []
average_item = np.average(R,axis = 0)

for user in range(0,num_users):
    ratings_u = R[user,:]
    I_non_max_u = ratings_u[ratings_u<max_rating]
    phi_u = sum(np.power(ratings_u[I_non_max_u] - average_item[I_non_max_u],2))/(sum(I_non_max_u))
    phi.append(phi_u)
    

# Factor (g) over Each User
beta_1 = 0.1
tau_1 = 0.1

g = []

for user in range(0,num_users):
    g_u = 1/(91 + np.exp(np.power(-1,1-m[user])*beta_1*(phi[user] - tau_1)))
    g.append(g_u)



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



# Rating matrix
R = [[3, 4, 5],
     [4, 3, 5],
     [2, 5, 4]]

#todo: 1) Change discrete factors to continuous factors! 2) Code rating bias, f, g, h
