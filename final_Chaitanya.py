from __future__ import print_function
from __future__ import division
import sys
import numpy as np
from pgmpy.models import FactorGraph
from pgmpy.factors.discrete import DiscreteFactor
from data import build_movies_dict, generate_matrix

# Set paths
movies_data = './Data/movies.csv'
ratings_data = './Data/ratings.csv'

# Hyperparameters
alpha_t = -3
delta_r = 0.35
beta_1 = -1
tau_1 = 0.5
beta_2 = 1
tau_2 = 1.5
small = 1e-9

# User-item rating matrix
movies_dict = build_movies_dict(movies_data)
R = generate_matrix(ratings_data, movies_dict)


# Data Statistics
num_users = np.shape(R)[0]
num_items = np.shape(R)[1]




# Create nodes : node_list = ['m1', 'm2', 'm3', 't1', 't2', 't3']
user_nodes = []
for i in range(1, num_users+1):
    user_nodes.append('m' + str(i))

item_nodes = []
for i in range(1, num_items+1):
    item_nodes.append('t' + str(i))


# Spam Users and Target Items Initializations
m = np.random.rand(num_users)
m = [1 if i > 0.5 else 0 for i in m]

t = np.random.rand(num_items)
t = [1 if i > 0.5 else 0 for i in t]


# Initialize Factor Graph
G = FactorGraph()

# Add Nodes to Factor Graph
G.add_nodes_from(user_nodes)
G.add_nodes_from(item_nodes)


# Item Rating Bias
min_rating = 1
max_rating = 5

rating_bias = []

for i in range(num_items):
    r_ui = [R[u, i] for u in range(num_users) if R[u,i] != 0]
    m_i = [m[u] for u in range(num_users) if R[u,i] == 5]

    first = sum(r_ui)/(len(r_ui) + small)
    second = (sum(r_ui) - max_rating * sum(m_i)) / (len(r_ui) - sum(m_i) + small)

    rating_bias.append(np.abs(first - second))

# Factor (f) over each Item
def almost_softmax(x, scale, feature, threshold):
    return 1/(1 + np.exp(np.power(-1,(1-x)) * scale * (feature - threshold)))

f = []
h = []
psi = np.var(R,axis = 0)

for item in range(num_items):
    f.append(almost_softmax(t[item], alpha_t, rating_bias[item], delta_r))
    h.append(almost_softmax(t[item], beta_2, psi[item], tau_2))

sys.exit(0)
# Phi for Every User
phi = []
average_item = np.average(R,axis=0)

for user in range(num_users):
    ratings_u = R[user,:]
    I_non_max_u = ratings_u[ratings_u<max_rating]
    phi_u = sum(np.power(ratings_u[I_non_max_u] - average_item[I_non_max_u],2))/(sum(I_non_max_u))
    phi.append(phi_u)
    

# Factor (g) over Each User
beta_1 = 0.1
tau_1 = 0.1

g = []

for user in range(num_users):
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

#todo: 1) Change discrete factors to continuous factors! 2) Code rating bias, f, g, h
