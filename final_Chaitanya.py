from __future__ import print_function
from __future__ import division
import sys
import numpy as np

from pgmpy.models import FactorGraph
from pgmpy.factors.distributions import CustomDistribution
from pgmpy.factors.continuous import ContinuousFactor


from data import build_movies_dict, generate_matrix
import features

# Set paths
movies_data = './Data/movies.csv'
ratings_data = './Data/ratings.csv'

# Hyper-parameters
alpha_t = -3
delta_r = 0.35
beta_1 = -1
tau_1 = 0.5
beta_2 = 1
tau_2 = 1.5
min_rating = 0.5
max_rating = 5
small = 1e-9

print('Generating User-Item Matrix...\n')
# User-item rating matrix
movies_dict = build_movies_dict(movies_data)
R = generate_matrix(ratings_data, movies_dict)

# Data Statistics
num_users = np.shape(R)[0]
num_items = np.shape(R)[1]

print('Initializing...\n')
# Initialize Factor Graph
G = FactorGraph()


# Create nodes : node_list = ['m1', 'm2', 'm3', 't1', 't2', 't3']
user_nodes = []
for i in range(1, num_users+1):
    user_nodes.append('m' + str(i))

item_nodes = []
for i in range(1, num_items+1):
    item_nodes.append('t' + str(i))

# Spam Users and Target Items Initializations
m = np.random.rand(num_users)
m = [1 if i > 0.5 else 0 for i in np.random.rand(num_users)]

t = np.random.rand(num_items)
t = [1 if i > 0.5 else 0 for i in np.random.rand(num_items)]

# Dict to map nosed to their values
user_dict = {}
item_dict = {}

for user_id, user_node in enumerate(user_nodes):
    user_dict[user_node] = m[user_id]

for item_id, item_node in enumerate(item_nodes):
    item_dict[item_node] = t[item_id]

# Add Nodes to Factor Graph
G.add_nodes_from(user_nodes)
G.add_nodes_from(item_nodes)

# Factor Helper Functions
def almost_sigmoid(x, scale, feature, threshold):
    return 1/(1 + np.exp(np.power(-1,(1-x)) * scale * (feature - threshold)))


print('Building Factors...\n')
# Init factors and factor_vals
f = []
g = []
h = []

# Features
rating_bias = features.item_rating_bias(R, m, num_users, num_items)
psi_i = features.variance(R)
phi_u = features.mean_var(R, num_users, num_items)

# Create factors

def g_dist(user_node, user_id):
    return almost_sigmoid(user_node, beta_1, phi_u[user_id], tau_1)

def f_dist(item_node, item_id):
    return almost_sigmoid(item_node, beta_2, psi_i[item_id], tau_2)

def h_dist(item_node, item_id):
    return almost_sigmoid(item_node, alpha_t, rating_bias[item_id], delta_r)


# Create Factors
for user_id, user_node in enumerate(user_nodes):
    g_pdf = CustomDistribution(variables=[user_node, user_id], distribution=g_dist)
    g.append(ContinuousFactor([user_node], g_pdf))

for item_id, item_node in enumerate(item_nodes):
    h_pdf = CustomDistribution(variables=[item_node, item_id], distribution=h_dist)
    h.append(ContinuousFactor([item_node], h_pdf))

    f_pdf = CustomDistribution(variables=[item_node, item_id], distribution=f_dist)
    f.append(ContinuousFactor([item_node], f_pdf))

print('Adding Factors to graph...\n')

# Add factors to graph
for g_factor in g:
    G.add_factors(g_factor)

for h_factor in h:
    G.add_factors(h_factor)

for f_factor in f:
    G.add_factors(f_factor)

# print(len(user_nodes), len(g))

# sys.exit(0)
print('Adding edges...\n')

# Add edges
for idx in range(len(user_nodes)):
    G.add_edge(user_nodes[idx], g[idx])

for idx in range(len(item_nodes)):
    G.add_edge(item_nodes[idx], h[idx])
    G.add_edge(item_nodes[idx], f[idx])

for i in range(len(f)):
    for j in range(len(user_nodes)):
        G.add_edge(f[i], user_nodes[j])

# print(G.check_model())

print('Done dana done done \n')
