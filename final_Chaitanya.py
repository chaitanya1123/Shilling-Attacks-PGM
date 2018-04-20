from __future__ import print_function
from __future__ import division
import sys
import time
import numpy as np

from pgmpy.models import FactorGraph
from pgmpy.factors.distributions import CustomDistribution
from pgmpy.factors.continuous import ContinuousFactor
from pgmpy.factors.discrete import DiscreteFactor

import factorgraph as fg

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

D = 8


# print('Generating User-Item Matrix...\n')
# User-item rating matrix
movies_dict = build_movies_dict(movies_data)
R = generate_matrix(ratings_data, movies_dict)

# R = np.random.randint(0, 5, (5, 5))

# Data Statistics
num_users = np.shape(R)[0]
num_items = np.shape(R)[1]

print('Initializing...\n')
# Initialize Factor Graph
Graph = fg.Graph()


# Create nodes : node_list = ['m1', 'm2', 'm3', 't1', 't2', 't3']
user_nodes = []
for i in range(num_users):
    user_nodes.append('m' + str(i))

item_nodes = []
for i in range(num_items):
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
for user_node in user_nodes:
    Graph.rv(user_node, 2)

for item_node in item_nodes:
    Graph.rv(item_node, 2)


# Factor Helper Functions
def almost_sigmoid(x, scale, feature, threshold):
    return 1/(1 + np.exp(np.power(-1,(1-x)) * scale * (feature - threshold)))

def binarize(num, num_users):
    b = bin(num)[2:]
    op = list('0'*(num_users - len(b)) + b)
    return op

print('Building Factors...\n')
# Init factors and factor_vals
f = []
g = []
h = []

# Features
# rating_bias = features.item_rating_bias(R, m, num_users, num_items)
psi_i = features.variance(R)
phi_u = features.mean_var(R, num_users, num_items)

# Create factors

def g_dist(user_node, user_id):
    return almost_sigmoid(user_node, beta_1, phi_u[user_id], tau_1)

def h_dist(item_node, item_id):
    return almost_sigmoid(item_node, beta_2, psi_i[item_id], tau_2)

# def f_dist(item_node, item_id):
#     return almost_sigmoid(item_node, alpha_t, rating_bias[item_id], delta_r)

# Create Factors
for user_id, user_node in enumerate(user_nodes):
    Graph.factor([user_node], potential=np.array([g_dist(0, user_id), g_dist(1, user_id)]))

for item_id, item_node in enumerate(item_nodes):
    Graph.factor([item_node], potential=np.array([h_dist(0, item_id), h_dist(1, item_id)]))

# Calc Mi
M_i = []
for i in range(num_items):
    m_i = [m[u] for u in range(num_users) if R[u,i] == 5]
    M_i.append(m_i)

def split_list(list, jump):
    temp = []
    for i in range(0, len(list), jump):
        temp.append(list[i:i+jump])
    return temp

def group_rating_bias(R, m, num_users, num_items, m_i_k, G_i):
    rating_bias = []
    for k in range(G_i):
        group_len = D if k < G_i else len(M_i) % D
        U_i_cap = [m[u] for u in range(num_users) if R[u,i] != 5]
        r_ui_cap = [R[u, i] for u in U_i_cap]
        R_i_cap = sum(r_ui_cap)
        w_i_k = group_len/len(M_i)
        first = (R_i_cap * w_i_k + max_rating * group_len) / (len(U_i_cap) * w_i_k + group_len)
        second = (R_i_cap * w_i_k + max_rating * sum(m_i_k)) / (len(U_i_cap) * w_i_k + sum(m_i_k))

        rating_bias.append(np.abs(first - second))
    return rating_bias


M_i_k_all_items = []
r_i_M_i_k = []

# D = 8
for item_node in item_nodes:
    M_i_k = []
    for m_idx in M_i:
        # Divide users into groups
        l = len(m_idx)
        G_i = int(np.abs(l) / D) + 1 # Randomly divide user nodes in M_i into G_i groups
        M_i_k.append(split_list(m_idx, G_i))
        # print(len(M_i_k))

    M_i_k_all_items.append(M_i_k)
    # r_i_M_i_k.append(group_rating_bias(R, m, num_users, num_items, m_i_k, G_i))

print(len(M_i_k_all_items))

        # Get rating bias for that group



sys.exit(0)


# f_list = []
#
# for item_id, item_node in enumerate(item_nodes):
#     for user_id, user_node in enumerate(user_nodes):
#         f_list.append(almost_sigmoid())
#
#     f.append(DiscreteFactor([user_nodes, item_node], [32, 2], rating_bias))

# print(len(user_nodes), len(g))

# sys.exit(0)
# print('Done dana done done \n')
