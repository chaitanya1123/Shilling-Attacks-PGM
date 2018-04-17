from __future__ import print_function
from __future__ import division
import sys
import numpy as np
from pgmpy.models import FactorGraph
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.factors.continuous import ContinuousFactor
from data import build_movies_dict, generate_matrix

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

print('Generating User-Item Matrix...')
# User-item rating matrix
movies_dict = build_movies_dict(movies_data)
R = generate_matrix(ratings_data, movies_dict)

# Data Statistics
num_users = np.shape(R)[0]
num_items = np.shape(R)[1]

print('Initializing...')
# Initialize Factor Graph
G = FactorGraph()

# Create nodes : node_list = ['m1', 'm2', 'm3', 't1', 't2', 't3']
# user_nodes = []
# for i in range(1, num_users+1):
#     user_nodes.append('m' + str(i))
#
# item_nodes = []
# for i in range(1, num_items+1):
#     item_nodes.append('t' + str(i))

# Spam Users and Target Items Initializations
m = np.random.rand(num_users)
m = [1 if i > 0.5 else 0 for i in np.random.rand(num_users)]
user_nodes = m

t = np.random.rand(num_items)
t = [1 if i > 0.5 else 0 for i in np.random.rand(num_items)]
item_nodes = t

# Add Nodes to Factor Graph
G.add_nodes_from(user_nodes)
G.add_nodes_from(item_nodes)

# Factor Helper Functions
def almost_sigmoid(x, scale, feature, threshold):
    return 1/(1 + np.exp(np.power(-1,(1-x)) * scale * (feature - threshold)))

def item_rating_bias(R, num_users, num_items):
    rating_bias = []

    for i in range(num_items):
        r_ui = [R[u, i] for u in range(num_users) if R[u,i] != 0]
        m_i = [m[u] for u in range(num_users) if R[u,i] == 5]

        first = sum(r_ui)/(len(r_ui) + 1e-9) #Avoid by 0 division
        second = (sum(r_ui) - max_rating * sum(m_i)) / (len(r_ui) - sum(m_i) + 1e-9)

        rating_bias.append(np.abs(first - second))
    return rating_bias

def mean_var(R, num_users, num_items):
    mean_var = []
    r_i_bar = np.average(R, axis=0)

    for u in range(num_users):
        I_u_subset = [R[u, i] for i in range(num_items) if R[u,i] !=0 and R[u,i] != 5]
        avg_u_subset = [r_i_bar[i] for i in range(num_items) if R[u,i] !=0 and R[u,i] != 5]
        mean_var_value = [(first - second)**2 for first, second in zip(I_u_subset, avg_u_subset)]
        mean_var.append(sum(mean_var_value)/len(mean_var_value))

    return mean_var

print('Building Factors...')
# Init factors and factor_vals
f = []
g = []
h = []

# Features
rating_bias = item_rating_bias(R, num_users, num_items)
psi = np.var(R, axis = 0)
phi_u = mean_var(R, num_users, num_items)

# Create factors
# for item in range(num_items):
#     f_values.append(almost_sigmoid(t[item], alpha_t, rating_bias[item], delta_r))
#     h_values.append(almost_sigmoid(t[item], beta_2, psi[item], tau_2))
#
# for user in range(num_users):
#     g_values.append(almost_sigmoid(m[user], beta_1, phi_u[user], tau_1))

def g_pdf(user_node, uid):
    return almost_sigmoid(user_node, beta_1, phi_u[uid], tau_1)
def f_pdf(item_node, iid):
    return almost_sigmoid(item_node, beta_2, psi[iid], tau_2)
def h_pdf(item_node, iid):
    return almost_sigmoid(item_node, alpha_t, rating_bias[iid], delta_r)

# Create Factors
for uid, user_node in enumerate(user_nodes):
    print(uid, user_node)
    g.append(ContinuousFactor([user_node], pdf=g_pdf(user_node, uid)))

for iid, item_node in enumerate(item_nodes):
    h.append(ContinuousFactor([item_node], pdf=h_pdf(item_node, iid)))

for uid, user_node in enumerate(user_nodes):
    for iid, item_node in enumerate(item_nodes):
        f.append(ContinuousFactor([user_node, item_node], f_pdf(item_node, iid)))

G.add_factors(g[0])

# Add factors to graph
# for factor in g:
#     print(factor)
    # G.add_factors(factor)
# for factor in h:
    # G.add_factors(factor)
# for factor in f:
    # G.add_factors(factor)

sys.exit(0)

print('Adding edges...')

# Add edges
for idx in range(len(user_nodes)):
    G.add_edge(user_nodes[idx], g[idx])

for idx in range(len(item_nodes)):
    G.add_edge(item_nodes[idx], h[idx])
    G.add_edge(item_nodes[idx], f[idx])

for i in range(len(f)):
    for j in range(len(user_nodes)):
        G.add_edge(f[i], user_nodes[j])


print('Done dana done done')