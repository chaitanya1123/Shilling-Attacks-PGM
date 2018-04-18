from __future__ import print_function
from __future__ import division
import sys
import numpy as np

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

def item_rating_bias(R, m, num_users, num_items):
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

def variance(R):
    return np.var(R, axis=0)
