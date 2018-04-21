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
        M_i = [m[u] for u in range(num_users) if R[u,i] == 5]

        first = sum(r_ui)/(len(r_ui) + small) #Avoid by 0 division
        second = (sum(r_ui) - max_rating * sum(M_i)) / (len(r_ui) - sum(M_i) + small)

        rating_bias.append(np.abs(first - second))
    return rating_bias

def mean_var(R, num_users, num_items):

    mean_var = []
    r_i_bar = []

    for i in range(num_items):
        u_i = [R[u, i] for u in range(num_users) if R[u,i] !=0]
        r_i_bar.append(sum(u_i)/(len(u_i) + 1e-9))

    for u in range(num_users):
        I_u_subset = [R[u, i] for i in range(num_items) if R[u,i] !=0 and R[u,i] != 5]
        avg_u_subset = [r_i_bar[j] for j in range(num_items) if R[u,j] !=0 and R[u,j] != 5]
        mean_var_value = [(first - second)**2 for first, second in zip(I_u_subset, avg_u_subset)]
        mean_var.append(sum(mean_var_value)/(len(mean_var_value) + 1e-9))

    return mean_var

def variance(R, num_users, num_items):

    vari = []
    for i in range(num_items):
        u_i = [R[u, i] for u in range(num_users) if R[u,i] != 0]
        m = sum(u_i)/(len(u_i) + 1e-9)
        vari.append(sum([(xi - m) ** 2 for xi in u_i]) / (len(u_i) + 1e-9))

    return vari

def WDMA(R, num_users, num_items):
    WDMA = []
    r_i_bar = np.average(R, axis=0)
    l_i = []
    for i in range(num_items):
        l_i_val = [R[u,i] for u in range(num_users) if R[u, i]!=0]
        l_i.append(len(l_i_val))


    for u in range(1):
        I_u_subset = [R[u, i] for i in range(num_items) if R[u, i]!=0]
        print(I_u_subset)
        avg_u_subset = [r_i_bar[i] for i in range(num_items) if R[u,i]!=0]
        print(avg_u_subset)
        print(l_i)
        WDMA_val = [(abs(first - second))/(l_i[u]**2) for first,second in zip(I_u_subset, avg_u_subset)]
        print(WDMA_val)
        WDMA.append(sum(WDMA_val)/len(I_u_subset))

    return WDMA

def WDA(R, num_users, num_items):
    WDA = []
    l_i = []
    r_i_bar = np.average(R, axis=0)
    for i in range(num_items):
        l_i_val = [R[u, i] for u in range(num_users) if R[u, i] != 0]
        l_i.append(len(l_i_val))

    for u in range(num_users):
        I_u_subset = [R[u, i] for i in range(num_items) if R[u, i] != 0]
        avg_u_subset = [r_i_bar[i] for i in range(num_items) if R[u, i] != 0]
        WDA_val = [(abs(first - second))/l_i[u] for first,second in zip(I_u_subset, avg_u_subset)]
        WDA.append(sum(WDA_val))

    return WDA

def LengthVar(R, num_users, num_items):
    Lengthvar = []
    n_u =[]
    avg_no_ratings=[]
    for u in range(num_users):
        no_ratings = [R[u,i] for i in range(num_items) if R[u,i]!=0]
        n_u.append(len(no_ratings))
        avg_no_ratings.append(len(no_ratings))
    n_u_bar = sum(avg_no_ratings)/num_users

    den =[]
    for u in range(num_users):
        den_val = (n_u[u] - n_u_bar)**2
        den.append(den_val)
    den = sum(den)

    for u in range(num_users):
        Lengthvar.append(abs(n_u[u]-n_u_bar)/(den+1e-5))

    return Lengthvar