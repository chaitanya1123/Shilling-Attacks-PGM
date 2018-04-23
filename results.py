from __future__ import print_function
from __future__ import division
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

import heapq


from data import build_movies_dict, generate_100k_matrix, generate_dirty_matrix, simulate_shilling_attack, \
    generate_matrix_from_csv, generate_user_spam_list
import features

label_name = 'labels-rand-s10-f5-t5'
profile_name = 'profiles-rand-s10-f5-t5'

filename = 'rand-s10-f5-t5.txt'

# Set paths

dirty_ratings_data = './Data/dirty/MovieLens/100k/' + profile_name
spam_users_file = 'Data/dirty/MovieLens/100k/' + label_name

R = generate_dirty_matrix(dirty_ratings_data)
user_ground_truth = generate_user_spam_list(spam_users_file)

# Data Statistics
num_users = np.shape(R)[0]
num_items = np.shape(R)[1]


p0 = []
p1 = []
with open(filename, 'r') as f:
    for idx, line in enumerate(f):
        first = line.split(' ')[1]
        second = line.split(' ')[3]
        second = second[0:len(second) - 2]
        # print(first, second, idx)
        # first = float(first)
        p0.append(float(first))
        p1.append(float(second))

# md = 120
md = 93 #10% attack
# md = 46
# top_md = heapq.nlargest(md, p0)
top_md = heapq.nlargest(md, p1)

# top_md_users = heapq.nlargest(md, range(len(p0)), p0.__getitem__)
top_md_users = heapq.nlargest(md, range(len(p1)), p1.__getitem__)

# print(top_md_users)

user_predictions = np.zeros((num_users, 1))
for idx, u in enumerate(top_md_users):
    user_predictions[u] = 1

pre = metrics.precision_score(user_ground_truth, user_predictions)
rkl = metrics.recall_score(user_ground_truth, user_predictions)

print(pre, rkl)

print(metrics.confusion_matrix(user_ground_truth, user_predictions))

# print(metrics.classification_report(user_ground_truth, user_predictions))
