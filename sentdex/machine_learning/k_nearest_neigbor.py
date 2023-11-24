'''
This is the k nearest neighbors
algorithm writen from scratch
AKA Euclidean Distance

EX:  -- ⎇ == square root
q = (1,3)
p = (2,5)
ed =  ⎇ (1 - 2)^2 + (3 - 5)^2


p1 = [1,3]
p2 = [2,5]
# def ecluidean_distance(plot1, plot2):
#     ecl = sqrt((plot1[0]-plot2[0])**2 + (plot1[1]-plot2[1])**2)
#     return ecl
# print(ecluidean_distance(p1, p2))
'''
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
style.use('fivethirtyeight')


dataset = {'k': [[1,2], [2,3], [3,1]], 'r': [[6,5], [7,7], [8,6]]}
new_features = [5,7]
# [[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
# plt.scatter(new_features[0], new_features[1], s=150, color='orange')
# plt.show()

# def k_nearest_neighbors(data, predict, k=3):
#     if len(data) >= k:
#         warnings.warn('K is set to a value less than total voting groups! ')
        # ##################
        #PAUSED VIDEO 16 LAST FINISHED
        # START 17
        # ##################
#     # knnalogs 
#     # return vote_result