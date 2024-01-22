from kmeans import kmeanspp
import torch
import time
import numpy as np

M = np.random.rand(10000, 1024)
t = time.time()
clusters,centers,d = kmeanspp.func_Kmeans(M, 5)
print('Clustering time: ',time.time()-t)
print(clusters)
# print(M.shape)