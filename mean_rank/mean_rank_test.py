
import libMeanRanker
import numpy as np
import random
vecs=[]
for i in range(2000):
    vecs.append(np.random.normal(loc=0.0,scale=1.0,size=[768]))

targets=[]
for i in range(2000):
    targets.append(random.randint(0,2000))

print(libMeanRanker.mean_rank(vecs,targets))