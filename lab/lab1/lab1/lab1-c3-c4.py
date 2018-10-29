#!/usr/bin/python
### NYU HPC Course Lab1
### Yuqiong Li
### yl5090
### 20181008


import time

######################################################################################
#                                 C3: Python inferencing
######################################################################################

## 0. Hyperparameters
INPUT_1 = 256  # first dim of inputs
INPUT_2 = 256  # second dim of inputs
HIDDEN = 4000
OUT = 1000

## 1. initialization
x = []
for i in range(INPUT_2):
    curr = []  # current layer
    for j in range(INPUT_1):
        curr.append(0.4 + ((i + j) % 40 - 20) / 40.0)
    x.append(curr)

weights1 = []
for i in range(HIDDEN):
    curr = []
    for j in range(INPUT_1 * INPUT_2):
        curr.append(0.4 + ((i + j) % 40 - 20) / 40.0)
    weights1.append(curr)

weights2 = []
for i in range(OUT):
    curr = []
    for j in range(HIDDEN):
       curr.append(0.4 + (i % 40 - 20) / 40.0)
    weights2.append(curr)

# 2. inferencing
start=time.monotonic()
## Hidden layer
x_flat = [item for sublist in x for item in sublist]  # flatten input
hid = []
for i in range(HIDDEN):
    curr = sum([a * b for a,b in zip(x_flat, weights1[i])])  # the i-th element of hidden layer
    hid.append(curr)

## output layer
out = []
for i in range(OUT):
    curr = sum([a * b for a, b in zip(hid, weights2[i])])  # the i-th element of output layer
    out.append(curr)

# 3. output
S = sum(out)
print(S)

end=time.monotonic()
t = round((end-start) * 10 **(-9), 10)
print("Python time : " + str(t) + " secs")

######################################################################################
#                                 C4: Numpy inferencing
######################################################################################
import numpy as np

## 1. initialization
x_numpy = np.matrix(x)
weights1_numpy = np.matrix(weights1)
weights2_numpy = np.matrix(weights2)

# 2. inferencing
start = time.monotonic()

x_numpy_flat = x_numpy.flatten()
hid_numpy = np.matmul(x_numpy_flat, weights1_numpy.T)
out_numpy = np.matmul(weights2_numpy, hid_numpy.T)

# 3. output
S_numpy = np.sum(out_numpy)
print(S_numpy)

end=time.monotonic()
t = round((end-start) * 10 **(-9), 10)
print("Numpy time : " + str(t) + " secs")
