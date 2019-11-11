import csv
import numpy as np
from numpy import linalg as LA

# Pre-calculated offset from beginnings of labeled_data
mocap_offset = 375
ultrasound_offset = 155

# Initialize arrays for reading in labeled_data
index_data_1 = np.ones((0,3))
index_data_2 = np.ones((0,3))
index_data_3 = np.ones((0,3))
index_data_4 = np.ones((0,3))

print("getting mocap")
with open("original_data/Pinch_and_Relax01_Mocap.csv", "r") as f:
    r = csv.reader(f)
    for i in range(5):
        next(r)
    for row in r:
        # Each of these holds the labeled_data from one joint marker
        index_data_1 = np.append(index_data_1, [row[2:5]], axis=0)
        index_data_2 = np.append(index_data_2, [row[5:8]], axis=0)
        index_data_3 = np.append(index_data_3, [row[8:11]], axis=0)
        index_data_4 = np.append(index_data_4, [row[11:14]], axis=0)

# Pre-size the array otherwise it takes up too much memory from
# numpy reallocating the array every time
lines = np.ones((2000, 39680))
print("getting ultrasound")
with open("original_data/pinch_relax_thevalues.csv", "r") as f:
    r = csv.reader(f)
    i = 0
    for line in r:
        lines[i,:] = line
        i += 1

# Reformat the labeled_data as float values for analysis
index_data_1 = index_data_1.astype(float)
index_data_2 = index_data_2.astype(float)
index_data_3 = index_data_3.astype(float)
index_data_4 = index_data_4.astype(float)

# Compute the dot product of the vectors of the "palm" markers and
# the first joint of the index finger
vectorA = np.subtract(index_data_2, index_data_1)
vectorB = np.subtract(index_data_4, index_data_3)
dotProd = np.einsum('ij, ij->i', vectorA, vectorB)
# Make them all unit vectors
vectorAnorm = LA.norm(vectorA, 2, axis=1)
vectorBnorm = LA.norm(vectorB, 2, axis=1)
magProduct = np.multiply(vectorAnorm, vectorBnorm)
cosVals = np.divide(dotProd, magProduct)
# Get the joint angles using the dot product formula
angles = np.arccos(cosVals)

# RMS average the motion capture labeled_data to smooth it out for processing
M = 10
angles_adj = np.power(np.convolve(np.power(angles, 2), np.ones(M)/M, 'same'), 0.5)[mocap_offset:]

# Repeat the ultrasound labeled_data because the framerate is half of the motion capture
new_lines = lines[ultrasound_offset:,:]
clipped_lines = np.repeat(np.clip(new_lines, 0, None),2,axis=0)

# Make the labeled_data the same length to label it
print(angles_adj.shape, clipped_lines.shape)
angles_len = angles_adj.shape[0]
us_len = clipped_lines.shape[0]
data_len = min(angles_len, us_len)
new_ang = angles_adj[:data_len]
print(new_ang.shape, clipped_lines.shape)

# Split the labeled_data into training and testing labeled_data
split = int(data_len*0.7)
ang_train = new_ang[:split]
ang_test = new_ang[split:]
us_train = clipped_lines[:split,:]
us_test = clipped_lines[split:,:]

np.save('labeled_data/pinch_relax/ang_train.npy', ang_train)
np.save('labeled_data/pinch_relax/ang_test.npy', ang_test)
np.save('labeled_data/pinch_relax/us_train.npy', us_train)
np.save('labeled_data/pinch_relax/us_test.npy', us_test)
print('Data saved')
