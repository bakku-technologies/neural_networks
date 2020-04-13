import csv
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

# Pre-calculated offset from beginnings of labeled_data
mocap_offset = 375
ultrasound_offset = 155

# Initialize arrays for reading in labeled_data
index_data_1 = np.ones((0,3))
index_data_2 = np.ones((0,3))
index_data_3 = np.ones((0,3))
index_data_4 = np.ones((0,3))

middle_data_1 = np.ones((0,3))
middle_data_2 = np.ones((0,3))
middle_data_3 = np.ones((0,3))
middle_data_4 = np.ones((0,3))

ring_data_1 = np.ones((0,3))
ring_data_2 = np.ones((0,3))
ring_data_3 = np.ones((0,3))
ring_data_4 = np.ones((0,3))

pinky_data_1 = np.ones((0,3))
pinky_data_2 = np.ones((0,3))
pinky_data_3 = np.ones((0,3))
pinky_data_4 = np.ones((0,3))

print("getting mocap")
with open("../original_data/Fist_and_Relax01_Mocap.csv", "r") as f:
    r = csv.reader(f)
    for i in range(5):
        next(r)
    for row in r:
        # Each of these holds the labeled_data from one joint marker
        # INDEX FINGER
        index_data_1 = np.append(index_data_1, [row[2:5]], axis=0)
        index_data_2 = np.append(index_data_2, [row[5:8]], axis=0)
        index_data_3 = np.append(index_data_3, [row[8:11]], axis=0)
        index_data_4 = np.append(index_data_4, [row[11:14]], axis=0)

        # MIDDLE FINGER
        middle_data_1 = np.append(middle_data_1, [row[14:17]], axis=0)
        middle_data_2 = np.append(middle_data_2, [row[17:20]], axis=0)
        middle_data_3 = np.append(middle_data_3, [row[20:23]], axis=0)
        middle_data_4 = np.append(middle_data_4, [row[23:26]], axis=0)

        # RING FINGER
        ring_data_1 = np.append(ring_data_1, [row[26:29]], axis=0)
        ring_data_2 = np.append(ring_data_2, [row[29:32]], axis=0)
        ring_data_3 = np.append(ring_data_3, [row[32:35]], axis=0)
        ring_data_4 = np.append(ring_data_4, [row[35:38]], axis=0)

        # PINKY FINGER
        pinky_data_1 = np.append(pinky_data_1, [row[38:41]], axis=0)
        pinky_data_2 = np.append(pinky_data_2, [row[41:44]], axis=0)
        pinky_data_3 = np.append(pinky_data_3, [row[44:47]], axis=0)
        pinky_data_4 = np.append(pinky_data_4, [row[47:50]], axis=0)


# Pre-size the array otherwise it takes up too much memory from
# numpy reallocating the array every time
lines = np.ones((2000, 39680))
print("getting ultrasound")
with open("../original_data/fist_relax_thevalues.csv", "r") as f:
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

middle_data_1 = middle_data_1.astype(float)
middle_data_2 = middle_data_2.astype(float)
middle_data_3 = middle_data_3.astype(float)
middle_data_4 = middle_data_4.astype(float)

ring_data_1 = ring_data_1.astype(float)
ring_data_2 = ring_data_2.astype(float)
ring_data_3 = ring_data_3.astype(float)
ring_data_4 = ring_data_4.astype(float)

pinky_data_1 = pinky_data_1.astype(float)
pinky_data_2 = pinky_data_2.astype(float)
pinky_data_3 = pinky_data_3.astype(float)
pinky_data_4 = pinky_data_4.astype(float)


# Compute the dot product of the vectors of the "palm" markers and
# the first joint of the index finger
def compute_angles(data_1, data_2, data_3, data_4):
    vectorA = np.subtract(data_2, data_1)
    vectorB = np.subtract(data_4, data_3)
    dotProd = np.einsum('ij, ij->i', vectorA, vectorB)
    # Make them all unit vectors
    vectorAnorm = LA.norm(vectorA, 2, axis=1)
    vectorBnorm = LA.norm(vectorB, 2, axis=1)
    magProduct = np.multiply(vectorAnorm, vectorBnorm)
    cosVals = np.divide(dotProd, magProduct)
    # Get the joint angles using the dot product formula
    angles = np.arccos(cosVals)
    return angles


index_angles = compute_angles(index_data_1, index_data_2, index_data_3, index_data_4)
middle_angles = compute_angles(middle_data_1, middle_data_2, middle_data_3, middle_data_4)
ring_angles = compute_angles(ring_data_1, ring_data_2, ring_data_3, ring_data_4)
pinky_angles = compute_angles(pinky_data_1, pinky_data_2, pinky_data_3, pinky_data_4)

# RMS average the motion capture labeled_data to smooth it out for processing
M = 10
index_angles_adj = np.power(np.convolve(np.power(index_angles, 2), np.ones(M)/M, 'same'), 0.5)[mocap_offset:]
middle_angles_adj = np.power(np.convolve(np.power(middle_angles, 2), np.ones(M)/M, 'same'), 0.5)[mocap_offset:]
ring_angles_adj = np.power(np.convolve(np.power(ring_angles, 2), np.ones(M)/M, 'same'), 0.5)[mocap_offset:]
pinky_angles_adj = np.power(np.convolve(np.power(pinky_angles, 2), np.ones(M)/M, 'same'), 0.5)[mocap_offset:]

# Repeat the ultrasound labeled_data because the framerate is half of the motion capture
new_lines = lines[ultrasound_offset:,:]
clipped_lines = np.repeat(np.clip(new_lines, 0, None),2,axis=0)

# Make the labeled_data the same length to label it
angles_len = index_angles_adj.shape[0]
us_len = clipped_lines.shape[0]
data_len = min(angles_len, us_len)

index_new_ang = index_angles_adj[:data_len]
middle_new_ang = middle_angles_adj[:data_len]
ring_new_ang = ring_angles_adj[:data_len]
pinky_new_ang = pinky_angles_adj[:data_len]

new_ang = np.array([[index_new_ang[0], middle_new_ang[0], ring_new_ang[0], pinky_new_ang[0]]])
for i in range(1, len(index_new_ang)):
    new_ang = np.append(new_ang, [[index_new_ang[i], middle_new_ang[i], ring_new_ang[i], pinky_new_ang[i]]], axis=0)

print(new_ang.shape)

# Split the labeled_data into training and testing labeled_data
split = int(data_len*0.7)
ang_train = new_ang[:split]
ang_test = new_ang[split:]
us_train = clipped_lines[:split,:]
us_test = clipped_lines[split:,:]

plt.plot(new_ang[:, 0], 'b')
plt.savefig('pinch_relax_mocap')

np.save('../labeled_data/fist_relax/four_fingers/ang_train.npy', ang_train)
np.save('../labeled_data/fist_relax/four_fingers/ang_test.npy', ang_test)
np.save('../labeled_data/fist_relax/four_fingers/us_train.npy', us_train)
np.save('../labeled_data/fist_relax/four_fingers/us_test.npy', us_test)
print('Data saved')
