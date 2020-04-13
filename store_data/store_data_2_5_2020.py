import csv
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

thumb_data_1 = np.ones((0, 3))
thumb_data_2 = np.ones((0, 3))
thumb_data_3 = np.ones((0, 3))
thumb_data_4 = np.ones((0, 3))

index_data_1 = np.ones((0, 3))
index_data_2 = np.ones((0, 3))
index_data_3 = np.ones((0, 3))
index_data_4 = np.ones((0, 3))

middle_data_1 = np.ones((0, 3))
middle_data_2 = np.ones((0, 3))
middle_data_3 = np.ones((0, 3))
middle_data_4 = np.ones((0, 3))

ring_data_1 = np.ones((0, 3))
ring_data_2 = np.ones((0, 3))
ring_data_3 = np.ones((0, 3))
ring_data_4 = np.ones((0, 3))

pinky_data_1 = np.ones((0, 3))
pinky_data_2 = np.ones((0, 3))
pinky_data_3 = np.ones((0, 3))
pinky_data_4 = np.ones((0, 3))

print("Getting MOCAP:")
with open("../data_2_5_2020/vicon/BallGrab-200205-01-Mocap.csv") as f:
    reader = csv.reader(f)
    for i in range(5):
        next(r)
    for row in reader:
        # THUMB FINGER
        thumb_data_1 = np.append(index_data_1, [row[2:5]], axis=0)
        thumb_data_2 = np.append(index_data_2, [row[5:8]], axis=0)
        thumb_data_3 = np.append(index_data_3, [row[8:11]], axis=0)
        thumb_data_4 = np.append(index_data_4, [row[11:14]], axis=0)

        # INDEX FINGER
        index_data_1 = np.append(index_data_1, [row[14:17]], axis=0)
        index_data_2 = np.append(index_data_2, [row[17:20]], axis=0)
        index_data_3 = np.append(index_data_3, [row[20:23]], axis=0)
        index_data_4 = np.append(index_data_4, [row[23:26]], axis=0)

        # MIDDLE FINGER
        middle_data_1 = np.append(middle_data_1, [row[26:29]], axis=0)
        middle_data_2 = np.append(middle_data_2, [row[29:32]], axis=0)
        middle_data_3 = np.append(middle_data_3, [row[32:35]], axis=0)
        middle_data_4 = np.append(middle_data_4, [row[35:38]], axis=0)

        # RING FINGER
        ring_data_1 = np.append(ring_data_1, [row[38:41]], axis=0)
        ring_data_2 = np.append(ring_data_2, [row[41:44]], axis=0)
        ring_data_3 = np.append(ring_data_3, [row[44:47]], axis=0)
        ring_data_4 = np.append(ring_data_4, [row[47:50]], axis=0)

        # PINKY FINGER
        pinky_data_1 = np.append(pinky_data_1, [row[50:53]], axis=0)
        pinky_data_2 = np.append(pinky_data_2, [row[53:56]], axis=0)
        pinky_data_3 = np.append(pinky_data_3, [row[56:59]], axis=0)
        pinky_data_4 = np.append(pinky_data_4, [row[59:62]], axis=0)

# Pre-size the array otherwise it takes up too much memory from
# numpy reallocating the array every time
lines = np.ones((636, 358400))
print("getting ultrasound")
with open("../data_2_5_2020/verasonics/BallGrab-200205-01.csv", "r") as f:
    reader = csv.reader(f)
    i = 0
    for line in reader:
        lines[i, :] = line
        i += 1

thumb_data_1 = thumb_data_1.astype(float)
thumb_data_2 = thumb_data_2.astype(float)
thumb_data_3 = thumb_data_3.astype(float)
thumb_data_4 = thumb_data_4.astype(float)

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


thumb_angles = compute_angles(thumb_data_1, thumb_data_2, thumb_data_3, thumb_data_4)
index_angles = compute_angles(index_data_1, index_data_2, index_data_3, index_data_4)
middle_angles = compute_angles(middle_data_1, middle_data_2, middle_data_3, middle_data_4)
ring_angles = compute_angles(ring_data_1, ring_data_2, ring_data_3, ring_data_4)
pinky_angles = compute_angles(pinky_data_1, pinky_data_2, pinky_data_3, pinky_data_4)

# RMS average the motion capture labeled_data to smooth it out for processing
M = 10
thumb_angles_adj = np.power(np.convolve(np.power(thumb_angles, 2), np.ones(M) / M, 'same'), 0.5)
index_angles_adj = np.power(np.convolve(np.power(index_angles, 2), np.ones(M) / M, 'same'), 0.5)
middle_angles_adj = np.power(np.convolve(np.power(middle_angles, 2), np.ones(M) / M, 'same'), 0.5)
ring_angles_adj = np.power(np.convolve(np.power(ring_angles, 2), np.ones(M) / M, 'same'), 0.5)
pinky_angles_adj = np.power(np.convolve(np.power(pinky_angles, 2), np.ones(M) / M, 'same'), 0.5)
