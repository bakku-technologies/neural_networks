import csv
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

ang_train = np.load('labeled_data/fist_and_relax/ang_train.npy')
us_train = np.load('labeled_data/fist_and_relax/us_train.npy')  # .8 gigabyte
us_test = np.load('labeled_data/fist_and_relax/us_test.npy')
ang_test = np.load('labeled_data/fist_and_relax/ang_test.npy')

# Train an SVM for regression on the labeled_data
machine = SVR(gamma='auto')
machine.fit(us_train, ang_train)

# Predict the values on the testing labeled_data
pred_ang = machine.predict(us_test)
# # Plot the actual angles and the predicted angles
plt.plot(ang_test, 'b')
plt.plot(pred_ang, 'r')
plt.show()

# np.save('predictions/svm_pred', pred_ang)

print(mean_squared_error(ang_test, pred_ang))

### Error: 0.1264725988955141