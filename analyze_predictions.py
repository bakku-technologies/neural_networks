import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

cnn_pred = np.load('predictions/cnn_train_pred.npy').flatten()
# svm_pred = np.load('predictions/svm_pred.npy')
test_y = np.load('labeled_data/ang_train.npy')

for i in range(len(cnn_pred)):
    if cnn_pred[i] < 0 or cnn_pred[i] > 2:
        cnn_pred[i] = 0.25

plt.plot(test_y, 'b')
plt.plot(cnn_pred, 'r')
# plt.plot(svm_pred, 'g')
# plt.show()
plt.savefig('plots/cnn_train_performance')

# svm_mse = mean_squared_error(test_y, svm_pred)
cnn_mse = mean_squared_error(test_y, cnn_pred)
# print('SVM Mean Squared Error: ' + str(svm_mse))
print('CNN Mean Squared Error: ' + str(cnn_mse))
