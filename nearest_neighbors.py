from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

x_train = np.load('labeled_data/fist_and_relax/us_train.npy')
y_train = np.load('labeled_data/fist_and_relax/ang_train.npy').flatten()
x_test = np.load('labeled_data/fist_and_relax/us_test.npy')
y_test = np.load('labeled_data/fist_and_relax/ang_test.npy').flatten()

print(np.shape(y_train))

knn = KNeighborsRegressor(n_neighbors=5, metric='euclidean')
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

# # Plot the actual angles and the predicted angles
plt.plot(y_test, 'b')
plt.plot(y_pred, 'r')
plt.show()

print(mean_squared_error(y_test, y_pred))

### Error: 0.0768398038747982