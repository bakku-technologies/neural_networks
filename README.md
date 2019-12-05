Data used for the result was Fist_and_Relax01_Mocap and the converted ultrasound data in CSV form
- add Fist_and_Relax01_Mocap.csv to directory
- add thevalues.csv to directory

First, run store_data.py to save processed training and testing data as npy files to a folder `labeled_data/`. Neural network look in this folder for processed data.

- convolutional.py: convolutional neural network (functional)
- svm_regression.py: support vector machine (functional)
- recurrent_nn.py: recurrent neural network (NOT functional)

To Use WebApp
- cd web_app
- python app.py