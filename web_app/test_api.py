import numpy as np
import requests
import time
import matplotlib.pyplot as plt

url = 'http://localhost:5000/api/predictAngles'


def predict(index, images):

    start_time = time.time()

    test_image = images[index].tolist()
    json_obj = {'image': test_image}

    req = requests.post(url, json=json_obj)

    print(req.text)
    print('total time: ' + str(time.time() - start_time))


def main():
    images = np.load('../labeled_data/fist_relax/four_fingers/us_test.npy')

    predict(500, images)
    predict(501, images)


if __name__ == '__main__':
    main()
