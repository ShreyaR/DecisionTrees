__author__ = 'shreyarajpal'

from csv import reader
import numpy as np

class DecisionTree:

    def __init__(self, mode):
        train_data = []
        test_data = []
        valid_data = []

        with open('train.csv') as f:
            raw_trainData = reader(f)
        with open('validation.csv') as f:
            raw_valiData = reader(f)
        with open('test.csv') as f:
            raw_testData = reader(f)

        for row in raw_trainData:
            train_data.append(row)
        for row in raw_testData:
            test_data.append(row)
        for row in raw_valiData:
            valid_data.append(row)

        train_data = np.array(train_data)
        test_data = np.array(test_data)
        valid_data = np.array(valid_data)







'''def preprocess():
    with open('train.csv') as f:
        data = reader(f)

    print type(data)
    return

preprocess()'''