"""Practical implementation of the K-Nearest Neighbors (KNN) algorithm applied on the CIFAR-10 dataset."""


import keras
import numpy as np


def prepare_data(x_train:np.ndarray, y_train:np.ndarray, x_test:np.ndarray, y_test:np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten and normalize the input datasets and convert labels to one-hot encoding."""
    # Reshape the dataset from 32*32*3 to a vector
    x_train_flatten = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2]*x_train.shape[3]).astype('float64')
    x_test_flatten = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2]*x_test.shape[3]).astype('float64')

    # Normalize the input values
    x_train_flatten = x_train_flatten / 255
    x_test_flatten = x_test_flatten / 255

    # Transform the classes labels into a binary matrix
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    return x_train_flatten, y_train, x_test_flatten, y_test

def main():
    """Implement KNN classification."""
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Preprocess the data
    x_train, y_train, x_test, y_test = prepare_data(x_train, y_train, x_test, y_test)

    pass

if __name__ == "__main__":
    main()