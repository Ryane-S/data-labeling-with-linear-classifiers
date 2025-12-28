"""Practical implementation of the K-Nearest Neighbors (KNN) algorithm applied on the CIFAR-10 dataset."""


import keras
import numpy as np


def predictLabelNN(x_train:np.ndarray, y_train:np.ndarray, img:np.ndarray, distance_metric_L1_or_L2:bool) -> np.ndarray:
    """Implement NN classifier."""
    # Initialize the minimum score with infinity
    scoreMin = float('inf')

    # Loop over all training images to compute the distance with the test image
    for idx, imgT in enumerate(x_train):
        # Compute L1 (Manhattan) distance if specified
        if distance_metric_L1_or_L2:
            difference = np.abs(img-imgT)
            score = np.sum(difference)
        # Otherwise compute L2 (Euclidean) distance
        else:
            difference = np.square(img-imgT)
            score = np.sqrt(np.sum(difference))

        # Update the predicted label if this training image is closer than previous ones
        if score < scoreMin :
            scoreMin = score
            predictedLabel = y_train[idx]

    return predictedLabel

def prepare_data(x_train:np.ndarray, y_train:np.ndarray, x_test:np.ndarray, y_test:np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten and normalize the input datasets and convert labels to one-hot encoding."""
    # Reshape the dataset from 32*32*3 to a vector
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2]*x_train.shape[3]).astype('float64')
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2]*x_test.shape[3]).astype('float64')

    # Normalize the input values
    x_train = x_train / 255
    x_test = x_test / 255

    # Transform the classes labels into a binary matrix
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    return x_train, y_train, x_test, y_test

def main():
    """Implement KNN classification."""
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Preprocess the data
    x_train, y_train, x_test, y_test = prepare_data(x_train, y_train, x_test, y_test)

    # Predictions
    numberOfCorrectPredictedImages = 0

    for idx, img in enumerate(x_test[0:200]):
        print(f'Make prediction for image {idx}')

        # Predict label using NN algorithm
        predictedLabel = predictLabelNN(x_train, y_train, img, True) # Set the last parameter to True to use the L1 (Manhattan) distance or to False to use the L2 (Euclidian) distance.

        # Predict label using KNN algorithm

        # Compare the predicted label with the groundtruth and increment the counter
        if (predictedLabel == y_test[idx]).all():
            numberOfCorrectPredictedImages += 1

    # Compute the accuracy
    accuracy = 100 * numberOfCorrectPredictedImages
    print(f'System accuracy = {accuracy}')

if __name__ == "__main__":
    main()