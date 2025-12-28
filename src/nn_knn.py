"""Practical implementation of the K-Nearest Neighbors (KNN) algorithm applied on the CIFAR-10 dataset."""


import keras
import numpy as np


def most_frequent_one_hot(labels: list[np.ndarray]) -> np.ndarray:
    """Return the most frequent one-hot vector in the list of one-hot vectors."""
    # Convert list of arrays to 2D array
    arr = np.stack(labels) 
    
    # Count occurrences of each unique row
    unique_rows, counts = np.unique(arr, axis=0, return_counts=True)
    
    # Return the row with the highest count
    return unique_rows[np.argmax(counts)]


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


def predictLabelNN(x_train:np.ndarray, y_train:np.ndarray, img:np.ndarray, distance_metric:bool) -> np.ndarray:
    """Implement NN classifier."""
    # Initialize the minimum score with infinity
    scoreMin = float('inf')

    # Loop over all training images to compute the distance with the test image
    for idx, imgT in enumerate(x_train):
        # Compute L1 (Manhattan) distance if specified
        if distance_metric:
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


def predictLabelKNN(x_train:np.ndarray, y_train:np.ndarray, img:np.ndarray, k:int, distance_metric:bool) -> np.ndarray:
    """Implement KNN classifier."""
    # Initialize a list to store the distances between the test image and each training image
    predictions = []

    # Loop over all training images to compute the distance with the test image
    for idx, imgT in enumerate(x_train):
        # Compute L1 (Manhattan) distance if specified and store it
        if distance_metric:
            difference = np.abs(img-imgT)
            score = np.sum(difference)
            predictions.append((score, y_train[idx]))
        # Otherwise compute L2 (Euclidean) distance and store it
        else:
            difference = np.square(img-imgT)
            score = np.sqrt(np.sum(difference))
            predictions.append((score, y_train[idx]))

    # Sort all elements in the predictions list in ascending order based on scores
    predictions.sort(key=lambda item: item[0])
   
    # Retain only the top k predictions
    predictions = predictions[:k]
    
    # Extract in a separate vector only the labels for the top k predictions
    predictedLabels = [x[1] for x in predictions]

    # Determine the dominant class from the predicted labels
    predictedLabel = most_frequent_one_hot(predictedLabels)

    return predictedLabel


def main():
    """Implement KNN classification."""
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Preprocess the data
    x_train, y_train, x_test, y_test = prepare_data(x_train, y_train, x_test, y_test)

    # Predictions
    numberOfCorrectPredictedImages = 0

    for idx, img in enumerate(x_test):
        print(f'Make prediction for image {idx}')

        # Predict label using NN algorithm
        predictedLabel = predictLabelNN(x_train, y_train, img, True) # Set the last parameter to True to use the L1 (Manhattan) distance or to False to use the L2 (Euclidian) distance.

        # Predict label using KNN algorithm
        K = 3
        predictedLabel = predictLabelKNN(x_train, y_train, img, K, True)

        # Compare the predicted label with the groundtruth and increment the counter
        if (predictedLabel == y_test[idx]).all():
            numberOfCorrectPredictedImages += 1

    # Compute the accuracy
    accuracy = 100 * numberOfCorrectPredictedImages
    print(f'System accuracy = {accuracy}')


if __name__ == "__main__":
    main()