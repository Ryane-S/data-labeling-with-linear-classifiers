"""Practical implementation of a linear classifier trained with multiclass SVM loss applied on the Iris dataset."""


import matplotlib.pyplot as plt
import numpy as np


def predict(x_sample:np.ndarray, W:np.ndarray) -> np.ndarray:
    """Compute the scores s for a data point."""
    # Compute scores using a linear transformation
    return np.dot(x_sample, W)


def compute_Loss_For_Sample(s:np.ndarray, labelForSample:int, delta:float) -> float:
    """Compute the loss for a data point."""
    loss_i = 0
    
    # The score for the correct class corresponding to the current input sample based on the label yi
    syi = s[labelForSample] 

    # Compute SVM loss
    for j, sj in enumerate(s):
        dist = sj - syi + delta

        if j == labelForSample:
            continue

        if dist > 0:
            loss_i += dist

    return loss_i


def compute_Loss_Gradient_For_Sample(W:np.ndarray, s:np.ndarray, currentDataPoint:np.ndarray, labelForSample:int, delta:float) -> np.ndarray:
    """Compute the gradient loss for a data point."""
    dW_i = np.zeros(W.shape)

    # Establish the score obtained for the true class
    syi = s[labelForSample]

    # Compute gradient contributions for each class
    for j, sj in enumerate(s):
        dist = sj - syi + delta

        if j == labelForSample:
            continue

        if dist > 0:
            dW_i[j] = currentDataPoint
            dW_i[labelForSample] = dW_i[labelForSample] - currentDataPoint

    return dW_i


def load_data(filename:str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load, shuffle and split the Iris dataset."""
    # Load input samples
    data = np.loadtxt(filename, delimiter=",", skiprows=1, dtype=str)

    # Separate data and labels
    X = data[:, :-1].astype(float)
    y = data[:, -1]

    # Encode labels to associate each one with a specific number
    classes, Y = np.unique(y, return_inverse=True)

    # Shuffle
    indexes = np.random.permutation(len(X))
    X, Y = X[indexes], Y[indexes]

    # Split train/test 80/20
    split_ratio = int(len(X)*0.8)
    x_train, x_test = X[:split_ratio], X[split_ratio:]
    y_train, y_test = Y[:split_ratio], Y[split_ratio:]

    return x_train, y_train, x_test, y_test, classes


def main():
    """Implement Linear classification with SVM loss."""
    # Load and prepare the Iris dataset
    x_train, y_train, x_test, y_test, classes = load_data("src/data/Iris.csv")
    
    # Initialize weight matrix
    W = np.random.random_sample((len(classes), x_train[0].shape[0]))

    # Hyperparameters
    delta = 1.0           # margin for the SVM loss
    step_size = 0.1       # learning rate / weight adjustment ratio
    number_steps = 0      # counter for the number of optimization steps performed

    # Performance indicator
    accuracy = 0 # accuracy on training

    # History trackers for monitoring training
    loss_history = []     # store loss values over iterations
    accuracy_history = [] # store accuracy values over iterations
    steps_history = []    # store step numbers (iteration indices)

    # Enable interactive mode for real-time plot updates
    plt.ion()

    # Create a figure with two subplots:
    # - ax1: to display the evolution of the loss
    # - ax2: to display the evolution of the accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

    while True:
        dW = np.zeros(W.shape)  # global gradient loss matrix
        loss_L = 0 # global loss

        for idx, x_sample in enumerate(x_train):
            # Compute the scores s for all classes
            s = predict(x_sample,W)

            # Compute the loss for a data point
            loss_i = compute_Loss_For_Sample(s, y_train[idx], delta=1)

            # Compute the gradient loss for a data point
            dW_i = compute_Loss_Gradient_For_Sample(W, s, x_train[idx], y_train[idx], delta)

            # Update the global loss
            loss_L += loss_i

            # Update the global gradient loss matrix
            dW += dW_i

        # Compute the global normalized loss
        loss_L = loss_L / x_train.shape[0]

        # Compute the global normalized gradient loss
        dW = dW / x_train.shape[0]

        # Adjust the weight matrix
        W = W - step_size*dW

        # Compute classification accuracy on the test set
        correctPredicted = 0
        for idx, x_sample in enumerate(x_test):

            # Compute class scores and predicted label
            s = predict(x_sample, W)
            labelPredicted = np.argmax(s)

            # Compare prediction with ground truth
            if y_test[idx] == labelPredicted:
                correctPredicted += 1

        # Compute accuracy in percentage
        accuracy = 100 * correctPredicted / x_test.shape[0]

        # Store training metrics for visualization
        loss_history.append(loss_L)
        accuracy_history.append(accuracy)
        steps_history.append(number_steps)

        # Update loss plot
        ax1.clear()
        ax1.plot(steps_history, loss_history, color="blue")
        ax1.set_ylim(0, max(loss_history) + 1)
        ax1.set_title("Loss evolution")
        ax1.set_xlabel("Steps")
        ax1.set_ylabel("Loss")

        # Update accuracy plot
        ax2.clear()
        ax2.plot(steps_history, accuracy_history, color="green")
        ax2.set_ylim(0, 100)
        ax2.set_title("Accuracy evolution")
        ax2.set_xlabel("Steps")
        ax2.set_ylabel("Accuracy (%)")

        # Pause to allow interactive plot refresh
        plt.pause(0.01)

        # Stop training after a fixed number of steps and save the figure
        if number_steps == 400:
            plt.savefig(fname="accuracy_loss_evolution")
            break

        # Update previous loss and increment step counter
        number_steps += 1



if __name__ == "__main__":
    main()