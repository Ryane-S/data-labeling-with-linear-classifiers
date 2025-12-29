import numpy as np

from linear_classifier_svm_loss import (
    compute_Loss_For_Sample,
    compute_Loss_Gradient_For_Sample,
    predict,
)


def test_predict():
    W = np.array([[1, 0], [0, 1]])
    x = np.array([2, 3])
    s = predict(x, W)

    assert np.allclose(s, np.array([2, 3]))

def test_compute_loss_for_sample():
    s = np.array([3.0, 1.0, 0.0])
    y = 0
    delta = 1.0

    loss = compute_Loss_For_Sample(s, y, delta)

    assert loss == 0

def test_loss_gradient_shape():
    W = np.zeros((3, 4))
    x = np.array([1, 2, 3, 4])
    s = np.array([2.0, 1.0, 0.0])
    y = 0
    delta = 1.0

    dW = compute_Loss_Gradient_For_Sample(W, s, x, y, delta)

    assert dW.shape == W.shape