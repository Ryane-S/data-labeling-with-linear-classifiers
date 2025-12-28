import numpy as np
import pytest

from nn_knn import prepare_data


@pytest.fixture
def sample_dataset():
    # Training dataset with 2 images
    x_train = np.array([
        [
            [[0, 0, 0], [10, 20, 30], [40, 50, 60], [70, 80, 90]],
            [[10, 20, 30], [20, 30, 40], [50, 60, 70], [80, 90, 100]],
            [[20, 30, 40], [30, 40, 50], [60, 70, 80], [90, 100, 110]],
            [[30, 40, 50], [40, 50, 60], [70, 80, 90], [100, 110, 120]]
        ],
        [
            [[255, 255, 255], [200, 200, 200], [150, 150, 150], [100, 100, 100]],
            [[200, 200, 200], [180, 180, 180], [130, 130, 130], [80, 80, 80]],
            [[150, 150, 150], [130, 130, 130], [100, 100, 100], [50, 50, 50]],
            [[100, 100, 100], [80, 80, 80], [60, 60, 60], [30, 30, 30]]
        ]
    ], dtype=np.uint8)

    # Training labels
    y_train = np.array([[0], [1]])

    # Testing dataset with 2 images
    x_test = np.array([
        [
            [[5, 5, 5], [15, 25, 35], [35, 45, 55], [65, 75, 85]],
            [[15, 25, 35], [25, 35, 45], [45, 55, 65], [75, 85, 95]],
            [[25, 35, 45], [35, 45, 55], [55, 65, 75], [85, 95, 105]],
            [[35, 45, 55], [45, 55, 65], [65, 75, 85], [95, 105, 115]]
        ],
        [
            [[250, 250, 250], [210, 210, 210], [160, 160, 160], [110, 110, 110]],
            [[210, 210, 210], [190, 190, 190], [140, 140, 140], [90, 90, 90]],
            [[160, 160, 160], [140, 140, 140], [110, 110, 110], [60, 60, 60]],
            [[110, 110, 110], [90, 90, 90], [70, 70, 70], [40, 40, 40]]
        ]
    ], dtype=np.uint8)

    # Testing labels
    y_test = np.array([[0], [1]])

    return x_train, y_train, x_test, y_test


def test_prepare_data(sample_dataset):
    x_train_raw, y_train_raw, x_test_raw, y_test_raw = sample_dataset
    x_train, y_train, x_test, y_test = prepare_data(x_train_raw, y_train_raw, x_test_raw, y_test_raw)
    
    # Check shapes
    assert x_train.shape == (2, 48)
    assert x_test.shape == (2, 48)
    assert y_train.shape == (2, 2)
    assert y_test.shape == (2, 2)
    
    # Check dtype
    assert x_train.dtype == np.float64
    assert x_test.dtype == np.float64
    
    # Check normalization
    assert np.all((x_train >= 0) & (x_train <= 1))
    assert np.all((x_test >= 0) & (x_test <= 1))