import numpy as np
from src.weak1.linear_regression import price_prediction

x = np.array([20, 10, 15, 17])
x = x.T


def test_compute_output_model():
    assert price_prediction(2.294, 9, 4) * 10000 > 206786
    assert price_prediction(2.854, 9, 5) * 10000 < 308586
    assert price_prediction(1.994, 8, 4) * 10000 < 206786
