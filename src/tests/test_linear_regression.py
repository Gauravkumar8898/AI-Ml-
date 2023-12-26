import pytest
from src.weak1.linear_regression import computeOutputModel


def test_computeOutputModel():
    dw = computeOutputModel(20,10,10)
    assert dw>0
