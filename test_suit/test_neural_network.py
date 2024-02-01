import pytest
import pandas as pd
import numpy as np
from src.utils.helpers import load_dataset, drop_unused_feature, standardization_features


@pytest.fixture
def data_path_sample(tmp_path):
    sample_data = pd.DataFrame({
        'A': [1, 2, 3, 4],
        'B': ['a', 'b', 'c', 'd']
    })
    sample_file_path = tmp_path / 'sample_data.csv'
    sample_data.to_csv(sample_file_path, index=False)
    return sample_file_path


@pytest.fixture
def dataset_test():
    return pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]
    })


@pytest.fixture
def sample_datasets():
    x_train = np.array([[1, 2], [3, 4], [5, 6]])
    x_test = np.array([[7, 8], [9, 10]])
    return x_train, x_test


# Test the load_dataset function
def test_load_dataset(data_path_sample):
    result = load_dataset(data_path_sample)
    assert isinstance(result, pd.DataFrame)


def test_drop_unused_feature(dataset_test):
    drop_li = ['B', 'C']
    result = drop_unused_feature(dataset_test, drop_li)
    assert isinstance(result, pd.DataFrame)
    # Check if the dropped features are not present in the result
    assert 'A' in result.columns
    assert 'B' not in result.columns
    assert 'C' not in result.columns


def test_standardization_features(sample_datasets):
    x_train, x_test = sample_datasets

    # Call the function to get the scaled datasets
    result_train, result_test = standardization_features(x_train, x_test)

    # Check if the result is a tuple
    assert isinstance(result_train, np.ndarray)
    assert isinstance(result_test, np.ndarray)

    # Check if the scaled datasets have the same shape as the original datasets
    assert result_train.shape == x_train.shape
    assert result_test.shape == x_test.shape
