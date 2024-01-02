import unittest

from .test_constant import test_data_path
from src.preprocessing.svm_preprocessing import load_dataset, drop_unused_feature, set_feature_target_variable


class TestPreprocessing(unittest.TestCase):

    # In this function we have test the funtion loading and feature dropping
    def test_load_dataset_and_drop_features(self):
        test_dataset = load_dataset(test_data_path)
        self.assertIsNotNone(test_dataset)
        test_dataset1 = drop_unused_feature(test_dataset)
        assert len(test_dataset1.columns) < len(test_dataset.columns)

    def test_set_feature_target_variable(self):
        dataset = load_dataset(test_data_path)
        test_f, test_t = set_feature_target_variable(dataset)
        print(test_f)
        self.assertIs(test_t.columns, "Level")


if __name__ == '__main__':
    unittest.main()
