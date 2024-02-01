import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_dataset(data_path):
    """
       Use Pandas read_csv function to read the dataset from the specified file path.
       Return the loaded dataset.
       """
    data_set = pd.read_csv(data_path)
    return data_set


def drop_unused_feature(dataset, lists):
    """
        :param dataset: The input dataset in a Pandas DataFrame.
        :param lists: A list of feature names to be dropped from the dataset.
        :return: Return the modified dataset.
        """
    datasets = dataset.drop(lists, axis=1)
    return datasets


def standardization_features(x_train, x_test):
    """
    :param x_train:A dataset of x_train
    :param x_test: A dataset of x_train
    :return: return scaled dataset
    """
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled
