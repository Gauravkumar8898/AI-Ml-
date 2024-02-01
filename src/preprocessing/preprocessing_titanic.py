from src.utils.helpers import load_dataset, drop_unused_feature, standardization_features
from src.utils.constant import (x_train_data_path, y_train_data_path,
                                x_test_data_path, y_test_data_path)


class TitanicPreprocessing:

    @staticmethod
    def merge_data_set(data1, data2):
        """
        Merge two dataframes on the 'Id' column using an inner join.
        Parameters:
            data1: First dataframe to be merged.
            data2: Second dataframe to be merged.
        Returns:
            Merged dataframe.
        """
        data_set = data1.merge(data2, how='inner', on='Id')
        return data_set

    @staticmethod
    def drop_na_data(data):
        """
        Drop rows with missing values (NaN) from the given dataframe.
        Parameters:
            data: The dataframe to process.
        Returns:
            Dataframe without rows containing missing values.
        """
        data = data.dropna()
        return data

    @staticmethod
    def split_data_feature_target(dataset, target_label):
        """
        Split the dataset into feature data and target data.
        Parameters:
            dataset: The original dataset containing both features and the target variable.
            target_label: The label of the target variable in the dataset.
        Returns:
            feature_dataset: The dataset containing only the feature columns.
            target_label_data: The target variable data.
        """
        feature_dataset = dataset.drop([target_label], axis=1)
        target_label_data = dataset[target_label]
        return feature_dataset, target_label_data

    @staticmethod
    def work_flow():
        x_train = load_dataset(x_train_data_path)
        y_train = load_dataset(y_train_data_path)
        x_test = load_dataset(x_test_data_path)
        y_test = load_dataset(y_test_data_path)
        instance = TitanicPreprocessing()
        train_dataset = instance.merge_data_set(x_train, y_train)
        test_dataset = instance.merge_data_set(x_test, y_test)
        train_dataset = drop_unused_feature(train_dataset, ['Id'])
        test_dataset = drop_unused_feature(dataset=test_dataset, lists=['Id'])
        train_dataset = instance.drop_na_data(train_dataset)
        test_dataset = instance.drop_na_data(test_dataset)
        x_train, y_train = instance.split_data_feature_target(train_dataset, 'Survived')
        x_test, y_test = instance.split_data_feature_target(test_dataset, 'Survived')
        x_train_scaled, x_test_scaled = standardization_features(x_train, x_test)
        return x_train_scaled, y_train, x_test_scaled, y_test
