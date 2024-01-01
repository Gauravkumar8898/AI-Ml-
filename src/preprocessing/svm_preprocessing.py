import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# In this class we have to done data loading and preprocessing
def load_dataset(dataset):
    loaded_dataset = pd.read_csv(dataset)
    return loaded_dataset


def drop_unused_feature(dataset):
    dataset = dataset.drop(['index', 'Patient Id'], axis=1)
    return dataset


def set_feature_target_variable(dataset):
    df_features = dataset.drop("Level", axis='columns')
    df_target = dataset['Level']
    return df_features, df_target


def transform_data(df_target):
    encoder = LabelEncoder()
    df_target = encoder.fit_transform(df_target)
    return df_target


def split_dataset(df_features, df_target):
    x_train, x_test, y_train, y_test = (
        train_test_split(df_features, df_target, random_state=91, test_size=0.2))
    return x_train, x_test, y_train, y_test


def standardize_feature(x_train, x_test):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled
