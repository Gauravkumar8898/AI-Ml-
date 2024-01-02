import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# In this function we have to done data loading
def load_dataset(data_path):
    loaded_dataset = pd.read_csv(data_path)
    return loaded_dataset


# In this function we have to remove are features which not use for data training
def drop_unused_feature(dataset):
    dataset = dataset.drop(['index', 'Patient Id'], axis=1)
    return dataset


# In this function we have to set the feature and target variable
def set_feature_target_variable(dataset):
    df_features = dataset.drop("Level", axis='columns')
    df_target = dataset['Level']
    return df_features, df_target


# In this function we change categorical data into numeric form
def transform_data(df_target):
    encoder = LabelEncoder()
    df_target = encoder.fit_transform(df_target)

    return df_target, encoder


# In this function we split the data into training and testing on both (features and target)
def split_dataset(df_features, df_target):
    x_train, x_test, y_train, y_test = (
        train_test_split(df_features, df_target, random_state=91, test_size=0.2))
    return x_train, x_test, y_train, y_test


# In this function we standardize the features of training and testing
def standardize_feature(x_train, x_test):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled
