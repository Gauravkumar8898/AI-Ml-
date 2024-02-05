from src.preprocessing.preprocessing_cat_dogs import Preprocessing
from src.neaural_network_model.transfer_learning import transfer_learning_runner


def cats_dogs_tf_pipeline():
    train_dataset, test_dataset = Preprocessing.runner_preprocessing()
    transfer_learning_runner(train_dataset, test_dataset)
