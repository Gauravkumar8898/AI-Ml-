from src.preprocessing.preprocessing_cat_dogs import Preprocessing
from src.neaural_network_model.neural_network_cats_dogs import CatsDogsModel


def cats_dogs_pipeline():
    train_dataset, test_dataset = Preprocessing.runner_preprocessing()
    CatsDogsModel.model_runner(train_dataset, test_dataset)