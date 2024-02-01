from src.preprocessing.preprocessing_titanic import TitanicPreprocessing
from src.neaural_network_model.neaural_network_titanic import TitanicModel


def pipeline():
    x_train, y_train, x_test, y_test = TitanicPreprocessing.work_flow()
    TitanicModel.work_flow_neural_network(x_train, y_train, x_test, y_test)
