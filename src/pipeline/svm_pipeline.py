from src.utils.constant import lung_cancer_data_path
from src.preprocessing.svm_preprocessing import (load_dataset, drop_unused_feature,
                                                 set_feature_target_variable, transform_data,
                                                 split_dataset, standardize_feature)
from src.module.support_vector_machine import (train_model_svm, display_score_for_model,
                                               prediction_svm)

from src.Flask.make_pkl import make_pickle
def pipeline():
    dataset = load_dataset(lung_cancer_data_path)
    dataset = drop_unused_feature(dataset)
    df_features, df_target = set_feature_target_variable(dataset)
    df_target = transform_data(df_target)
    x_train, x_test, y_train, y_test = split_dataset(df_features, df_target)
    x_train_scaled, x_test_scaled = standardize_feature(x_train, x_test)
    svm = train_model_svm(x_train_scaled, y_train)
    y_prediction = prediction_svm(svm, x_test_scaled)
    display_score_for_model(svm, y_test, y_prediction)
    svm = svm.best_estimator_
    make_pickle(svm)
