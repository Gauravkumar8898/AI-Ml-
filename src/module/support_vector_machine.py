from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import SVC


def train_model_svm(x_train_scaled, y_train):
    params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                    'C': [1, 10, 100, 1000]},
                   {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    svm = GridSearchCV(SVC(), params_grid, cv=5)
    svm = svm.fit(x_train_scaled, y_train)
    return svm


def display_score_for_model(svm, y_test, y_prediction):
    print('Best score for training data:', svm.best_score_, "\n")

    # View the best parameters for the model found using grid search
    print('Best C:', svm.best_estimator_.C, "\n")
    print('Best Kernel:', svm.best_estimator_.kernel, "\n")
    print('Best Gamma:', svm.best_estimator_.gamma, "\n")
    accuracy = accuracy_score(y_test, y_prediction)
    report = classification_report(y_test, y_prediction)

    # Print the results
    print(f'Accuracy: {accuracy}')
    print(f'Classification Report:\n{report}')


def prediction_svm(svm, x_test_scaled):
    model = svm.best_estimator_
    y_prediction = model.predict(x_test_scaled)
    return y_prediction

