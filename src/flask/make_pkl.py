import pickle


def make_pickle(svm):
    pickle.dump(svm, open("src/flask/model.pkl", "wb"))
