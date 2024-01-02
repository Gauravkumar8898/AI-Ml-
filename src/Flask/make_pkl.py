import pickle


def make_pickle(svm):
    pickle.dump(svm, open("src/Flask/model.pkl", "wb"))
