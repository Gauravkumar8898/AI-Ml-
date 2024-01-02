from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify
import pickle
import pandas as pd
from src.pipeline.svm_pipeline import pipeline

app = Flask(__name__)

# load pickle file
model = pickle.load(open("src/Flask/model.pkl", "rb"))


@app.route("/predict", methods=['POST'])
def predict():
    json_ = request.json
    query_df = pd.DataFrame(json_)
    scaler = StandardScaler()
    x_test = scaler.fit_transform(query_df)
    prediction = model.predict(x_test)
    return jsonify({"Prediction": str(prediction)})


# WSGI Application
@app.route('/')
def welcome():
    return "Hello World"


# make pickle file of our model


if __name__ == "__main__":
    app.run(debug=True)

