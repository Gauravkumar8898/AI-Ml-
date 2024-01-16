import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from src.utils.constant import data_path_mall_customers
from src.utils.helpers import load_dataset

data = pd.read_csv(data_path_mall_customers)


class kMeanClustering:

    @staticmethod
    def plot_data_point(data_set):
        plt.figure(figsize=(15, 7))
        plt.title('Scatter plot of Age v/s Spending Score', fontsize=20)
        plt.xlabel('Age')
        plt.ylabel('Spending Score')
        plt.scatter(x='Age', y='Spending Score (1-100)', data=data_set)
        plt.show()

    @staticmethod
    def data_for_kmean(data_set):
        new_data = data_set[['Age', 'Spending Score (1-100)']].iloc[:, :].values
        return new_data

    @staticmethod
    def deciding_k_value(new_data):
        inertia = []
        for n in range(1, 15):
            algorithm = (KMeans(n_clusters=n, init='k-means++', n_init=10, max_iter=300,
                                tol=0.0001, random_state=71, algorithm='lloyd'))
            algorithm.fit(new_data)
            inertia.append(algorithm.inertia_)
        return inertia

    @staticmethod
    def plot_elbow(inertia):
        plt.figure(1, figsize=(15, 6))
        plt.plot(np.arange(1, 15), inertia, 'o')
        plt.plot(np.arange(1, 15), inertia, '-', alpha=0.5)
        plt.xlabel('Number of Clusters'), plt.ylabel('Inertia')
        plt.show()

    @staticmethod
    def k_mean_for_k(k, dataset):
        model = (KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300,
                        random_state=81, algorithm='lloyd'))
        model.fit(dataset)
        return model

    @staticmethod
    def work_flow_kmean():
        data_set = load_dataset(data_path_mall_customers)
        kMeanClustering.plot_data_point(data_set)
        new_data_set = kMeanClustering.data_for_kmean(data_set)
        inertia = kMeanClustering.deciding_k_value(new_data=new_data_set)
        kMeanClustering.plot_elbow(inertia)
        model = kMeanClustering.k_mean_for_k(k=3, dataset=new_data_set)



