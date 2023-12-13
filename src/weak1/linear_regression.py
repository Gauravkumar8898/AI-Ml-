import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("src/data/housing_price_dataset.csv")
# print((dataset.head(5)))
# print(dataset.tail())
# print(dataset.tail())
# print(dataset.info())

house_x_train = dataset.SquareFeet[:-39999]
house_y_train = dataset.Price[:-39999]
# house_xtest=  dataset.SquareFeet[]

# plt.scatter(house_x_train,house_y_train,c="r",marker="*")
# plt.show()

# ligression model
w = 20
b = 50


def computeOutputModel(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb


temp_f_wb = computeOutputModel(house_x_train, w, b)
# print(temp_f_wb)
plt.scatter(house_x_train, house_y_train, c='r', marker="*", label="actual value")
plt.plot(house_x_train, temp_f_wb, c='b', label="our prediction")
plt.title('price prediction')
plt.ylabel('price in dollar')
plt.xlabel('size in sq feet')
plt.legend()
# plt.show()
plt.savefig("src/image/squares.png")


