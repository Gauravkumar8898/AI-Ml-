#  -----------------------------Assignment 2-------------------------------------
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

dataset = pd.read_csv("/home/knoldus/PycharmProjects/AI/Ml-Intern/src/data/housing_price_dataset.csv")

dataset['SquareFeet'] = dataset['SquareFeet']/1000
house_x_train = dataset.SquareFeet[:39999]
dataset['Price'] = (dataset['Price']/10000).round(2)
house_y_train = dataset.Price[:39999]

def computeOutputModel(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb



def costForLinearrigresion(x, y, w, b):
    m=x.shape[0]
    costSum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i])**2
        costSum += cost
    totalCost = (1/(2*m))*costSum
    return totalCost


def Gradient(x, y, w, b):
    m = x.shape[0]
    der_w = 0
    der_b = 0
    for i in range(m):
        f_wb = w * x[i] + b
        der_w_i = (f_wb - y[i]) * x[i]
        der_b_i = (f_wb - y[i])
        # der_w = der_w + der_w_i
        der_w = np.add(der_w, der_w_i)
        # der_b = der_b + der_b_i
        der_b = np.add(der_b, der_b_i)
    der_w /= m
    der_b /= m
    return der_w, der_b


def gradientDicent(x, y, w_in, b_in, alpha, num_iters, gradientFunction,):
    w = w_in
    b = b_in
    for i in range(num_iters):
        dj_w, dj_b = gradientFunction(x, y, w, b)
        b = b - alpha * dj_b
        w = w - alpha * dj_w
    return w, b


w_int = 0
b_int = 0
itreration = 100
tmp_alpha = 1.0e-2
w_f, b_f = gradientDicent(house_x_train, house_y_train, w_int, b_int,
                          tmp_alpha, itreration, Gradient)

houseSize1 = 1.495
houseSize2 = 1.955
houseSize3 = 2.235

print(f"(w,b) found by gradient descent: ({w_f:8.4f},{b_f:8.4f})")

print(f"predicted price for {houseSize1*1000} is : {(w_f*houseSize1+b_f)*10000:8.2f}")
print(f"predicted price for {houseSize2*1000} is : {(w_f*houseSize2+b_f)*10000:8.2f}")
print(f"predicted price for {houseSize3*1000} is : {(w_f*houseSize3+b_f)*10000:8.2f}")