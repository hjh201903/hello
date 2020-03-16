# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 19:26:46 2020

@author: lenovo
"""


import numpy as np
import xlrd
import matplotlib.pyplot as plt
import math
from pandas import DataFrame,Series
import statsmodels.api as sm
import pandas as pd
from sklearn.linear_model import LinearRegression
from pandas import DataFrame,Series

data = xlrd.open_workbook('steady4.xlsx')
table = data.sheets()[0]
  
Mw_e = np.array([float(i) for i in table.col_values(3)[1:40000]])  #冷冻水流量
Tcw_i  = np.array([float(i) for i in table.col_values(2)  [1:40000]])  #冷冻水进水温度
Tcw_o  = np.array([float(i) for i in table.col_values(0)  [1:40000]])  #冷冻水出水温度

Tcd_i  = np.array([float(i) for i in table.col_values(4)  [1:40000]])  #冷却水进水温度
Tcd_o  = np.array([float(i) for i in table.col_values(5)  [1:40000]])  #冷却水出水温度
y  = np.array([float(i) for i in table.col_values(1)  [1:40000]])  #冷冻侧功率

def X_1(Tcd_o,Tcw_o):
    x_1=Tcd_o-Tcw_o
    return x_1

x1=X_1(Tcd_o,Tcw_o)

def X_2(x1):
    x_2=x1**2
    return x_2

x2=X_2(x1)

def X_3():
    x_3=(Tcw_i-Tcw_o)*Mw_e
    return x_3

x3=X_3()

def X_4(x3):
    x_4=x3**2
    return x_4

x4=X_4(x3)

def X_5():
    x_5=(Tcd_o -Tcw_o)*(Tcw_i-Tcw_o)*Mw_e
    return x_5
x5=X_5()

x=np.column_stack((x1,x2,x3,x4,x5,))

#建立初步的数据集模型之后将训练集中的特征值与标签值放入LinearRegression()模型中且使用fit函数进行训练,在模型训练完成之后会得到所对应的方程式（线性回归方程式）需要利用函数中的intercept_与coef_。
model = LinearRegression()

model.fit(x,y)

a  = model.intercept_#截距
b = model.coef_#回归系数
print("最佳拟合线:截距",a,",回归系数：",b)

#对线性回归进行预测
Y_pred = model.predict(x) 
print(Y_pred) 

plt.plot(range(len(Y_pred)),Y_pred,'b',label="predict")
plt.plot(range(len(y)),y,'r',label="test")
#显示图像
plt.savefig("predict.jpg")
plt.show()

from sklearn import metrics
MSE = metrics.mean_squared_error(y, Y_pred)
RMSE = np.sqrt(metrics.mean_squared_error(y, Y_pred))

print('标准差：',RMSE)

