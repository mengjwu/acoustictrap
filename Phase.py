# -*- coding: utf-8 -*-

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
import time
import csv
import datetime

data_all = np.loadtxt("training_sample.csv", delimiter=",")
testing_all = np.loadtxt("sample_testing_50.csv", delimiter=",")
err = open('predict2lmd1.csv','w')

x = data_all[0:290,0:18]  
y = data_all[0:290,18:26]  # num*8
xt = testing_all[0:50,0:18]
yt = testing_all[0:50,18:26]

scalerx = StandardScaler().fit(x)
scalery = StandardScaler().fit(y)
x = scalerx.transform(x)
y = scalery.transform(y)

# X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.1)

print("X trainning size is:",X_train.shape)
print("Y trainning size is:",Y_train.shape)
print("X Testing size is:",X_test.shape)
print("Y Testing size is:",Y_test.shape)
# print("xt Testing size is:",xt)
# X_train = X_train.reshape(-1, 1)
# X_test = X_test.reshape(-1, 1)

# alpha:L2 ：MLP是可以支持正则化的，默认为L2，具体参数需要调整
#'logistic'，logistic sigmoid函数，返回f（x）= 1 /（1 + exp（-x））。
#'tanh'，双曲tan函数，返回f（x）= tanh（x）。
#'relu'，整流后的线性单位函数，返回f（x）= max（0，x）
model_mlp = MLPRegressor(
    hidden_layer_sizes=(30,80,150,80,30),  activation='tanh', solver='adam', alpha=0.0001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=5000, shuffle=True,
    random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False,beta_1=0.9, beta_2=0.999, epsilon=1e-08)

# model_mlp.fit(X_train, y_train)
# 将创建的模型对象作为参数传入
wrapper = MultiOutputRegressor(model_mlp)
# 训练模型
wrapper.fit(X_train, Y_train)
# startTime = time.time()
# x1 = x.reshape(-1,1)
# mlp_score=model_mlp.score(x1,y)
# print('sklearn多层感知器-回归模型得分',mlp_score)#预测正确/总数
# result = model_mlp.predict(x1)
#stopTime = time.time()
#sumTime = stopTime - startTime
#print('总时间是：', sumTime)
# inp = [[ele] for ele in X_train]
# pre = clf.predict(inp)
# #print(pre)
# plt.plot(X_train, Y_train, 'bo')
# plt.plot(X_test, result, 'ro')
# plt.show()

startTime = time.time()
print("Start time is:",startTime)
wrapper_score = wrapper.score(X_test,Y_test)
# print(X_test)
print('sklearn多层感知器-回归模型得分',wrapper_score)#预测正确/总数
result = wrapper.predict(X_test)
stopTime = time.time()
print("Stop time is:",stopTime)
sumTime = stopTime - startTime
print('总时间是：', sumTime)

for i in range(0,50,1):
    # print("i is:", i)
    data_all_testing_x = np.array([xt[i]])
    # print("xt[i]] is:",  data_all_testing_x)
    data_all_testing = scalerx.transform(data_all_testing_x)
    result_testing_y = wrapper.predict(data_all_testing)
    result_testing_y_trans = scalery.inverse_transform(result_testing_y)
    # print(result_testing_y_trans.shape)
    #writer = csv.writer(err)
    #for j in result_testing_y_trans:
    #    writer.writerow(j)
#err.close()

# 8*2=16 elements position (x,y)
data_all_testing_x1 = np.array([[126,188,171.37,175.01,178.64,182.28,185.92,189.56,193.19,196.83,262.37,260.28,258.18,256.08,253.97,251.88,x_axis_coordinate,y_axis_coordinate]])
print("data_all_testing_x1 size is:",data_all_testing_x1.shape)
data_all_testing1 = scalerx.transform(data_all_testing_x1)
result_testing_y1 = wrapper.predict(data_all_testing1)

result_testing_y_trans1 = scalery.inverse_transform(result_testing_y1)
print("ToF预测结果是",result_testing_y_trans1)
min_tof=np.min(result_testing_y_trans1)
print("ToF minmum value是：",min_tof)
normalized_tof=result_testing_y_trans1-min_tof
print("ToF 减去最小值后归一化是：",normalized_tof)
phase_normalized=2*3.1415*normalized_tof

# result1 = wrapper.predict([[125,143]])
# print(result1)
# inp = [[ele] for ele in X_train]
# pre = clf.predict(inp)
# print(pre)
# print(X_test.shape, Y_test.shape)
Dim = 6
print(Y_test[:,Dim])
# plt.plot(list(range(0,Y_test.shape[0])), Y_test[:,Dim], 'b')
# plt.plot(list(range(0,Y_test.shape[0])), result[:,Dim], 'r')
plt.plot(list(range(0,Y_test.shape[0])), Y_test[:,Dim], 'bo')
plt.plot(list(range(0,Y_test.shape[0])), result[:,Dim], 'ro')
plt.ylim(-2,2)
# plt.show()
print("Time Reversed后的Trap Phase is",phase_normalized)