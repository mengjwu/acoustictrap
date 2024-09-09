from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
import time
import csv
import datetime
import csv
import datetime
startTime = time.time()
print("Training process start time is:",startTime)

data_all = np.loadtxt("extract_amplitude_for_train_1lmd.csv", delimiter=",")
testing_all = np.loadtxt("extract_amplitude_for_test_1lmd.csv", delimiter=",")
err = open('amp_predict_1lmd.csv','w')

x = data_all[0:805,0:18]  #805
y = data_all[0:805,18:26]  # num*8
xt = testing_all[0:115,0:18]
yt = testing_all[0:115,18:26]

scalerx = StandardScaler().fit(x)
scalery = StandardScaler().fit(y)
x = scalerx.transform(x)
y = scalery.transform(y)

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.13)

print("X trainning size is:",X_train.shape)
print("Y trainning size is:",Y_train.shape)
print("X Testing size is:",X_test.shape)
print("Y Testing size is:",Y_test.shape)
print("xt  size is:",xt.shape)
# X_train = X_train.reshape(-1, 1)
# X_test = X_test.reshape(-1, 1)

# alpha:L2的参数：MLP是可以支持正则化的，默认为L2，具体参数需要调整
#'logistic'，logistic sigmoid函数，返回f（x）= 1 /（1 + exp（-x））。
#'tanh'，双曲tan函数，返回f（x）= tanh（x）。
#'relu'，整流后的线性单位函数，返回f（x）= max（0，x）
model_mlp = MLPRegressor(
    hidden_layer_sizes=(70,160,250,160,70),  activation='tanh', solver='adam', alpha=0.0001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.0004, power_t=0.7, max_iter=7000, shuffle=True,
    random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.95, nesterovs_momentum=True,
    early_stopping=False, beta_1=0.95, beta_2=0.999, epsilon=1e-08)

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

wrapper_score = wrapper.score(X_test,Y_test)
# print(X_test)
print('sklearn多层感知器-回归模型得分',wrapper_score)#预测正确/总数
result = wrapper.predict(X_test)
stopTime = time.time()
print("Training process stop time is:",stopTime)
sumTime = stopTime - startTime
print('Training process time for AMPLITUDE [units: second]:', sumTime)

startTime1 = time.time()
for i in range(0,115,1):
    data_all_testing_x = np.array([xt[i]])
    data_all_testing = scalerx.transform(data_all_testing_x)
    result_testing_y = wrapper.predict(data_all_testing)
    result_testing_y_trans = scalery.inverse_transform(result_testing_y)
    writer = csv.writer(err)
    for j in result_testing_y_trans:
        writer.writerow(j)
err.close()
stopTime1 = time.time()
print('Testing process time for 115 samples in AMPLITUDE [units: second]:', stopTime1-startTime1)


data_all_testing_x1 = np.array([[133, 184, 160.94, 164.57, 168.21, 171.85, 175.49, 179.12, 182.76, 186.4, 262.62, 260.53, 258.42, 256.33, 254.23, 252.13, 250.03, 247.93]])
data_all_testing1 = scalerx.transform(data_all_testing_x1)
result_testing_y1 = wrapper.predict(data_all_testing1)
result_testing_y_trans1 = scalery.inverse_transform(result_testing_y1)
print("amplitude prediction:",result_testing_y_trans1)
max_amp=np.max(result_testing_y_trans1)
coff_amp=max_amp/result_testing_y_trans1
print("amplitude modulated (AM) coefficients ：",coff_amp)

# result1 = wrapper.predict([[125,143]])
# print(result1)
# inp = [[ele] for ele in X_train]
# pre = clf.predict(inp)
# print(pre)
# print(X_test.shape, Y_test.shape)
Dim = 6
# print(Y_test[:,Dim])
# plt.plot(list(range(0,Y_test.shape[0])), Y_test[:,Dim], 'b')
# plt.plot(list(range(0,Y_test.shape[0])), result[:,Dim], 'r')
plt.plot(list(range(0,Y_test.shape[0])), Y_test[:,Dim], 'bo')
plt.plot(list(range(0,Y_test.shape[0])), result[:,Dim], 'ro')
plt.ylim(-2,2)
plt.show()
