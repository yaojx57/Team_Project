import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import copy
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor



def estimation_2d(train_loader):
    predictions_2d = []
    train_data = pd.DataFrame(train_loader)
    training_data = copy.copy(train_data)
    
    for i in training_data:
        training_data[i] = pd.DataFrame(training_data[i])
        for j in range(len(training_data)):
            training_data[i][j] = list(training_data[i][j])
    
    list_training_2d = training_data['2d'].to_list()
    list_training_3d = training_data['3d'].to_list()

    def to_array(li):
        for i in range(len(li)):
            li[i] = np.array(li[i])
            for j in range(len(li[i])):
                li[i][j] = np.array(li[i][j])
                for k in range(len(li[i][j])):
                    li[i][j][k] = np.array(li[i][j][k])
        return li
    
    x = to_array(list_training_2d)
    y = to_array(list_training_3d)

    def mlp_reg(x_train, y_train, d):
        nsamples, nx, ny = np.array(x_train).shape
        nsamplesy, na, nb = np.array(y_train).shape
        regr = MLPRegressor(random_state=1, max_iter=500)\
        .fit(x_train.reshape(nsamples, nx*ny), y_train.reshape(nsamplesy, na*nb))
        loss = regr.loss_
        if d != 3:
            prediction = regr.predict(x_train.reshape(nsamples, nx*ny)).reshape(nsamples, nx, ny)
        else:prediction = None
        return loss, prediction

    y_2d = []
    for i in range(len(y)):
        l2 = []
        for j in range(len(y[i])):
            l1 = []
            for k in range(len(y[i][j])):
                l = y[i][j][k][:2]
                l1.append(l)
            l2.append(l1)
        y_2d.append(l2)
    y_2d = to_array(y_2d)

    predictions = []
    sum_loss = 0
    for i in range(len(x)):
        loss, temp =  mlp_reg(x[i], y_2d[i])
        sum_loss += loss
        predictions.append(temp)
        sum += 1
        if i > 10:
            break
    mean_loss = loss/sum
    print(mean_loss)
    predictions_2d = to_array(predictions)


    # sum_loss = 0
    # sum = 0
    # for i in range(len(predictions)):
    #     loss, temp = mlp_reg(predictions[i], y[i], 3)
    #     sum+= 1
    #     sum_loss += loss
    #     if i > 10:
    #         break
    # print(sum_loss/sum)

    return predictions_2d