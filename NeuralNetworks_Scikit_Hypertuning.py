"""
Developer : Naveen
This file contains the code to create a neural network. Neural Network is tuned using Scikit GridSearchCV. In order to tune each parameter one neeeds to uncomment the section of code while commenting the rest of 
the parameters
"""

import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from math import sqrt
from sklearn.model_selection import cross_val_predict
import os
from    matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from sklearn.model_selection import GridSearchCV
from keras.optimizers import SGD
from keras.layers import Dropout




#reading the dataframe and filling the null values
dataframe = pandas.read_csv(r'DataSet.csv')
dataframe = dataframe.fillna(value=0)



def larger_model(dropout_rate=0.0, weight_constraint=0):
    model = Sequential()
    model.add(Dense(12, input_dim=13, kernel_initializer='zero', activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, kernel_initializer='zero'))
    # optimizer = SGD(lr=learn_rate, momentum=momentum)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def main(trait):
    X = dataframe.loc[:,'mediaUsage':'Scheduling_OfficeTools_Weather'].values
    Y = dataframe.loc[:,trait].values
    seed=7
    numpy.random.seed(seed)

    NN = KerasRegressor(build_fn=larger_model,verbose=0,epochs=100,batch_size=10)
	
	
	#tuning batch_size
    batch_size = [10, 20, 40, 60, 80, 100]
    epochs = [10, 50, 100]
    param_grid = dict(batch_size=batch_size, epochs=epochs)
    
	
	#tuning Optimizer
    ptimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    param_grid = dict(optimizer=optimizer)
    #
    
	#tuning the learning rate
    #define the grid search parameters
    learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
    momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    param_grid = dict(learn_rate=learn_rate, momentum=momentum)
    
	
	#tuning the neurons weight initializtion
    # define the grid search parameters
    init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    param_grid = dict(init_mode=init_mode)
    
	
	#tuning the activation functions
    activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    param_grid = dict(activation=activation)
    
	
	#tuning the number of neurons in hiddden layers
    neurons = [1, 5, 10, 15, 20, 25, 30]
    param_grid = dict(neurons=neurons)




    #tuning the weight constraints
    weight_constraint = [1, 2, 3, 4, 5]
    dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    param_grid = dict(dropout_rate=dropout_rate, weight_constraint=weight_constraint)
    
    CV_NN = GridSearchCV(estimator= NN, param_grid=param_grid, cv=5)
    CV_NN.fit(X,Y)
    print(CV_NN.best_params_)



#check for all big5 traits
traits=['Openness','Conscientiousness','Extraversion','Agreeableness','Neuroticism']
for trait in traits:
    main(trait)



