"""
Developer : Naveen
This file contains the code to create a neural network. Neural Network is tuned manually with out any libraries.
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

hidden_nodes=0
activator=''

#reading the dataframe and filling the null values
dataframe = pandas.read_csv(r'DataSet.csv')
dataframe = dataframe.fillna(value=0)





def larger_model():
    """
	method to create the neural net layers  using Keras layer on top of Tensorflow
	"""
    global hidden_nodes,activator

    model = Sequential()
    model.add(Dense(hidden_nodes, input_dim=13, kernel_initializer='normal', activation='sigmoid'))
    model.add(Dense(5, kernel_initializer='normal', activation='tanh'))
    model.add(Dense(5, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def main(trait,splits,batchsize,hiddennodes,activation):

    # split into input (X) and output (Y) variables
    X = dataframe.loc[:,'mediaUsage':'Scheduling_OfficeTools_Weather'].values
    Y = dataframe.loc[:,trait].values
    
	#getting the NN parameters
	global hidden_nodes,activator
    hidden_nodes=hiddennodes
    activator=activation
    seed=7
    numpy.random.seed(seed)
    
	
	#adding the scalar and NN model
	estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasRegressor(build_fn=larger_model, nb_epoch=300, batch_size=batchsize , verbose=0)))


    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=splits, random_state=seed)
    
    #predicted the values using cross validation
    results = cross_val_predict(pipeline,X,Y,cv=kfold)

    dataframe['predicted'] = results
	
	#grouping by day
    grouped = dataframe.groupby(['ID'])
    outputlist=[(key,numpy.average(value[trait]),numpy.average(value['predicted'])) for (key,value) in grouped.__iter__()]
    outputdf= pandas.DataFrame(outputlist,columns=['ID','Actual','Predicted'])
    outputdf=outputdf.round({'Actual': 2, 'Predicted': 2}) # Rounding off two two decimals

    #Getting the Error and converting in to percentage

    outputdf['Error'] =  outputdf['Actual']-outputdf['Predicted']
    outputdf['Error']= (numpy.abs(outputdf['Error']) *100)
    
    Actual=  outputdf.loc[:,'Actual'].values
    Predicted=outputdf.loc[:,'Predicted'].values

    #Plotting Fit line for predicted and actual values
    fig, ax = plt.subplots()
    ax.scatter(Actual,Predicted, edgecolors=(0, 0, 0))
	
	#drawing fit line at 45 degrees
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    
    ax.set_xlabel('Actual Values (Big Five PreSurvey)')
    ax.set_ylabel('Predicted Values')
    plt.title("Neural Networks - "+trait)
    plt.show()
    # plt.savefig('/home/naveen/Desktop/Plots1/'+trait+': batch size'+str(batchsize)+':'+': hidden'+str(hiddennodes)+':splits'+str(split)+'.png')
    

    return (sqrt(mean_squared_error(Actual,Predicted))*100)




traits=['Openness','Conscientiousness','Extraversion','Agreeableness','Neuroticism']

splits=[4,10,20]
batchsizes=[10,20,50,100]
hiddennodes=[5,10,20,30,50]
results={}
activators=['relu','sigmoid','tanh']


#running the NN model for Big5 traits
for trait in traits:
    for split in splits:
        # print("Split",split)
        for batch in batchsizes:

            for hidden in hiddennodes:

                for activation in activators:
                    list=[]
                    for i in range(500):
                        # running the method for 500 times
                        list.append(main(trait,split,batch,hidden,activation))
                    model_parameters= trait+' '+'Splits -'+str(split)+' ' +'Batch Size -'+str(batch)+' '+'Hidden Nodes-'+str(hidden)+' '+'Activation-'+str(activation)
                    results[model_parameters]= numpy.average(list)


#printing results for each trait
for key,val in results.items():
    print(key,"-------",val)
