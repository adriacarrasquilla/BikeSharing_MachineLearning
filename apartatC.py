
# -*- coding: utf-8 -*-

from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import datetime

# Visualitzarem només 3 decimals per mostra
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# Funcio per a llegir dades en format csv
def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    return dataset

#Without numpy
def mean_squeared_error(y1, y2):
    # comprovem que y1 i y2 tenen la mateixa mida
    assert(len(y1) == len(y2))
    mse = 0
    for i in range(len(y1)):
        mse += (y1[i] - y2[i])**2
    return mse / len(y1)

#with numpy
def mse(v1, v2):
    return ((v1 - v2)**2).mean()

def regression(x, y):
    # Creem un objecte de regressió de sklearn
    regr = LinearRegression()

    # Entrenem el model per a predir y a partir de x
    regr.fit(x, y)

    # Retornem el model entrenat
    return regr

def standarize(x_train):
    mean = x_train.mean(0)
    std = x_train.std(0)
    x_t = x_train - mean[None, :]
    x_t /= std[None, :]
    return x_t


# Carreguem dataset d'exemple
dataset = pd.read_csv('bike-sharing-dataset/day.csv', header=0, delimiter=',')
datasetHora = pd.read_csv('bike-sharing-dataset/hour.csv', header=0, delimiter=',')
dataset = dataset.fillna(dataset.mean())

#creem columna dia(substitut data)
#dataset.insert(0, 'dia', int(0))
"""
#omplim aquesta columna
i=-1
print(range(dataset['dia'].shape[0]))
for index in range(dataset['dia'].shape[0]):
	if index%24 == 0:
		i+=1
	dataset['dia'][index]=i
"""

"""
#Esborrem atributs que no usarem
del dataset["dteday"]
del dataset["instant"]
del dataset["casual"]
del dataset["registered"]
del dataset["atemp"]
del dataset['holiday']
del dataset['windspeed']

del dataset['yr']
del dataset['mnth']
del dataset['weekday']
del dataset['workingday']
del dataset['weathersit']

#del dataset['temp']
del dataset['season']
del dataset['hum']


#del dataset['dia']
#del dataset['hr']
#PROVES
#dataset = dataset.drop(['casual','registered','instant','holiday','windspeed','atemp'],axis=1)
"""




# GUIA

# https://www.kaggle.com/abdul002/bike-sharing-rental-prediction







#Esborrem camps inutils
dataset = dataset.drop(["dteday","instant","casual","registered","atemp",'holiday','windspeed'], axis=1)

#comprovem si hi ha elements nulls
dataset.info()

#per entendre la naturalesa del nostre atribut a predir
datasetHora['cnt'].describe()

ax = datasetHora[['hr','cnt']].groupby(['hr']).sum().reset_index().plot(kind='bar', figsize=(8, 6),
                                       legend = False, title ="Lloguer total per hores", 
                                       color='#054466', fontsize=12, width=1.5)
ax.set_xlabel("Hora", fontsize=10)
ax.set_ylabel("Cnt", fontsize=10)
plt.show()

print (dataset.keys())


data = dataset.values
dataNorm = standarize(data)
x_t = dataNorm[:, :8] 
y = dataNorm[:, 8] 




regr = regression(x_t[:,:8], y) 
predicted = regr.predict(x_t[:,:8]) 

MSE = mse(y, predicted)
r2 = r2_score(y, predicted)





regr = regression(x_t, y) 
predicted = regr.predict(x_t)

MSE = mse(y, predicted)
r2 = r2_score(y, predicted)

print("Mean squeared error: ", MSE)
print("R2 score: ", r2)



plt.figure()
ax = plt.scatter(x_t[:,6], y)
plt.plot(x_t[:,6], predicted, '#054466')
plt.show()





def split_data(x, y, train_ratio=0.8):
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    n_train = int(np.floor(x.shape[0]*train_ratio))
    indices_train = indices[:n_train]
    indices_val = indices[n_train:] 
    x_train = x[indices_train, :]
    y_train = y[indices_train]
    x_val = x[indices_val, :]
    y_val = y[indices_val]
    return x_train, y_train, x_val, y_val


# Dividim dades d'entrenament
x_train, y_train, x_val, y_val = split_data(x_t, y)

for i in range(x_train.shape[1]):
    x_t = x_train[:,i] # seleccionem atribut i en conjunt de train
    x_v = x_val[:,i] # seleccionem atribut i en conjunt de val.
    x_t = np.reshape(x_t,(x_t.shape[0],1))
    x_v = np.reshape(x_v,(x_v.shape[0],1))

    regr = regression(x_t, y_train)    
    error = mse(y_val, regr.predict(x_v)) # calculem error
    r2 = r2_score(y_val, regr.predict(x_v))

    print("Error en atribut %d: %f" %(i, error))
    print("R2 score en atribut %d: %f\n" %(i, r2))




