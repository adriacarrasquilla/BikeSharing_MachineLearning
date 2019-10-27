
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
import matplotlib.ticker as ticker

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
correlacio = dataset.corr().round(2)

plt.figure(figsize=(10,8))

ax = sb.heatmap(correlacio, annot=True, linewidths=.5, )
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.savefig('heatmap_tot_bo.png')

#Executar aquest codi quan s'hagi d'executar tot, sino ens donarà error al eliminar una columna ja eliminada

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





# GUIA

# https://www.kaggle.com/abdul002/bike-sharing-rental-prediction







#Esborrem camps inutils
#dataset = dataset.drop(["dteday","instant","casual","registered","atemp",'holiday','windspeed'], axis=1)

#comprovem si hi ha elements nulls
dataset.info()

#per entendre la naturalesa del nostre atribut a predir
datasetHora['cnt'].describe()

ax = datasetHora[['temp','cnt']].groupby(['temp']).sum().reset_index().plot(kind='bar', figsize=(8, 6),
                                       legend = False, title ="Lloguer total en funció la temperatura", 
                                       color='#a80027', fontsize=12, width=1.5)
ax.set_facecolor('#ffecea')
ax.set_xlabel("temp", fontsize=10)
ax.set_ylabel("Cnt", fontsize=10)
plt.show()

"""
count = datasetHora['workingday'].value_counts()
cntCount = datasetHora.groupby('workingday').sum()['cnt']
print(cntCount)
print(count)
res = cntCount/count
work=['no-laborable', 'laborable']

fig = plt.figure()
plt.bar(work, res, color = ["#ba343a", "#a80027"])
plt.xticks(work, ('No Laborable', 'Laborable'))

plt.show()
"""

def rand_jitter(arr):
    stdev = .01*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev

def jitter(x, y, s=20, c='b', marker='o', cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, **kwargs):
    return plt.scatter(rand_jitter(x), rand_jitter(y), s=s, c=c, marker=marker, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, alpha=alpha, linewidths=linewidths, verts=verts, **kwargs)


plt.figure(figsize=(12,6))
ax = jitter(datasetHora['hr'], datasetHora['cnt'], c = datasetHora['temp'], cmap='inferno')
plt.xlabel("Hora")
plt.ylabel("Cnt")
cbar = plt.colorbar()
cbar.set_label('temp', rotation=270)
plt.xticks(datasetHora['hr'])
plt.savefig("tmp_hour_cnt_jigger")

plt.figure(figsize=(12,6))
ax = plt.scatter(datasetHora['hr'], datasetHora['cnt'], c = datasetHora['temp'], cmap='inferno')
plt.xlabel("Hora")
plt.ylabel("Cnt")
cbar = plt.colorbar()
cbar.set_label('temp', rotation=270)
plt.xticks(datasetHora['hr'])
plt.savefig("tmp_hour_cnt")


"""
count = datasetHora['weathersit'].value_counts()
temps=['Clear','Cloudy','Low rainfall', 'High rainfall']
fig = plt.figure()
plt.bar(temps, count, color = ["#461220","#ca343a", "#ba343a","#a80027",])
plt.title("Count del weathersit")
plt.show()
"""
"""
plt.figure()
ax = sb.countplot(x='workingday', data=datasetHora)
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
ax.xaxis.set_major_locator(ticker.MultipleLocator(base=5))
plt.show()
"""

print (dataset.keys())


data = dataset.values
dataNorm = standarize(data)
x_t = dataNorm[:, :1]
y = dataNorm[:, 1]




regr = regression(x_t[:,:1], y)
predicted = regr.predict(x_t[:,:1])

MSE = mse(y, predicted)
r2 = r2_score(y, predicted)





regr = regression(x_t, y) 
predicted = regr.predict(x_t)

MSE = mse(y, predicted)
r2 = r2_score(y, predicted)

print("Mean squeared error: ", MSE)
print("R2 score: ", r2)


"""
plt.figure()
#ax.set_facecolor('#ffecea')
ax = plt.scatter(x_t[:,0], y, c='#ba343a', alpha=0.7, edgecolors = 'none', facecolor='#ffecea')
#plt.plot(x_t[:,0], predicted, '#461220')
plt.title("Cnt en funció la temperatura (Normalitzat)")
plt.xlabel("Temperatura")
plt.ylabel("Cnt")
plt.show()
"""





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

    print(y_val)
    print(x_train[:,i])
    plt.figure()
    ax = plt.scatter(x_val[:,i], y_val)
    plt.plot(x_val[:,i], regr.predict(x_v), '#054466')
    plt.show()



