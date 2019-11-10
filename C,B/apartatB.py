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
dataset = pd.read_csv('bike-sharing-dataset/hour.csv', header=0, delimiter=',')
datasetHora = pd.read_csv('bike-sharing-dataset/hour.csv', header=0, delimiter=',')
dataset = dataset.fillna(dataset.mean())

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
#del dataset['hr']
del dataset['season']
del dataset['hum']

print(dataset.head())


class RegressorO(object):
    def __init__(self, arrayTheta, alpha, x, y, maxIt, epsilon, reg):
        # Inicialitzem theta0 i theta1 (per ser ampliat amb altres theta's)
        self.arrayTheta = arrayTheta
        self.alpha = alpha
        self.maxIt = maxIt
        self.epsilon = epsilon
        self.reg = reg
        self.x = x
        self.y = y


    def predict(self):
        # implementar aqui la funció de prediccio
        loss = np.dot(self.x, self.arrayTheta)
        cost = np.sum(loss ** 2) / (2 * self.x.shape[0])
        print("Cost: "+str(cost))
        return loss

    def __update(self, hy):
        # actualitzar aqui els pesos donada la prediccio (hy) i la y real. pass
        #loss = hy - y
        #cost = np.sum(loss**2)/(2*y.shape[0])
        #gradient = np.dot()
        self.arrayTheta = self.arrayTheta*(1-self.alpha*(self.epsilon/self.x.shape[0])) -(1/self.y.shape[0])*alpha*( self.x.T.dot((hy - self.y)))

    def train(self):
        # Entrenar durant max_iter iteracions o fins que la millora sigui inferior a epsilon
        i=0
        millora=epsilon+1
        costAnt=0
        while( i < self.maxIt and millora > self.epsilon):
            pred = self.predict()
            loss=pred-self.y
            self.cost=(np.sum(loss**2)+reg*np.sum(thetas**2))/(2*self.x.shape[0])
            millora = abs(self.cost-costAnt)
            self.__update(pred)
            costAnt=self.cost
        return self.arrayTheta

def Regressor(x, y, arrayTheta, max_iter, epsilon, aplha, reg):
    i = 1
    millora=epsilon+1
    costAnt=0
    while( i < max_iter and millora > epsilon):
        predict = np.dot(x, arrayTheta)
        loss = predict - y
        cost = (np.sum(loss**2)+reg*np.sum(thetas**2))/(2*x.shape[0])#np.sum(loss ** 2) / (2 * x.shape[0])
        millora = abs(costAnt-cost)
        costAnt = cost
        gradient = np.dot(x.T, loss) / x.shape[0]
        print("Iter: "+str(i)+"  Cost: "+str(cost))
        arrayTheta = arrayTheta - alpha * gradient
        i+=1
    return arrayTheta

def desnormalitzar(x, mean, std): 
    return x * std + mean

# dades que utilitzarem

data = dataset.values
dataNorm = data#standarize(data)
x_t = dataNorm[:, :1]
y = dataNorm[:, 2]
y = y[:, np.newaxis]
X_b = np.c_[np.ones((len(x_t),1)),x_t]
ymean=y.mean(0)
ystd=y.std(0)
xmean= data[:,0].mean(0)
xstd= data[:,0].std(0)



"""
X = 2 * np.random.rand(100,1)
y = 4 +3 * X+np.random.randn(100,1)
X_b = np.c_[np.ones((len(X),1)),X]
"""


thetas= np.random.randn(2,2)
print(thetas)
alpha = 1
it = 1000
epsilon= 1.1e-10
reg=0.01


#arrayTheta, alpha, x, y, maxIt, epsilon, reg
regr = RegressorO(thetas,alpha,X_b,y, it, epsilon, reg)
print(regr.train())

#print(regr.train(it,0.1))

thetas = Regressor(X_b,y,thetas,it,epsilon,alpha,reg)
print(thetas)
#y=desnormalitzar(y, ymean, ystd)

#thetas = desnormalitzar(thetas, xmean, xstd)
print(thetas)
#x_t= desnormalitzar(x_t, xmean, xstd)
#cost 3gra 1136010.1278228892
#cost rect 1136010.1278228872

plt.figure()
#ax.set_facecolor('#ffecea')
ax = plt.scatter(x_t, y, c='#ba343a', alpha=0.7, edgecolors = 'none', facecolor='#ffecea')

x_rect = np.linspace(0,1,100)
y_rect = thetas[1][1]*x_rect+thetas[1][0]
#y_rect = thetas[3]*x_rect**3 + thetas[2]*x_rect**2+ thetas[1]*x_rect + thetas[0]
plt.plot(x_rect, y_rect, '#461220')
#ValueError: x and y must have same first dimension, but have shapes (100, 1) and (2, 1)
plt.title("Cnt en funció la temp (Descens del Gradient)")
plt.xlabel("temp")
plt.ylabel("cnt")
plt.show()



from mpl_toolkits.mplot3d import axes3d, Axes3D # generem dades 3D d'exemple
x_val = np.random.random((100, 2))
y_val = np.random.random((100, 1))
#regr = regression(x_val, y_val)

predX3D = regr.predict()
# Afegim els 1's
A = np.hstack((x_val,np.ones([x_val.shape[0],1])))
w = np.linalg.lstsq(A,predX3D)[0]
#1r creem una malla acoplada a la zona de punts per tal de representar el pla
malla = (range(20) + 0 * np.ones(20)) / 10
malla_x1 =  malla * (max(x_val[:,0]) - min(x_val[:,0]))/2 + min(x_val[:,0])
malla_x2 =  malla * (max(x_val[:,1]) - min(x_val[:,1]))/2 + min(x_val[:,1])
#la fucnio meshgrid ens aparella un de malla_x1 amb un de malla_x2, per atot #element de mallax_1 i per a tot element de malla_x2.
xplot, yplot = np.meshgrid(malla_x1 ,malla_x2)
# Cal desnormalitzar les dades
def desnormalitzar(x, mean, std): return x * std + mean
#ara creem la superficies que es un pla
zplot = w[0] * xplot + w[1] * yplot + w[2]
#Dibuixem punts i superficie
plt3d = plt.figure('Coeficiente prismatico -- Relacio longitud desplacament 3D', dpi=100.0).gca(projection='3d')
plt3d.plot_surface(xplot,yplot,zplot, color='red')
plt3d.scatter(x_val[:,0],x_val[:,1],y_val)



