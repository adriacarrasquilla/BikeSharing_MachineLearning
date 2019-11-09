import numpy as np
import pandas as pd

def hypothesis(theta, X, n):
    h = np.ones((X.shape[0],1))
    theta = theta.reshape(1,n+1)
    for i in range(0,X.shape[0]):
        h[i] = float(np.matmul(theta, X[i]))
    h = h.reshape(X.shape[0])
    return h

def BGD(theta, alpha, num_iters, h, X, y, n):
    cost = np.ones(num_iters)
    for i in range(0,num_iters):
        theta[0] = theta[0] - (alpha/X.shape[0]) * sum(h - y)
        for j in range(1,n+1):
            theta[j] = theta[j] - (alpha/X.shape[0]) * sum((h-y) * X.transpose()[j])
        h = hypothesis(theta, X, n)
        cost[i] = (1/X.shape[0]) * 0.5 * sum(np.square(h - y))
    theta = theta.reshape(1,n+1)
    return theta, cost

def linear_regression(X, y, alpha, num_iters):
    n = X.shape[1]
    one_column = np.ones((X.shape[0],1))
    X = np.concatenate((one_column, X), axis = 1)
    # initializing the parameter vector...
    theta = np.zeros(n+1)
    # hypothesis calculation....
    h = hypothesis(theta, X, n)
    # returning the optimized parameters by Gradient Descent...
    theta, cost = BGD(theta,alpha,num_iters,h,X,y,n)
    return theta, cost


dataset = pd.read_csv('bike-sharing-dataset/hour.csv', header=0, delimiter=',')
dataset = dataset.fillna(dataset.mean())

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

data = dataset.values

X_train = data[:,[0,1]] #feature set
y_train = data[:,2] #label set

mean = np.ones(X_train.shape[1])
std = np.ones(X_train.shape[1])
for i in range(0, X_train.shape[1]):
    mean[i] = np.mean(X_train.transpose()[i])
    std[i] = np.std(X_train.transpose()[i])
    for j in range(0, X_train.shape[0]):
        X_train[j][i] = (X_train[j][i] - mean[i])/std[i]

theta, cost = linear_regression(X_train, y_train, 0.001, 100)

print(theta)

import matplotlib.pyplot as plt
cost = list(cost)
n_iterations = [x for x in range(1,101)]
plt.plot(n_iterations, cost)
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
plt.show()


from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
sequence_containing_x_vals = list(X_train.transpose()[0])
sequence_containing_y_vals = list(X_train.transpose()[1])
sequence_containing_z_vals = list(y_train)
fig = pyplot.figure()
ax = Axes3D(fig)
ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals)
ax.set_xlabel('tmp', fontsize=10)
ax.set_ylabel('hr', fontsize=10)
ax.set_zlabel('cnt', fontsize=10)
plt.show()


# Getting the predictions...
X_train = np.concatenate((np.ones((X_train.shape[0],1)), X_train) ,axis = 1)
predictions = hypothesis(theta, X_train, X_train.shape[1] - 1)
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
sequence_containing_x_vals = list(X_train.transpose()[1])
sequence_containing_y_vals = list(X_train.transpose()[2])
sequence_containing_z_vals = list(predictions)
fig = pyplot.figure()
ax = Axes3D(fig)
ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals)
ax.set_xlabel('tmp', fontsize=10)
ax.set_ylabel('hr', fontsize=10)
ax.set_zlabel('cnt', fontsize=10)
plt.show()



from mpl_toolkits.mplot3d import axes3d, Axes3D # generem dades 3D d'exemple
#regr = regression(x_val, y_val)

predX3D= np.array(predictions)
# Afegim els 1's
A = np.hstack((X_train,np.ones([X_train.shape[0],1])))
w = np.linalg.lstsq(A,predX3D)[0]
#1r creem una malla acoplada a la zona de punts per tal de representar el pla
malla = (range(20) + 0 * np.ones(20)) / 10
malla_x1 =  malla * (max(X_train[:,0]) - min(X_train[:,0]))/2 + min(X_train[:,0])
malla_x2 =  malla * (max(X_train[:,1]) - min(X_train[:,1]))/2 + min(X_train[:,1])
#la fucnio meshgrid ens aparella un de malla_x1 amb un de malla_x2, per atot #element de mallax_1 i per a tot element de malla_x2.
xplot, yplot = np.meshgrid(malla_x1 ,malla_x2)
# Cal desnormalitzar les dades
def desnormalitzar(x, mean, std): return x * std + mean
#ara creem la superficies que es un pla
zplot = w[0] * xplot + w[1] * yplot + w[2]
#Dibuixem punts i superficie
plt3d = plt.figure('Coeficiente prismatico -- Relacio longitud desplacament 3D', dpi=100.0).gca(projection='3d')
plt3d.plot_surface(xplot,yplot,zplot, color='red')
plt3d.scatter(X_train[:,0],X_train[:,1],y_train)
plt.show()