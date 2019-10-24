#!/usr/bin/env python
# coding: utf-8

# # Pràctica 1C: Anàlisi de les dades

# En aquesta primera peça de codi s'importen totes les llibreries necessàries, hi apliquem certes opcions i carreguem el nostre dataset en forma d'array pandas.


# -*- coding: utf-8 -*-

from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sb
#get_ipython().run_line_magic('matplotlib', 'notebook')
from sklearn.linear_model import LinearRegression

# Visualitzarem només 3 decimals per mostra
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Funcio per a llegir dades en format csv
def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    return dataset

# Carreguem dataset d'exemple
dataset = pd.read_csv('bike-sharing-dataset/hour.csv', header=0, delimiter=',')


#Omplim
dataset = dataset.fillna(dataset.mean())

print("Dimensionalitat de la BBDD:", dataset.shape)
#print("Dimensionalitat de les entrades X", x.shape)
#print("Dimensionalitat de l'atribut Y", y.shape)

print("5 Primeres mostres de la BBDD:")
dataset.head()



print("Estadistiques dels atributs numerics de la BBDD:")
dataset.describe()


# #### Aquest primer apartat té com a objectiu respondre les següents preguntes:
# 
# 1. Quin són els atributs més importants per fer una bona predicció?
# 
# 2. Amb quin atribut s'assoleix un MSE menor?
# 
# 3. Quina correlació hi ha entre els atributs de la vostra base de dades?
# 
# 4. Com influeix la normalització en la regressió?
# 
# 5. Com millora la regressió quan es filtren aquells atributs de les mostres que no contenen informació?
# 

# #### Mirem la correlació entre els atributs d'entrada per entendre millor les dades



correlacio = dataset.corr().round(2)

plt.figure(figsize=(10,8))

ax = sb.heatmap(correlacio, annot=True, linewidths=.5, )


# # ANOTACIONS DESPRES DE VEURE EL HEATMAP:
# Com era d'esperar, cada variable te un índex del 100% de correlació amb si mateixa. Però això no ens interessa, sinò el fet de si dues variables estàn molt relacionades entre si o no. 
# 
# Les variables que més relació tenen entre elles són principalment registered i cnt, doncs això és perquè la majoria de bicis compartides es fan amb usuaris registrats i pràcticament una equival a l'altre. Un altre variable que té un comportament similar amb cnt és casual, tot i que al ser menys usual que un casual comparteixi bici comparat amb un registered, el valor de correlació és menor. Llavors la millor manera de predir el cnt (el nostre objectiu) seria utilitzant l'atribut registered, però com la informació que ens dona és pràcticament la mateixa que la informació a predir, no ens és útil.
# 
# També trobem que les variables yr i instant tenen relació. Això es deu a que les instàncies estan ordenades cronològicament i per tant a majors instàncies, majors anys. Realment no és una relació útil perquè instant és un identificador únic de la mostra en qüestió. Tenim altre cop un comportament similar amb month i season, doncs és evident que cada estació està delimitada per 4 mesos en qüestió i la seva correlació ha de ser elevada. Per últim observem que temp i atemp (temperatura i sensació de temperatura) per raons evidents són variables molt relacionades.
# 
# Llavors observant això podríem concloure que les variables instant i atemp ens són prescindibles. Podem dir el mateix (per ara tot i que no descartem revertir aquest canvi posteriorment) de les variables casual i registered, doncs considerem que amb cnt ja tenim aquesta informació prou representada. Per ara no prescindirem de month o season, doncs ens podría ser útil estudiar el model amb ambdós atributs.
#     



#Executar aquest codi quan s'hagi d'executar tot, sino ens donarà error al eliminar una columna ja eliminada
del dataset["dteday"]
del dataset["instant"]
del dataset["casual"]
del dataset["registered"]
del dataset["atemp"]





#Repetim el heatmap amb els atributs eliminats
correlacio = dataset.corr().round(2)

plt.figure(figsize=(10,8))

ax = sb.heatmap(correlacio, annot=True, linewidths=.5 )


# ## ANOTACIONS DESPRÉS DE LA MODIFICACIÓ: HEATMAP
# Ara que hem prescindit del atributs que hem considerat innecessaris, podem observar quines variables tenen més relació amb l'atribut target cnt.
# Els dos atributs principals que tenen un valor de correlació molt similar i considerem més rellevants són:
# 
# * temp
# * hr
# 
# Seguit d'un valor menor però significant:
# * yr: Probablement el fet de compartir bici ha estat més popular el segon any que no pas el primer.
# 
# Noves relacions que se'ns havien escapat en el comentari previ serien les següents:
# * temp - season: Evidentment a l'estiu la temperatura serà més elevada que no pas a l'hivern
# * hum - weathersit: Tot i que no sempre es el cas, normalment un dia assolejat la humitat no és tan elevada com ho és un dia que neva o plou.
# 



data = dataset.values

print(data.shape)

x = data[:, 0:11]
y = data[:, 11]

def standarize(x_train):
    mean = x_train.mean(0)
    std = x_train.std(0)
    x_t = x_train - mean[None, :]
    x_t /= std[None, :]
    return x_t

x_t = standarize(x)



import matplotlib.ticker as ticker
for col in dataset.columns:
    plt.figure()
    ax = sb.countplot(x=col, data=dataset)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=5))
    #Per alguna raó plt.show() dona problemes a la meva terminal, al notebook funciona correctament
    #plt.show()


# ## Anàlisis del comportament dels atributs
# Com podem observar, els atributs amb una distribució semblant a la normal són:
# * temp
# * hum
# * windspeed
# 

# Creem la funció standarize, que mitjançant procediments de normalització, modifica tots els atributs del dataset, per identificar els valors que tinguin distribució normal, millors per fer regressió, i descartar els que no siguin representatius.
# Per visualitzar la millora que suposarà normalitzar les dades, calcularem el mean squared error per a la regressió aplicada directament sobre el dataset i per a la regressió després de normalitzar els atributs.
# 


x = np.array(dataset.values)
def standarize(x_train):
    mean = x_train.mean(0)
    std = x_train.std(0)
    x_t = x_train - mean[None, :]
    x_t /= std[None, :]
    return x_t

x_t = standarize(dataset.values)
#print(dataset.values)

for i in range(12):
    plt.figure()
    plt.title("Histograma de l'atribut"+ str(i))
    plt.xlabel("Attribute Value")
    plt.ylabel("Count")
    hist = plt.hist(x_t[:,i], bins=11, range=[np.min(x_t[:,i]), np.max(x_t[:,i])], histtype="bar", rwidth=0.8)
    



# ## DESPRÉS DE STANDARIZE
# Podem observar que el comportament es similar al que ja teniem, però ara els valors estàn estandaritzats.

# **Definim la funció regression, que entrena el model perquè predigui y a partir d'x


from sklearn.linear_model import LinearRegression
def regression(x, y):
    # Creem un objecte de regressió de sklearn
    regr = LinearRegression()

    # Entrenem el model per a predir y a partir de x
    regr.fit(x, y)

    # Retornem el model entrenat
    return regr


plt.figure()
ax= plt.scatter(x[:,8],y)


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
# Extraiem el primer atribut de x i canviem la mida a #exemples, #dimensions de l'atribut.
# En el vostre cas, haureu de triar un atribut com a y, i utilitzar la resta com a x.
atribut1 = x_t[:,8].reshape(x_t.shape[0], 1) 
regr = regression(atribut1, y) 
predicted = regr.predict(atribut1)

# Mostrem la predicció del model entrenat en color vermell a la Figura anterior 1
plt.figure()
ax = plt.scatter(x_t[:,0], y)
plt.plot(atribut1[:,0], predicted, 'r')

# Mostrem l'error (MSE i R2)
MSE = mean_squared_error(y, predicted)
r2 = r2_score(y, predicted)

print("Mean squeared error: ", MSE)
print("R2 score: ", r2)



from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
# Extraiem el primer atribut de x i canviem la mida a #exemples, #dimensions de l'atribut.
# En el vostre cas, haureu de triar un atribut com a y, i utilitzar la resta com a x.
atribut1 = x_t[:,0].reshape(x_t.shape[0], 1) 
regr = regression(atribut1, y) 
predicted = regr.predict(atribut1)

# Mostrem la predicció del model entrenat en color vermell a la Figura anterior 1
plt.figure()
ax = plt.scatter(x_t[:,0], y)
plt.plot(atribut1[:,0], predicted, 'r')

# Mostrem l'error (MSE i R2)
MSE = mean_squared_error(y, predicted)
r2 = r2_score(y, predicted)

print("Mean squeared error: ", MSE)
print("R2 score: ", r2)


