
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn import svm, datasets
from sklearn.linear_model import LogisticRegression
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


# Visualitzarem només 3 decimals per mostra
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# Funcio per a llegir dades en format csv
def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    return dataset

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

dataset = load_dataset("ds/mushrooms.csv")
print(dataset.columns)

target = 'class'




### REGRESSIÓ LOGÍSTICA

x = np.random.random((100, 3))
y = np.array([-1,1])[np.random.randint(0, 2, 100)]
plt.figure()
ax = plt.scatter(x[:,0], y, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
particions = [0.5, 0.7, 0.8]

for part in particions:

	x_t, y_t, x_v, y_v = split_data(x, y, part)

	#Creem el regresor logístic
	logireg = LogisticRegression(C=2.0, fit_intercept=True, penalty='l2', tol=0.001) # l'entrenem
	logireg.fit(x_t, y_t)
	print ("Correct classification Logistic ", part, "%: ", logireg.score(x_v, y_v))

	y_pred = logireg.predict(x_v)
	percent_correct_log = np.mean(y_v == y_pred).astype('float32')
	print ("Correct classification Logistic ", part, "%: ", percent_correct_log, "\n")




from sklearn.model_selection import train_test_split 
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import cross_val_predict

#Creem el regresor logístic
logireg = LogisticRegression(C=2.0, fit_intercept=True, penalty='l2', tol=0.001) 

#Creem la partició K-fold, K=5 i apliquem el regressor logístic
scores = cross_val_score(logireg, x, y, cv=5)

# mirem el resultat
score = scores.mean()
y_pred = cross_val_predict(logireg, x, y, cv=5)
print("Accuracy Kfold:",score)

#Creem la partició LOOCV, i apliquem el regressor logístic
loo = LeaveOneOut()

scores=[]
for train_index, test_index in loo.split(x):
	X_train, X_test = x[train_index], x[test_index]
	y_train, y_test = y[train_index], y[test_index]
	logireg.fit(X_train,y_train)
	scores.append(logireg.score(X_test,y_test))

# mirem el resultat
score = np.mean(scores)
y_pred = logireg.predict(x)
print("Accuracy LOOCV",score)


# SVM
def train_svm(x, y, kernel='linear', C=0.01, gamma=0.001, probability=True): 
	if(kernel =='linear'):
		svclin = svm.SVC(C=C, kernel=kernel, gamma=gamma, probability=probability) 
	if(kernel =='poly'):
		svclin = svm.SVC(C=C, kernel=kernel, gamma=gamma, probability=probability) 
	if(kernel =='rbf'):
		svclin = svm.SVC(C=C, kernel=kernel, gamma=gamma, probability=probability) 

	# l'entrenem
	return svclin.fit(x, y)



# CORVA PRECISION/RECALL

from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score,

# Create random dataset
n_classes = 5
x = np.random.randn(100,3)
y = np.random.randint(0, n_classes, 100)
x_t, y_t, x_v, y_v = split_data(x, y, 0.8)

# Train model
model = train_svm(x_t, y_t)

# Get class probabilities
probs = model.predict_proba(x_v)

# Compute Precision-Recall and plot curve
precision = {}
recall = {}
average_precision = {}
plt.figure()

for i in range(n_classes):
	precision[i], recall[i], _ = precision_recall_curve(y_v == i, probs[:, i]) 
	average_precision[i] = average_precision_score(y_v == i, probs[:, i])
	plt.plot(recall[i], precision[i], label='Precision-recall curve of class {0} (area = {1:0.2f})' ''.format(i, average_precision[i]))
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.legend(loc="upper right")


# ROC
