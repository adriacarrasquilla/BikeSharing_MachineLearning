```python

for parametre in testParametre:
    regr = RegressorO(thetas,alpha,X_b,y, it, epsilon, reg) #substituim un dels parametres per parametre
    thetas = regr.train()
    costosEstudi.append(regr.costos[-1])
    iteracionsPerEstudi.append(regr.it)


#Cost en funció del paràmetre
plt.plot(testParametre,costosEstudi, '#461220' )
plt.title("Cost en funció del parametre")
plt.xlabel("parametre")
plt.ylabel("cost (x100)")
#plt.xscale('log') #opcional
plt.show()

#Iteracions en funció del paràmetre
plt.plot(regTest,iteracionsPerEstudi, '#461220' )
plt.title("Iteracions en funció del parametre")
plt.xlabel("parametre")
plt.ylabel("iteracions")
#plt.xscale('log') #opcional
plt.show()
```
