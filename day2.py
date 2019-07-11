import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
'''data=pd.read_csv('headbrain.csv')
a=data["Head Size(cm^3)"].values
y=data.iloc[:,c].values'''
data=pd.read_csv('Salary_Data.csv')
b=data["YearsExperience"].values
x=data.iloc[:,0:1].values
y=data.iloc[:,1].values
plt.scatter(x,y,color='red')
plt.plot(x,y,color='red')

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.2,
                                                 random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
m=regressor.coef_
c=regressor.intercept_
'''#i=int(input("Enter the year experience"))
y75=(m*8)+c
yp75=regressor.predict([[8]])
plt.scatter(x_train,y_train,color='green')
plt.plot(x_train,regressor.predict(x_train),color='blue')'''

i=input("Enter the year experience")
i=i.split(',')
#i=[float(x) for x in a]
res=[]
for a in i:
    res.append(float(a))
print(res)
res=np.array(res).reshape((len(res),1))
regressor.predict(res)

#plt.scatter(x_train,y_train,color='red')
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
sum=0
y_pred=regressor.predict(x_test)
for i in range(0,len(y_test)):
    sum+=(y_test[i] - y_pred[i])**2
resu=sum/len(y_test)
import math
math.sqrt(resu)
print(resu)

from sklearn.metrics import mean_squared_error
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
print(rmse)

