import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


base='C:\\Users\\mitts\\Desktop\\sem5\\ML\\Practical\\3'

df_train=pd.read_csv(os.path.join(base,'Train.csv'))
df_test=pd.read_csv(os.path.join(base,'Test.csv'))


x_train=np.ones((len(df_train),2))
x_train[:,1]=df_train.iloc[:,0]
y_train=df_train.iloc[:,1]

x_test=df_test.iloc[:,0]
y_test=df_test.iloc[:,1]

theta=np.zeros((2,))

epochs=10
lr=4e-4
m=len(df_train)
m_test=len(df_train)

y=df_train.iloc[:,1]

los=[]
los_test=[]

for i in range(epochs):
    y_pred=(np.dot(x_train,theta))
    y_pred_test=theta[0]*x_test+theta[1]
    loss=((y-y_pred)**2)/(2*m)
    loss_test=((y_test-y_pred_test)**2)/(2*m_test)
    los.append(np.sum(loss))
    los_test.append(np.sum(loss_test))

    n=x_train.shape[1]
    temp = np.zeros(theta.shape)

    for j in range(n):
        temp[j]=theta[j]-(lr/m)*np.sum((np.dot(y_pred-y,x_train)))
    theta[:]=temp[:]


y_pred=(np.dot(x_train,theta))
y_pred_test=theta[0]*x_test+theta[1]

plt.title('Training')
plt.scatter(df_train.iloc[:,0],y)
plt.plot(df_train.iloc[:,0],y_pred,color='r')
plt.show()
plt.title('Training')
plt.plot(range(epochs),los)
plt.show()

plt.title('Testing')
plt.scatter(df_test.iloc[:,0],y_test)
plt.plot(df_test.iloc[:,0],y_pred_test,color='r')
plt.show()
plt.title('Testing')
plt.plot(range(epochs),los_test)
plt.show()

