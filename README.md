# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the required libraries.
2. Load the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary. 6.Define a function to predict the Regression value.
   
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: SUDHAKARAN S
RegisterNumber: 212222220051
*/
```
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()


plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
np.mean(predict(res.x,X)==y)

```

## Output:

## Array Value of x
  
   ![Screenshot 2024-04-15 033700](https://github.com/23013743/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/161271714/53ea9891-469f-4302-b2ab-89f5ae570930)

## Array Value of y
   ![Screenshot 2024-04-15 033716](https://github.com/23013743/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/161271714/09a7d024-6494-441d-9483-47d4a87e2a46)

## Exam 1 - score graph

   ![Screenshot 2024-04-15 033728](https://github.com/23013743/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/161271714/f194cf9b-90f5-40ef-9e82-bf44cbae1fdf)

## Sigmoid function grapH
   ![Screenshot 2024-04-15 033738](https://github.com/23013743/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/161271714/41108f65-f4ad-45cb-868c-3f8178e6bd25)

## X_train_grad value

   ![Screenshot 2024-04-15 033745](https://github.com/23013743/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/161271714/eebcecce-8f51-4a50-956f-25f8b20f183c)
## Y_train_grad value

   ![Screenshot 2024-04-15 033753](https://github.com/23013743/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/161271714/302c3a41-65bb-441e-8191-41c2c7990be5)
## Print res.x

  ![Screenshot 2024-04-15 034005](https://github.com/23013743/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/161271714/7c6dbfca-4999-41da-99f9-cc693310faa8)
## Decision boundary - graph for exam score
  
  ![Screenshot 2024-04-15 034016](https://github.com/23013743/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/161271714/9824c702-6751-4557-a856-c704625201de)
## Proability value

  ![Screenshot 2024-04-15 034027](https://github.com/23013743/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/161271714/69062590-586b-42b6-8e2e-6857864c7dda)

## Prediction value of mean

  ![Screenshot 2024-04-15 034034](https://github.com/23013743/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/161271714/337bfe1a-0008-4817-8b99-6976c0b5dce8)
 
## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

