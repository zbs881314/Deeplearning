# Linear regression example: linear prediction
# for Lecture 7, Exercise 7.2
import numpy as np
import matplotlib.pyplot as plt

## generated training data set with model x(i)=a x(i-1)+ e(i)
N=5000  # total number of training data
a=0.99
ei=np.random.randn(N)*np.sqrt(0.02)  # generate e(n)
xi=np.zeros((N),dtype=np.float)
for i in range(N):
    if i==0:
        xi[i]=ei[i]
    else:
        xi[i]=a*xi[i-1]+ei[i]

## LMS algorithm to estimate a as w
w=0.0
eta=0.001
err=np.zeros((N),dtype=np.float)  # save E(n) to draw learning curve
for i in range(N):
    if i==0: continue
    err[i]=xi[i]-w*xi[i-1]
    w=w+eta*err[i]*xi[i-1]

## output results and draw learning curve
print("Ture value a = %f. Learned value w= %f." %(a,w))
plt.plot(np.arange(N), np.square(err),'b-'), plt.grid(True),
plt.xlabel('Iteration n'), plt.ylabel('MSE'), plt.show()