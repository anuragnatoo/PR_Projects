import pandas as pd 
import numpy as np
import math
from sympy import symbols, Eq, solve
import matplotlib.pyplot as plt
from sympy.plotting import plot_implicit
from sympy import simplify
data1= pd.read_csv("iris.csv", usecols = ['petalwidth'])
data2= pd.read_csv("iris.csv", usecols = ['petallength'])

dataarr1=np.array(data1)
dataarr2=np.array(data2)

#print(len(dataarr1))

set1x=[]
set1y=[]
set2x=[]
set2y=[]
set3x=[]
set3y=[]

for i in range(0,50):
	set1x.append(float(dataarr1[i][0]))
	set1y.append(float(dataarr2[i][0]))
	

for i in range(50,100):
	set2x.append(float(dataarr1[i][0]))
	set2y.append(float(dataarr2[i][0]))
	

for i in range(100,150):
	set3x.append(float(dataarr1[i][0]))
	set3y.append(float(dataarr2[i][0]))
	

setc1=np.vstack((set1x,set1y)).T
setc2=np.vstack((set2x,set2y)).T
setc3=np.vstack((set3x,set3y)).T

set1=np.array(setc1)
set2=np.array(setc2)
set3=np.array(setc3)

na=len(set1)
nb=len(set2)
nc=len(set3)

print(set1)
print(set2)
print(set3)

x1=np.array(set1[ : , 0]) #for all x-values of set1
x2=np.array(set2[ : , 0]) #for all x-values of set2
x3=np.array(set3[ : , 0]) #for all x-values of set3


y1=np.array(set1[ : , 1]) #for all y-values of set1
y2=np.array(set2[ : , 1]) #for all y-values of set2
y3=np.array(set3[ : , 1]) #for all y-values of set3


mu1=np.array([np.sum(x1)/na,np.sum(y1)/na]) #mean matrix of set1
mu2=np.array([np.sum(x2)/nb,np.sum(y2)/nb]) #mean matrix of set2
mu3=np.array([np.sum(x3)/nc,np.sum(y3)/nc]) #mean matrix of set3


print(f'MEAN OF CLASS A\n{mu1}\n')
print(f'MEAN OF CLASS B\n{mu2}\n')
print(f'MEAN OF CLASS C\n{mu3}\n')

cov1=np.cov(np.transpose(set1))
cov2=np.cov(np.transpose(set2))
cov3=np.cov(np.transpose(set3))

print(cov1)

covinv1 = np.linalg.inv(cov1)
covinv2 = np.linalg.inv(cov2)
covinv3 = np.linalg.inv(cov3)

A1= -(1/2)*(covinv1)
B1= np.dot(mu1,covinv1)
C1=	-(1/2)*(np.dot(np.dot(mu1.transpose(),covinv1),mu1))-(1/2)*math.log(np.linalg.det(cov1))+math.log(1/3)

A2= -(1/2)*(covinv2)
B2= np.dot(mu2,covinv2)
C2=	-(1/2)*(np.dot(np.dot(mu2.transpose(),covinv2),mu2))-(1/2)*math.log(np.linalg.det(cov2))+math.log(1/3)

A3= -(1/2)*(covinv3)
B3= np.dot(mu2,covinv3)
C3=	-(1/2)*(np.dot(np.dot(mu3.transpose(),covinv3),mu3))-(1/2)*math.log(np.linalg.det(cov3))+math.log(1/3)

A111=A1[0][0]-A2[0][0]
A121=A1[1][1]-A2[1][1]
A131=A1[0][1]+A1[1][0]-A2[0][1]-A2[1][0]
B121=B1[1]-B2[1]
B111=B1[0]-B2[0]
C11=C1-C2

A112=A2[0][0]-A3[0][0]
A122=A2[1][1]-A3[1][1]
A132=A2[0][1]+A2[1][0]-A3[0][1]-A3[1][0]
B122=B2[1]-B3[1]
B112=B2[0]-B3[0]
C12=C2-C3

A113=A1[0][0]-A3[0][0]
A123=A1[1][1]-A3[1][1]
A133=A1[0][1]+A1[1][0]-A3[0][1]-A3[1][0]
B123=B1[1]-B3[1]
B113=B1[0]-B3[0]
C13=C1-C3
p=None
x,y=symbols("x y")
g=A111*x*x+A121*y*y+A131*x*y+B111*x+B121*y+C11
lk=plot_implicit(g,(x,-10,10),(y,-10,10),show=False)
if p:
	p.extend(lk)
else:
	p=lk
h=A112*x*x+A122*y*y+A132*x*y+B112*x+B122*y+C12
lk1=plot_implicit(h,(x,-10,10),(y,-10,10),show=False)
if p:
	p.extend(lk1)
else:
	p=lk1
f=A113*x*x+A123*y*y+A133*x*y+B113*x+B123*y+C13
lk2=plot_implicit(f,(x,-10,10),(y,-10,10),show=False)
if p:
	p.extend(lk2)
else:
	p=lk2
p.show()
a,b=set1.T
plt.scatter(a,b)
c,d=set2.T
plt.scatter(c,d)
i,j=set3.T
plt.scatter(i,j)
plt.show()

