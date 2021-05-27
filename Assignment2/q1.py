import numpy as np
import math
from sympy import symbols, Eq, solve
import matplotlib.pyplot as plt

a=np.array([[1,6],[3,4],[3,8],[5,6]])
b=np.array([[3,0],[1,-2],[3,-4],[5,-2]])

na=len(a)
nb=len(b)

pa=0.8
pb=0.2

x = symbols('x')
y = symbols('y')
X=np.array([[x],[y]])

print(f'CLASS A\n{a}')
print(f'CLASS B\n{b}\n')

x1=np.array(a[ : , 0]) #for all x-values of a
x2=np.array(b[ : , 0]) #for all x-values of b

#print (x1)
#print (x2)

y1=np.array(a[ : , 1]) #for all y-values of a
y2=np.array(b[ : , 1]) #for all y-values of b

#print (y1)
#print (y2)

ua=np.array([np.sum(x1)/na,np.sum(y1)/na]) #mean matrix of a
ub=np.array([np.sum(x2)/nb,np.sum(y2)/nb]) #mean matrix of b


print(f'MEAN OF CLASS A\n{ua}\n')
print(f'MEAN OF CLASS B\n{ub}\n')

at=a.transpose()
bt=b.transpose()

cova=np.cov(at)
covb=np.cov(bt)

print(f'COV OF CLASS A\n{cova}\n')
print(f'COV OF CLASS B\n{covb}\n')


covinva = np.linalg.inv(cova)
covinvb = np.linalg.inv(covb)
print(f'INVCOV OF CLASS A\n{covinva}\n')
print(f'INVCOV OF CLASS B\n{covinvb}\n')

A1= -(1/2)*(covinva)
B1= np.dot(ua,covinva)
C1=-(1/2)*(np.dot(np.dot(ua.transpose(),covinva),ua))-(1/2)*math.log(np.linalg.det(cova))+math.log(pa)

A2= -(1/2)*(covinvb)
B2= np.dot(ub,covinvb)
C2=-(1/2)*(np.dot(np.dot(ub.transpose(),covinvb),ub))-(1/2)*math.log(np.linalg.det(covb))+math.log(pb)

print(f'A1\n{A1}\n')
print(f'B1\n{B1}\n')
print(f'C1\n{C1}\n')

print(f'A2\n{A2}\n')
print(f'B2\n{B2}\n')
print(f'C2\n{C2}\n')


l=A1[0][0]-A2[0][0]
m=A1[1][1]-A2[1][1]
n=A1[0][1]+A1[1][0]-A2[0][1]-A2[1][0]
p=B1[1]-B2[1]
q=B1[0]-B2[0]
r=C1-C2
print(f'EQUATION FOR DECISON BOUNDARY : {l}x^2+{m}y^2+{n}xy+{p}y+{q}x+{r}=0')

e1=np.linspace(-10,10,1000)
e2=(-1/p)*(l*e1**2+q*e1+r)
fig, ax = plt.subplots()
ax.plot(e1, e2)
plt.title('Q-1 PA=0.8 PB=0.2')
g,h=a.T
plt.scatter(g,h)
i,j=b.T
plt.scatter(i,j)
plt.show()


	

	
