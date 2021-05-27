import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statistics import mean
from sympy import *
dataset=pd.read_csv("iris.csv")
'''
Printing all feature vectors in this cell.
'''
#Setosa---
train_set_dim_1=np.array(dataset["sepal.length"][0:40])
#print(train_set_dim_1)
train_set_dim_2=np.array(dataset["sepal.width"][0:40])
#print(train_set_dim_2)
train_set_dim_3=np.array(dataset["petal.length"][0:40])
#print(train_set_dim_3)
train_set_dim_4=np.array(dataset["petal.width"][0:40])
#print(train_set_dim_4)

#Versicolor---
train_ver_dim_1=np.array(dataset["sepal.length"][50:90])
#print(train_ver_dim_1)
train_ver_dim_2=np.array(dataset["sepal.width"][50:90])
#print(train_ver_dim_2)
train_ver_dim_3=np.array(dataset["petal.length"][50:90])
#print(train_ver_dim_3)
train_ver_dim_4=np.array(dataset["petal.width"][50:90])
#print(train_ver_dim_4)

#Virginica---
train_vir_dim_1=np.array(dataset["sepal.length"][100:140])
#print(train_vir_dim_1)
train_vir_dim_2=np.array(dataset["sepal.width"][100:140])
#print(train_vir_dim_2)
train_vir_dim_3=np.array(dataset["petal.length"][100:140])
#print(train_vir_dim_3)
train_vir_dim_4=np.array(dataset["petal.width"][100:140])
#print(train_vir_dim_4)

'''
Printing all the test vectors in this cell
'''
#Setosa---
test_set_dim_1=np.array(dataset["sepal.length"][40:50])
#print(test_set_dim_1)
test_set_dim_2=np.array(dataset["sepal.width"][40:50])
#print(test_set_dim_2)
test_set_dim_3=np.array(dataset["petal.length"][40:50])
#print(test_set_dim_3)
test_set_dim_4=np.array(dataset["petal.width"][40:50])
#print(test_set_dim_4)

#Versicolor---
test_ver_dim_1=np.array(dataset["sepal.length"][90:100])
#print(test_ver_dim_1)
test_ver_dim_2=np.array(dataset["sepal.width"][90:100])
#print(test_ver_dim_2)
test_ver_dim_3=np.array(dataset["petal.length"][90:100])
#print(test_ver_dim_3)
test_ver_dim_4=np.array(dataset["petal.width"][90:100])
#print(test_ver_dim_4)

#Virginica---
test_vir_dim_1=np.array(dataset["sepal.length"][140:150])
#print(test_vir_dim_1)
test_vir_dim_2=np.array(dataset["sepal.width"][140:150])
#print(test_vir_dim_2)
test_vir_dim_3=np.array(dataset["petal.length"][140:150])
#print(test_vir_dim_3)
test_vir_dim_4=np.array(dataset["petal.width"][140:150])
print(test_vir_dim_4)

# All Test Cases in one Array
final_test_case_dim1=np.concatenate([test_set_dim_1,test_ver_dim_1,test_vir_dim_1])
#print(np.shape(final_test_case_dim1))
final_test_case_dim2=np.concatenate([test_set_dim_2,test_ver_dim_2,test_vir_dim_2])
final_test_case_dim3=np.concatenate([test_set_dim_3,test_ver_dim_3,test_vir_dim_3])
final_test_case_dim4=np.concatenate([test_set_dim_4,test_ver_dim_4,test_vir_dim_4])

#Covariance Matrix Calculation
#Setosa---
set_train_matrix=np.matrix([train_set_dim_1,train_set_dim_2,train_set_dim_3,train_set_dim_4])
#print(set_train_matrix)
train_set_matrix=np.transpose(set_train_matrix)
#print(train_set_matrix)

#Versicolor--
ver_train_matrix=np.matrix([train_ver_dim_1,train_ver_dim_2,train_ver_dim_3,train_ver_dim_4])
#print(ver_train_matrix)
train_ver_matrix=np.transpose(ver_train_matrix)
#print(train_ver_matrix)

#Virginica---
vir_train_matrix=np.matrix([train_vir_dim_1,train_vir_dim_2,train_vir_dim_3,train_vir_dim_4])
#print(vir_train_matrix)
train_vir_matrix=np.transpose(vir_train_matrix)
#print(train_vir_matrix)

set_cov_matrix=np.cov(set_train_matrix)
#print(set_cov_matrix)
#print(np.shape(set_cov_matrix))
#print(np.shape(set_train_matrix))

ver_cov_matrix=np.cov(ver_train_matrix)
#print(ver_cov_matrix)
#print(np.shape(ver_cov_matrix))

vir_cov_matrix=np.cov(vir_train_matrix)
print(vir_cov_matrix)
print(np.shape(vir_cov_matrix))

'''
Inverse of Covariance Matrices
'''
set_covinv_matrix=np.linalg.inv(set_cov_matrix)
set_coeff_a=(-0.5)*set_covinv_matrix
#print(set_covinv_matrix)
#print(set_coeff_a)
#print(np.shape(set_coeff_a))

ver_covinv_matrix=np.linalg.inv(ver_cov_matrix)
ver_coeff_a=(-0.5)*ver_covinv_matrix
#print(ver_covinv_matrix)
#print(ver_coeff_a)

vir_covinv_matrix=np.linalg.inv(vir_cov_matrix)
vir_coeff_a=(-0.5)*vir_covinv_matrix
#print(np.shape(vir_coeff_a))
#print(vir_coeff_a)
#print(vir_covinv_matrix)



set_mean_dim_1=mean(train_set_dim_1)
print(set_mean_dim_1)
set_mean_dim_2=mean(train_set_dim_2)
print(set_mean_dim_2)
set_mean_dim_3=mean(train_set_dim_3)
print(set_mean_dim_3)
set_mean_dim_4=mean(train_set_dim_4)
print(set_mean_dim_4)

ver_mean_dim_1=mean(train_ver_dim_1)
print(ver_mean_dim_1)
ver_mean_dim_2=mean(train_ver_dim_2)
print(ver_mean_dim_2)
ver_mean_dim_3=mean(train_ver_dim_3)
print(ver_mean_dim_3)
ver_mean_dim_4=mean(train_ver_dim_4)
print(ver_mean_dim_4)

vir_mean_dim_1=mean(train_vir_dim_1)
print(vir_mean_dim_1)
vir_mean_dim_2=mean(train_vir_dim_2)
print(vir_mean_dim_2)
vir_mean_dim_3=mean(train_vir_dim_3)
print(vir_mean_dim_3)
vir_mean_dim_4=mean(train_vir_dim_4)
print(vir_mean_dim_4)

set_mean_matrix=np.matrix([set_mean_dim_1,set_mean_dim_2,set_mean_dim_3,set_mean_dim_4])
print(np.shape(set_mean_matrix))
ver_mean_matrix=np.matrix([ver_mean_dim_1,ver_mean_dim_2,ver_mean_dim_3,ver_mean_dim_4])
print(np.shape(ver_mean_matrix))
vir_mean_matrix=np.matrix([vir_mean_dim_1,vir_mean_dim_2,vir_mean_dim_3,vir_mean_dim_4])
print(np.shape(vir_mean_matrix))

set_coeff_b=set_mean_matrix*set_covinv_matrix
print(set_coeff_b)
print(np.shape(set_coeff_b))

ver_coeff_b=ver_mean_matrix*ver_covinv_matrix
print(ver_coeff_b)
print(np.shape(ver_coeff_b))

vir_coeff_b=vir_mean_matrix*vir_covinv_matrix
print(vir_coeff_b)
print(np.shape(vir_coeff_b))

det_set_1=np.linalg.det(set_covinv_matrix)
det_ver_1=np.linalg.det(ver_covinv_matrix)
det_vir_1=np.linalg.det(vir_covinv_matrix)
det_set_2=-0.5*np.log(abs(det_set_1))
print(det_set_2)
det_ver_2=-0.5*np.log(abs(det_ver_1))
print(det_ver_2)
det_vir_2=-0.5*np.log(abs(det_vir_1))
print(det_vir_2)

#Coefficient C
set_coeff_c=-0.5*set_mean_matrix*((set_covinv_matrix*np.transpose(set_mean_matrix)))+det_set_2
print(set_coeff_c)
print(np.shape(set_mean_matrix))

ver_coeff_c=-0.5*ver_mean_matrix*((ver_covinv_matrix*np.transpose(ver_mean_matrix)))+det_ver_2
print(ver_coeff_c)
print(np.shape(ver_mean_matrix))

vir_coeff_c=-0.5*vir_mean_matrix*((vir_covinv_matrix*np.transpose(vir_mean_matrix)))+det_vir_2
print(vir_coeff_c)
print(np.shape(vir_mean_matrix))

x1,x2,x3,x4=symbols('x y z u')
matrix_var=Matrix([[x1],[x2],[x3],[x4]])
print(np.shape(matrix_var))
g_set_val=np.dot(np.dot(matrix_var.T,set_coeff_a),matrix_var)[0][0]+np.dot(set_coeff_b,matrix_var)[0][0]+set_coeff_c
print(g_set_val)
g_ver_val=np.dot(np.dot(matrix_var.T,ver_coeff_a),matrix_var)[0][0]+np.dot(ver_coeff_b,matrix_var)[0][0]+ver_coeff_c
print(g_ver_val)
g_vir_val=np.dot(np.dot(matrix_var.T,vir_coeff_a),matrix_var)[0][0]+np.dot(vir_coeff_b,matrix_var)[0][0]+vir_coeff_c
print(g_vir_val)

#Calculation--
final_test_vector_t=np.array([final_test_case_dim1,final_test_case_dim2,final_test_case_dim3,final_test_case_dim4])
final_test_vector=np.transpose(final_test_vector_t)
print(np.shape(final_test_vector))
predicted_class=[]
actual_class=[]
for i in range(30):
    if i<10:
        actual_class.append(1)
    elif i>=10 and i<20:
        actual_class.append(2)
    else:
        actual_class.append(3)
print(actual_class)

for i in range(30):
    a1=final_test_vector[i][0]
    b1=final_test_vector[i][1]
    c1=final_test_vector[i][2]
    d1=final_test_vector[i][3]
    mat=np.array([a1,b1,c1,d1])
    g_set_val=np.dot(np.dot(mat.T,set_coeff_a),mat)+np.dot(set_coeff_b,mat)+set_coeff_c
    #print(g_set_val)
    g_ver_val=np.dot(np.dot(mat.T,ver_coeff_a),mat)+np.dot(ver_coeff_b,mat)+ver_coeff_c
    #print(g_ver_val)
    g_vir_val=np.dot(np.dot(mat.T,vir_coeff_a),mat)+np.dot(vir_coeff_b,mat)+vir_coeff_c
    #print(g_vir_val)
    if g_set_val>g_ver_val and g_set_val>g_vir_val:
        predicted_class.append(1)
        print('Flower',i,"belongs to Setosa")
    elif g_ver_val>g_set_val and g_ver_val>g_vir_val:
        predicted_class.append(2)
        print('Flower',i,"belongs to Versicolor")
    elif g_vir_val>g_set_val and g_vir_val>g_set_val:
        predicted_class.append(3)
        print('Flower',i,"belongs to Virginica")
        
print(predicted_class)
accuracy=0
for i in range(30):
    if actual_class[i]==predicted_class[i]:
        accuracy=accuracy+1
accuracy_percentage=(accuracy/30)*100
print("Percentage Accuracy of Classifier is",accuracy_percentage,"%")