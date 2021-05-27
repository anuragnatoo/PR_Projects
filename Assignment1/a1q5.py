import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statistics import mean


dataset=pd.read_csv("iris.csv")
setosa_list_length=np.array(dataset["petal.length"][1:50])
print(setosa_list_length)
setosa_list_width=np.array(dataset["petal.width"][1:50])
print(setosa_list_width)
versicolor_list_length=np.array(dataset["petal.length"][51:100])
print(versicolor_list_length)
versicolor_list_width=np.array(dataset["petal.width"][51:100])
print(versicolor_list_width)
virginica_list_length=np.array(dataset["petal.length"][101:150])
print(virginica_list_length)
virginica_list_width=np.array(dataset["petal.width"][101:150])
print(virginica_list_width)


plt.plot(setosa_list_length,setosa_list_width,'r+')
plt.plot(versicolor_list_length,versicolor_list_width,'bo')
plt.plot(virginica_list_length,virginica_list_width,'gx')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')

setosa_list=[]
versicolor_list=[]
virginica_list=[]
setosa_list.append(mean(setosa_list_length))
setosa_list.append(mean(setosa_list_width))
setosa_mean_l=mean(setosa_list_length)
setosa_mean_w=mean(setosa_list_width)
versicolor_list.append(mean(versicolor_list_length))
versicolor_list.append(mean(versicolor_list_width))
versicolor_mean_l=mean(versicolor_list_length)
versicolor_mean_w=mean(versicolor_list_width)
virginica_list.append(mean(virginica_list_length))
virginica_list.append(mean(virginica_list_width))
virginica_mean_l=mean(virginica_list_length)
virginica_mean_w=mean(virginica_list_width)

setosa_test_l=dataset['petal.length'].values[0]
setosa_test_w=dataset['petal.width'].values[0]
versicolor_test_l=dataset['petal.length'].values[50]
versicolor_test_w=dataset['petal.width'].values[50]
virginica_test_l=dataset['petal.length'].values[100]
virginica_test_w=dataset['petal.width'].values[100]

print(setosa_list)
print(versicolor_list)
print(virginica_list)
print(setosa_test_l,setosa_test_w)
print(versicolor_test_l,versicolor_test_w)
print(virginica_test_l,virginica_test_w)


setosal_mat_calc=setosa_list_length-setosa_mean_l
setosab_mat_calc=setosa_list_width-setosa_mean_w
print(setosal_mat_calc)
print(setosab_mat_calc)

versicolorl_mat_calc=versicolor_list_length-versicolor_mean_l
versicolorb_mat_calc=versicolor_list_width-versicolor_mean_w
print(versicolorl_mat_calc)
print(versicolorb_mat_calc)

virginical_mat_calc=virginica_list_length-virginica_mean_l
virginicab_mat_calc=virginica_list_width-virginica_mean_w
print(virginical_mat_calc)
print(virginicab_mat_calc)

setosa_matrix_transpose=np.matrix([setosal_mat_calc,setosab_mat_calc])
setosa_matrix=np.transpose(setosa_matrix_transpose)

versicolor_matrix_transpose=np.matrix([versicolorl_mat_calc,versicolorb_mat_calc])
versicolor_matrix=np.transpose(versicolor_matrix_transpose)

virginica_matrix_transpose=np.matrix([virginical_mat_calc,virginicab_mat_calc])
virginica_matrix=np.transpose(virginica_matrix_transpose)

print(setosa_matrix)
print(setosa_matrix.shape)
print(versicolor_matrix)
print(versicolor_matrix.shape)
print(virginica_matrix)
print(virginica_matrix.shape)

setosa_covariance=(1/48)*(setosa_matrix_transpose.dot(setosa_matrix))
versicolor_covariance=(1/48)*(versicolor_matrix_transpose*versicolor_matrix)
virginica_covariance=(1/48)*(virginica_matrix_transpose*virginica_matrix)
setosa_covariance=np.linalg.inv(setosa_covariance)
versicolor_covariance=np.linalg.inv(versicolor_covariance)
virginica_covariance=np.linalg.inv(virginica_covariance)
print(setosa_covariance)
print(versicolor_covariance)
print(virginica_covariance)

#Distance Calculation
test11l=setosa_test_l-setosa_mean_l
test11b=setosa_test_w-setosa_mean_w
test12l=setosa_test_l-versicolor_mean_l
test12b=setosa_test_w-versicolor_mean_w
test13l=setosa_test_l-virginica_mean_l
test13b=setosa_test_w-virginica_mean_w

test21l=versicolor_test_l-setosa_mean_l
test21b=versicolor_test_w-setosa_mean_w
test22l=versicolor_test_l-versicolor_mean_l
test22b=versicolor_test_w-versicolor_mean_w
test23l=versicolor_test_l-virginica_mean_l
test23b=versicolor_test_w-virginica_mean_w

test31l=virginica_test_l-setosa_mean_l
test31b=virginica_test_w-setosa_mean_w
test32l=virginica_test_l-versicolor_mean_l
test32b=virginica_test_w-versicolor_mean_w
test33l=virginica_test_l-virginica_mean_l
test33b=virginica_test_w-virginica_mean_w

testmat11_trans=np.matrix([test11l,test11b])
testmat12_trans=np.matrix([test12l,test12b])
testmat13_trans=np.matrix([test13l,test13b])

testmat21_trans=np.matrix([test21l,test21b])
testmat22_trans=np.matrix([test22l,test22b])
testmat23_trans=np.matrix([test23l,test23b])

testmat31_trans=np.matrix([test31l,test31b])
testmat32_trans=np.matrix([test32l,test32b])
testmat33_trans=np.matrix([test33l,test33b])
print(testmat11_trans)
print(testmat12_trans)
print(testmat13_trans)

print(testmat21_trans)
print(testmat22_trans)
print(testmat23_trans)

print(testmat31_trans)
print(testmat32_trans)
print(testmat33_trans)


testmat11=np.transpose(testmat11_trans)
testmat12=np.transpose(testmat12_trans)
testmat13=np.transpose(testmat13_trans)

testmat21=np.transpose(testmat21_trans)
testmat22=np.transpose(testmat22_trans)
testmat23=np.transpose(testmat23_trans)

testmat31=np.transpose(testmat31_trans)
testmat32=np.transpose(testmat32_trans)
testmat33=np.transpose(testmat33_trans)

print(testmat11)
print(testmat12)
print(testmat13)

print(testmat21)
print(testmat22)
print(testmat23)

print(testmat31)
print(testmat32)
print(testmat33)

distance1from1=testmat11_trans.dot(setosa_covariance.dot(testmat11))
distance1from2=testmat12_trans*(versicolor_covariance*testmat12)
distance1from3=testmat13_trans*(virginica_covariance*testmat13)
distance1from1=np.sqrt(distance1from1)
distance1from2=np.sqrt(distance1from2)
distance1from3=np.sqrt(distance1from3)
testvalues1=[]
testvalues1.append(distance1from1[0,0])
testvalues1.append(distance1from2[0,0])
testvalues1.append(distance1from3[0,0])

print(testvalues1)

print(distance1from1)
print(distance1from2)
print(distance1from3)
if testvalues1.index(min(testvalues1)) == 0:
    print("Flower 0 belongs to class Senotsa")
elif testvalues1.index(min(testvalues1)) ==1:
    print("Flower 0 belongs to class Versicolor")
elif testvalues1.index(min(testvalues1)) ==2:
    print("Flower 0 belongs to class Virginica")



distance2from1=testmat21_trans*(setosa_covariance*testmat21)
distance2from2=testmat22_trans*(versicolor_covariance*testmat22)
distance2from3=testmat23_trans*(virginica_covariance*testmat23)
distance2from1=np.sqrt(distance2from1)
distance2from2=np.sqrt(distance2from2)
distance2from3=np.sqrt(distance2from3)

testvalues2=[]
testvalues2.append(distance2from1[0,0])
testvalues2.append(distance2from2[0,0])
testvalues2.append(distance2from3[0,0])
print(testvalues2)

print(distance2from1)
print(distance2from2)
print(distance2from3)
if testvalues2.index(min(testvalues2)) == 0:
    print("Flower 50 belongs to class Senotsa")
elif testvalues2.index(min(testvalues2)) ==1:
    print("Flower 50 belongs to class Versicolor")
elif testvalues2.index(min(testvalues2)) ==2:
    print("Flower 50 belongs to class Virginica")

distance3from1=testmat31_trans*(setosa_covariance*testmat31)
distance3from2=testmat32_trans*(versicolor_covariance*testmat32)
distance3from3=testmat33_trans*(virginica_covariance*testmat32)
distance3from1=np.sqrt(distance3from1)
distance3from2=np.sqrt(distance3from2)
distance3from3=np.sqrt(distance3from3)


testvalues3=[]
testvalues3.append(distance3from1[0,0])
testvalues3.append(distance3from2[0,0])
testvalues3.append(distance3from3[0,0])
print(testvalues3)

print(distance3from1)
print(distance3from2)
print(distance3from3)
if testvalues3.index(min(testvalues3)) == 0:
    print("Flower 100 belongs to class Senotsa")
elif testvalues3.index(min(testvalues3)) ==1:
    print("Flower 100 belongs to class Versicolor")
elif testvalues3.index(min(testvalues3)) ==2:
    print("Flower 100 belongs to class Virginica")