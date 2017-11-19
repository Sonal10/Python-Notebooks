#Normalizing data

import pandas as pd
import numpy as np
import math

#Read from file the data, put it into a dataframe in pandas
full = pd.read_csv('iris.csv',skipinitialspace=True,sep=",")

#Initialize class labels dictionary
class_labels = {'Iris-setosa': '1', 'Iris-versicolor': '2', 'Iris-virginica' : '3'}

Class_Lab=[]
#Assigning class labels
for index,row in full.iterrows() :
	for key,value in class_labels.items():
		if key == row['Class']:
			#row['Class']=value
			#print(row['Class'])
			#print value
			Class_Lab.append(value)

full['Class_Lab']=Class_Lab

#for index,row in full.iterrows() :
	#print(row['Sepal_Length'],row['Class'],row['Class_Lab'])

#Normalizing data
max1=full['Sepal_Length'].max()
#print max1
max2=full['Sepal_Width'].max()
max3=full['Petal_length'].max()
max4=full['Petal_Width'].max()

full['Norm_Sepal_Length'] = full['Sepal_Length'].apply(lambda x: x/max1)
full['Norm_Sepal_Width'] = full['Sepal_Width'].apply(lambda x: x/max2)
full['Norm_Petal_Length'] = full['Petal_length'].apply(lambda x: x/max3)
full['Norm_Petal_Width'] = full['Petal_Width'].apply(lambda x: x/max4)
#print full['Sepal_Length'] ,full['Norm_Sepal_Length']

print full.head()

#New Dataframe with new classs lables and normalized columns
new_df = full.filter(['Norm_Sepal_Length','Norm_Sepal_Width','Norm_Petal_Length','Norm_Petal_Width','Class_Lab'], axis=1)

#Take first 30 observations from each class and append to train.csv
new_df[0:30].to_csv('train.csv',index=False)
new_df[50:80].to_csv('train.csv',index=False,mode='a', header=False)
new_df[100:130].to_csv('train.csv',index=False,mode='a', header=False)

#Take last 20 observations from each class and append to test.csv
new_df[30:50].to_csv('test.csv',index=False)
new_df[80:100].to_csv('test.csv',index=False,mode='a', header=False)
new_df[130:150].to_csv('test.csv',index=False,mode='a', header=False)
