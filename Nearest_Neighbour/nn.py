import pandas as pd
import numpy as np
import math

training = pd.read_csv('iris_training.csv',skipinitialspace=True,sep=",")
#print(training.head())

test = pd.read_csv('iris_test.csv',skipinitialspace=True,sep=",")
#print(test.head())

features = training.iloc[:, :4]
class_labels1 = training.iloc[:, 4:]
class_labels = class_labels1.as_matrix()
#print(features)
#print(class_labels)

predict = test.iloc[:, :4]
actual_test_labels1 = test.iloc[:, 4:]
actual_test_labels = actual_test_labels1.as_matrix()
#print(predict)
#print(actual_test_labels)


correct = 0
training_label_index=[]
test_label_index=[]

for index, predict_features in predict.iterrows():
	min_dist=100000
	for idx, feature in features.iterrows():
		distance = 0
		distance = (((feature["Sepal_Length"]-predict_features["Sepal_Length"])**(2.0))+((feature["Sepal_Width"]-predict_features["Sepal_Width"])**(2.0))+((feature["Petal_length"]-predict_features["Petal_length"])**(2.0))+((feature["Petal_Width"]-predict_features["Petal_Width"])**(2.0)))
		#print("Distance calculated is %f" % (distance))
		euc_distance = math.sqrt(distance)
		#print("Euclidean distance of this data point is %f" % (euc_distance))
		#print("Index of training data is %d and Index of test data is %d" % (idx,index))

		if min_dist > euc_distance:
				min_dist = euc_distance
				#print("Minimum distance so far is %f" % (min_dist))
				training_label_index.append(idx)
				test_label_index.append(index)

	#print(training_label_index)
	#print(test_label_index)
	j = training_label_index[-1]
	i = test_label_index[-1]
	print("First Test Point has predicted label as %s from training_index %d" %(class_labels[j],j))
	print("Test point %d's actual label is %s" %(index,actual_test_labels[i]))

	if class_labels[j] == actual_test_labels[i]:
		correct = correct + 1

print("Correct number of predictions are %d" % (correct))
CA = (correct/60.00)*100.00

print("Classification accuracy is %0.2f " % (CA) )
