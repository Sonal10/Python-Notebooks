#Implementing k-means

import pandas as pd
import numpy as np
import math

#Read from file the data, put it into a dataframe in pandas

training = pd.DataFrame()
training = pd.read_csv('train.csv',skipinitialspace=True,sep=",")
#print(training.head())
print len(training)

#test= pd.read_csv('test.csv',skipinitialspace=True,sep=",")
#print(test.head())

train_features = training.filter(['Norm_Sepal_Length','Norm_Sepal_Width','Norm_Petal_Length','Norm_Petal_Width'], axis=1)
train_label = training.filter(['Class_Lab'],axis=1)
train_label.columns=['Label']

#test_features = test.filter(['Norm_Sepal_Length','Norm_Sepal_Width','Norm_Petal_Length','Norm_Petal_Width'], axis=1)
#test_label = test.filter(['Class_Lab'],axis=1)
#test_label.columns=['Label']

#Choose initial seeds, initialise as centroids, initialise cluster buckets lists
list1 = train_features.values.tolist()

initial_c1 = list1[0]
c1 = [ round(elem, 2) for elem in initial_c1 ]

print "Centroid c1 is {0}".format(c1)
cluster1=[]

initial_c2 = list1[50]
c2 = [ round(elem, 2) for elem in initial_c2 ]

print "Centroid c2 is {0}".format(c2)
cluster2=[]

initial_c3 = list1[100]
c3 = [ round(elem, 2) for elem in initial_c3 ]

print "Centroid c3 is {0}".format(c3)
cluster3=[]

Centroids = c1,c2,c3
print len(list1)
print "Initial Centroids are {0}".format(Centroids)
sq_distance=0
dist=[]
diffs=[]
cluster=[]
count = 0

#Take in all data points, calculate distance from each centroid, pick min and assign to cluster bucket
while (True):
    count = count+1
    print "Pass {}".format(count)
    for x in list1:
        for c in Centroids:
            diffs = []
            for num,cent in zip(x,c):
                diffs.append(abs(num - cent)**2)
            sq_distance = (diffs[-1] + diffs[-2] + diffs[-3] + diffs[-4])
            sq_distance = math.sqrt(sq_distance)
            dist.append(sq_distance)
        #square_distance += abs(num-cent)**2
    #min_cluster_distance = min(dist[-1],dist[-2],dist[-3])
        if (min(dist[-1],dist[-2],dist[-3]) == dist[-1]):
            cluster.append('3.0')
            cluster3.append(x)
        elif (min(dist[-1],dist[-2],dist[-3]) == dist[-2]):
            cluster.append('2.0')
            cluster2.append(x)
        else:
            cluster.append('1.0')
            cluster1.append(x)

# Recalculate centroid and repeat till iterations doesnt result a change in the point allocations
    initial_c4 = np.mean(cluster1, axis=0)
    initial_c5 = np.mean(cluster2, axis=0)
    initial_c6 = np.mean(cluster3, axis=0)
    c4= [ round(elem, 2) for elem in initial_c4 ]
    c5= [ round(elem, 2) for elem in initial_c5 ]
    c6= [ round(elem, 2) for elem in initial_c6 ]

    New_Centroids = c4,c5,c6
    print "New centroids are {}".format(New_Centroids)

    #print (lab)

    if (count == 1):
        set_x = set([i[0] for i in Centroids])
        #print set_x
        set_y = set([i[0] for i in New_Centroids])
        #print set_y
        matches = list(set_x & set_y)
        #print matches
        lab1 = pd.DataFrame(cluster)
        lab1.columns = ['Class_Lab']
        intermediate_df1=pd.DataFrame()
        intermediate_df1 = pd.concat([train_features, lab1],axis=1)
        check_first_df=pd.DataFrame()
        check_first_df['new'] = intermediate_df1.apply(lambda row: ','.join(map(str, row)), axis=1)

        if (len(matches) == 3):
            break

        else:
            Centroids = New_Centroids
            cluster1=[]
            cluster2=[]
            cluster3=[]
            cluster=[]




    if (count > 1) :
        lab = pd.DataFrame(cluster)
        lab.columns = ['Class_Lab']
        intermediate_df=pd.DataFrame()
        intermediate_df = pd.concat([train_features, lab],axis=1)
        check_second_df=pd.DataFrame()
        check_second_df['new'] = intermediate_df.apply(lambda row: ','.join(map(str, row)), axis=1)
        check_new_df=pd.DataFrame()
        check_new_df['Match']=check_first_df['new'].isin(check_second_df['new'])
        check_count_True= len(check_new_df[(check_new_df['Match'] == True)])
        if (check_count_True == 90):
            break
        set_x = set([i[0] for i in Centroids])
        #print set_x
        set_y = set([i[0] for i in New_Centroids])
        #print set_y
        matches = list(set_x & set_y)
        if (len(matches) == 3):
            break


        else:
            Centroids = New_Centroids
            cluster1=[]
            cluster2=[]
            cluster3=[]
            cluster=[]
            #count = count + 1
            del check_first_df
            check_first_df=pd.DataFrame()
            check_first_df = check_second_df
            del check_second_df
            del intermediate_df
            del lab


print "Total no. of passes made are {0}".format(count)
print "Out of while loop, final Centroids are: {0}".format(New_Centroids)
print "Finally, Cluster1 contains these points - {0}".format(cluster1)
print "Finally, Cluster2 contains these points - {0}".format(cluster2)
print "Finally, Cluster3 contains these points - {0}".format(cluster3)

#Adding labels to each of the points in the final cluster

lab = pd.DataFrame(cluster)
lab.columns = ['Class_Lab']
#print (lab)

end_df = pd.concat([train_features, lab],axis=1)

#print "Final Dataframe"
#print end_df

print "Cluster1 length is {}".format(len(cluster1))
print "Cluster2 length is {}".format(len(cluster2))
print "Cluster3 length is {}".format(len(cluster3))

#Checking for accuracy

#print "I am before matching statement"

first_df=pd.DataFrame()
second_df=pd.DataFrame()
first_df['new'] = training.apply(lambda row: ','.join(map(str, row)), axis=1)
second_df['new'] = end_df.apply(lambda row: ','.join(map(str, row)), axis=1)
new_df=pd.DataFrame()
new_df['Match']=first_df['new'].isin(second_df['new'])
#print new_df

#print "I am after matching statement"

count_True= len(new_df[(new_df['Match'] == True)])
print "This is true count {0}".format(count_True)

accuracy=count_True/float(len(list1))
print "Classification Accuracy is {0}".format(accuracy)

new_df1=pd.DataFrame()
first_df_1=pd.DataFrame()
second_df_1=pd.DataFrame()
first_df_2=pd.DataFrame()
second_df_2=pd.DataFrame()
first_df_1 = training[training['Class_Lab'] == 1.0]
second_df_1 = end_df[end_df['Class_Lab'] == '1.0']
first_df_2['new'] = first_df_1.apply(lambda row: ','.join(map(str, row)), axis=1)
second_df_2['new'] = second_df_1.apply(lambda row: ','.join(map(str, row)), axis=1)
new_df1['Match']=first_df_2['new'].isin(second_df_2['new'])
count_correct_1 = len(new_df1[(new_df1['Match'] == True)])
#print count_correct_1
#print len(second_df_1)
precision_1 = count_correct_1 / float(len(second_df_1))
print "Precision for Class 1 is {0}".format(precision_1)

new_df1=pd.DataFrame()
first_df_1=pd.DataFrame()
second_df_1=pd.DataFrame()
first_df_2=pd.DataFrame()
second_df_2=pd.DataFrame()
first_df_1 = training[training['Class_Lab'] == 2.0]
second_df_1 = end_df[end_df['Class_Lab'] == '2.0']
first_df_2['new'] = first_df_1.apply(lambda row: ','.join(map(str, row)), axis=1)
second_df_2['new'] = second_df_1.apply(lambda row: ','.join(map(str, row)), axis=1)
new_df1['Match']=first_df_2['new'].isin(second_df_2['new'])
count_correct_1 = len(new_df1[(new_df1['Match'] == True)])
#print count_correct_1
len_2 = float(len(second_df_1))
precision_2 = count_correct_1 / len_2
print "Precision for Class 2 is {0}".format(precision_2)

new_df1=pd.DataFrame()
first_df_1=pd.DataFrame()
second_df_1=pd.DataFrame()
first_df_2=pd.DataFrame()
second_df_2=pd.DataFrame()
first_df_1 = training[training['Class_Lab'] == 3.0]
second_df_1 = end_df[end_df['Class_Lab'] == '3.0']
first_df_2['new'] = first_df_1.apply(lambda row: ','.join(map(str, row)), axis=1)
second_df_2['new'] = second_df_1.apply(lambda row: ','.join(map(str, row)), axis=1)
new_df1['Match']=first_df_2['new'].isin(second_df_2['new'])
count_correct_1 = len(new_df1[(new_df1['Match'] == True)])
#print count_correct_1
len_2 = float(len(second_df_1))
precision_3 = count_correct_1 / len_2
print "Precision for Class 3 is {0}".format(precision_3)

new_df1=pd.DataFrame()
first_df_1=pd.DataFrame()
second_df_1=pd.DataFrame()
first_df_2=pd.DataFrame()
second_df_2=pd.DataFrame()
first_df_1 = training[training['Class_Lab'] == 1.0]
second_df_1 = end_df[end_df['Class_Lab'] == '1.0']
first_df_2['new'] = first_df_1.apply(lambda row: ','.join(map(str, row)), axis=1)
second_df_2['new'] = second_df_1.apply(lambda row: ','.join(map(str, row)), axis=1)
new_df1['Match']=first_df_2['new'].isin(second_df_2['new'])
count_correct_1 = len(new_df1[(new_df1['Match'] == True)])
#print count_correct_1
len_1 = float(len(first_df_1))
recall_1 = count_correct_1 / len_1
print "Recall for Class 1 is {0}".format(recall_1)

new_df1=pd.DataFrame()
first_df_1=pd.DataFrame()
second_df_1=pd.DataFrame()
first_df_2=pd.DataFrame()
second_df_2=pd.DataFrame()
first_df_1 = training[training['Class_Lab'] == 2.0]
second_df_1 = end_df[end_df['Class_Lab'] == '2.0']
first_df_2['new'] = first_df_1.apply(lambda row: ','.join(map(str, row)), axis=1)
second_df_2['new'] = second_df_1.apply(lambda row: ','.join(map(str, row)), axis=1)
new_df1['Match']=first_df_2['new'].isin(second_df_2['new'])
count_correct_1 = len(new_df1[(new_df1['Match'] == True)])
#print count_correct_1
len_1 = float(len(first_df_1))
recall_1 = count_correct_1 / len_1
print "Recall for Class 2 is {0}".format(recall_1)

new_df1=pd.DataFrame()
first_df_1=pd.DataFrame()
second_df_1=pd.DataFrame()
first_df_2=pd.DataFrame()
second_df_2=pd.DataFrame()
first_df_1 = training[training['Class_Lab'] == 3.0]
second_df_1 = end_df[end_df['Class_Lab'] == '3.0']
first_df_2['new'] = first_df_1.apply(lambda row: ','.join(map(str, row)), axis=1)
second_df_2['new'] = second_df_1.apply(lambda row: ','.join(map(str, row)), axis=1)
new_df1['Match']=first_df_2['new'].isin(second_df_2['new'])
count_correct_1 = len(new_df1[(new_df1['Match'] == True)])
#print count_correct_1
len_1 = float(len(first_df_1))
recall_1 = count_correct_1 / len_1
print "Recall for Class 3 is {0}".format(recall_1)
