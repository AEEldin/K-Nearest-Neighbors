import pandas as pd
from sklearn import cluster
from sklearn.neighbors import KNeighborsClassifier


# build a fake dataset
titles = ['user','Jaws','Star Wars','Exorcist','Omen']
records = [['john',5,5,2,1],['mary',4,5,3,2],['lisa',2,2,4,5],['bob',4,4,4,3],['lee',1,2,3,4],['harry',2,1,5,5]]
dataSet = pd.DataFrame(records,columns=titles)

ratings = dataSet.drop('user',axis='columns')

# define the class (or the labels) for each record
classes = ['Class0', 'Class0', 'Class1', 'Class1','Class1','Class0']
#classes = [0,0,1,1,1,0]


# let's train KNN classifier model
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(ratings,classes)


# now we can use the model to predict the class of new data points
newPoints = [[5,5,2,1]]

prediction = knn.predict(newPoints)
print(prediction)

newPoints = [[3,1,7,8], [4,3,1,3]]
prediction = knn.predict(newPoints)
print(prediction)

# you can also test the accuracy of your predicitions using sklearn.metrics.accuracy_score() function
