# K-Nearest-Neighbors
In this repository, we will discuss the work of K-Nearest Neighbors (KNN) algorithm using Python's Sklearn (scikit-learn). The K-Nearest Neighbor (K-NN) algorithm is a supervised machine learning technique to classify data points into classes (K classes) and to create imaginary boundaries between such classes. The goal of the algorithm next, after building these classes, is to predict the classes of the new data points.

The algorithm uses labeled datapoints to build these boundaries (the classification model) in a process called training. Labeled datapoints are data from your dataset with predefined classes. If you visualize your datapoints as records in a table format, then the columns are known as the features (or independent variable), while the final predefined class is known as the label (or target variable or dependent variable). Classification is a prediction task with a categorical target variable. The kNN algorithm is a supervised machine learning model, which means the algorithm predicts a target variable using one or multiple independent variables.

Some of these labeled datapoints are used to train the model, while others are used to test the accuracy of the model. With a trained model can then be used to predict future unlabeled datapoints (datapoints we do not know their classes beforehand). The goal of the designer, after analyzing the dataset and the requirements, is to choose the right value of K to avoid overfitting and underfitting of the dataset.


The K-NN algorithm follows a set of iterative steps
- Step1: Create feature (the information within your datapoints) and target variables (the categories)
- Step2: Split the dataset into train datapoints and test datapoints
- Step3: Initialize the classifier with the value of K (the number of classes)
- Step4: Train the classifier using the train datapoints
- Step5: Test the classifier using the test datapoints
- Step6: Use the classifier to predict the classes of new datapoints.



## Step1: Prepare the required libraries

Python version 3.8 is the most stable version, used in the different tutorials. Scikit-learn (or commonly referred to as sklearn) is probably one of the most powerful and widely used Machine Learning libraries in Python.

```
python3 --version
pip3 --version
pip3 install --upgrade pip  
pip3 install pandas
pip3 install numpy
pip3 install scikit-learn
```

## Step2: Prepare/build your Dataset


```
import pandas as pd
from sklearn import cluster
```
Your dataset is composed of a set of records in a table format, and the titles of each column. Next, we build a dataframe out of the our dataset and display columns. A DataFrame is a 2D data structure (you can visualize it as a table or a spreadsheet) and is most commonly used pandas object. A DataFrame optionally accepts index (row labels) and columns (column lables) arguments

```
titles = ['user','Jaws','Star Wars','Exorcist','Omen']
records = [['john',5,5,2,1],['mary',4,5,3,2],['lisa',2,2,4,5],['bob',4,4,4,3],['lee',1,2,3,4],['harry',2,1,5,5]]

dataSet = pd.DataFrame(records,columns=titles)
print(dataSet)
```

We will not need the users' names in our clustering mission, so let's drop them first. The drop() is used to remove specific row or entire column and two important arguments to consider with drop(). The labels (can be a single label or a list) to specify Index or column labels to drop.
- the axis to specifiy whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’).
- By specifying the column axis (axis='columns'), the drop() method removes the specified column.
- By specifying the row axis (axis='index'), the drop() method removes the specified row.

```
ratings = dataSet.drop('user',axis='columns')
```

Finally, you should assign a label (or target) for each data record:
```
classes = ['Class0', 'Class0', 'Class1', 'Class1','Class1','Class0']   #or classes = [0,0,1,1,1,0]
```


## Step3: Build and train the model

```
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(ratings,classes)
```




## Step4: testing the model on new data point(s)

```
newPoints = [[5,5,2,1]]

prediction = knn.predict(newPoints)
print(prediction)


newPoints = [[3,1,7,8], [4,3,1,3]]

prediction = knn.predict(newPoints)
print(prediction)
```
