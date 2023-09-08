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
