# Module 3: Datasets
# Iris Flower Dataset
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Step 1 Get Data
from sklearn import datasets
iris = datasets.load_iris()
data = iris.data
target = iris.target
# print(data)
# print(target)

# Step 3 Shuffle Data and Split Data to Train/Test
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(data, target, test_size=0.33, random_state=42)
# print(train_X)
# print(train_y)
