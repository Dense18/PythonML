# PythonML
Collection of various machine learning models made from stratch in Python. 

The code is based on working with numpy arrays as the arguments, and only supports numerical values in the dataset.

## Setup
```
pip install -e . # On root directory
pip install requirements.txt
```
## Supported models
- Linear Regression using Ordinary Least Squares
- Stochastic gradient descent Regression
- Decision Tree Classifier
- Random Forest Classifier
- KNN Classifier

- K means clustering

## Usage
All the provided models have a predict and fit function:
- fit(X, y): Fits the model based on the training set (X, y)
- predict(X): Predicts class/regression value for X
  
 where X is a 2D numpy array of independent variables and y is a 1D numpy array

Here is an example:
```
from sklearn import datasets
from sklearn.model_selection import train_test_split

from models.tree.DecisionTreeClassifier import DecisionTreeClassifier
from utils.metrics import accuracy_score

# Load data
breast = datasets.load_breast_cancer()
X, y = breast.data, breast.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Build model
knn = DecisionTreeClassifier(min_samples_split = 2, max_depth=10, random_state=123) # Look at the class docstrings to identify which parameters a specific model supports
knn.fit(X_train, y_train) 

# Use model
y_pred = knn.predict(X_test)
accuracy_score(y_test, y_pred)
# >>> 0.9298245614035088
```

More examples can be found here: [usage folder](usage)
# Note
The main purpose of this project is to :
- understand how various machine learning models works under the hood
- deepen my understanding of the supported models
