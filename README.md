# Support-Vector-Machine
## Open Source Machine Learning Library Built With Numpy

### SVM Introduction

The goal of an svm is to separate a n-dimensional feature space with a single hyperplane. We can use this to classify groups of related data. The hyperplane allows the SVM algorithm to make a prediction on an input based on which side it falls on.

This library leverages the power of the svm algorithm while only using one dependency - numpy. Numpy is a python math library that contains functions to assist when working with linear algebra. 

### Solving the SVM dual problem
Below is the dual form of the SVM. This library is able to tackle this problem effienctly by utilizing the SMO algorithm. 
![svm diagram](https://i.stack.imgur.com/mDQfb.png "SVM")

### Classification examples including kernal functions & linear classification
Below are examples of data seperated by a hyperlane utilizing different kernal functions
![svm diagram](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fnbt1206-1565/MediaObjects/41587_2006_BFnbt12061565_Fig1_HTML.gif "SVM")
<sub><sup>Noble, W. What is a support vector machine?. Nat Biotechnol 24, 1565–1567 (2006). https://doi.org/10.1038/nbt1206-1565</sup></sub>

#### Getting Started 

#### Install Numpy
In order to start classifying your own data with the svm you must first install numpy

`pip install numpy`

after installing numpy your ready to start using the library. below is an example classifier which is able to determine if one has parkinsons disease from pitch differeances in audio samples. 

```
from SVM import *

X = []
y = []

# open parkinsons dataset text file. This file is filled with training data
file = open('datasets/parkinson\'s disease/dataset.txt')

for line in file.readlines():
    lineArr = line.strip().split(',')
    xx = []
    for i, datapoint in enumerate(lineArr):
        if i > 0 and i != 17: xx.append(float(datapoint))
        if i == 17: y.append(datapoint)
    X.append(xx)

# init svm with radial basis function kernal
svm = RBFSVM(x=X, y=y, C=2, tol=0.000001, max_Passes = 500, min_Alpha=0.00001, gamma=0.001, checkAccuracy=True, OVR=False)

# train/fit classifier
svm.fit()

# allow user to input samples into trained svm algorithm. 
# the algorithm will then output a confidence score and predictions
while True:
    vector = input('Input a sample\n')
    try:
        sample = [float(x) for x in vector.split(',')];
        predictSample = svm.predict([sample]);
        print("{0}. confidence {1}%".format(predictSample[0],predictSample[1]));
    except:
        pass
```
