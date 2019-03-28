from SVM import *

print("gathering features")
X = []
y = []
file = open('datasets/breast cancer/dataset.txt')
for line in file.readlines():
    lineArr = line.strip().split(',')
    xx = []
    for i, datapoint in enumerate(lineArr):
        if i > 1: xx.append(float(datapoint))
        if i == 1: y.append(datapoint)
    X.append(xx)
svm = LSVM(x=X, y=y, C=1, tol=0.000001, max_Passes = 50, min_Alpha=0.00001, checkAccuracy=True, OVR=False)
print('Training classifier please wait')
svm.train()
while True:
    vector = input('Input a sample\n')
    sample = [float(x) for x in vector.split(',')];
    predictSample = svm.classifySample([sample]);
    print("Sample classified as {0}. confidence {1}%".format(predictSample[0],predictSample[1]));

'''
https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
Test
malignant
13.86,16.93,90.96,578.9,0.1026,0.1517,0.09901,0.05602,0.2106,0.06916,0.2563,1.194,1.933,22.69,0.00596,0.03438,0.03909,0.01435,0.01939,0.00456,15.75,26.93,104.4,750.1,0.146,0.437,0.4636,0.1654,0.363,0.1059
benign
11.89,18.35,77.32,432.2,0.09363,0.1154,0.06636,0.03142,0.1967,0.06314,0.2963,1.563,2.087,21.46,0.008872,0.04192,0.05946,0.01785,0.02793,0.004775,13.25,27.1,86.2,531.2,0.1405,0.3046,0.2806,0.1138,0.3397,0.08365
'''
