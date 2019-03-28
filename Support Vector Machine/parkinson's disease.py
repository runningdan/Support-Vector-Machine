from SVM import *

print("gathering features")
X = []
y = []
file = open('datasets/parkinson\'s disease/dataset.txt')
for line in file.readlines():
    lineArr = line.strip().split(',')
    xx = []
    for i, datapoint in enumerate(lineArr):
        if i > 0 and i != 17: xx.append(float(datapoint))
        if i == 17: y.append(datapoint)
    X.append(xx)

svm = RBFSVM(x=X, y=y, C=2, tol=0.000001, max_Passes = 500, min_Alpha=0.00001, gamma=0.001, checkAccuracy=True, OVR=False)
print('Training classifier please wait')
svm.fit()
while True:
    vector = input('Input a sample\n')
    try:
        sample = [float(x) for x in vector.split(',')];
        predictSample = svm.predict([sample]);
        print("{0}. confidence {1}%".format(predictSample[0],predictSample[1]));
    except:
        pass

'''
https://archive.ics.uci.edu/ml/datasets/parkinsons
Test
parkinsons subject
187.73300,202.32400,173.01500,0.00316,0.00002,0.00168,0.00182,0.00504,0.01663,0.15100,0.00829,0.01003,0.01366,0.02488,0.00265,26.31000,0.396793,0.758324,-6.006647,0.266892,2.382544,0.160691
healthy subject
197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569
'''
