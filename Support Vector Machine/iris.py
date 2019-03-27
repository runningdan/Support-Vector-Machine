from SVM import *
print("gathering features")
X = []
y = []
file = open('datasets/iris flowers/dataset.txt')
for line in file.readlines():
    lineArr = line.strip().split(',')
    xx = []
    for i, datapoint in enumerate(lineArr):
        if i < 4: xx.append(float(datapoint))
        else: y.append(datapoint)
    X.append(xx)

svm = RBFSVM(x=X, y=y, C=1.0, tol=10**-6, max_Passes = 50, min_Alpha=10**-5, gamma=0.1, checkAccuracy=True, OVR=True)
print('Training classifier please wait')
svm.trainClassifier()
while True:
    vector = input('Input a sample\n')
    sample = [float(x) for x in vector.split(',')];
    predictSample = svm.classifySample([sample]);
    print("Sample classified as {0}. confidence {1}%".format(predictSample[0],predictSample[1]));

'''
https://archive.ics.uci.edu/ml/datasets/iris
test
Iris-versicolor
5.0,2.3,3.3,1.0
Iris-setosa
5.0,3.0,1.6,0.2
Iris-virginica
6.4,3.2,5.3,2.3
'''
