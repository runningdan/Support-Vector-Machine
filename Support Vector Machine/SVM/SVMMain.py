
'''
By Dan White
'''

from numpy import *


class SVM:

    '''SVM Base class'''

    def __init__(
                self, x, y, C = 1.0, tol = 10**-6,
                max_Passes = 50, min_Alpha = 10**-5,
                checkAccuracy = True, OVR = False, **kwargs
                ):

        self._x = mat(x) 
        self._OVR = OVR
        self._y = y 
        self._C = C 
        self._max_Passes = max_Passes 
        self._tol = tol
        self._min_Alpha = min_Alpha 
        self._m, self._n = shape(self._x) 
        self._checkAccuracyRate = checkAccuracy
        self._multiLabel_y = []
        self._multiLabel_alpha = []
        self._multiLabel_b = []
        super().__init__(**kwargs)

    @property
    def get_alpha(self): return self._multiLabel_alpha

    @property
    def get_b(self): return self.__multiLabel_b

    def kernal(self, va, vb):
        K = va * vb.T
        return K

    def train(self, yy):
        yy = mat(yy).transpose()
        passes = 0; alpha = mat(zeros((self._m,1)))
        b = 0; K = mat(zeros((self._m,self._m)))
        for i in range(self._m):
            for j in range(self._m):
                K[i,j] = self.kernal(mat(self._x[i]), mat(self._x[j]))
        while passes < self._max_Passes:
            num_changed_alphas = 0
            for i in range(self._m):
                Ei = self.calcEk(i, self._x, yy, alpha, b, i, K)
                if (alpha[i] < self._C and yy[i]*Ei < -self._tol)\
                    or \
                    (alpha[i] > 0 and yy[i]*Ei > self._tol):
                    j = self.calcJ(self._m, i)
                    Ej = self.calcEk(j, self._x, yy, alpha, b, j, K)
                    alphaIold = alpha[i].copy(); alphaJold = alpha[j].copy()
                    L, H = self.optimizeBounds(i, j, yy, alpha, self._C)
                    if (L==H): continue
                    N = self.calcN(i, j, self._x, K)
                    if (N >= 0): continue
                    alpha[j] = self.calcAlphaJ(alpha[j], yy[j], Ei, Ej, N)
                    alpha[j] = self.clipAlpha(alpha[j], L, H)
                    if (abs(alpha[j] - alphaJold) < self._min_Alpha): continue
                    alpha[i] = self.calcAlphaI(i, j, yy, alpha, alphaJold)
                    b = self.KKTConditions(
                                            i, j, self._x, yy, alpha,
                                            Ei, Ej, b, alphaIold, alphaJold, self._C, K
                                          )
                    num_changed_alphas += 1
            if num_changed_alphas == 0: passes += 1
            else: passes = 0
        w = self.calcW(self._x, yy, self._m, self._n, alpha)
        return alpha, w, b

    def trainClassifier(self):
        types = list(dict.fromkeys(self._y))
        for i, j in enumerate(types):
            y=[]
            for k, x in enumerate(self._y):
                if x == j: y.append(-1)
                else: y.append(1)
            alpha, w, b = self.train(y)
            self._multiLabel_alpha.append(alpha)
            self._multiLabel_b.append(b); self._multiLabel_y.append([y])
            if(self._OVR == False): break
            else: print("{0}% complete".format((i+1)*100/len(types)))
        if (self._checkAccuracyRate):
            stats = 100.0
            if(self._OVR): stats = self.calcStats(self._m)
            print( "Accuracy rate is {0}%".format(stats))

    def classifySample(self, nVector):
        types = list(dict.fromkeys(self._y))
        prob = [0]*len(types); score = 0
        for i, j in enumerate(types):
            predict = self.classifyPoint(
                                        nVector, self._multiLabel_y[i],
                                        self._multiLabel_alpha[i], self._multiLabel_b[i]
                                        )
            if(predict == -1): prob[i] += 1
            else:
                for k, e in enumerate(types):
                     if e != types[i]:
                         prob[k] += 1
            if(self._OVR == False): score = 100; break;
            else : score = max(prob)/len(prob)*100;
        return types[argmax(prob)], score

    def classifyPoint(self, nVector, y, alpha, b):
        y = mat(y).transpose()
        vect = mat(nVector); Iter = shape(vect)[0]
        supportVectorIndex=nonzero(alpha.A>0)[0]
        supportVectors=self._x[supportVectorIndex]; supportVectorLabel = y[supportVectorIndex]
        for i in range(Iter):
            kernal = self.kernal(supportVectors,vect)
            predict=kernal.T * multiply(supportVectorLabel,alpha[supportVectorIndex]) + b
        classify = -1
        if (sign(predict[0]).item(0, 0) == 1): classify = 1
        return classify

    def calcStats(self, m):
        accuracyRate = 0
        for i, x in enumerate(self._x):
            predict = self.classifySample(x)
            if predict[0] == self._y[i]: accuracyRate += 1
        return accuracyRate/len(self._x)*100

    def calcW(self, x, y, m, n, alpha):
        w = zeros((n, 1))
        for i in range(m):
            w += multiply(y[i] * alpha[i], x[i].T)
        return w

    def KKTConditions(self, i, j, x, y, alpha, Ei, Ej, b, alphaIold, alphaJold, C, K):
        b1 = b-Ei-y[i]*(alpha[i]-alphaIold)*K[i,i]- \
             y[j]*(alpha[j]-alphaJold)*K[i,j]
        b2 = b-Ej-y[i]*(alpha[i]-alphaIold)*K[i,j]- \
             y[j]*(alpha[j]-alphaJold)*K[j,j]
        if 0 < alpha[i] and C > alpha[i]: b = b1
        elif 0 < alpha[j] and C > alpha[j]: b = b2
        else: b = (b1 + b2) / 2.0
        return b

    def calcEk(self, index, x, y, alpha, b, i, K):
        Ei = (float(multiply(y, alpha).T* \
             K[:,i] + b)) - float(y[index])
        return Ei

    def calcJ(self, m, i):
        j=i
        while (j==i): j = int(random.uniform(0,m))
        return j

    def optimizeBounds(self, i, j, y, alpha, C):
        if (y[i] != y[j]):
            L = max(0, alpha[j] - alpha[i])
            H = min(C,  C + alpha[j] - alpha[i])
        else:
            L = max(0, alpha[j] + alpha[i] - C)
            H = min(C, alpha[j] + alpha[i])
        return L, H

    def calcN(self, i, j, x, K):
        N = 2.0 * K[i,j] - K[i,i] \
                - K[j,j]
        return N

    def calcAlphaJ(self, alphaJ, yJ, Ei, Ej, N):
        alphaJ -= yJ * (Ei - Ej) / N
        return alphaJ

    def calcAlphaI(self,i, j, y, alpha, alphaJold):
        alpha[i] += y[j] * y[i] * (alphaJold - alpha[j])
        return alpha[i]

    def clipAlpha(self, alphaJ, L, H):
        if alphaJ < L: alphaJ = L
        if alphaJ > H: alphaJ = H
        return alphaJ

class RBFSVM(SVM):

    def __init__(self, gamma=1, **kwargs):
        self._gamma = gamma
        super().__init__(**kwargs)

    def kernal(self, va, vb): #k(x,y)=exp(−gamma*∥x−y∥2)
        K = zeros((va.shape[0],vb.shape[0]))
        for i,x in enumerate(va):
            for j,y in enumerate(vb):
                K[i,j] = exp(-self._gamma*linalg.norm(x-y)**2)
        return K

class LSVM(SVM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
