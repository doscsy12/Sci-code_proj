import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from scipy import optimize

r = requests.get('https://api.coindesk.com/v1/bpi/historical/close.json?start=2012-01-01&end=2019-01-01')
r2 = requests.get('https://api.coindesk.com/v1/bpi/historical/close.json?start=2019-01-01&end=2019-03-01')

#x_scaled = min_max_scaler.fit_transform(xalu)
#dfScaled = pandas.DataFrame(x_scaled)

trainX = []
trainY = []

testX = []
testY = []

def convertX(df, array):
    for i in range(len(df)):
       array.append([df[i]])
    NewArray = np.asarray(array)
    return NewArray

StartingDataMax = 1
StartingDataMin = 1

def prepareDF(BitcoinReq, dataX, dataY):
    dfX = pd.DataFrame(BitcoinReq.json()).bpi
    dfX =  dfX.dropna()    
    dfX_y= dfX.drop(dfX.index[0])
    # dfX_y= dfX_y.drop(dfX_y.index[0])
    dfX =  dfX[:-1]
    minV = dfX.min()
    maxV = dfX.max()
    normalized_df=(dfX-dfX.min())/(dfX.max()-dfX.min())
    normalized_dfY=(dfX_y-dfX_y.min())/(dfX_y.max()-dfX_y.min())
    XData= convertX(normalized_df,dataX)
    YData = convertX(normalized_dfY,dataY)
    return XData, YData, minV, maxV
 
trainX,trainY,StartingDataMin,StartingDataMax = prepareDF(r, trainX, trainY)
testX, testY, StartingDataMin,StartingDataMax = prepareDF(r2, testX, testY)
# trainY = convertY(normalized_dfY, trainY)

class Neural_Network(object):
    def __init__(self, Lambda=0):        
        #Define Hyperparameters
        self.inputLayerSize = 1
        self.hiddenLayerSize = 3
        self.outputLayerSize = 1

        #Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)
        
        #Regularization Parameter:
        self.Lambda = Lambda
        
    def forward(self, X):
        #Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3) 
        return yHat

    def relu(self,z):
        return np.maximum(0,z)
    def reluPrime(self,z):
        return np.where(z<=0,0,1)

    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        # return self.relu(z)
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        # return self.reluPrime(z)
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)/X.shape[0] + (self.Lambda/2)*(np.sum(self.W1**2)+np.sum(self.W2**2))
        return J
    def prediction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        return self.yHat
    
    def output(self, X, y):
        return y

    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)

        delta3 = -(y-self.yHat)* self.sigmoidPrime(self.z3)
        #Add gradient of regularization term:
        dJdW2 = np.dot(self.a2.T, delta3)/X.shape[0] + self.Lambda*self.W2

        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        #Add gradient of regularization term:
        dJdW1 = np.dot(X.T, delta2)/X.shape[0] + self.Lambda*self.W1
        
        return dJdW1, dJdW2
    
    #Helper functions for interacting with other methods/classes
    def getParams(self):
        #Get W1 and W2 Rolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
    
    def setParams(self, params):
        #Set W1 and W2 using single parameter vector:
        W1_start = 0
        W1_end = self.hiddenLayerSize*self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], \
                             (self.inputLayerSize, self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], \
                             (self.hiddenLayerSize, self.outputLayerSize))
        
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))
    
def computeNumericalGradient(N, X, y):
        paramsInitial = N.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-4

        for p in range(len(paramsInitial)):
            #Set perturbation vector
            perturb[p] = e
            N.setParams(paramsInitial + perturb)
            loss2 = N.costFunction(X, y)
            
            N.setParams(paramsInitial - perturb)
            loss1 = N.costFunction(X, y)

            #Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2*e)

            #Return the value we changed to zero:
            perturb[p] = 0
            
        #Return Params to original value:
        N.setParams(paramsInitial)

        return numgrad 
    
class trainer(object):
    def __init__(self, N):
        #Make Local reference to network:
        self.N = N
        
    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))
        self.testJ.append(self.N.costFunction(self.testX, self.testY))
        self.yHatValue.append(self.N.prediction(self.testX, self.testY))
        
        
    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X,y)
        
        return cost, grad
        
    def train(self, trainX, trainY, testX, testY):
        #Make an internal variable for the callback function:
        self.X = trainX
        self.y = trainY
        
        self.testX = testX
        self.testY = testY

        #Make empty list to store training costs:
        self.J = []
        self.testJ = []
        self.yHatValue = []
  
        params0 = self.N.getParams()

        options = {'maxiter': 200, 'disp' : True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
                                 args=(trainX, trainY), options=options, callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res


originalY=[]
originalYhat = []
def convertBack(XData, StartingDataMax,StartingDataMin,storage):

    for i in range(len(XData)):
        original = XData[i]*(StartingDataMax-StartingDataMin)+StartingDataMin

        storage.append(original)
    return storage


NN = Neural_Network(Lambda=0.0001)

T = trainer(NN)
T.train(trainX, trainY, testX, testY)

originalY = convertBack(testY,StartingDataMax, StartingDataMin,originalY)
originalYhat= convertBack(T.yHatValue[T.optimizationResults.nit-1],StartingDataMax, StartingDataMin,originalYhat)

plt.plot(originalY)
plt.plot(originalYhat)
plt.grid(1)
plt.title('Bitcoin prediction')
plt.xlabel('Days')
plt.ylabel('Price')
plt.savefig('/usercode/myfig')
plt.show()
