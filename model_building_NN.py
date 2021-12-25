# Overall process
# 1. Import cleaned/processed dataset
# 2. Multiplies the input by a set weights (performs a dot product aka matrix multiplication) 
# 3. Applies an activation function 
# 4. Returns an output 
# 5. Error is calculated by taking the difference from the desired output from the data and the predicted output. This creates our gradient descent, which we can use to alter the weights 
# 6. The weights are then altered slightly according to the error. 
# 7. To train, this process is repeated N times. 

import pandas as pd
import numpy as np

# housing price data 
df = pd.read_csv("NO_housing.csv")
# X = df.values
# X = df['age','sq_metres','nr_rm','nr_bath','nr_garage','heating_type','house_type','nr_flr','city','suburb','nr_hut']
X = df.drop(columns=['price'])
y = df[['price']]
# xPredicted = np.array(([4,8]), dtype=float)

# scale units
X = X/np.amax(X, axis=0) 
y = y/100 # scaled/normalised price/score is 100
# xPredicted = xPredicted/np.amax(xPredicted, axis=0) # maximum of xPredicted (our input data for the prediction)

class Neural_Network(object):
    def __init__(self):        
        #Define Hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3
        
        #Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
        
    def forward(self, X):
        #forward propagation through our network
        self.z = np.dot(X, self.W1) # dot product of X (input) and first set of 3x2 weights
        self.z2 = self.sigmoid(self.z) # activation function
        self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of 3x1 weights
        out = self.sigmoid(self.z3) # final activation function
        return out

    def backward(self, X, y, o):
        # backward propagate through the network
        self.o_error = y - o # error in output
        self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to z3 error

        self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z error

        self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
        # self.W1 += learningrate.X.T.dot(self.z2_delta)
        self.W2 += self.z2.T.dot(self.o_delta) # adjusting second set (hidden --> output) weights

    def train (self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)
 
    def sigmoid(self,z):
        # Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
        
    def sigmoidPrime(self, s):
        #derivative of sigmoid
        return s * (1 - s)
    
    #self.o_delta = self.o_error * self.sigmoidPrime(self.z3)
    #self.z2_delta = self.z2_error * self.sigmoidPrime(self.z)
    #d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
    #d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

    def predict(self):
        print "Predicted data based on trained weights: ";
        print "Input (scaled): \n" + str(xPredicted);
        print "Output: \n" + str(self.forward(xPredicted));

NN = Neural_Network()

for i in xrange(1000): # trains the NN 1,000 times
    print "# " + str(i) + "\n"
    print "Input (scaled): \n" + str(X)
    print "Actual Output: \n" + str(y)
    print "Predicted Output: \n" + str(NN.forward(X))
    print "Loss: \n" + str(np.mean(np.square(y - NN.forward(X)))) # mean sum squared loss
    print "\n"
    NN.train(X, y)

NN.predict()
