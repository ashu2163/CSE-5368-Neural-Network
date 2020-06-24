# Mehta, Ashutosh
# 1001-709-115
# 2020-02-16
# Assignment-01-01

import numpy as np
import math
from collections import deque

class Perceptron(object):
    
    def __init__(self, input_dimensions,number_of_nodes):
        """
        Initialize Perceptron model and set all the weights and biases to random numbers.
        :param input_dimensions: The number of dimensions of the input data
        :param number_of_nodes: Note that number of neurons in the model is equal to the number of classes.
        """
        self.number_of_nodes=number_of_nodes
        self.input_dimensions=input_dimensions
        self.weights=1
        self.initialize_weights()
        self.error=0

    def initialize_weights(self,seed=2):
        """
        Initialize the weights, initalize using random numbers.
        If seed is given, then this function should
        use the seed to initialize the weights to random numbers.
        :param seed: Random number generator seed.
        :return: None
        """
        if seed != None:
            np.random.seed(seed)
        
        W=np.random.randn(self.number_of_nodes,self.input_dimensions+1)
        self.set_weights(W)
         

    def set_weights(self, W):
        """
        This function sets the weight matrix (Bias is included in the weight matrix).
        :param W: weight matrix
        :return: None if the input matrix, w, has the correct shape.
        If the weight matrix does not have the correct shape, this function
        should not chanweights=Perceptron.get_weights(self)ge the weight matrix and it should return -1.
        """ 
        del self.weights
        self.weights=W

        

    def get_weights(self):
        """
        This function should return the weight matrix(Bias is included in the weight matrix).
        :return: Weight matrix
        """
        return self.weights

    def predict(self, X):
        """
        Make a prediction on a batach of inputs.
        :param X: Array of input [input_dimensions,n_samples]
        :return: Array of model [number_of_nodes ,n_samples]
        Note that the activation function of all the nodes is hard limit.
        """
        X=np.append([np.ones(X.shape[1])],X,axis=0)
        X1=np.transpose(X)

        print(X1)
        #temp=np.transpose(self.weights)
        #y,x = self.weights.shape
        #w=self.crop(x-2,y)
        #print(w,"C")
        summation=deque()
        print(X1[0].shape[0])
        print(X1[0].shape,self.weights.shape,"S")

        for i in range(X1.shape[0]):
            #X1[i]=np.transpose([X1[i][:,]])

            #print(X1[i].shape, "S2")
            summation.append(np.dot(self.weights,X1[i]))
            
        result=np.array(summation)
        print(result,"A3")
        result[result>=0]=1
        result[result<0]=0

        #print(result,"Ok5")
        return np.transpose(result)
        

    '''def crop(self,cropx,cropy):
        W=np.array(self.weights)
        y,x = self.weights.shape
        startx = x//2-(cropx//2)    
        return W[0:y,startx:startx+cropx+1]
    '''

    def train(self, X, Y, num_epochs,  alpha):
        """
        Given a batch of input and desired outputs, and the necessary hyperparameters (num_epochs and alpha),
        this function adjusts the weights using Perceptron learning rule.
        Training should be repeated num_epochs times.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes ,n_samples]
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :return: None
        """
        X1=np.append([np.ones(X.shape[1])],X,axis=0)
        input=np.transpose(X1)
        target=np.transpose(Y)
        
        for _ in range(num_epochs):
            

            print("-----------------------iteration_no_",_+1,"--------------------")
            for i in range(input.shape[0]):   
                print("-----------------------sample_no_",i+1,"--------------------")    
                actual= np.transpose(self.predict(X))
                error=target[i]-actual[i]
                print(error, target[i], actual[i],"A")
                v=deque()
                for k in range(error.shape[0]):    
                    v.append(alpha*error[k]*np.transpose(input[i]))
                        #w0 += error[k]
                v1=np.array(v)    
                print(v1,"learning rate*error*transpose(input)")
                        
                self.weights=np.add(self.weights,v1)
                    #w0=np.delete(w0,0,1)
                print(self.weights,"Updated Weight")
        

    def calculate_percent_error(self,X, Y):
        """
        Given a batch of input and desired outputs, this function calculates percent error.
        For each input sample, if the output is not the same as the desired output, Y,
        then it is considered one error. Percent error is 100*(number_of_errors/ number_of_samples).
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes ,n_samples]
        :return percent_error
        """
        target=self.predict(X)
        print(target,Y,"Z")
        error=0
        for i in range(target.shape[1]):
            for j in range(target.shape[0]):
                if(target[j][i]!=Y[j][i]):
                    error+=1
                    break
        
        total_errors=100*(error/X.shape[1])
        return total_errors




if __name__ == "__main__":
    input_dimensions = 2
    number_of_nodes = 2

    model = Perceptron(input_dimensions=input_dimensions, number_of_nodes=number_of_nodes)
    model.initialize_weights(seed=2)
    '''X_train = np.array([[-1.43815556, 0.10089809, -1.25432937, 1.48410426],
                        [-1.81784194, 0.42935033, -1.2806198, 0.06527391]])

    X_train=np.append([np.ones(X_train.shape[1])],X_train,axis=0)
    print(X_train)
    #print(np.ones([X_train.shape[1]]),X_train)
    Y_train = np.array([[1, 0, 0, 1], [0, 1, 1, 0],[0,0,1,1]])
    '''
    X_train = np.array([[-1.43815556, 0.10089809, -1.25432937, 1.48410426],
                        [-1.81784194, 0.42935033, -1.2806198, 0.06527391]])
    #X_train=np.append([np.ones(X_train.shape[1])],X_train,axis=0)
    Y_train = np.array([[1, 1, 1, 1], [1, 0, 1, 1]])
    #model.set_weights(np.ones((number_of_nodes, input_dimensions + 1)))
    #model.set_weights(np.array([[1,2,3],[4,5,6]]))
    model.predict(X_train)
    print("****** Model weights ******\n",model.get_weights())
    print("****** Input samples ******\n",X_train)
    print("****** Desired Output ******\n",Y_train)
    percent_error=[]
    for k in range (20):
        model.train(X_train, Y_train, num_epochs=1, alpha=0.1)
        percent_error.append(model.calculate_percent_error(X_train,Y_train))
    print("******  Percent Error ******\n",percent_error)
    print("****** Model weights ******\n",model.get_weights())