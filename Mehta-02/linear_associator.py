# Mehta, Ashutosh
# 1001-709-115
# 2020-03-01
# Assignment-02-01

import numpy as np
import math
from collections import deque



class LinearAssociator(object):
    def __init__(self, input_dimensions=2, number_of_nodes=4, transfer_function="Linear"):
        """
        Initialize linear associator model
        :param input_dimensions: The number of dimensions of the input data
        :param number_of_nodes: number of neurons in the model
        :param transfer_function: Transfer function for each neuron. Possible values are:
        "Hard_limit", "Linear".
        """
        self.number_of_nodes=number_of_nodes
        self.input_dimensions=input_dimensions
        self.weights=1
        self.transfer_function=transfer_function

        self.initialize_weights()

    def initialize_weights(self, seed=None):
        """
        Initialize the weights, initalize using random numbers.
        If seed is given, then this function should
        use the seed to initialize the weights to random numbers.
        :param seed: Random number generator seed.
        :return: None
        """
        if seed != None:
            np.random.seed(seed)
        
        W=np.random.randn(self.number_of_nodes,self.input_dimensions)
        self.set_weights(W)
        

    def set_weights(self, W):
        """
         This function sets the weight matrix.
         :param W: weight matrix
         :return: None if the input matrix, w, has the correct shape.
         If the weight matrix does not have the correct shape, this function
         should not change the weight matrix and it should return -1.
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
        Make a prediction on an array of inputs
        :param X: Array of input [input_dimensions,n_samples].
        :return: Array of model outputs [number_of_nodes ,n_samples]. This array is a numerical array.
        """
        summation=np.dot(self.weights,X)    
        #print(self.transfer_function,"T_f")
        if(self.transfer_function=="Hard_limit"): 
            summation[summation>=0]=1
            summation[summation<0]=0
        #print(summation)
        return summation
        
        #print(summation,"Ok5")
        
        


    def fit_pseudo_inverse(self, X, y):
        """
        Given a batch of data, and the targets,
        this function adjusts the weights using pseudo-inverse rule.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        """
        # print(X,"input")
        # print(self.weights, "Ok1")
        # p=np.dot(np.transpose(X),X)
        # print(p,"pT,p")
        # inverse = np.linalg.inv(p)
        # print(inverse, "inverse")
        # p_plus=np.dot(inverse,np.transpose(X))
        # print(p_plus,"p_plus")
        p_plus=np.linalg.pinv(X)
        W=np.dot(y,p_plus)
        #print(W,"out")
        self.weights=W
        


    def train(self, X, y, batch_size=5, num_epochs=10, alpha=0.1, gamma=0.9, learning="Delta"):
        """
        Given a batch of data, and the necessary hyperparameters,
        this function adjusts the weights using the learning rule.
        Training should be repeated num_epochs time.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples].
        :param num_epochs: Number of times training should be repeated over all input data
        :param batch_size: number of samples in a batch
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :param gamma: Controls the decay
        :param learning: Learning rule. Possible methods are: "Filtered", "Delta", "Unsupervised_hebb"
        :return: None
        """
        
        #input=np.transpose(X)
        #target=np.transpose(y)
        #actual= self.predict(X)
        
        batch_split_input=np.array_split(X,batch_size,axis=1)
        if learning=="delta" or learning=="Delta":
            print(learning,"ABC")
            for _ in range(num_epochs):
               # print("-----------------------iteration_no_",_+1,"--------------------")       
                #print(error, y, actual,"A")
                 
                batch_split_target=np.array_split(y,batch_size,axis=1)
                
                for i in range(X.shape[1]//batch_size):
                    actual= self.predict(batch_split_input[i])
                    error=batch_split_target[i]-actual
                    print(error,"Error")
                    v=alpha*np.dot(error,np.transpose(batch_split_input[i]))
                    self.weights=np.add(self.weights,v)
                #    print(self.weights,"Updated Weight")

        
        elif learning=="Unsupervised_hebb":
            for _ in range(num_epochs):
                print("-----------------------iteration_no_",_+1,"--------------------")       
                actual= self.predict(X)
                batch_split_actual=np.array_split(actual,batch_size,axis=1) 
                
                for i in range(X.shape[1]//batch_size):
                    v=alpha*np.dot(batch_split_actual[i],np.transpose(batch_split_input[i]))
                self.weights=np.add(self.weights,v)
        
        elif learning=="filtered":
            for _ in range(num_epochs):
                print("-----------------------iteration_no_",_+1,"--------------------")       
                batch_split_target=np.array_split(y,batch_size,axis=1) 
                
                for i in range(X.shape[1]//batch_size):
                    print(batch_split_target[i])
                    v=alpha*np.dot(batch_split_target[i],np.transpose(batch_split_input[i]))
                self.weights=np.add((1-gamma)*self.weights,v)
                    
        

    def calculate_mean_squared_error(self, X, y):
        """
        Given a batch of data, and the targets,
        this function calculates the mean squared error (MSE).
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        :return mean_squared_error
        """
        actual=self.predict(X)
        mse = ((y - actual)**2).mean()

        print(mse,"mse")
        return mse
