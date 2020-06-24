# Mehta, Ashutosh
# 1001-709-115
# 2020_03_22
# Assignment-03-01

# %tensorflow_version 2.x
# %tensorflow_version 2.x
import tensorflow as tf
import numpy as np


class MultiNN(object):
    def __init__(self, input_dimension):
        """
        Initialize multi-layer neural network
        :param input_dimension: The number of dimensions for each input data sample
        """
        self.input_dimension=input_dimension
        self.weights=[]
        self.biases=[]
        self.activation=[]


    def add_layer(self, num_nodes, transfer_function="Linear"):
        """
         This function adds a dense layer to the neural network
         :param num_nodes: number of nodes in the layer
         :param transfer_function: Activation function for the layer. Possible values are:
        "Linear", "Relu","Sigmoid".
         :return: None
        """
        if not self.weights:
            w=tf.Variable(tf.random.normal([self.input_dimension,num_nodes]))
        else:
            w=tf.Variable(tf.random.normal([self.weights[-1].shape[1],num_nodes]))

        self.weights.append(w)
        b=tf.Variable(tf.random.normal([num_nodes,]))
        self.biases.append(b)
        self.activation.append(transfer_function)

    def get_weights_without_biases(self, layer_number):
        """
        This function should return the weight matrix (without biases) for layer layer_number.
        layer numbers start from zero.
         :param layer_number: Layer number starting from layer 0. This means that the first layer with
          activation function is layer zero
         :return: Weight matrix for the given layer (not including the biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
        """
        return self.weights[layer_number]

    def get_biases(self, layer_number):
        """
        This function should return the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0
         :return: Weight matrix for the given layer (not including the biases).
         Note that the biases shape should be [1][number_of_nodes]
        """
        return self.biases[layer_number]
            

    def set_weights_without_biases(self, weights, layer_number):
        """
        This function sets the weight matrix for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param weights: weight matrix (without biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         :param layer_number: Layer number starting from layer 0
         :return: none
         """ 
        self.weights[layer_number]=weights
        

    def set_biases(self, biases, layer_number):
        """
        This function sets the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
        :param biases: biases. Note that the biases shape should be [1][number_of_nodes]
        :param layer_number: Layer number starting from layer 0
        :return: none
        """
        self.biases[layer_number]=biases


    def calculate_loss(self, y, y_hat):
        """
        This function calculates the sparse softmax cross entropy loss.
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
        :param y_hat: Array of actual output values [n_samples][number_of_classes].
        :return: loss
        """
        return tf.math.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y,y_hat,name=None))

    def predict(self, X):
        """
        Given array of inputs, this function calculates the output of the multi-layer network.
        :param X: Array of input [n_samples,input_dimensions].
        :return: Array of outputs [n_samples,number_of_classes ]. This array is a numerical array.
        """
        test=np.array(self.weights)
        a=[]
        z=X
        print(self.activation, "Ok3")
        print(X.shape, self.biases[0], test[0].shape, "Ok2")
        for i in range(test.shape[0]):
            mul=tf.matmul(z,test[i])
            print(mul,"XX")        
            summation=tf.add(mul,self.biases[i])    
            if self.activation[i]=="Sigmoid":
                a=tf.sigmoid(summation)
            elif self.activation=="Relu":
                a=tf.nn.relu(summation)
            else:
                a=summation
            z=a
        return a
    
    def train(self, X_train, y_train, batch_size, num_epochs, alpha=0.8):
        """
         Given a batch of data, and the necessary hyperparameters,
         this function trains the neural network by adjusting the weights and biases of all the layers.
         :param X: Array of input [n_samples,input_dimensions]
         :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
         :param batch_size: number of samples in a batch
         :param num_epochs: Number of times training should be repeated over all input data
         :param alpha: Learning rate
         :return: None
        """
        z=X_train
        batch_split_input=np.array_split(z,batch_size,axis=1)
        for _ in range(num_epochs):
               # print("-----------------------iteration_no_",_+1,"--------------------")       
                #print(error, y, actual,"A")
                 
            batch_split_target=np.array_split(y_train,batch_size,axis=1)
                
            for i in range(z.shape[1]//batch_size):
                with tf.GradientTape() as tape:
                    predictions= self.predict(batch_split_input[i])
                    loss= self.calculate_loss(batch_split_target[i],predictions)
                    l_w,l_b=tape.gradient(loss,[self.weights,self.biases])
                    w=tf.Variable(self.weights)
                    for i in range(w.shape[0]):
                        self.weights[i]=tf.subtract(self.weights[i], alpha*l_w[i])    
                        self.biases[i]=tf.subtract(self.biases[i],alpha*l_b[i])
                    

    def calculate_percent_error(self, X, y):
        """
        Given input samples and corresponding desired (true) output as indexes,
        this method calculates the percent error.
        For each input sample, if the predicted class output is not the same as the desired class,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        Note that the predicted class is the index of the node with maximum output.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return percent_error
        """
        target=self.predict(X)
        print(target,y,"Z")
        error=0
        for i in range(target.shape[1]):
            for j in range(target.shape[0]):
                if(target[j][i]!=y[j][i]):
                    error+=1
                    break
        
        total_errors=100*(error/X.shape[1])
        return total_errors


    def calculate_confusion_matrix(self, X, y):
        """
        Given input samples and corresponding desired (true) outputs as indexes,
        this method calculates the confusion matrix.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return confusion_matrix[number_of_classes,number_of_classes].
        Confusion matrix should be shown as the number of times that
        an image of class n is classified as class m.
        """
    
        x=self.predict(X)
        print(tf.math.confusion_matrix(y,np.argmax(x,axis=1)))
        return tf.math.confusion_matrix(y,np.argmax(x,axis=1))