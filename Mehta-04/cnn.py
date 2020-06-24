# Mehta, Ashutosh
# 1001-709-115
# 2020_04_19
# Assignment-04-01

'''
References: https://keras.io/layers/core/
            https://github.com/chaitanya72/API_For_Multi_Layer_Network/blob/master/cnn.py#L1
'''

import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,MaxPool2D,InputLayer,Flatten
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import sparse_categorical_crossentropy,hinge,mean_squared_error
from tensorflow.keras.optimizers import Adam,RMSprop,SGD,Adagrad
import tensorflow.keras.backend as K
import os
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.initializers import Zeros
if tf.executing_eagerly() == False:
    tf.enable_eager_execution()

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class CNN(object):
    def __init__(self):
        """
        Initialize multi-layer neural network
        """
        self.model = Sequential()
        self.loss=None
        self.metric = []
        self.optimizer=None
        self.initilizer = Zeros()


    def add_input_layer(self, shape=(2,),name="" ):
        """
         This function adds an input layer to the neural network. If an input layer exist, then this function
         should replcae it with the new input layer.
         :param shape: input shape (tuple)
         :param name: Layer name (string)
         :return: None
        """
        self.input_shape = shape
        self.model.add(InputLayer(input_shape=shape,name=name))


    def append_dense_layer(self, num_nodes,activation="relu",name="",trainable=True):
        """
         This function adds a dense layer to the neural network
         :param num_nodes: Number of nodes
         :param activation: Activation function for the layer. Possible values are "Linear", "Relu", "Sigmoid",
         "Softmax"
         :param name: Layer name (string)
         :param trainable: Boolean
         :return: None
        """
        self.model.add(Dense(units=num_nodes,activation=activation,trainable=trainable,name=name,kernel_initializer=self.initilizer,bias_initializer=self.initilizer))
    
    def append_conv2d_layer(self, num_of_filters, kernel_size=3, padding='same', strides=1,
                         activation="Relu",name="",trainable=True):
        """
         This function adds a conv2d layer to the neural network
         :param num_of_filters: Number of nodes
         :param num_nodes: Number of nodes
         :param kernel_size: Kernel size (assume that the kernel has the same horizontal and vertical size)
         :param padding: "same", "Valid"
         :param strides: strides
         :param activation: Activation function for the layer. Possible values are "Linear", "Relu", "Sigmoid"
         :param name: Layer name (string)
         :param trainable: Boolean
         :return: Layer object
        """
        self.model.add(Conv2D(filters=num_of_filters,kernel_size=kernel_size,padding=padding,strides=strides,activation=activation,trainable=trainable,name=name,kernel_initializer=self.initilizer,bias_initializer=self.initilizer))
        return self.model.layers[-1]

    def append_maxpooling2d_layer(self, pool_size=2, padding="same", strides=2,name=""):
        """
         This function adds a maxpool2d layer to the neural network
         :param pool_size: Pool size (assume that the pool has the same horizontal and vertical size)
         :param padding: "same", "valid"
         :param strides: strides
         :param name: Layer name (string)
         :return: Layer object
        """
        self.model.add(MaxPool2D(pool_size=pool_size,padding=padding,strides=strides,name=name))
        return self.model.layers[-1]
    
    def append_flatten_layer(self,name=""):
        """
         This function adds a flattening layer to the neural network
         :param name: Layer name (string)
         :return: Layer object
        """
        self.model.add(Flatten(name=name))
        return self.model.layers[-1]
    
    def set_training_flag(self,layer_numbers=[],layer_names="",trainable_flag=True):
        """
        This function sets the trainable flag for a given layer
        :param layer_number: an integer or a list of numbers.Layer numbers start from layer 0.
        :param layer_names: a string or a list of strings (if both layer_number and layer_name are specified, layer number takes precedence).
        :param trainable_flag: Set trainable flag
        :return: None
        """
        if len(layer_numbers)!=0:
            for layer_number in layer_numbers:
                self.model.layers[layer_number-1].trainable = trainable_flag
        # else:
        #     for layer in self.model.layers:
        #         if layer.name == layer_names:
        #             print(layer.name, "Ok1" )
        #             layer.trainable = trainable_flag

    def get_weights_without_biases(self,layer_number=None,layer_name=""):
        """
        This function should return the weight matrix (without biases) for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0.
         :param layer_name: Layer name (if both layer_number and layer_name are specified, layer number takes precedence).
         :return: Weight matrix for the given layer (not including the biases). If the given layer does not have
          weights then None should be returned.
        """
        
        if layer_number!= None:
            if layer_number == 0:
                return None
            if layer_number < 0:
                return self.model.layers[layer_number].get_weights()[0]
            #print(self.model.layers[layer_number].get_weights().shape)
            weight = self.model.layers[layer_number-1].get_weights()
            if len(weight)>0:
                #print(self.model.layers[layer_number-1].get_weights()[0].shape)
                return self.model.layers[layer_number-1].get_weights()[0]
            else:
                #print("None")
                return None
        else:
            for layer in self.model.layers:
                if layer.name==layer_name:
                    weight = layer.get_weights()
                    if len(weight)>0:
                        return layer.get_weights()[0]
                    else:
                        return None


    def get_biases(self,layer_number=None,layer_name=""):
        """
        This function should return the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0
         :param layer_name: Layer name (if both layer_number and layer_name are specified, layer number takes precedence).
         :return: biases for the given layer (If the given layer does not have bias then None should be returned)
        """
        if layer_number!= None:
            if layer_number==0:
                return None
            bias = self.model.layers[layer_number-1].get_weights()
            if layer_number < 0:
                return self.model.layers[layer_number].get_weights()[1]
            if len(bias)>0:
                return self.model.layers[layer_number-1].get_weights()[1]
            else:
                return None
        else:
            for layer in self.model.layers:
                if layer.name==layer_name:
                    bias = layer.get_weights()
                    if len(bias)>0:
                        return layer.get_weights()[1]
                    else:
                        return None

    def set_weights_without_biases(self,weights,layer_number=None,layer_name=""):
        """
        This function sets the weight matrix for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param weights: weight matrix (without biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         :param layer_number: Layer number starting from layer 0
         :param layer_name: Layer name (if both layer_number and layer_name are specified, layer number takes precedence).
         :return: None
        """
        #print(layer_number)
        if layer_number!= None:
            print(self.model.layers[layer_number-1].weights[0])
            #K.set_value(self.model.layers[layer_number-1].weights[0],weights )
        else:
            for layer in self.model.layers:
                if layer.name==layer_name:
                    K.set_value(layer.weights[0],weights)
    
    
    
    def set_biases(self,biases,layer_number=None,layer_name=""):
        """
        This function sets the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
        :param biases: biases. Note that the biases shape should be [1][number_of_nodes]
        :param layer_number: Layer number starting from layer 0
        :param layer_name: Layer name (if both layer_number and layer_name are specified, layer number takes precedence).
        :return: none
        """
        if layer_number!= None:
            K.set_value(self.model.layers[layer_number-1].weights[1],biases )
        else:
            for layer in self.model.layers:
                if layer.name==layer_name:
                    K.set_value(layer.weights[1],biases)
    
    
    
    def remove_last_layer(self):
        """
        This function removes a layer from the model.
        :return: removed layer
        """

        layer_popped = self.model.layers.pop()
        self.model.pop()
        return layer_popped

    def load_a_model(self,model_name="",model_file_name=""):
        """
        This function loads a model architecture and weights.
        :param model_name: Name of the model to load. model_name should be one of the following:
        "VGG16", "VGG19"
        :param model_file_name: Name of the file to load the model (if both madel_name and
         model_file_name are specified, model_name takes precedence).
        :return: model
        """
        '''if model_name!="":
            if model_name=="VGG16":
                self.model = keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
            else:
                self.model = keras.applications.vgg19.VGG19(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
        else:
            self.model = load_model(model_file_name)'''
        self.model = Sequential()
        if model_name!="":
            if model_name=="VGG16":
                model_temp = keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
                for i in range(0,len(model_temp.layers)):
                    self.model.add(model_temp.layers[i])
                
            else:
                model_temp = keras.applications.vgg19.VGG19(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
                for i in range(0,len(model_temp.layers)):
                    self.model.add(model_temp.layers[i])
        else:
            model_temp = load_model(model_file_name)
            for i in range(0, len(model_temp.layers)):
                self.model.add(model_temp.layers[i])
        return self.model
    
    
    def save_model(self,model_file_name=""):
        """
        This function saves the current model architecture and weights together in a HDF5 file.
        :param file_name: Name of file to save the model.
        :return: model
        """
        self.model.save(model_file_name)


    def set_loss_function(self, loss="SparseCategoricalCrossentropy"):
        """
        This function sets the loss function.
        :param loss: loss is a string with the following choices:
        "SparseCategoricalCrossentropy",  "MeanSquaredError", "hinge".
        :return: none
        """
        if loss== "SparseCategoricalCrossentropy":
            self.loss = sparse_categorical_crossentropy
        else:
            if loss == "MeanSquaredError":
                self.loss = mean_squared_error
            else:
                self.loss = hinge

    def set_metric(self,metric):
        """
        This function sets the metric.
        :param metric: metric should be one of the following strings:
        "accuracy", "mse".
        :return: none
        """
        self.metric.append(metric)

    def set_optimizer(self,optimizer="SGD",learning_rate=0.01,momentum=0.0):
        """
        This function sets the optimizer.
        :param optimizer: Should be one of the following:
        "SGD" , "RMSprop" , "Adagrad" ,
        :param learning_rate: Learning rate
        :param momentum: Momentum
        :return: none
        """
        if optimizer=="SGD":
            self.optimizer = SGD(learning_rate=learning_rate,momentum=momentum)
        else:
            if optimizer =="RMSprop":
                self.optimizer=RMSprop(learning_rate=learning_rate)
            else:
                self.optimizer=Adagrad(learning_rate=learning_rate)

    def predict(self, X):
        """
        Given array of inputs, this function calculates the output of the multi-layer network.
        :param X: Input tensor.
        :return: Output tensor.
        """
        #self.model.reset_states()
        temp_model = Sequential()
        #temp_model = self.model
        temp_model.add(InputLayer(input_shape=self.input_shape))
        for i in range(0,len(self.model.layers)):
            temp_model.add(self.model.layers[i])
        temp_model.set_weights(self.model.get_weights())
        
        output = temp_model.predict(X)
        return output.copy()


    def evaluate(self,X,y):
        """
         Given array of inputs and desired ouputs, this function returns the loss value and metrics of the model.
         :param X: Array of input
         :param y: Array of desired (target) outputs
         :return: loss value and metric value
         """
        return self.model.evaluate(x=X,y=y)

    def train(self, X_train, y_train, batch_size, num_epochs):
        """
         Given a batch of data, and the necessary hyperparameters,
         this function trains the neural network by adjusting the weights and biases of all the layers.
         :param X_train: Array of input
         :param y_train: Array of desired (target) outputs
         :param X_validation: Array of input validation data
         :param y: Array of desired (target) validation outputs
         :param batch_size: number of samples in a batch
         :param num_epochs: Number of times training should be repeated over all input data
         :return: list of loss values. Each element of the list should be the value of loss after each epoch.
        """
        self.model.compile(optimizer=self.optimizer,metrics=self.metric,loss=self.loss)
        history = self.model.fit(x=X_train,y=y_train,batch_size=batch_size,epochs=num_epochs,shuffle=False)
        return history.history['loss']

if __name__ == "__main__":

    my_cnn=CNN()
    print(my_cnn)
    my_cnn.add_input_layer(shape=(32,32,3),name="input")
    my_cnn.append_conv2d_layer(num_of_filters=16, kernel_size=(3,3),padding="same", activation='linear', name="conv1")
    my_cnn.append_maxpooling2d_layer(pool_size=2, padding="same", strides=2,name="pool1")
    my_cnn.append_conv2d_layer(num_of_filters=8, kernel_size=3, activation='relu', name="conv2")
    my_cnn.append_flatten_layer(name="flat1")
    my_cnn.append_dense_layer(num_nodes=10,activation="relu",name="dense1")
    my_cnn.append_dense_layer(num_nodes=2,activation="relu",name="dense2")
    # my_cnn.append_conv2d_layer(num_of_filters=32,kernel_size=3,activation='linear',name="conv1")
    #print(my_cnn.model.summary())
    #print(my_cnn.model.summary())
    #print(my_cnn.remove_last_layer())
    #from tensorflow.keras.datasets import cifar10

    # my_cnn.append_conv2d_layer(num_of_filters=32,kernel_size=3,activation='linear',name="conv1")
    # print(my_cnn.model.summary())
    weights=my_cnn.get_weights_without_biases(layer_number=0)
    biases=my_cnn.get_biases(layer_number=0)
    print("w0",None if weights is None else weights.shape,type(weights))
    print("b0",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_number=1)
    biases=my_cnn.get_biases(layer_number=1)
    print("w1",None if weights is None else weights.shape,type(weights))
    print("b1",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_number=2)
    biases=my_cnn.get_biases(layer_number=2)
    print("w2",None if weights is None else weights.shape,type(weights))
    print("b2",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_number=3)
    biases=my_cnn.get_biases(layer_number=3)
    print("w3",None if weights is None else weights.shape,type(weights))
    print("b3",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_number=4)
    biases=my_cnn.get_biases(layer_number=4)
    print("w4",None if weights is None else weights.shape,type(weights))
    print("b4",None if biases is None else biases.shape,type(biases))
    weights = my_cnn.get_weights_without_biases(layer_number=5)
    biases = my_cnn.get_biases(layer_number=5)
    print("w5", None if weights is None else weights.shape, type(weights))
    print("b5", None if biases is None else biases.shape, type(biases))

    weights=my_cnn.get_weights_without_biases(layer_name="input")
    biases=my_cnn.get_biases(layer_number=0)
    print("input weights: ",None if weights is None else weights.shape,type(weights))
    print("input biases: ",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_name="conv1")
    biases=my_cnn.get_biases(layer_number=1)
    print("conv1 weights: ",None if weights is None else weights.shape,type(weights))
    print("conv1 biases: ",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_name="pool1")
    biases=my_cnn.get_biases(layer_number=2)
    print("pool1 weights: ",None if weights is None else weights.shape,type(weights))
    print("pool1 biases: ",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_name="conv2")
    biases=my_cnn.get_biases(layer_number=3)
    print("conv2 weights: ",None if weights is None else weights.shape,type(weights))
    print("conv2 biases: ",None if biases is None else biases.shape,type(biases))
    weights=my_cnn.get_weights_without_biases(layer_name="flat1")
    biases=my_cnn.get_biases(layer_number=4)
    print("flat1 weights: ",None if weights is None else weights.shape,type(weights))
    print("flat1 biases: ",None if biases is None else biases.shape,type(biases))
    weights = my_cnn.get_weights_without_biases(layer_name="dense1")
    biases = my_cnn.get_biases(layer_number=4)
    print("dense1 weights: ", None if weights is None else weights.shape, type(weights))
    print("dense1 biases: ", None if biases is None else biases.shape, type(biases))
    weights = my_cnn.get_weights_without_biases(layer_name="dense2")
    biases = my_cnn.get_biases(layer_number=4)
    print("dense2 weights: ", None if weights is None else weights.shape, type(weights))
    print("dense2 biases: ", None if biases is None else biases.shape, type(biases))
