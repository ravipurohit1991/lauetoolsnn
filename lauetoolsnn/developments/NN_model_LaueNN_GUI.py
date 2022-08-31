# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 13:46:07 2022

@author: PURUSHOT

Numpy neural network implementation of the DNN model used in LaueNN GUI
Note: This does not include any kernel or bias regularization terms 
For larger output models please install tensorflow


Activate: Relu
Dropout: 0.3
optimizer: Adam (with learning rate)
Loss: categorical_crossentropy

For other models such as CNN, install tensorflow and keras (could be done in Numpy but not efficient)

TODO
Softmax gradient
Adam optimizer for gradient
Dropout

"""
# =============================================================================
# Data to be used for both Keras model and Numpy model to evalute the performance
# a simple Si crystal model with 1 grain and 200 simulations with 9output classes
# =============================================================================
import sys
sys.path.append(r"C:\Users\purushot\Desktop\github_version_simple\lauetoolsnn")
from NNmodels import model_arch_general
from utils_lauenn import array_generator_verify, array_generator
import numpy as np
import _pickle as cPickle

save_directory = r"C:\Users\purushot\Desktop\github_version_simple\lauetoolsnn\models\Si"
classhkl = np.load(save_directory+"//MOD_grain_classhkl_angbin.npz")["arr_0"]
angbins = np.load(save_directory+"//MOD_grain_classhkl_angbin.npz")["arr_1"]
loc_new = np.load(save_directory+"//MOD_grain_classhkl_angbin.npz")["arr_2"]
with open(save_directory+"//class_weights.pickle", "rb") as input_file:
    class_weights = cPickle.load(input_file)
class_weights = class_weights[0]
batch_size = 50
epochs = 5

trainy_inbatch = array_generator_verify(save_directory+"//training_data", batch_size, len(classhkl), loc_new)
print("Number of spots in a batch of %i files : %i" %(batch_size, len(trainy_inbatch)))
print("Min, Max class ID is %i, %i" %(np.min(trainy_inbatch), np.max(trainy_inbatch))) 
## Batch loading for numpy grain files (Keep low value to avoid overcharging the RAM)
steps_per_epoch = int((1 * 200) / batch_size)
    
val_steps_per_epoch = int(steps_per_epoch / 5)
if steps_per_epoch == 0:
    steps_per_epoch = 1
if val_steps_per_epoch == 0:
    val_steps_per_epoch = 1   
## Load generator objects from filepaths
training_data_generator = array_generator(save_directory+"//training_data", batch_size, len(classhkl), loc_new)
testing_data_generator = array_generator(save_directory+"//testing_data", batch_size, len(classhkl), loc_new)
#%%
# =============================================================================
# Keras neural network
# =============================================================================
## load model and train
learning_rate, kernel_coeff, bias_coeff = 1e-3,1e-5,1e-6
model_keras = model_arch_general(len(angbins)-1, len(classhkl),
                                     kernel_coeff= None, bias_coeff=None, 
                                     lr=learning_rate)
######### TRAIN THE DATA
stats_model = model_keras.fit(
                            training_data_generator, 
                            epochs=epochs, 
                            steps_per_epoch=steps_per_epoch,
                            validation_data=testing_data_generator,
                            validation_steps=val_steps_per_epoch,
                            verbose=1,
                            class_weight=class_weights,
                            )
## lets predict on the test data
test_data = next(testing_data_generator)
class_testdata = np.argmax(test_data[1], axis = 1)     
groundtruth_hkl = classhkl[class_testdata]

prediction_keras = model_keras.predict(test_data[0])
class_predicted = np.argmax(prediction_keras, axis = 1)     
predicted_hkl = classhkl[class_predicted]
#%%
# =============================================================================
# NUMPY version of DNN code
# =============================================================================

import numpy as np
# =============================================================================
# Activation functions for forward and backward pass
# define new if required
# =============================================================================
def relu(Z):
    return np.maximum(0,Z)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0
    return dZ

def softmax(x):
    exps = np.exp(x - x.max())
    return exps / np.sum(exps, axis=0)

def softmax_backward(x, Z):
    exps = np.exp(x - x.max())
    return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))

# =============================================================================
# A full forward pass : OK works well
# =============================================================================
def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation="relu"):
    # calculation of the input value for the activation function
    # A_prev should be a vector and not matrix!!!!!!!
    Z_curr = np.dot(W_curr, A_prev) + b_curr
    # selection of activation function
    if activation == "relu":
        activation_func = relu
    elif activation == "softmax":
        activation_func = softmax
    else:
        raise Exception('Non-supported activation function')
    # return of calculated activation A and the intermediate Z matrix
    return activation_func(Z_curr), Z_curr

def full_forward_propagation(X, params_values, nn_architecture):
    # creating a temporary memory to store the information needed for a backward step
    memory = {}
    # X vector is the activation for layer 0 
    A_curr = X
    # iteration over network layers
    for idx, layer in enumerate(nn_architecture):
        # we number network layers from 1
        layer_idx = idx + 1
        # transfer the activation from the previous iteration
        A_prev = A_curr
        # extraction of the activation function for the current layer
        activ_function_curr = layer["activation"]
        # extraction of W for the current layer
        W_curr = params_values["W" + str(layer_idx)]
        # extraction of b for the current layer
        b_curr = params_values["b" + str(layer_idx)]
        # calculation of activation for the current layer
        A_curr, Z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)
        # saving calculated values in the memory
        memory["A" + str(idx)] = A_prev
        memory["Z" + str(layer_idx)] = Z_curr
    # return of prediction vector and a dictionary containing intermediate values
    return A_curr, memory

# =============================================================================
# A full Backward propagation pass
# =============================================================================
def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
    # number of examples
    m = A_prev.shape[1]
    
    # selection of activation function
    if activation == "relu":
        backward_activation_func = relu_backward
    elif activation == "softmax":
        backward_activation_func = softmax_backward
    else:
        raise Exception('Non-supported activation function')
    
    # calculation of the activation function derivative
    dZ_curr = backward_activation_func(dA_curr, Z_curr)   
    
    # derivative of the matrix W
    dW_curr = np.dot(dZ_curr, A_prev.T) / m
    # derivative of the vector b
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    # derivative of the matrix A_prev
    dA_prev = np.dot(W_curr.T, dZ_curr)

    return dA_prev, dW_curr, db_curr

def full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture):
    grads_values = {}
    # a hack ensuring the same shape of the prediction vector and labels vector
    # commented as this should be the case
    # Y = Y.reshape(Y_hat.shape)
    
    # initiation of gradient descent algorithm
    dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat));
    
    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        # we number network layers from 1
        layer_idx_curr = layer_idx_prev + 1
        # extraction of the activation function for the current layer
        activ_function_curr = layer["activation"]
        
        dA_curr = dA_prev
        
        A_prev = memory["A" + str(layer_idx_prev)]
        Z_curr = memory["Z" + str(layer_idx_curr)]
        
        W_curr = params_values["W" + str(layer_idx_curr)]
        b_curr = params_values["b" + str(layer_idx_curr)]
        
        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)
        
        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["db" + str(layer_idx_curr)] = db_curr
    
    return grads_values

def init_layers(nn_architecture, seed = 99, verbose=False):
    # random seed initiation
    np.random.seed(seed)
    # parameters storage initiation
    param_shape = {}
    params_values = {}
    # iteration over network layers
    for idx, layer in enumerate(nn_architecture):
        # we number network layers from 1
        layer_idx = idx + 1
        # extracting the number of units in layers
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]
        # initiating the values of the W matrix
        # and vector b for subsequent layers
        params_values['W' + str(layer_idx)] = np.random.randn(layer_output_size, layer_input_size) * 0.0005
        params_values['b' + str(layer_idx)] = np.random.randn(layer_output_size, 1) * 0.0005
        param_shape['W' + str(layer_idx)] = params_values['W' + str(layer_idx)].shape
        param_shape['b' + str(layer_idx)] = params_values['b' + str(layer_idx)].shape
    if verbose:
        print(param_shape)
    return params_values

# =============================================================================
# Optimizer
# using ADAm optimizer for learning
# =============================================================================
# 2 dependent values beta1 and beta2; beta 1is the exponential decay of the 
# rate for the first moment estimates, and its literature value is 0.9. beta2 is 
# the exponential decay rate for the second-moment estimates, and its literature value is 0.999
# compute m and v and update weights with it
# class AdamOptim():
#     def __init__(self, eta=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
#         self.m_dw, self.v_dw = 0, 0
#         self.m_db, self.v_db = 0, 0
#         self.beta1 = beta1
#         self.beta2 = beta2
#         self.epsilon = epsilon
#         self.eta = eta
#     def update(self, t, params_values, nn_architecture, dw, db):
#         ## dw, db are from current minibatch
#         for layer_idx, layer in enumerate(nn_architecture, 1):
#             w = params_values["W" + str(layer_idx)]
#             b = params_values["b" + str(layer_idx)]
            # ## momentum beta 1
            # # *** weights *** #
            # self.m_dw = self.beta1*self.m_dw + (1-self.beta1)*dw
            # # *** biases *** #
            # self.m_db = self.beta1*self.m_db + (1-self.beta1)*db
            # ## rms beta 2
            # # *** weights *** #
            # self.v_dw = self.beta2*self.v_dw + (1-self.beta2)*(dw**2)
            # # *** biases *** #
            # self.v_db = self.beta2*self.v_db + (1-self.beta2)*(db)
            # ## bias correction
            # m_dw_corr = self.m_dw/(1-self.beta1**t)
            # m_db_corr = self.m_db/(1-self.beta1**t)
            # v_dw_corr = self.v_dw/(1-self.beta2**t)
            # v_db_corr = self.v_db/(1-self.beta2**t)
            
#             # derivative of the matrix W
#             dW_curr = np.dot(dZ_curr, A_prev.T) / m
#             # derivative of the vector b
#             db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
#             # derivative of the matrix A_prev
#             dA_prev = np.dot(W_curr.T, dZ_curr)
            
#             ## update weights and biases
#             w = w - self.eta*(m_dw_corr/(np.sqrt(v_dw_corr)+self.epsilon))
#             b = b - self.eta*(m_db_corr/(np.sqrt(v_db_corr)+self.epsilon))
#         return w, b

# =============================================================================
# Update parameters
# =============================================================================
def update(params_values, grads_values, nn_architecture, learning_rate):
    # iteration over network layers
    for layer_idx, layer in enumerate(nn_architecture, 1):
        params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]        
        params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]
    return params_values

## Mulit-class cross entropy
# =============================================================================
# Loss function / cost function
# =============================================================================
def get_cost_value(outputs, targets, derivative=False, epsilon=1e-11 ):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and outputs. 
    Input: outputs (N, k) ndarray
           targets (N, k) ndarray        
    Returns: scalar
    """
    outputs = np.clip(outputs, epsilon, 1. - epsilon)
    N = outputs.shape[0]
    ce = -np.sum(targets*np.log(outputs+1e-9))/N
    return ce

# =============================================================================
# Calculating accuracy
# =============================================================================
def get_accuracy_value(Y_hat, Y):
    """ Compute Accuracy
      Input: (in one hot encoding)
          Y_hat: K X N numpy array (K: number of classes)
          Y:  K X N numpy array
      Output:
          acc: scalar
    """
    num_sample = Y_hat.shape[1]
    pred_idx = np.argmax(Y_hat, axis=0)
    gt_idx = np.argmax(Y, axis=0)
    acc = np.sum(np.equal(pred_idx, gt_idx).astype(float)) / num_sample
    return acc 


#%%

n_bins = len(angbins)-1
n_outputs = len(classhkl)
param = n_outputs
    
nn_architecture = [
                    {"input_dim": n_bins, "output_dim": n_bins, "activation": "relu"},
                    {"input_dim": n_bins, "output_dim": ((param)*15 + n_bins)//2, "activation": "relu"},
                    {"input_dim": ((param)*15 + n_bins)//2, "output_dim": (param)*15, "activation": "relu"},
                    {"input_dim": (param)*15, "output_dim": n_outputs, "activation": "softmax"},
                    ]


params_values = init_layers(nn_architecture, np.random.randint(1e6), verbose=True)

learning_rate = 0.001

verbose = True
# initiation of lists storing the history 
# of metrics calculated during the learning process 
cost_history = []
accuracy_history = []


# optimizer = AdamOptim(eta=0.001)

# performing calculations for subsequent iterations
for ii in range(5):
    for jj in range(steps_per_epoch):
        X, Y = next(training_data_generator)        
        X = X.reshape((X.shape[1], X.shape[0]))
        Y = Y.reshape((Y.shape[1], Y.shape[0]))
        
        Y_hat, cashe = full_forward_propagation(X, params_values, nn_architecture)
        
        # calculating metrics and saving them in history
        cost = get_cost_value(Y_hat, Y)
        cost_history.append(cost)

        accuracy = get_accuracy_value(Y_hat, Y)
        accuracy_history.append(accuracy)
        
        # step backward - calculating gradient
        grads_values = full_backward_propagation(Y_hat, Y, cashe, params_values, nn_architecture)
        
        # updating model state
        ## iteratively to check gradient
        params_values = update(params_values, grads_values, nn_architecture, learning_rate)
        #params_values = optimizer.update(ii+1, params_values, dw, db)

        if(verbose):
            print("Iteration: {:05} - cost: {:.5f} - accuracy: {:.5f}".format(jj, cost, accuracy))

















