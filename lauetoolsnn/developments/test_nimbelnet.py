# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 21:30:15 2022

@author: PURUSHOT

Test nimbelnet function
"""

from nimbelnet_1D_DNN import softmax_categorical_cross_entropy_cost, categorical_accuracy
from nimbelnet_1D_DNN import ReLU_function, softmax_function
from nimbelnet_1D_DNN import print_test, Adam, NeuralNet


# =============================================================================
# Data to be used for both Keras model and Numpy model to evalute the performance
# a simple Si crystal model with 1 grain and 200 simulations with 9output classes
# =============================================================================
import sys
sys.path.append(r"C:\Users\purushot\Desktop\github_version_simple\lauetoolsnn")
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


n_bins = len(angbins)-1
n_outputs = len(classhkl)
param = n_outputs
#%%
# Training set
# dataset             = [ Instance( [0,0], [0] ), Instance( [1,0], [1] ), Instance( [0,1], [1] ), Instance( [1,1], [1] ) ]
# preprocess          = construct_preprocessor( dataset, [standarize] ) 
# training_data       = preprocess( dataset )
# test_data           = preprocess( dataset )


settings = {
            # Required settings
            "n_inputs"              : n_bins,       # Number of network input signals
            "layers"                : [  
                                       (n_bins, ReLU_function), 
                                       (((param)*15 + n_bins)//2, ReLU_function), 
                                       ((param)*15, ReLU_function), 
                                       (1, softmax_function)    
                                       ],
            # Optional settings
            "initial_bias_value"    : 0.0,
            "weights_low"           : -0.1,     # Lower bound on the initial weight value
            "weights_high"          : 0.1,      # Upper bound on the initial weight value
            }
# initialize the neural network
network             = NeuralNet( settings )
network.check_gradient(next(training_data_generator), softmax_categorical_cross_entropy_cost )

## load a stored network configuration
# network = NeuralNet.load_network_from_file( "network0.pkl" )

# Train the network using backpropagation with Adam optimizer
Adam(
        network,                            # the network to train
        training_data,                      # specify the training set
        test_data,                          # specify the test set
        softmax_categorical_cross_entropy_cost,                      # specify the cost function to calculate error
        ERROR_LIMIT             = 1e-2,     # define an acceptable error limit 
        max_iterations          = epochs,      # continues until the error limit is reach if this argument is skipped
        batch_size              = batch_size,        # 1 := no batch learning, 0 := entire trainingset as a batch, anything else := batch size
        print_rate              = 1,     # print error status every `print_rate` epoch.
        learning_rate           = 0.001,      # learning rate
        momentum_factor         = 0.9,      # momentum
        input_layer_dropout     = 0.3,      # dropout fraction of the input layer
        hidden_layer_dropout    = 0.3,      # dropout fraction in all hidden layers
        save_trained_network    = False     # Whether to write the trained weights to disk
    )

# Print a network test
print_test(network, training_data, softmax_categorical_cross_entropy_cost)


"""
Prediction Example
"""
prediction_set = [ Instance([0,1]), Instance([1,0]) ]
prediction_set = preprocess( prediction_set )
print(network.predict( prediction_set )) # produce the output signal