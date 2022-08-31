# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 20:44:38 2022

@author: PURUSHOT

nimbelnets functions

"""

import numpy as np
import copy
import math
import random

# =============================================================================
# Preprocessor functions
# =============================================================================
def construct_preprocessor( trainingset, list_of_processors, **kwargs ):
    combined_processor = lambda x: x
    for entry in list_of_processors:
        if type(entry) == tuple:
            processor, processor_configuration = entry
            assert type(processor_configuration) == dict, \
                "The second argument to a preprocessor entry must be a dictionary of settings."
            combined_processor = processor( combined_processor( trainingset ), **processor_configuration )
        else:
            processor = entry
            combined_processor = processor( combined_processor( trainingset ))
    return lambda dataset: combined_processor( copy.deepcopy( dataset ))

def standarize( trainingset ):
    """
    Morph the input signal to a mean of 0 and scale the signal strength by 
    dividing with the standard deviation (rather that forcing a [0, 1] range)
    """
    def encoder( dataset ):
        for instance in dataset:
            if np.any(stds == 0):
                nonzero_indexes = np.where(stds!=0)
                instance.features[nonzero_indexes] = (instance.features[nonzero_indexes] - means[nonzero_indexes]) / stds[nonzero_indexes]
            else:
                instance.features = (instance.features - means) / stds
        return dataset
    training_data = np.array( [instance.features for instance in trainingset ] )
    means = training_data.mean(axis=0)
    stds = training_data.std(axis=0)
    return encoder

# =============================================================================
# constructor instance
# =============================================================================
class Instance:
    # This is a simple encapsulation of a `input signal : output signal`
    # pair in our training set.
    def __init__(self, features, target = None ):
        self.features = np.array(features)
        
        if target != None:
            self.targets  = np.array(target)
        else:
            self.targets  = None
            
# =============================================================================
# Activation functions
# =============================================================================
def ReLU_function( signal, derivative=False ):
    if derivative:
        return (signal > 0).astype(float)
    else:
        # Return the activation signal
        return np.maximum( 0, signal )

def softmax_function( signal, derivative=False ):
    # Calculate activation signal
    e_x = np.exp( signal - np.max(signal, axis=1, keepdims = True) )
    signal = e_x / np.sum( e_x, axis = 1, keepdims = True )
    if derivative:
        return np.ones( signal.shape )
    else:
        # Return the activation signal
        return signal

# =============================================================================
# Loss function
# =============================================================================
def softmax_categorical_cross_entropy_cost( outputs, targets, derivative=False, epsilon=1e-11 ):
    """
    The output signals should be in the range [0, 1]
    """
    outputs = np.clip(outputs, epsilon, 1 - epsilon)
    if derivative:
        return outputs - targets
    else:
        return np.mean(-np.sum(targets * np.log( outputs ), axis=1))

# =============================================================================
# Evaluation function
# =============================================================================
def categorical_accuracy( outputs, targets ):
    return 1.0 - 1.0 * np.count_nonzero(np.argmax(outputs, axis=1) == np.argmax(targets, axis=1)) / outputs.shape[0]

# =============================================================================
# Misc functions
# =============================================================================
def print_test( network, testset, cost_function ):
    assert testset[0].features.shape[0] == network.n_inputs, \
        "ERROR: input size varies from the defined input setting"
    assert testset[0].targets.shape[0]  == network.layers[-1][0], \
        "ERROR: output size varies from the defined output setting"
    
    test_data              = np.array( [instance.features for instance in testset ] )
    test_targets           = np.array( [instance.targets  for instance in testset ] )
    
    input_signals, derivatives = network.update( test_data, trace=True )
    out                        = input_signals[-1]
    error                      = cost_function(out, test_targets )
    
    print( "[testing] Network error: %.4g" % error)
    print( "[testing] Network results:")
    print( "[testing]   input\tresult\ttarget")
    for entry, result, target in zip(test_data, out, test_targets):
        print( "[testing]   %s\t%s\t%s" % tuple(map(str, [entry, result, target])))
    

def dropout( X, p = 0. ):
    if p != 0:
        retain_p = 1 - p
        X = X * np.random.binomial(1,retain_p,size = X.shape)
        X /= retain_p
    return X


def add_bias(A):
    # Add a bias value of 1. The value of the bias is adjusted through
    # weights rather than modifying the input signal.
    return np.hstack(( np.ones((A.shape[0],1)), A ))


def confirm( promt='Do you want to continue?' ):
	prompt = '%s [%s|%s]: ' % (promt,'y','n')
	while True:
		ans = input(prompt).lower()
		if ans in ['y','yes']:
			return True
		if ans in ['n','no']:
			return False
		print( "Please enter y or n.")

# def check_network_structure(network, cost_function ):
#     assert softmax_function != network.layers[-1][1] or cost_function == softmax_categorical_cross_entropy_cost,\
#         "When using the `softmax` activation function, the cost function MUST be `softmax_neg_loss`."
#     assert cost_function != softmax_categorical_cross_entropy_cost or softmax_function == network.layers[-1][1],\
#         "When using the `softmax_neg_loss` cost function, the activation function in the final layer MUST be `softmax`."

# def verify_dataset_shape_and_modify( network, random_data_features, random_data_targets ):   
#     ## for generators version
#     assert random_data_features.shape[1] == network.n_inputs, "ERROR: input size varies from the defined input setting"
#     assert random_data_targets.shape[1]  == network.layers[-1][0], "ERROR: output size varies from the defined output setting"
#     return random_data_features, random_data_targets

# def verify_dataset_shape_and_modify( network, dataset ):   
#     assert dataset[0].features.shape[0] == network.n_inputs, "ERROR: input size varies from the defined input setting"
#     assert dataset[0].targets.shape[0]  == network.layers[-1][0], "ERROR: output size varies from the defined output setting"
#     data              = np.array( [instance.features for instance in dataset ] )
#     targets           = np.array( [instance.targets  for instance in dataset ] )
#     return data, targets 

# def apply_regularizers( dataset, cost_function, regularizers, network ):
#     dW_regularizer = lambda x: np.zeros( shape = x.shape )
#     if regularizers != None:
#         # Modify the cost function to add the regularizer
#         for entry in regularizers:
#             if type(entry) == tuple:
#                 regularizer, regularizer_settings = entry
#                 cost_function, dW_regularizer  = regularizer( dataset, cost_function, dW_regularizer, network, **regularizer_settings )
#             else:
#                 regularizer    = entry
#                 cost_function, dW_regularizer  = regularizer( dataset, cost_function, dW_regularizer, network )
#     return cost_function, dW_regularizer
# =============================================================================
# The NETWORK
# =============================================================================

class NeuralNet:
    def __init__(self, settings ):
        # self.__dict__.update( default_settings )
        self.__dict__.update( settings )
        
        assert not softmax_function in map(lambda x: x[1], self.layers) or softmax_function == self.layers[-1][1],\
            "The `softmax` activation function may only be used in the final layer."
        
        # Count the required number of weights. This will speed up the random number generation when initializing weights
        self.n_weights = (self.n_inputs + 1) * self.layers[0][0] +\
                         sum( (self.layers[i][0] + 1) * layer[0] for i, layer in enumerate( self.layers[1:] ) )
        
        # Initialize the network with new randomized weights
        self.set_weights( self.generate_weights( self.weights_low, self.weights_high ) )
        
        # Initalize the bias to 0.01
        for index in range(len(self.layers)):
            self.weights[index][:1,:] = self.initial_bias_value    
    
    def generate_weights(self, low = -0.1, high = 0.1):
        # Generate new random weights for all the connections in the network
        return np.random.uniform(low, high, size=(self.n_weights,))    
    
    def set_weights(self, weight_list ):
        # This is a helper method for setting the network weights to a previously defined list
        # as it's useful for loading a previously optimized neural network weight set.
        # The method creates a list of weight matrices. Each list entry correspond to the 
        # connection between two layers.
        start, stop         = 0, 0
        self.weights        = [ ]
        previous_shape      = self.n_inputs + 1 # +1 because of the bias
        for n_neurons, activation_function in self.layers:
            stop           += previous_shape * n_neurons
            self.weights.append( weight_list[ start:stop ].reshape( previous_shape, n_neurons ))
            previous_shape  = n_neurons + 1     # +1 because of the bias
            start           = stop    
    
    def get_weights(self, ):
        # This will stack all the weights in the network on a list, which may be saved to the disk.
        return [w for l in self.weights for w in l.flat]    
    
    def error(self, weight_vector, training_data, training_targets, cost_function ):
        # assign the weight_vector as the network topology
        self.set_weights( np.array(weight_vector) )
        # perform a forward operation to calculate the output signal
        out = self.update( training_data )
        # evaluate the output signal with the cost function
        return cost_function(out, training_targets )    
    
    def measure_quality(self, training_data, training_targets, cost_function ):
        # perform a forward operation to calculate the output signal
        out = self.update( training_data )
        # calculate the mean error on the data classification
        mean_error = cost_function( out, training_targets ) / float(training_data.shape[0])
        # calculate the numeric range between the minimum and maximum output value
        range_of_predicted_values = np.max(out) - np.min(out)
        # return the measured quality 
        return 1 - (mean_error / range_of_predicted_values)    
    
    def gradient(self, weight_vector, training_data, training_targets, cost_function ):
        # assign the weight_vector as the network topology
        self.set_weights( np.array(weight_vector) )
        input_signals, derivatives  = self.update( training_data, trace=True )                  
        out                         = input_signals[-1]
        cost_derivative             = cost_function(out, training_targets, derivative=True).T
        delta                       = cost_derivative * derivatives[-1]
        layer_indexes               = range( len(self.layers) )[::-1]    # reversed
        n_samples                   = float(training_data.shape[0])
        deltas_by_layer             = []
        
        for i in layer_indexes:
            # Loop over the weight layers in reversed order to calculate the deltas
            deltas_by_layer.append(list((np.dot( delta, add_bias(input_signals[i]) )/n_samples).T.flat))
            if i!= 0:
                # i!= 0 because we don't want calculate the delta unnecessarily.
                weight_delta        = np.dot( self.weights[ i ][1:,:], delta ) # Skip the bias weight
                # Calculate the delta for the subsequent layer
                delta               = weight_delta * derivatives[i-1]        
        return np.hstack( reversed(deltas_by_layer) )    
    
    def check_gradient(self, trainingset, cost_function, epsilon = 1e-4 ):
        # check_network_structure(self, cost_function ) # check for special case topology requirements, such as softmax
        training_data, training_targets = trainingset #verify_dataset_shape_and_modify( self, trainingset_features,  trainingset_target)
        # assign the weight_vector as the network topology
        initial_weights         = np.array(self.get_weights())
        numeric_gradient        = np.zeros( initial_weights.shape )
        perturbed               = np.zeros( initial_weights.shape )
        
        print ("[gradient check] Running gradient check...")
        
        for i in range( self.n_weights ):
            perturbed[i]        = epsilon
            right_side          = self.error( initial_weights + perturbed, training_data, training_targets, cost_function )
            left_side           = self.error( initial_weights - perturbed, training_data, training_targets, cost_function )
            numeric_gradient[i] = (right_side - left_side) / (2 * epsilon)
            perturbed[i]        = 0
        
        # Reset the weights
        self.set_weights( initial_weights )
        # Calculate the analytic gradient
        analytic_gradient       = self.gradient( self.get_weights(), training_data, training_targets, cost_function )
        # Compare the numeric and the analytic gradient
        ratio                   = np.linalg.norm(analytic_gradient - numeric_gradient) / np.linalg.norm(analytic_gradient + numeric_gradient)
        if not ratio < 1e-6:
            print( "[gradient check] WARNING: The numeric gradient check failed! Analytical gradient differed by %g from the numerical." % ratio)
            if not confirm("[gradient check] Do you want to continue?"):
                print( "[gradient check] Exiting.")
                import sys
                sys.exit(2)
        else:
            print( "[gradient check] Passed!")
        return ratio    
    
    def update(self, input_values, trace=False ):
        # This is a forward operation in the network. This is how we 
        # calculate the network output from a set of input signals.
        output          = input_values
        if trace: 
            derivatives = [ ]        # collection of the derivatives of the act functions
            outputs     = [ output ] # passed through act. func.
        
        for i, weight_layer in enumerate(self.weights):
            # Loop over the network layers and calculate the output
            signal      = np.dot( output, weight_layer[1:,:] ) + weight_layer[0:1,:] # implicit bias
            output      = self.layers[i][1]( signal )
            if trace: 
                outputs.append( output )
                derivatives.append( self.layers[i][1]( signal, derivative = True ).T ) # the derivative used for weight update
        if trace: 
            return outputs, derivatives
        return output    
    
    def predict(self, predict_set ):
        """
        This method accepts a list of Instances
        
        Eg: list_of_inputs = [ Instance([0.12, 0.54, 0.84]), Instance([0.15, 0.29, 0.49]) ]
        """
        predict_data           = np.array([instance.features for instance in predict_set ] )
        return self.update( predict_data )
    
    def save_network_to_file(self, filename = "network0.pkl" ):
        import cPickle, os, re
        """
        This save method pickles the parameters of the current network into a 
        binary file for persistant storage.
        """
        if filename == "network0.pkl":
            while os.path.exists( os.path.join(os.getcwd(), filename )):
                filename = re.sub('\d(?!\d)', lambda x: str(int(x.group(0)) + 1), filename)
    
        with open( filename , 'wb') as file:
            store_dict = {
                "n_inputs"             : self.n_inputs,
                "layers"               : self.layers,
                "n_weights"            : self.n_weights,
                "weights"              : self.weights,
            }
            cPickle.dump( store_dict, file, 2 )

    @staticmethod
    def load_network_from_file( filename ):
        import cPickle
        """
        Load the complete configuration of a previously stored network.
        """
        network = NeuralNet( {"n_inputs":1, "layers":[[0,None]]} )
    
        with open( filename , 'rb') as file:
            store_dict                   = cPickle.load(file)
        
            network.n_inputs             = store_dict["n_inputs"]            
            network.n_weights            = store_dict["n_weights"]           
            network.layers               = store_dict["layers"]
            network.weights              = store_dict["weights"]             
    
        return network

# =============================================================================
#     Learning function
# =============================================================================
def backpropagation_foundation(network, trainingset, testset, cost_function, calculate_dW, evaluation_function = None, ERROR_LIMIT = 1e-3, max_iterations = (), batch_size = 0, input_layer_dropout = 0.0, hidden_layer_dropout = 0.0, print_rate = 1000, save_trained_network = False, **kwargs):
    # check_network_structure( network, cost_function ) # check for special case topology requirements, such as softmax
    
    training_data, training_targets = trainingset #verify_dataset_shape_and_modify( network, trainingset )
    test_data, test_targets    = testset #verify_dataset_shape_and_modify( network, testset)
    
    # Whether to use another function for printing the dataset error than the cost function. 
    # This is useful if you train the network with the MSE cost function, but are going to 
    # classify rather than regress on your data.
    if evaluation_function != None:
        calculate_print_error = evaluation_function
    else:
        calculate_print_error = cost_function
    
    batch_size                 = batch_size if batch_size != 0 else training_data.shape[0] 
    batch_training_data        = np.array_split(training_data, math.ceil(1.0 * training_data.shape[0] / batch_size))
    batch_training_targets     = np.array_split(training_targets, math.ceil(1.0 * training_targets.shape[0] / batch_size))
    batch_indices              = range(len(batch_training_data))       # fast reference to batches
    error                      = calculate_print_error(network.update( test_data ), test_targets )
    reversed_layer_indexes     = range( len(network.layers) )[::-1]
    
    epoch                      = 0
    while error > ERROR_LIMIT and epoch < max_iterations:
        epoch += 1
        
        random.shuffle(batch_indices) # Shuffle the order in which the batches are processed between the iterations
        
        for batch_index in batch_indices:
            batch_data                 = batch_training_data[    batch_index ]
            batch_targets              = batch_training_targets[ batch_index ]
            batch_size                 = float( batch_data.shape[0] )
            input_signals, derivatives = network.update( batch_data, trace=True )
            out                        = input_signals[-1]
            cost_derivative            = cost_function( out, batch_targets, derivative=True ).T
            delta                      = cost_derivative * derivatives[-1]
            
            for i in reversed_layer_indexes:
                # Loop over the weight layers in reversed order to calculate the deltas
                # perform dropout
                dropped = dropout( 
                                    input_signals[i], 
                                    # dropout probability
                                    hidden_layer_dropout if i > 0 else input_layer_dropout
                                )
            
                # calculate the weight change
                dX = (np.dot( delta, add_bias(dropped) )/batch_size).T
                dW = calculate_dW( i, dX )
                
                if i != 0:
                    """Do not calculate the delta unnecessarily."""
                    # Skip the bias weight
                    weight_delta = np.dot( network.weights[ i ][1:,:], delta )
                    # Calculate the delta for the subsequent layer
                    delta = weight_delta * derivatives[i-1]
                # Update the weights with Nestrov Momentum
                network.weights[ i ] += dW
            #end weight adjustment loop
        error = calculate_print_error(network.update( test_data ), test_targets )
        if epoch%print_rate==0:
            # Show the current training status
            print ("[training] Current error:", error, "\tEpoch:", epoch)
    print ("[training] Finished:")
    print ("[training]   Converged to error bound (%.4g) with error %.4g." % ( ERROR_LIMIT, error ))
    print ("[training]   Measured quality: %.4g" % network.measure_quality( training_data, training_targets, cost_function ))
    print ("[training]   Trained for %d epochs." % epoch)
    if save_trained_network:
        network.save_network_to_file()

default_configuration = {
    'ERROR_LIMIT'           : 0.001, 
    'learning_rate'         : 0.001, 
    'batch_size'            : 50, 
    'print_rate'            : 1, 
    'save_trained_network'  : False,
    'input_layer_dropout'   : 0.3,
    'hidden_layer_dropout'  : 0.3, 
    'evaluation_function'   : None,
    'max_iterations'        : 5
}

def Adam(network, trainingset, testset, cost_function, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, **kwargs ):
    configuration = dict(default_configuration)
    configuration.update( kwargs )
    learning_rate = configuration["learning_rate"]
    m = [ np.zeros( shape = weight_layer.shape ) for weight_layer in network.weights ]
    v = [ np.zeros( shape = weight_layer.shape ) for weight_layer in network.weights ]
    
    def calculate_dW( layer_index, dX ):
        m[ layer_index ] = beta1 * m[ layer_index ] + ( 1 - beta1 ) * dX
        v[ layer_index ] = beta2 * v[ layer_index ] + ( 1 - beta2 ) * ( dX**2 )
        return -learning_rate * m[ layer_index ] / ( np.sqrt(v[ layer_index ]) + epsilon )
    
    return backpropagation_foundation( network, trainingset, testset, cost_function, calculate_dW, **configuration  )























