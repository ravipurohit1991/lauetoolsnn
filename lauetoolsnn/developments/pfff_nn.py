# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 17:23:15 2022

@author: PURUSHOT


Numpy neural network implementation of the DNN model used in LaueNN GUI
Note: This does not include any kernel or bias regularization terms 
For larger output models please install tensorflow

Test script to get a working DNN model with numpy (from https://mlfromscratch.com/neural-network-tutorial/#/)

So far works well but is extremely slow in comparison to Keras

TODO
Softmax gradient
Adam optimizer for gradient
Dropout

"""
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

X, Y = next(training_data_generator)
for jj in range(steps_per_epoch-1):
    X1, Y1 = next(training_data_generator)
    X = np.vstack((X,X1))
    Y = np.vstack((Y,Y1))

#%%
import numpy as np
from sklearn.model_selection import train_test_split
import time

x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.15, random_state=42)

class DeepNeuralNetwork():
    def __init__(self, sizes, epochs=10, l_rate=0.001):
        self.sizes = sizes
        self.epochs = epochs
        self.l_rate = l_rate
        # we save all parameters in the neural network in this dictionary
        self.params = self.initialization()

    def ReLU(self, x, derivative=False):
        if derivative:
            return 1 * (x > 0)
        return np.maximum(0,x)

    def softmax(self, x, derivative=False, array=False):
        # Numerically stable with large exponentials
        if array:
            exps = np.exp(x.T - x.max(axis=1))
        else:
            exps = np.exp(x - x.max())
            
        if derivative:
            return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)

    def initialization(self):
        # number of nodes in each layer
        input_layer = self.sizes[0]
        hidden_1 = self.sizes[1]
        hidden_2 = self.sizes[2]
        hidden_3 = self.sizes[3]
        output_layer = self.sizes[4]
        params = {
                'W1':np.random.randn(hidden_1, input_layer) * np.sqrt(1. / hidden_1),
                'W2':np.random.randn(hidden_2, hidden_1) * np.sqrt(1. / hidden_2),
                'W3':np.random.randn(hidden_3, hidden_2) * np.sqrt(1. / hidden_3),
                'W4':np.random.randn(output_layer, hidden_3) * np.sqrt(1. / output_layer)
                }
        return params

    def forward_pass(self, x_train):
        params = self.params

        # input layer activations becomes sample
        params['A0'] = x_train

        # input layer to hidden layer 1
        params['Z1'] = np.dot(params["W1"], params['A0'])
        params['A1'] = self.ReLU(params['Z1'])

        # hidden layer 1 to hidden layer 2
        params['Z2'] = np.dot(params["W2"], params['A1'])
        params['A2'] = self.ReLU(params['Z2'])
        
        # hidden layer 2 to hidden layer 3
        params['Z3'] = np.dot(params["W3"], params['A2'])
        params['A3'] = self.ReLU(params['Z3'])
        
        # hidden layer 3 to output layer
        params['Z4'] = np.dot(params["W4"], params['A3'])
        params['A4'] = self.softmax(params['Z4'])

        return params['A4']
    
    def predict(self, x_train): ## array version of forward pass
        params = self.params

        # input layer activations becomes sample
        params['A0'] = x_train

        # input layer to hidden layer 1
        params['Z1'] = np.dot(params['A0'], params["W1"].T)
        params['A1'] = self.ReLU(params['Z1'])

        # hidden layer 1 to hidden layer 2
        params['Z2'] = np.dot(params['A1'], params["W2"].T)
        params['A2'] = self.ReLU(params['Z2'])
        
        # hidden layer 2 to hidden layer 3
        params['Z3'] = np.dot(params['A2'], params["W3"].T)
        params['A3'] = self.ReLU(params['Z3'])
        
        # hidden layer 3 to output layer
        params['Z4'] = np.dot(params['A3'], params["W4"].T)
        params['A4'] = self.softmax(params['Z4'], array=True)

        return params['A4']
    
    def backward_pass(self, y_train, output):
        '''
            This is the backpropagation algorithm, for calculating the updates
            of the neural network's parameters.

            Note: There is a stability issue that causes warnings. This is 
                  caused  by the dot and multiply operations on the huge arrays.
                  
                  RuntimeWarning: invalid value encountered in true_divide
                  RuntimeWarning: overflow encountered in exp
                  RuntimeWarning: overflow encountered in square
        '''
        params = self.params
        change_w = {}

        # Calculate W4 update
        error = 2 * (output - y_train) / output.shape[0] * self.softmax(params['Z4'], derivative=True)
        change_w['W4'] = np.outer(error, params['A3'])
        
        # Calculate W3 update
        error = np.dot(params['W4'].T, error) * self.ReLU(params['Z3'], derivative=True)
        change_w['W3'] = np.outer(error, params['A2'])
        
        # Calculate W2 update
        error = np.dot(params['W3'].T, error) * self.ReLU(params['Z2'], derivative=True)
        change_w['W2'] = np.outer(error, params['A1'])

        # Calculate W1 update
        error = np.dot(params['W2'].T, error) * self.ReLU(params['Z1'], derivative=True)
        change_w['W1'] = np.outer(error, params['A0'])

        return change_w

    def update_network_parameters(self, changes_to_w):
        '''
            Update network parameters according to update rule from
            Stochastic Gradient Descent.
            θ = θ - η * ∇J(x, y), 
                theta θ:            a network parameter (e.g. a weight w)
                eta η:              the learning rate
                gradient ∇J(x, y):  the gradient of the objective function,
                                    i.e. the change for a specific theta θ
        '''
        
        for key, value in changes_to_w.items():
            self.params[key] -= self.l_rate * value
    
    def compute_accuracy(self, x_val, y_val):
        '''
            This function does a forward pass of x, then checks if the indices
            of the maximum value in the output equals the indices in the label
            y. Then it sums over each prediction and calculates the accuracy.
        '''
        num_sample = x_val.shape[0]
        output = self.predict(x_val)
        pred_idx = np.argmax(output, axis=1)
        gt_idx = np.argmax(y_val, axis=1)
        acc = np.sum(np.equal(pred_idx, gt_idx).astype(float)) / num_sample
        loss = self.get_cost_value(output, y_val)
        return acc, loss
    
    def get_cost_value(self, outputs, targets, epsilon=1e-11 ):
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
    
    def compute_accuracy_v1(self, x_val, y_val):
        '''
            This function does a forward pass of x, then checks if the indices
            of the maximum value in the output equals the indices in the label
            y. Then it sums over each prediction and calculates the accuracy.
        '''
        predictions = []
        loss = []
        for x, y in zip(x_val, y_val):
            output = self.forward_pass(x)
            pred = np.argmax(output)
            predictions.append(pred == np.argmax(y))
            loss.append(self.get_cost_value(output, y))
        loss = -np.sum(loss)/len(loss)
        return np.mean(predictions), loss
    
    def get_cost_value_v1(self, y_pred, y_val, epsilon=1e-11 ):
        """
        Computes cross entropy between targets (encoded as one-hot vectors)
        and outputs. 
        Input: outputs (N, k) ndarray
                targets (N, k) ndarray        
        Returns: scalar
        """
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        ce = np.sum(y_val*np.log(y_pred+1e-9))
        return ce
    
    def train(self, x_train, y_train, x_val, y_val):
        start_time = time.time()
        for iteration in range(self.epochs):
            for x,y in zip(x_train, y_train):
                output = self.forward_pass(x)
                changes_to_w = self.backward_pass(y, output)
                self.update_network_parameters(changes_to_w)
            
            accuracy, loss = self.compute_accuracy(x_val, y_val)
            print('Epoch: {0}, Iter: {1}, Time Spent: {2:.2f}s, Loss: {3:.5f}, Accuracy: {4:.2f}%'.format(
                iteration+1, 1, time.time() - start_time, loss, accuracy * 100
            ))

n_bins = len(angbins)-1
n_outputs = len(classhkl)
param = n_outputs
#%%
dnn = DeepNeuralNetwork(sizes=[n_bins, n_bins, ((param)*15 + n_bins)//2, (param)*15, n_outputs])
dnn.train(x_train, y_train, x_val, y_val)


#%%# lets predict on the test data
class_testdata = np.argmax(y_val[0])     
groundtruth_hkl = classhkl[class_testdata]
print(groundtruth_hkl)
prediction_keras = dnn.predict(x_val)
class_predicted = np.argmax(prediction_keras, axis=1)     
predicted_hkl = classhkl[class_predicted]
print(predicted_hkl[0])




