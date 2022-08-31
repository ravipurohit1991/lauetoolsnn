# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 17:23:15 2022

@author: PURUSHOT


Numpy neural network implementation of the DNN model used in LaueNN GUI
Note: This does not include any kernel or bias regularization terms 
For larger output models please install tensorflow

Test script to get a working DNN model with numpy (from https://mlfromscratch.com/neural-network-tutorial/#/)
                                                   Modified version of Array for batch processing

So far works well but is slow in comparison to Keras

TODO
Softmax gradient
Adam optimizer for gradient
Dropout
"""
import numpy as np
import time

class DeepNeuralNetwork():
    def __init__(self, sizes, epochs=5, l_rate=0.001):
        self.sizes = sizes
        self.epochs = epochs
        self.l_rate = l_rate
        # we save all parameters in the neural network in this dictionary
        self.params = self.initialization()

    def ReLU(self, x, derivative=False):
        if derivative:
            return 1 * (x > 0)
        return np.maximum(0,x)

    def softmax(self, x, derivative=False):
        # Numerically stable with large exponentials
        exps = np.exp(x.T - x.max(axis=1))
        if derivative:
            return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)
    
    def initialization(self):
        # number of nodes in each layer
        input_layer=self.sizes[0]
        hidden_1=self.sizes[1]
        hidden_2=self.sizes[2]
        hidden_3=self.sizes[3]
        output_layer=self.sizes[4]

        params = {
                    'W1':np.random.randn(hidden_1, input_layer) * np.sqrt(1. / hidden_1),
                    'W2':np.random.randn(hidden_2, hidden_1) * np.sqrt(1. / hidden_2),
                    'W3':np.random.randn(hidden_3, hidden_2) * np.sqrt(1. / hidden_3),
                    'W4':np.random.randn(output_layer, hidden_3) * np.sqrt(1. / output_layer),
                    'B1':np.random.randn(hidden_1) * np.sqrt(1. / hidden_1),
                    'B2':np.random.randn(hidden_2) * np.sqrt(1. / hidden_2),
                    'B3':np.random.randn(hidden_3) * np.sqrt(1. / hidden_3),
                    'B4':np.random.randn(output_layer) * np.sqrt(1. / output_layer)
                }

        return params

    def forward_pass(self, x_train, y_train=None, compute_metric=False):
        params = self.params

        # input layer activations becomes sample
        params['A0'] = x_train

        # input layer to hidden layer 1
        params['Z1'] = np.dot(params['A0'], params["W1"].T) + params['B1']
        params['A1'] = self.ReLU(params['Z1'])

        # hidden layer 1 to hidden layer 2
        params['Z2'] = np.dot(params['A1'], params["W2"].T) + params['B2']
        params['A2'] = self.ReLU(params['Z2'])
        
        # hidden layer 2 to hidden layer 3
        params['Z3'] = np.dot(params['A2'], params["W3"].T) + params['B3']
        params['A3'] = self.ReLU(params['Z3'])
        
        # hidden layer 3 to output layer
        params['Z4'] = np.dot(params['A3'], params["W4"].T) + params['B4']
        params['A4'] = self.softmax(params['Z4'])
        
        if compute_metric:
            num_sample = x_train.shape[0]
            pred_idx = np.argmax(params['A4'], axis=0)
            gt_idx = np.argmax(y_train, axis=1)
            acc = np.sum(np.equal(pred_idx, gt_idx).astype(float)) / num_sample
            loss = self.get_cost_value(params['A4'], y_train)
            return params['A4'], acc, loss
        else:
            return params['A4']
    
    def predict(self, x_train):
        return self.forward_pass(x_train, y_train=None, compute_metric=False)
    
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
        change_b = {}
        change_w1 = {}
        change_b1 = {}
        # Calculate W4 update
        # print(output.shape, y_train.T.shape)
        error = 2 * (output - y_train.T) / output.shape[0] * self.softmax(params['Z4'], derivative=True)
        # print(error.shape, params['A3'].shape)
        change_w['W4'] = np.dot(error, params['A3'])
        change_b['B4'] = np.sum(error, axis=1, keepdims=True) / params['A3'].shape[0]
        # print(change_b['B4'].shape)
        ## momentum beta1, beta2
        self.change_adam['m_dw4'] = 0.9*self.change_adam['m_dw4'] + (1-0.9)*change_w['W4']
        self.change_adam['m_db4'] = 0.9*self.change_adam['m_db4'] + (1-0.9)*change_b['B4']
        self.change_adam['v_dw4'] = 0.999*self.change_adam['v_dw4'] + (1-0.999)*(change_w['W4']**2)
        self.change_adam['v_db4'] = 0.999*self.change_adam['v_db4'] + (1-0.999)*(change_b['B4'])
        ## bias correction
        self.change_adam['m_dw_corr4'] = self.change_adam['m_dw4']/(1-0.9**self.t)
        self.change_adam['m_db_corr4'] = self.change_adam['m_db4']/(1-0.9**self.t)
        self.change_adam['v_dw_corr4'] = self.change_adam['v_dw4']/(1-0.999**self.t)
        self.change_adam['v_db_corr4'] = self.change_adam['v_db4']/(1-0.999**self.t)
        
        change_w1['W4'] = (self.change_adam['m_dw_corr4']/(np.sqrt(self.change_adam['v_dw_corr4'])+1e-8))
        change_b1['B4'] = (self.change_adam['m_db_corr4']/(np.sqrt(self.change_adam['v_db_corr4'])+1e-8))

        # Calculate W3 update
        # print(error.T.shape, params['W4'].shape)
        error = np.dot(error.T, params['W4'] ) * self.ReLU(params['Z3'], derivative=True)
        change_w['W3'] = np.dot(error.T, params['A2'])
        change_b['B3'] = np.sum(error.T, axis=1, keepdims=True) / params['A2'].shape[0]
        # print(change_b['B3'].shape)
        ## momentum beta1, beta2
        self.change_adam['m_dw3'] = 0.9*self.change_adam['m_dw3'] + (1-0.9)*change_w['W3']
        self.change_adam['m_db3'] = 0.9*self.change_adam['m_db3'] + (1-0.9)*change_b['B3']
        self.change_adam['v_dw3'] = 0.999*self.change_adam['v_dw3'] + (1-0.999)*(change_w['W3']**2)
        self.change_adam['v_db3'] = 0.999*self.change_adam['v_db3'] + (1-0.999)*(change_b['B3'])
        ## bias correction
        self.change_adam['m_dw_corr3'] = self.change_adam['m_dw3']/(1-0.9**self.t)
        self.change_adam['m_db_corr3'] = self.change_adam['m_db3']/(1-0.9**self.t)
        self.change_adam['v_dw_corr3'] = self.change_adam['v_dw3']/(1-0.999**self.t)
        self.change_adam['v_db_corr3'] = self.change_adam['v_db3']/(1-0.999**self.t)
        
        change_w1['W3'] = (self.change_adam['m_dw_corr3']/(np.sqrt(self.change_adam['v_dw_corr3'])+1e-8))
        change_b1['B3'] = (self.change_adam['m_db_corr3']/(np.sqrt(self.change_adam['v_db_corr3'])+1e-8))
        
        # Calculate W2 update
        # print(error.shape, params['W3'].shape)
        error = np.dot(error, params['W3']) * self.ReLU(params['Z2'], derivative=True)
        change_w['W2'] = np.dot(error.T, params['A1'])
        change_b['B2'] = np.sum(error.T, axis=1, keepdims=True) / params['A1'].shape[0]
        # print(change_b['B2'].shape)
        ## momentum beta1, beta2
        self.change_adam['m_dw2'] = 0.9*self.change_adam['m_dw2'] + (1-0.9)*change_w['W2']
        self.change_adam['m_db2'] = 0.9*self.change_adam['m_db2'] + (1-0.9)*change_b['B2']
        self.change_adam['v_dw2'] = 0.999*self.change_adam['v_dw2'] + (1-0.999)*(change_w['W2']**2)
        self.change_adam['v_db2'] = 0.999*self.change_adam['v_db2'] + (1-0.999)*(change_b['B2'])
        ## bias correction
        self.change_adam['m_dw_corr2'] = self.change_adam['m_dw2']/(1-0.9**self.t)
        self.change_adam['m_db_corr2'] = self.change_adam['m_db2']/(1-0.9**self.t)
        self.change_adam['v_dw_corr2'] = self.change_adam['v_dw2']/(1-0.999**self.t)
        self.change_adam['v_db_corr2'] = self.change_adam['v_db2']/(1-0.999**self.t)
        
        change_w1['W2'] = (self.change_adam['m_dw_corr2']/(np.sqrt(self.change_adam['v_dw_corr2'])+1e-8))
        change_b1['B2'] = (self.change_adam['m_db_corr2']/(np.sqrt(self.change_adam['v_db_corr2'])+1e-8))
        
        # Calculate W1 update
        # print(error.shape, params['W2'].shape)
        error = np.dot(error, params['W2']) * self.ReLU(params['Z1'], derivative=True)
        change_w['W1'] = np.dot(error.T, params['A0'])
        change_b['B1'] = np.sum(error.T, axis=1, keepdims=True) / params['A0'].shape[0]
        # print(change_b['B1'].shape)
        ## momentum beta1, beta2
        self.change_adam['m_dw1'] = 0.9*self.change_adam['m_dw1'] + (1-0.9)*change_w['W1']
        self.change_adam['m_db1'] = 0.9*self.change_adam['m_db1'] + (1-0.9)*change_b['B1']
        self.change_adam['v_dw1'] = 0.999*self.change_adam['v_dw1'] + (1-0.999)*(change_w['W1']**2)
        self.change_adam['v_db1'] = 0.999*self.change_adam['v_db1'] + (1-0.999)*(change_b['B1'])
        ## bias correction
        self.change_adam['m_dw_corr1'] = self.change_adam['m_dw1']/(1-0.9**self.t)
        self.change_adam['m_db_corr1'] = self.change_adam['m_db1']/(1-0.9**self.t)
        self.change_adam['v_dw_corr1'] = self.change_adam['v_dw1']/(1-0.999**self.t)
        self.change_adam['v_db_corr1'] = self.change_adam['v_db1']/(1-0.999**self.t)
        
        change_w1['W1'] = (self.change_adam['m_dw_corr1']/(np.sqrt(self.change_adam['v_dw_corr1'])+1e-8))
        change_b1['B1'] = (self.change_adam['m_db_corr1']/(np.sqrt(self.change_adam['v_db_corr1'])+1e-8))
        
        return change_w1, change_b1

    def update_network_parameters(self, changes_to_w, changes_to_b):
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
            
        for key, value in changes_to_b.items():
            self.params[key] -= self.l_rate * value.flatten()
        
    def compute_accuracy(self, x_val, y_val):
        '''
            This function does a forward pass of x, then checks if the indices
            of the maximum value in the output equals the indices in the label
            y. Then it sums over each prediction and calculates the accuracy.
        '''
        num_sample = x_val.shape[0]
        output = self.predict(x_val)
        pred_idx = np.argmax(output, axis=0)
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
        N = outputs.shape[1]
        ce = -np.sum(targets.T*np.log(outputs+1e-9))/N
        return ce

    def train(self, training_generator, validation_generator, steps_training, steps_validation):
        start_time = time.time()
        self.m_dw, self.v_dw = 0, 0
        self.m_db, self.v_db = 0, 0
        self.t = 0
        self.change_adam = {}
        self.change_adam['m_dw1'] = 0
        self.change_adam['m_db1'] = 0
        self.change_adam['v_dw1'] = 0
        self.change_adam['v_db1'] = 0
        self.change_adam['m_dw2'] = 0
        self.change_adam['m_db2'] = 0
        self.change_adam['v_dw2'] = 0
        self.change_adam['v_db2'] = 0
        self.change_adam['m_dw3'] = 0
        self.change_adam['m_db3'] = 0
        self.change_adam['v_dw3'] = 0
        self.change_adam['v_db3'] = 0
        self.change_adam['m_dw4'] = 0
        self.change_adam['m_db4'] = 0
        self.change_adam['v_dw4'] = 0
        self.change_adam['v_db4'] = 0
        for iteration in range(self.epochs):
            for jj in range(steps_training):
                self.t = jj
                x, y = next(training_generator)

                output, accuracy_training, loss_training = self.forward_pass(x, y, compute_metric=True)

                changes_to_w, changes_to_b = self.backward_pass(y, output)
                
                self.update_network_parameters(changes_to_w, changes_to_b)
                
                print('Epoch: {0}, Iter: {1}, Time Spent: {2:.2f}s, Loss: {3:.5f}, Accuracy: {4:.2f}%'.format(
                    iteration+1, jj+1, time.time() - start_time, loss_training, accuracy_training * 100
                    ))
            
            acc, los = [], []
            for _ in range(steps_validation):    
                x_val, y_val = next(validation_generator)
                accuracy, loss = self.compute_accuracy(x_val, y_val)
                acc.append(accuracy)
                los.append(loss)
                
            print('Epoch: {0}, Time Spent: {1:.2f}s, Validation Loss: {2:.5f}, Validation Accuracy: {3:.2f}%'.format(
                        iteration+1, time.time() - start_time, np.mean(los), np.mean(acc) * 100
                        ))
#%% Test the network
import sys
sys.path.append(r"C:\Users\purushot\Desktop\github_version_simple\lauetoolsnn")
from utils_lauenn import array_generator_verify, array_generator
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
# steps_per_epoch = int((1 * 7500) / batch_size)
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
# =============================================================================
# Numpy model
# =============================================================================
dnn = DeepNeuralNetwork(sizes=[n_bins, n_bins, ((param)*15 + n_bins)//2, (param)*15, n_outputs],
                        epochs=epochs, 
                        l_rate=0.001)
dnn.train(training_data_generator, testing_data_generator,steps_per_epoch, val_steps_per_epoch)


#%%# lets predict on the test data
x_val, y_val = next(testing_data_generator)  

class_testdata = np.argmax(y_val[0])     
groundtruth_hkl = classhkl[class_testdata]
print(groundtruth_hkl)
prediction_keras = dnn.predict(x_val)
class_predicted = np.argmax(prediction_keras, axis=0)     
predicted_hkl = classhkl[class_predicted]
print(predicted_hkl[0])


#%%
# =============================================================================
# Keras neural network
# =============================================================================
import sys
sys.path.append(r"C:\Users\purushot\Desktop\github_version_simple\lauetoolsnn")
from NNmodels import model_arch_general_compnp
## load model and train
learning_rate = 1e-3
model_keras = model_arch_general_compnp(len(angbins)-1, len(classhkl), lr=learning_rate)
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
print(groundtruth_hkl[0])

prediction_keras = model_keras.predict(test_data[0])
class_predicted = np.argmax(prediction_keras, axis = 1)     
predicted_hkl = classhkl[class_predicted]
print(predicted_hkl[0])