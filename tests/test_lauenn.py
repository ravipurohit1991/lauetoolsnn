## as of now only test imports
## testing of GUI to be done after installation manually with example case
# import pytest

import keras
print("Keras passed import")

import scipy
print("scipy passed import")

import numpy
print("numpy passed import")

import h5py
print("h5py passed import")

import tensorflow
print("tensorflow passed import")

import PyQt5
print("PyQt5 passed import")

import sklearn
print("sklearn passed import")

import skimage
print("skimage passed import")

import fabio
print("fabio passed import")

import networkx
print("networkx passed import")

import tqdm
print("tqdm passed import")


import pkg_resources

print("scipy :", pkg_resources.require("scipy")[0].version)
print("numpy :", pkg_resources.require("numpy")[0].version)
print("h5py :", pkg_resources.require("h5py")[0].version)
print("keras :", pkg_resources.require("keras")[0].version)
print("tensorflow :", pkg_resources.require("tensorflow")[0].version)
print("PyQt5 :", pkg_resources.require("PyQt5")[0].version)
print("sklearn :", pkg_resources.require("scikit-learn")[0].version)
print("skimage :", pkg_resources.require("scikit-image")[0].version)
print("fabio :", pkg_resources.require("fabio")[0].version)
print("networkx :", pkg_resources.require("networkx")[0].version)
print("tqdm :", pkg_resources.require("tqdm")[0].version)


def test_method1():
    ##DUmmy test 
	a = 6
	b = 8
	assert a+2== b, "test failed"
	assert b-2 == a, "test failed"
    
    #TODO add an example test case that verifies all the functionality of GUI
    # For eample run the automated scripts from the example notebook directory
    
    
    
    
    
    
    
    
    