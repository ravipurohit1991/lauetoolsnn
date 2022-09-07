## as of now only test imports
## testing of GUI to be done after installation manually with example case
import pytest

import lauetoolsnn
import keras
import scipy
import numpy
import h5py
import tensorflow
import PyQt5
import sklearn
import skimage
import fabio
import networkx
import tqdm

import pkg_resources

print("Lauetoolsnn :", pkg_resources.require("lauetoolsnn")[0].version)
print("scipy :", pkg_resources.require("scipy")[0].version)
print("numpy :", pkg_resources.require("numpy")[0].version)
print("h5py :", pkg_resources.require("h5py")[0].version)
print("keras :", pkg_resources.require("keras")[0].version)
print("tensorflow :", pkg_resources.require("tensorflow")[0].version)
print("PyQt5 :", pkg_resources.require("PyQt5")[0].version)
print("sklearn :", pkg_resources.require("sklearn")[0].version)
print("skimage :", pkg_resources.require("skimage")[0].version)
print("fabio :", pkg_resources.require("fabio")[0].version)
print("networkx :", pkg_resources.require("networkx")[0].version)
print("tqdm :", pkg_resources.require("tqdm")[0].version)


def test_method1():
	a = 6
	b = 8
	assert a+2== b, "test failed"
	assert b-2 == a, "test failed"