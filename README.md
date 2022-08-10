# lauetoolsnn
A feed-forward neural network (FFNN) model to predict the HKL in single/multi-grain/multi-phase Laue patterns with high efficiency and accuracy is introduced. 

Version 3.0.39
### Video tutorial

- Video 1: Working with jupyter notebook scripts : https://cloud.esrf.fr/s/6q4DJfAn7K46BGN
- Video 2: Working with lauetoolsnn GUI : https://cloud.esrf.fr/s/AeGow4CoqZRJiyx


### Requirements: (latest version of each libraries accessed on 03/04/2022)  
- PyQt5 (GUI)
- matplotlib
- Keras
- tensorflow 
- fast_histogram (to be replaced with numpy in the future)
- numpy 
- scipy (scipy transform rotation is used)
- h5py (required for writing neural network model files)
- scikit-learn (required for generating trained model classification reports)
- fabio (used for opening raw Laue tiff images)
- networkx (to be replaced with numpy in the future)
- scikit-image (used for hough based analysis of Laue patterns)
- tqdm (required only for notebook scripts)

### Installation
Lauetoolsnn can be installed either via PYPI usiing the following command in terminal (this installs all dependencies automatically): 
https://pypi.org/project/lauetoolsnn/
``` bash
$ pip install lauetoolsnn
```

or can be compiled and installed locally via the setup.py file. Download the Github repository and type the following in terminal. In this case, the dependencies has to be installed manually. The latest version of each dependency works as of (01/04/2022).
``` bash
$ python setup.py install
```

See procedure_usage_lauetoolsnn.pdf for installation and how to write the configuration file to be used with GUI.

### Example case
Two example case studies are included in the lauetoolsnn\examples folder.
Run the GUI by either launching directly from the terminal using the 'lauetoolsnn' command or by running it locally with python lauetoolsneuralnetwork.py command.

First step is to load the config.txt from the example folder, it sets all the values of the GUI to the case study.
In the GUI: 
- Step1: File --> load config . Select the config file from the example directory. 
- Step1a: If config file is not available, one can set parameters in the configure parameters window directly.
- Step2: Press the configure parameters button and press Accept button at the end (the values are loaded from the config file).
- Step3: Press Generate Training dataset button. This will generate the training and validation dataset for neural network.
- Step4: Press Train Neural network button. This will start the training process and once finished will save the trained model.
- Step5: Press the Live prediction with IPF map to start the prediction on predefined experimental dataset. Example datafile is included in the examples folder.
- Step6: Once analyzed, the results can be saved using the save results button.

In addition, all the above mentioned steps can be done without the GUI and are detailed in the lauetoolsnn\example_notebook_scripts folder.
Jupyter notebook scripts are provided to run all the steps sequentially.

The indexed orientation matrix is also written in ".ctf" format, which can then be opened with channel 5 Aztec or MTEX software to do post processing related to orientations analysis. MTEX post processing script is also included in the lauetoolsnn\util_script\MTEX_plot.m
