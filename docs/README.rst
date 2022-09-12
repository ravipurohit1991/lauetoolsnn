
lauetoolsnn
------------
An autonomous feed-forward neural network (FFNN) model to predict the HKL in single/multi-grain/multi-phase Laue patterns with high efficiency and accuracy is introduced. 

Laue diffraction indexation (especially Laue images comprising of diffraction signal from several polycrystals/multi phase materials) can be a very tedious and CPU intensive process. To takle this, LaueNN or LauetoolsNN was developed employing the power of neural network to speed up a part of the indexation process. In the `LaueNN_presentation <https://github.com/ravipurohit1991/lauetoolsnn/blob/main/presentations/LaueNN_presentation.pdf>`_, several steps of Laue pattern indexation with classical approach is described. We have replaced the most CPU intensive step with the Neural Networks. The step where the Laue indices hkl of each spot os now determined with the Neural networks, alongside the spot hkl index, the neural network also predicts the Material that spot belongs to. This can be useful incase of Laue images comprising of diffraction signal from multi-phases. LaueNN uses the existing modules of Lauetools to generate simulated Laue patterns. The whole workflow and the application of this tool is illustrated in this article `LaueNN: neural-network-based hkl recognition of Laue spots and its application to polycrystalline materials <https://onlinelibrary.wiley.com/iucr/doi/10.1107/S1600576722004198>`_.


Video tutorial
----------------------------
- Video 1: Working with jupyter notebook scripts : https://cloud.esrf.fr/s/6q4DJfAn7K46BGN
- Video 2: Working with lauetoolsnn GUI : https://cloud.esrf.fr/s/AeGow4CoqZRJiyx


Requirements: (latest version of each libraries accessed on 03/04/2022) 
------------------------------------------------------------------------------------ 
- PyQt5 (GUI)
- matplotlib
- Keras
- tensorflow 
- numpy 
- scipy (scipy transform rotation is used)
- h5py (required for writing neural network model files)
- scikit-learn (required for generating trained model classification reports)
- fabio (used for opening raw Laue tiff images)
- networkx (to be replaced with numpy in the future)
- scikit-image (used for hough based analysis of Laue patterns)
- tqdm (required only for notebook scripts)
- opencv (for LOG based peak search)


Example case (end to end case)
------------------------------------------
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


Citation
--------------
If you use this software, please cite it using the metadata available in the `citation_bibtex <https://github.com/ravipurohit1991/lauetoolsnn/blob/main/citation_bibtex.cff>`_ file in root.

`
Purushottam Raj Purohit, R. R. P., Tardif, S., Castelnau, O., Eymery, J., Guinebretiere, R., Robach, O., Ors, T. & Micha, J.-S. (2022). J. Appl. Cryst. 55, 737-750.
`


Known Issues
--------------
So far, there is a issue with H5py and HDF5 version in the windows installation with conda. If error with H5py version mismatch exist after conda installation, please try "pip install lauetoolsnn" on windows as this should not have this problem. The other possibility is to install the H5py with pip before or after installing lauetoolsnn with conda.


Support
--------------
Do not hesitate to contact the development team at purushot@esrf.fr or micha@esrf.fr .


Maintainer(s)
--------------
* Ravi raj purohit PURUSHOTTAM RAJ PUROHIT (https://github.com/ravipurohit1991)


License
--------------

The project is licensed under the MIT license.