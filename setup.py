import os
import setuptools
from setuptools import find_packages, setup

with open(os.path.join(os.path.dirname(__file__), "README.md")) as readme:
    long_description = readme.read()

setuptools.setup(
    name="lauetoolsnn",
    
    version="3.0.39",
    
    author="Ravi raj purohit PURUSHOTTAM RAJ PUROHIT",
    
    author_email="purushot@esrf.fr",
    
    description="LaueNN- neural network training and prediction routine to index single and polycrystalline Laue patterns",

    long_description=long_description,
    
    long_description_content_type="text/markdown",
    
    include_package_data=True,
    
    packages=find_packages(),
    
    url="https://github.com/ravipurohit1991/lauetoolsnn",
    
    #install_requires=['matplotlib>=3.4.2', 'Keras>=2.4.3', 'fast_histogram>=0.10', 'numpy>=1.18.5', 'h5py>=2.10.0', 
                        #'tensorflow>=2.3.0','LaueTools>=3.0.0.71', 'PyQt5>=5.9', 'scikit-learn>=0.24.2', 'fabio>=0.11.0', 'networkx>=2.6.3']
    # Remove fast_histogram dependency by replacing it with Numpy
    # networkx library should also be replaced by numpy list intersect method
    
    #setup_requires=['matplotlib', 'Keras', 'fast_histogram', 'numpy', 'h5py', 'tensorflow', 'PyQt5', 'scikit-learn', 'fabio', 'networkx', 'scikit-image','tqdm'],
    #install_requires=['matplotlib>=3.4.2', 'Keras>=2.4.3', 'fast_histogram>=0.10', 'numpy>=1.18.5', 'h5py>=2.10.0', 'tensorflow>=2.3.0', 'PyQt5>=5.9', 'scikit-learn>=0.24.2', 'fabio>=0.11.0', 'networkx>=2.6.3', 'scikit-image>=0.18.0','tqdm>=4.60.0'],
    
    setup_requires=['matplotlib', 'Keras', 'numpy', 'h5py', 'tensorflow', 'PyQt5', 'scikit-learn', 'fabio', 'networkx', 'scikit-image','tqdm'],
    install_requires=['matplotlib>=3.4.2', 'Keras>=2.7.0', 'numpy>=1.18.5', 'h5py>=3.5.0', 'tensorflow>=2.7.0', 'PyQt5>=5.9', 'scikit-learn>=0.24.2', 'fabio>=0.11.0', 'networkx>=2.6.3', 'scikit-image>=0.18.0','tqdm>=4.60.0'],


    entry_points={
                 "console_scripts": ["lauetoolsnn=lauetoolsnn.lauetoolsneuralnetwork:start"]
                 },
                 
    classifiers=[
                    "Programming Language :: Python :: 3.6",
                    "Programming Language :: Python :: 3.7",
                    "Programming Language :: Python :: 3.8",
                    "Programming Language :: Python :: 3.9",
                    "Programming Language :: Python :: 3.10",
                    "Topic :: Scientific/Engineering :: Physics",
                    "Intended Audience :: Science/Research",
                    "Development Status :: 5 - Production/Stable",
                    "License :: OSI Approved :: MIT License "
                ],
                
    python_requires='~=3.6',
)