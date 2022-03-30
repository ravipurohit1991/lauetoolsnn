import os
import setuptools
from setuptools import find_packages, setup

with open(os.path.join(os.path.dirname(__file__), "README.md")) as readme:
    long_description = readme.read()

setuptools.setup(
    name="lauetoolsnn",
    version="3.0.38",
    author="Ravi raj purohit PURUSHOTTAM RAJ PUROHIT",
    author_email="purushot@esrf.fr",
    description="LaueNN- neural network training and prediction routine to index single and polycrystalline Laue patterns",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    packages=find_packages(),
    url="https://github.com/ravipurohit1991/lauetoolsnn",
    #install_requires=['matplotlib>=3.4.2', 'Keras>=2.4.3', 'fast_histogram>=0.10', 'numpy>=1.18.5', 'h5py>=2.10.0', 
                        #'tensorflow>=2.3.0','LaueTools>=3.0.0.71', 'PyQt5>=5.9', 'scikit_learn>=0.24.2', 'fabio>=0.11.0', 'networkx>=2.6.3']
    install_requires=['matplotlib', 'Keras', 'fast_histogram', 'numpy', 'h5py', 'tensorflow', 'PyQt5', 'scikit_learn', 'fabio', 'networkx', 'scikit-image'],
    entry_points={
                 "console_scripts": ["lauetoolsnn=lauetoolsnn.lauetoolsneuralnetwork:start"]
                 },
    classifiers=[
                "Programming Language :: Python :: 3",
                "License :: OSI Approved :: MIT License",
                "Operating System :: OS Independent",
                ],
    python_requires='>=3.6',
)