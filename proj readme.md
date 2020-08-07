# COMP 9517 20T2 Project

# Table of contents
1. [Introduction](#introduction)
1. [Instalation](#installation)
1. [Structure description](#Structure description)
1. [instruction](#instruction)

## 1. Introduction
In this project, after analyzing the different features of the given data sets, we programmed different programs for the data sets with different characteristics. First, we use the deep-water algorithm for the data sets that cannot be processed by the morphological algorithm, the second is the watershed algorithm for low contrast and poor imaging results, and finally, for the obvious image features and better imaging results Binarization algorithm of data set. In addition, we also realized the functions of detecting the mitotic cells, tracking and printing the cell paths, and calculating the corresponding characteristics.


## 2. Instalation

Prerequisites:
- Mac OS
- [anaconda](https://www.anaconda.com/)
- Jupyter Notebook 

Create an Anaconda environment with python dependencies.

Configure the following environment
```bash
OPENCV
$brew install cmake
$brew install opencv@2
```

```bash
## matplotlib
$pip install -U matplotlib
```

```bash
## scipy
$pip install numpy
$sudo pip install --upgrade scipy
$pip install scikit_image
```

## 3. Structure description

The datasets contains three different datasets: *DIC-C2DH_HeLa*, *PhC-C2DL-PSC* and *Fluo-N2DL-HeLa*.

# Structure of training data directory

```bash
--datasets
     -- 'DATASET_NAME'                    # Name of dataset e.g. 'DIC-C2DH-HeLa'
          -- Sequence 1                            # name of sequence 
               -- project.ipynb           # Project code
               -- t000.tif
               -- t001.tif
                  ...
               -- tXXX.tif
             ...
          -- Sequence 2
          
```
## 4. instruction

1. activate the anaconda *deepwater* environment.

2. Make sure project.ipynb is in the folder of the dataset you want to run.
Suppose we run on a second dataset *Fluo-N2DL-HeLa*.

```bash
/PhC-C2DL-PSC/Sequence 1 /project.ipynb/
```

3. Run the import part, preprocessing function, tracking function and function of Task 3.

```bash
import part
preprocessing function
tracking function 
function of Task 3
```

4.If you need the detections of the first and third datasets, run function of dataset 1 and dataset 3.
change the parameter in it to choose which dataset you need.
```bash
data_1 = True
data_2 = False
data_3 = False
```
That means we choose the first dateset.
If you need the second dataset, run function of dataset 2,also you can run the function of tracking part and division part
```bash
#### detetion and printing of the second data set ####
#### trackting part and division part ####
```
To get the caculatiion results of Task 3, run the function of Task 3, then run the printing part 
```bash
#### run this part before Task 3 ####
#### printing of Task 3 ####
```

5. Deactivate the anaconda environment.

