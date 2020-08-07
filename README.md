# Deepwater cell segmentation

# Table of contents
1. [Introduction](#introduction)
1. [Instalation](#installation)
1. [Examples](#examples)
1. [Support](#support)

## 1. Introduction
Deepwater is an implementation of our cell segmentation method. It is a universal cell segmentation tool, which performs well also for clustered cells and noisy frames. We primarily focus on datasets of the Cell Tracking Challenge. The method combines watershed transformation and deep learning models. A detailed description of the algorithm can be found in this [paper](https://arxiv.org/abs/2004.01607).

The package is under development.

![method schema](images/method_schema.png)



## 2. Instalation

Prerequisites:
- Linux OS
- [anaconda](https://www.anaconda.com/)
- NVIDIA GPU 


Clone this repository

```bash
git clone https://gitlab.fi.muni.cz/xlux/deepwater.git
cd deepwater
```

Create an Anaconda environment with python dependencies.

```bash
conda env create -f conda_env_deepwater.yml
```

The second option to resolve dependencies is to manually install the following libraries: [Python 3.7.](), [Keras](https://keras.io/), [Tensorflow](https://www.tensorflow.org/), [CV2](https://docs.opencv.org/3.4/d2/de6/tutorial_py_setup_in_ubuntu.html), [Scikit Learn](https://scikit-learn.org/stable/install.html), [numpy](https://numpy.org/), [tqdm](https://github.com/tqdm/tqdm)


## Examples

### 1. Evaluate default dataset

The package contains pre-trained models of four different datasets: *DIC-C2DH_HeLa*, *PhC-C2DL-PSC*, *BF-C2DL-MuSC* and *BF-C2DL-HSC*.
All the datasets were published in [Cell Tracking Challenge](http://celltrackingchallenge.net/).

1. activate the anaconda *deepwater* environment.

```bash
conda activate deepwater
```

2. download DIC-C2DH-HeLa dataset from the [Cell Tracking Challenge](http://celltrackingchallenge.net/2d-datasets/) webpages. 

```bash
wget http://data.celltrackingchallenge.net/training-datasets/DIC-C2DH-HeLa.zip
unzip DIC-C2DH-HeLa.zip -d datasets
rm DIC-C2DH-HeLa.zip
```

3. Run the segmentation process. It can take several minutes.
```bash
python3 deepwater.py --name DIC-C2DH-HeLa --sequence 01 --mode 2
python3 deepwater.py --name DIC-C2DH-HeLa --sequence 02 --mode 2
```
Datasets, as well as segmentation results, are stored in the directory _datasets/DIC-C2DH-HeLa_. The folder _YY_RES_ contains labeled segmentation masks. The folder _YY_VIZ_ contains colored segmentation images and raw neural network predictions.

4. Evaluate results using _SEG score_ and _DET score_.
```bash
python3 deepwater.py --name DIC-C2DH-HeLa --sequence 01 --mode 3
python3 deepwater.py --name DIC-C2DH-HeLa --sequence 02 --mode 3
```

5. Deactivate the anaconda environment.

```bash
conda deactivate
```

### 2. Train model at your own data

Deepwater allows to train new model on your own data. For a training, you need full annotations of training samples,
which describe mass of every cell in the image. The annotation is represented by 16bit image, where each pixel has a value
equal to the label of displayed cell. You can use also weak annotations to train cell detection.

1. activate the anaconda *deepwater* environment.

```bash
conda activate deepwater
```

2. Copy your dataset into the _dataset_ folder. Keep the following structure.

```
# Structure of training data directory
--datasets
     -- 'DATASET_NAME'                    # Name of dataset e.g. 'DIC-C2DH-HeLa'
          -- config.yml                   # Deepwater configuration file
          -- 01                           # YY - name of sequence, a two digit number, e.g. '01', '02'
               -- t000.tif
               -- t001.tif
                  ...
               -- tXXX.tif
          -- 01_'REFERENCE'               # 01_'REFERENCE' - type of reference, e.g. '01_GT', '01_ST'
               -- 'CELL_MASKS'            # reference cell segmentation (instance seg. or semantic seg.) e.g. 'SEG'
                    -- man_seg000.tif
                    -- man_seg001.tif
                       ...
                    -- man_segXXX.tif
               -- 'MARKER_MASKS'          # reference cell markers e.g. 'TRA'
                    -- man_track000.tif
                    -- man_track001.tif
                      ...
                    -- man_trackXXX.tif
```

3. Set configuration file

Copy configutation file 'conig_example.yml' into the dataset folder, name it 'config.yml'.
set several parameters:
a) DATASET_NAME - dataset name, corresponds to the folder, where the dataset is stored 

b) MARKER_ANNOTATIONS - type of annotation used to get markers; one of ['weak', 'full'] 

c) CELL\_DIAMETER - avarage cell diameter in pixels, mandatory if 'full' MARKER\_ANNOTATIONS 

d) MARKER\_DIAMETER - average marker diameter in pixels,  mandatory if 'weak' MARKER\_ANNOTATIONS 

*(OPTIONAL)*
Set other learning hyperparameters.
#### configuration arguments

| argument            | description                                                                   | default value |
| ------------------- | ----------------------------------------------------------------------------- | ------------- |
| **main arguments**           
| DATASET_NAME        | dataset name, corresponds to the folder, where the dataset is stored          | MY_DATASET    |
| MARKER_ANNOTATIONS  | type of annotation used to get markers; one of ['weak', 'full']               | weak          |
| CELL_DIAMETER       | in pixels, mandatory if 'full' MARKER_ANNOTATIONS                             | 160           |
| MARKER_DIAMETER     | parameter **c**, mandatory if 'weak' MARKER_ANNOTATIONS                       | 24            |
| REFERENCE           | source of reference annotations, in CTC corresponds to gold and silver truth  | GT            |
| CELL_MASKS          | directory with full annotations  (full annotations)                           | SEG           |
| MARKER_MASKS        | directory with reference markers (weak annotations)                           | TRA           |
| DIGITS              | number of digits to indexing images                                           | 3             |
| **postprocessing**  
| MIN_MARKER_DYNAMICS | parameter **h**                                                               | 5             |
| THR_MARKERS         | threshold value to get markers from predictions, parameter **t_m**            | 128           |
| THR_FOREGROUND      | threshold to get image foreground from prediction, parameter **t_f**          | 200           |
| **training**        
| LR                  | initial learning rate                                                          | 0.01         |
| BATCH_SIZE          | mini-batch size; reduce the size when you exceed the GPU memory limit          | 16           |
| STEPS_PER_EPOCH     | mini-batches in one epoch                                                      | 50           |
| EPOCHS              | maximal number of training epochs                                              | 120          |
| **img pre-processing**
| NORMALIZATION       | normalization function, one of [HE, median, CLAHE]                             | HE           |
| UNEVEN_ILLUMINATION | True if the dataset suffers from uneven illumination                           | False        |
| TRACKING            | consistent cell labeling through the sequence                                  | True         |

4. Train the model
```bash
python3 deepwater.py --name 'DATASET_NAME' --mode 1
```

Evaluate results
```bash
python3 deepwater.py --name 'DATASET_NAME' --sequence 01 --mode 3
python3 deepwater.py --name 'DATASET_NAME' --sequence 02 --mode 3
```

5. Deactivate the anaconda environment.

```bash
conda deactivate
```



### 3. Segment own dataset using a pre-trained model

This example shows how to run the method on your own data.
The implementation currently support segmentation using only pretrained models.
You cannot train your own model.
For the best performance, recommend to resize your images to have similar cell sizes in pixels with a chosen pre-trained model.

1. Store your data in the _datasets_ folder.
In this example, we named dataset _MY_DATASET_. The directory _MY_DATASET/01_ contains all the input frames.
If it is available, you can add segmentation ground truth to the directory _MY_DATASET/01_GT_. The schema follows the Cell Tracking Challenge [naming and formating convention](http://public.celltrackingchallenge.net/documents/Naming%20and%20file%20content%20conventions.pdf).

```
-- datasets
      -- MY_DATASET
            -- 01                           # input images
                 -- t000.tif
                 -- t001.tif
                    ...
                 -- tXXX.tif
            -- 01_RES                       # segmentation results     (optional)
            -- 01_VIZ                       # network outputs          (optional)
            -- 01_GT                        # dataset ground truth     (optional)
                 -- SEG                     
                      -- man_seg000.tif
                      -- man_seg001.tif
                         ...
                      -- man_segXXX.tif
                 -- DET
                      -- man_det000.tif
                      -- man_det001.tif
                         ...
                      -- man_detXXX.tif
                    

```

2. Choose proper pretrained model.

The essential input image properties are _cell apperiance_ and _cell size_. We recommend reshaping your input images to change a cell shape to correspond to the pre-trained dataset cell size.

[DIC-C2DH-HeLa](http://data.celltrackingchallenge.net/training-datasets/DIC-C2DH-HeLa.zip)  
<img src="./images/example_DIC-C2DH-HeLa.png"  width="300" height="300">

[PhC-C2DL-PSC](http://data.celltrackingchallenge.net/training-datasets/PhC-C2DL-PSC.zip)  
<img src="./images/example_PhC-C2DL-PSC.png"  width="300" height="300">

[BF-C2DL-MuSC](http://data.celltrackingchallenge.net/training-datasets/BF-C2DL-MuSC.zip)  
<img src="./images/example_BF-C2DL-MuSC.png"  width="300" height="300">

[BF-C2DL-HSC](http://data.celltrackingchallenge.net/training-datasets/BF-C2DL-HSC.zip)  
<img src="./images/example_BF-C2DL-HSC.png"  width="300" height="300">

3. Run the segmentation procedure.
```bash
python3 deepwater.py --name MY_DATASET --sequence 01 --mode 2 --model_name PRETRAINED_MODEL_NAME
```

The application store the results to folders _dataset/MY_DATASET/01_RES_ and _dataset/MY_DATASET/01_VIZ_.

| arguments            | value              |
| -------------------- | -------------------|
| --help               | help message and exit |
| --name               | dataset name (MY_DATASET) |
| --sequence           | sequence name (01) |
| --model_name         | name of pretrained model (PRETRAINED_MODEL_NAME) |
| --mode               | 1: training, 2:predicting, 3:evaluation |
| --regenerate         | True to recompute results |


4. Deactivate the anaconda environment.

```bash
conda deactivate
```



## Support
The software is under development. If you want to use it in your research, please contact us at [xlux@fi.muni.cz](xlux@fi.muni.cz), and we will help you with the application.
