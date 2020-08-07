## 1. Deepwater part

We found the code of Deepwater algorithm from Github lab. It is an open source code in https://gitlab.fi.muni.cz/xlux/deepwater.git

Documents deepwater.py, config.py, utils.py and deepwater_object.py are the procedures for the Deepwater Algorithm. We put the pictures in TASK1 into the training model, and then put the trained pictures into our function to get the results of the first set of data.

For using the code of Deepwater algorithm, the following libraries should be installed: [Python 3.7.](), [Keras](https://keras.io/), [Tensorflow](https://www.tensorflow.org/), [CV2](https://docs.opencv.org/3.4/d2/de6/tutorial_py_setup_in_ubuntu.html), [Scikit Learn](https://scikit-learn.org/stable/install.html), [numpy](https://numpy.org/), [tqdm](https://github.com/tqdm/tqdm)

1. Open the anaconda in the terminal.
2. Run the segmentation process
```bash
python3 deepwater.py --name DIC-C2DH-HeLa --sequence 01 --mode 2
python3 deepwater.py --name DIC-C2DH-HeLa --sequence 02 --mode 2
```
3. We can get the results in the folder  directory _datasets/DIC-C2DH-HeLa_/_YY_VIZ_ 
4. At last we put the results into our py file.

