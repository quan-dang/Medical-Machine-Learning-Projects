# Detecting Parkinson's Disease with OpenCV and Computer Vision

# Dataset
  i) This project used dataset from [NIATS of Federal University of Uberlândia](http://www.niats.feelt.ufu.br/en/node/81)
  ii) The dataset contains 204 images and was already split into a training set and a test set consisting of:
      a) Spiral: 102 images, 72 training and 30 testing
      b) Wave: 102 images, 72 training and 30 testing

# How to run the program
Run main.py only for the entire code
'''
python main.py -d data/spiral (for Spiral dataset)
OR
python main.py -d data/wave (for Wave dataset)
'''

# Results
1) Spiral dataset
    acc
    ===
    u=0.8267, o=0.0249

    sensitivity
    ===========
    u=0.7600, o=0.0327

    specificity
    ===========
    u=0.8933, o=0.0327


2) Wave dataset
    acc
    ===
    u=0.6867, o=0.0340

    sensitivity
    ===========
    u=0.6667, o=0.0422

    specificity
    ===========
    u=0.7067, o=0.0680

