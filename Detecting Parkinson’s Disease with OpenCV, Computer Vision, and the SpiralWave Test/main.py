import numpy as np 
import cv2 as cv 
import os
import argparse

from imutils import build_montages # using build_montages for visualization
from imutils import paths # using paths to extract  the file paths to each of the images in the dataset
from skimage import feature # HOG comes with the feature of skimage

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix


# construct the argument parser and parse the arguments
# --dataset: the path to the input (waves or spirals)
# --trials: number of trials to run, default = 5
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
            help="path to the input dataset")
ap.add_argument("-t", "--trials", type=int, default=5,
            help="number of trials to run")
args = vars(ap.parse_args())


# create feature vectors using HOG 
def quantify_img(img):
    features = feature.hog(img, orientations=9, 
                        pixels_per_cell=(10, 10), cells_per_block=(2,2),
                        transform_sqrt=True, block_norm="L1")

    return features


# extract data and corresponding labels from the dataset
def load_split(path):
    # retrieve list of paths of images in the input directory
    imgPaths = list(paths.list_images(path))
    
    # initialize data and labels
    data = []
    labels = []

    # loop over all the image paths
    for imgPath in imgPaths:
        # extract the label from the filename
        label = imgPath.split(os.path.sep)[-2]

        # load the input image
        img = cv.imread(imgPath)

        # grayscale the image
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # resize the image to 200x200 pixels, ignoring aspect ratio
        img = cv.resize(img, (200, 200))

        # threshold the image to obtain the white drawing on a black background
        img = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

        # convert the img to a feature vector
        img_vector = quantify_img(img)

        # append img_vector and label to data and labels respectively
        data.append(img_vector)
        labels.append(label)

    return (np.array(data), np.array(labels))


# define the paths to the training and testing datasets
trainingPath = os.path.sep.join([args["dataset"], "training"])
testingPath = os.path.sep.join([args["dataset"], "testing"])

# loading training and testing datasets
print("Loading data...")
(trainX, trainY) = load_split(trainingPath)
(testX, testY) = load_split(testingPath)

# encode the labels
le = LabelEncoder()
trainY = le.fit_transform(trainY)
testY = le.transform(testY)

# initialize our trials dictionary
trials = {}

# loop over all the number of trials
for i in range(args["trials"]):
    # train the model
    print("Training model {} of {}...".format(i+1, args["trials"]))
    model = RandomForestClassifier(n_estimators=100)
    model.fit(trainX, trainY)

    # make predictions on the testing dataset
    preds = model.predict(testX)
    metrics = {}

    # compute the confusion matrix
    cm = confusion_matrix(testY, preds).flatten()
    (tn, fp, fn, tp) = cm
    metrics["acc"] = (tp + tn) / float(cm.sum())
    metrics["sensitivity"] = tp / float(tp + fn)
    metrics["specificity"] = tn / float(tn + fp)

    # loop over the metrics
    for (k, v) in metrics.items():
        # update the trials dictionary with the list of values for the current metric
        l = trials.get(k, [])
        l.append(v)
        trials[k] = l

for metric in ("acc", "sensitivity", "specificity"):
    # get the list of values for the current metric and computer mean and stdv
    values = trials[metric]
    mean = np.mean(values)
    stdv = np.std(values)

    print(metric)
    print("=" * len(metric))
    print("u={:.4f}, o={:.4f}".format(mean, stdv))
    print("")


# randomly select a few images 
testingPaths = list(paths.list_images(testingPath))
idxs = np.arange(0, len(testingPaths))
idxs = np.random.choice(idxs, size=(25,), replace=False)

imgs = []

for i in idxs:
    # preprocess the testing image same as above
    img = cv.imread(testingPaths[i])
    output = img.copy()
    output = cv.resize(output, (128, 128))
    
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, (200, 200))
    img = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

    # create a feature vector from the testing image
    features = quantify_img(img)
    preds = model.predict([features])
    label = le.inverse_transform(preds)[0]

    # draw the colored class label on the output and add to the set of outputs
    green = (0, 255, 0) 
    red = (0, 0, 255)

    color = green if label == "healthy" else red
    cv.putText(output, label, (3, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    imgs.append(output)

# take a list of imgs and create a montage with size 5x5 and resize all the images inside to 128x128 
montage = build_montages(imgs, (128, 128), (5, 5))[0]

# show the output montage
cv.imshow("Output", montage)
cv.waitKey(0)