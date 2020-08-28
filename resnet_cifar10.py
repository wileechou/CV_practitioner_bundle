# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv.resnet import ResNet
from pyimagesearch.callbacks.epochcheckpoint import EpochCheckpoint
from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import numpy as np
import argparse
import sys

# solve failed to get convolution algorithm
import tensorflow as tf
cfg = tf.compat.v1.ConfigProto()
cfg.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=cfg)

# set a high recursion limit so Theano doesn't complain
sys.setrecursionlimit(5000)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True,
                help="path to output checkpoint directory")
ap.add_argument("-m", "--model", type=str,
                help="path to *specific* model checkpoint to load")
ap.add_argument("-s", "--start-epoch", type=int, default=0,
                help="epoch to restart training at")
args = vars(ap.parse_args())

# load the training and testing data, converting the images from
# integers to floats
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float")
testX = testX.astype("float")

# apply mean subtraction to the data
mean = np.mean(trainX, axis=0)
trainX -= mean
testX -= mean

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# construct the image generator for data augmentation
aug = ImageDataGenerator(width_shift_range=0.1,
                         height_shift_range=0.1, horizontal_flip=True,
                         fill_mode="nearest")

# if there is no specific model checkpoint supplied, then initialize
# the network (ResNet-56) and compile the model
if args["model"] is None:
    print("[INFO] compiling model...")
    opt = SGD(lr=1e-1)
    model = ResNet.build(32, 32, 3, 10, (9, 9, 9),
                         (64, 64, 128, 256), reg=0.0005)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

# other wise, load the checkpoint from disk
else:
    print("[INFO] loading {}...".format(args["model"]))
    model = load_model(args["model"])

    # update the learning rate
    print("[INFO] old learning rage: {}".format(
        K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, 1e-5)
    print("[INFO] new learning rate: {}".format(
        K.get_value(model.optimizer.lr)))

# construct the set of callbacks
callbacks = [
    EpochCheckpoint(args["checkpoints"], every=5,
                    startAt=args["start_epoch"]),
    TrainingMonitor("output/resnet56_cifar10.png",
                    jsonPath="output/resnet56_cifar10.json",
                    startAt=args["start_epoch"])]

# train the network
print("[INFO] training network...")
model.fit_generator(
    aug.flow(trainX, trainY, batch_size=128),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // 128, epochs=100,
    callbacks=callbacks, verbose=1)