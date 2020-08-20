# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from pyimagesearch.datasets.simpledatasetloader import SimpleDatasetLoader
from pyimagesearch.nn.conv.fcheadnet import FCHeadNet
from pyimagesearch.nn.conv.lenet import LeNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from imutils import paths
import numpy as np
import argparse
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--include-top", type=int, default=1,
                help="whether or not to include top of CNN")
args = vars(ap.parse_args())

# load the VGG16 network, ensuring the head FC layer sets are left
# off
baseModel = VGG16(weights="imagenet", include_top=False,
                  input_tensor=Input(shape=(224, 224, 3)))

leModel = LeNet.build(64, 64, 3, 10)


# initialize the new head of the network, a set of FC layers
# followed by a softmax classifier
output = Dense(100)(leModel.output)
#headModel = FCHeadNet.build(baseModel, 4, 256)

# place the head  FC model on top of the model -- this will
# become the actual model we will train
model = Model(inputs=leModel.input, outputs=output)

# loop over the layers in the network and display them to the
# console
for (i, layer) in enumerate(model.layers):
    print("[INFO] {}\t{}".format(i, layer.__class__.__name__))

