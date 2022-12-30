
## importing Librarise......

# ---------       STEP NO : 1...

import cv2
import numpy as np
import os
from sklearn.svm import LinearSVC
import joblib  # this one is for saving the model to .pkl file
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
#
# # Get the path of the training set

# --------   STEP NO:  2...

train_path = 'dataset/train_data'
training_names = os.listdir(train_path)

# Get all the path to the images and save them in a list
# image_paths and the corresponding label in image_paths
image_paths = []
image_classes = []
class_id = 0

def imglist(path):
    return [os.path.join(path, f) for f in os.listdir(path)]



for training_name in training_names:
    dir = os.path.join(train_path, training_name)
    class_path = imglist(dir)
    image_paths += class_path
    image_classes += [class_id] * len(class_path)
    class_id += 1

# -----------  STEP NO: 3...
# Create feature extraction and keypoint detector objects
fea_det = cv2.SIFT_create()
des_ext = cv2.xfeatures2d.SIFT_create()

# List where all the descriptors are stored
des_list = []

for image_path in image_paths:
    im = cv2.imread(image_path)
    kpts = fea_det.detect(im)
    kpts, des = des_ext.compute(im, kpts)
    des_list.append((image_path, des))

# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))

# -----------  STEP NO: 4...

# Perform k-means clustering
k = 100
voc, variance = kmeans(descriptors, k, 1)


im_features = np.zeros((len(image_paths), k), "float32")
for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        im_features[i][w] += 1

# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# here we are scaling our features.....
stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)


# -----------  Last STEP .....
# here we are gonna train our svm model on our keypints
# or features or keypoints which we extracted with sift detector.....

# Train the Linear SVM
clf = LinearSVC()
clf.fit(im_features, np.array(image_classes))

# Save the SVM
joblib.dump((clf, training_names, stdSlr, k, voc), "model.pkl", compress=3)


# here we saved our model to a model.pkl File in the next we will Load that File and do the detection ......



