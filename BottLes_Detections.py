
# importing libraries......

import cv2
import numpy as np
import os
import pylab as pl
from sklearn.svm import LinearSVC
import joblib
from scipy.cluster.vq import *


# loading our saved model....
clf, classes_names, stdSlr, k, voc = joblib.load("model.pkl")

# paths to our test dataset.. for testing purposes..... and detecting the bottle is either , coccola ,sprit or fanta ...etc
DIRECTORY='test_data'


##################3
# paths to images dataset for testing purposes....
Images_Names = os.listdir(DIRECTORY)
print(Images_Names)
images_paths=[]
image_list=[]
for image_name in Images_Names:
    image_list.append(image_name)

    images_path = os.path.join(DIRECTORY,image_name)
    images_paths.append(images_path)




#
# ##3   Please carefully read this .................
# # this is main part here we are gonna taking the key points and
# # descriptors of the images dataset and then we will compare these keypoints
# # with the our trained clf model keypoints and based on we will make the prediction
# # Create feature extraction and keypoint detector objects

fea_det = cv2.SIFT_create()
des_ext = cv2.xfeatures2d.SIFT_create()



# List where all the descriptors are stored
des_list = []
for img_path in images_paths:
    img = cv2.imread(img_path)
    kpts = fea_det.detect(img)
    kpts, des = des_ext.compute(img, kpts)
    des_list.append((img_path, des))





# # # Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]

print(descriptors)
for image_path, descriptor in des_list[0:]:
    descriptors = np.vstack((descriptors, descriptor))
    # print(descriptors)


# testing features....
test_features = np.zeros((len(images_paths), k), "float32")
for i in range(len(images_paths)):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        test_features[i][w] += 1


# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (test_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(images_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# here we are gonna scaling our features....
test_features = stdSlr.transform(test_features)

# Perform the predictions  ,, upto here we are gonna making the prediction...
predictions =  [classes_names[i] for i in clf.predict(test_features)]

# Report true class names so they can be compared with predicted classes
true_class = [images_paths[j] for j in range(len(images_paths))]
# Perform the predictions and report predicted class names.
predictions = [classes_names[i] for i in clf.predict(test_features)]

# Print the true class and Predictions
print("true_class =" + str(true_class))
print("prediction =" + str(predictions))



# checking the result of our code and making the presiction how our code is predicting and classifying the images ......


for image_path, prediction in zip(images_paths, predictions):
    img = cv2.imread(image_path)
    image = cv2.resize(img,(512,512))

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    # getting the possition to put the text on image window
    pt = (5, 120)
    cv2.putText(image, prediction, pt, cv2.FONT_HERSHEY_SIMPLEX , 2, [255, 0, 255], 3)
    cv2.imshow("Image", image)
    cv2.waitKey(2000)  #




