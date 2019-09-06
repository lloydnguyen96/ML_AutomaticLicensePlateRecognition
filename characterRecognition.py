import cv2
import numpy as np
import operator
import os


# Constants:
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30


def recognize_character(imgCharacter):
    try:
        npaClassifications = np.loadtxt("trainingData/classifications.txt", np.float32)
    except:
        print("Error: unable to open classifications.txt!!!\n")
        os.system("pause")
        return
    # end try
    try:
        npaFlattenedImages = np.loadtxt("trainingData/flattenedImages.txt", np.float32)
    except:
        print("Error: unable to open flattenedImages.txt!!!\n")
        os.system("pause")
        return
    # end try
    if imgCharacter is None:
        print("Error: image not read from file!!!\n")
        os.system("pause")
        return

    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))
    kNearest = cv2.ml.KNearest_create()
    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)
    imgCharacterResized = cv2.resize(imgCharacter, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))         # resize image, this will be more consistent for recognition and storage
    imgCharacterReshaped = imgCharacterResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT)) # flatten image into 1d numpy array
    imgCharacterReshaped = np.float32(imgCharacterReshaped)                                             # convert from 1d numpy array of ints to 1d numpy array of floats
    retval, npaResults, neigh_resp, dists = kNearest.findNearest(imgCharacterReshaped, k = 1)           # call KNN function find_nearest
    recognizedCharacter = str(chr(int(npaResults[0][0])))                                               # get character from results
    cv2.destroyAllWindows()
    return recognizedCharacter
# end characterRecognition