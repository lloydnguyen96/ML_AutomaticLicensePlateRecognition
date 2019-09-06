import numpy as np
import cv2
import os
import sys


from possiblePlateLocatedRegionDetection import *
from plateDetectionAndProcessing import *
from characterSegmentation import *
from characterRecognition import *


# Constants:
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30
MODEL_NEED_MORE_TRAINING = False
DELETE_TRAINING_EXAMPLE = False
ESC_BUTTON = 27
ENTER_BUTTON = 13
BACKSPACE_BUTTON = 8
NONE_CHARACTER = '.' # this code indicate that image does not represent a character
EMPTY_STRING = ""


def main():
	# Step: delete training example if needed
	if DELETE_TRAINING_EXAMPLE:
		print("-----PREDICTION MODEL TRAINING EXAMPLE ELIMINATION-----")
		delete_training_example()


	# Step: load processing image
	print("-----AUTOMATIC LICENSE PLATE RECOGNITION SYSTEM-----")
	input_image_path='C:/Users/MJ/Documents/Datasets/LicensePlatesData/1/133.jpg'
	if sys.platform=='linux':
		input_image_path='/media/Windows/C/Users/MJ/Documents/Datasets/LicensePlatesData/1/133.jpg'
	imgScene = cv2.imread(input_image_path)
	if imgScene is None:
		print("Error: input image was unsuccessfully loaded!!!")
		return


	# Step: find region of image in which plate is located
	imgPlateLocatedRegion, imgSecondStepChosenContour, imgSceneResized, boundingRect = detect_plate_located_region(imgScene)
	if imgPlateLocatedRegion is None or imgSecondStepChosenContour is None:
		print("Notification: couldn't find any possible plate-located region in this image.")
		cv2.destroyAllWindows()
		return


	# Step: process, crop, rotate, resize,... to obtain image of plate
	imgPlate = detect_plate(imgPlateLocatedRegion, imgSecondStepChosenContour)
	if imgPlate is None:
		print("Error: the processing process of possible plate-located region yielded an error!!!")
		cv2.destroyAllWindows()
		return


	# Step: segment image of plate into list of image of character
	listOfTopImgCharacter, listOfBottomImgCharacter = segment_character(imgPlate)
	if listOfTopImgCharacter == [] or listOfBottomImgCharacter == []:
		print("Error: couldn't crop image of plate into list of image of character!!!")
		cv2.destroyAllWindows()
		return
	listOfImgCharacter = listOfTopImgCharacter + listOfBottomImgCharacter


	# Step: recognize character from image of character
	strOfCharacters = EMPTY_STRING
	strOfTopCharacters = EMPTY_STRING
	strOfBottomCharacters = EMPTY_STRING
	for imgCharacter in listOfTopImgCharacter:
		character = recognize_character(imgCharacter)
		if character == NONE_CHARACTER:
			continue
		strOfTopCharacters = strOfTopCharacters + character
	for imgCharacter in listOfBottomImgCharacter:
		character = recognize_character(imgCharacter)
		if character == NONE_CHARACTER:
			continue
		strOfBottomCharacters = strOfBottomCharacters + character
	strOfCharacters = strOfTopCharacters + '-' + strOfBottomCharacters
	if strOfCharacters == EMPTY_STRING:
		print("Notification: system can't find any plate in image.")
		cv2.destroyAllWindows()
		return
	print('Notification: here is license plate number: ')
	# draw text
	print(strOfTopCharacters)
	print(strOfBottomCharacters)
	print(strOfCharacters)
	imgSceneResizedHeight, imgSceneResizedWidth = imgSceneResized.shape[:2]
	cv2.putText(imgSceneResized,
				strOfCharacters,
    			(0, imgSceneResizedHeight),
    			cv2.FONT_HERSHEY_COMPLEX,
    			1,
    			(0, 0, 255))
	# draw rectangle
	[boundingRectX, boundingRectY, boundingRectWidth, boundingRectHeight] = boundingRect
	cv2.rectangle(imgSceneResized, (boundingRectX, boundingRectY), (boundingRectX + boundingRectWidth, boundingRectY + boundingRectHeight), (0, 0, 255), 1)
	# show result
	cv2.imshow('imgSceneResized', imgSceneResized)


	# Step: train model if needed
	if MODEL_NEED_MORE_TRAINING:
		print("-----PREDICTION MODEL TRAINING-----")
		for imgCharacter in listOfImgCharacter:
			cv2.imshow('imgTrainingCharacter', imgCharacter)
			while True:
				print("Notification: choose one of these below options:\n\
					- Press (Esc) to stop training.\n\
					- Press (Enter) to train this character.\n\
					- Press (Backspace) to skip this character.\n")
				button = int(cv2.waitKey(0))
				if button == ESC_BUTTON or button == ENTER_BUTTON or button == BACKSPACE_BUTTON:
					break
				else:
					print("Notification: you just pressed the wrong key, please do it again.")
					continue
			if button == ESC_BUTTON:
				break
			elif button == ENTER_BUTTON:
				train_model(imgCharacter)
			elif button == BACKSPACE_BUTTON:
				continue
			else:
				print("Error: character input error!!!")
				cv2.destroyAllWindows()
				return


	print("Notification: ALPS process done:\n\
		- Press any key to exit.\n")
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return
# end main


def delete_training_example():
	try:
		npaClassifications = np.loadtxt("trainingData/classifications.txt", np.float32)
	except:
		print("Error: unable to open classifications.txt!!!")
		return
	# end try

	try:
		npaFlattenedImages = np.loadtxt("trainingData/flattenedImages.txt", np.float32)
	except:
		print("Error: unable to open flattenedImages.txt!!!")
		return
	# end try
	npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))
	numberOfDeletedTrainingExample = 0
	while True:
		print("Notification: choose one of these below options:\n\
			- Press (Enter) to delete one training example from bottom of training data table.\n\
			- Press (Esc) to stop deleting.\n")
		cv2.imshow('None', np.empty(1))
		button = int(cv2.waitKey(0))
		if button == ESC_BUTTON:
			break
		elif button == ENTER_BUTTON:
			numberOfDeletedTrainingExample = numberOfDeletedTrainingExample + 1
			shapeTuple = npaClassifications.shape
			shapeList = list(shapeTuple)
			firstDimensionShapeListNumberOfElement = shapeList[0]
			lastElementIndex = firstDimensionShapeListNumberOfElement - 1
			npaClassifications = np.delete(npaClassifications, lastElementIndex, 0)
			npaFlattenedImages = np.delete(npaFlattenedImages, lastElementIndex, 0)
		else:
			print("Notification: you just pressed the wrong key, please do it again.")
			continue
	np.savetxt("trainingData/classifications.txt", npaClassifications)
	np.savetxt("trainingData/flattenedImages.txt", npaFlattenedImages)
	return
# end delete_training_example


def train_model(imgCharacter):
	try:
		npaClassifications = np.loadtxt("trainingData/classifications.txt", np.float32)
	except:
		print("Error: unable to open classifications.txt!!!")
		return
	# end try

	try:
		npaFlattenedImages = np.loadtxt("trainingData/flattenedImages.txt", np.float32)
	except:
		print("Error: unable to open flattenedImages.txt!!!")
		return
	# end try

	intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
					ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
					ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
					ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z'), ord('.')]
	imgTrainingCharacterResized = cv2.resize(imgCharacter, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
	strImgTrainingCharacterResizedWindowName = "imgTrainingCharacterResized"
	cv2.imshow(strImgTrainingCharacterResizedWindowName, imgTrainingCharacterResized)
	while True:
		print("Notification: choose one of these below options:\n" +
			"- Press (ESC) to stop training this character.\n" +
			"- Press a (Character) (a number or a capital letter) corresponding to the character apprearing on window <%s>.\n" % strImgTrainingCharacterResizedWindowName +
			"- Press character ('.') from keyboard to inform that image appearing on window <%s> is not a license plate character.\n" % strImgTrainingCharacterResizedWindowName)
		intChar = cv2.waitKey(0)
		if intChar == ESC_BUTTON:
			return
		elif intChar in intValidChars:
			intClassification = [intChar]
			# appropriate format for storing image into txt file
			npaFlattenedImage = imgTrainingCharacterResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
			npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0)
			fltClassification = np.array(intClassification, np.float32)
			npaClassification = np.reshape(fltClassification, (1, 1))
			npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))
			npaClassifications = np.append(npaClassifications, npaClassification, 0)
			np.savetxt("trainingData/classifications.txt", npaClassifications)
			np.savetxt("trainingData/flattenedImages.txt", npaFlattenedImages)
			return
		else:
			print("Notification: you just pressed the wrong key, please do it again.")
			continue
# end train_model


if __name__ == "__main__":
	main()
# end if