import numpy as np
import cv2
import os


from possiblePlateLocatedRegionDetection import *
from plateDetectionAndProcessing import *
from characterSegmentation import *
from characterRecognition import *


# Constants:
DETECTION_CRITERION_THRESHOLD = 0.5
TEST_IMAGE_DIRECTORY = 'C:/Users/MJ/Documents/Datasets/LicensePlatesData/data_100Img/'
IMAGE_FILE_EXTENSION = '.jpg'
LABEL_FILE_EXTENSION = '.txt'
NONE_CHARACTER = '.' # this code indicate that image does not represent a character
EMPTY_STRING = ""


def main():
	detectionAccuracy, characterWiseRecognitionAccuracy, plateWiseRecognitionAccuracy = evaluate_alprs_on_a_test_set(TEST_IMAGE_DIRECTORY, IMAGE_FILE_EXTENSION, LABEL_FILE_EXTENSION)
	print('detectionAccuracy', detectionAccuracy)
	print('characterWiseRecognitionAccuracy', characterWiseRecognitionAccuracy)
	print('plateWiseRecognitionAccuracy', plateWiseRecognitionAccuracy)
	cv2.destroyAllWindows()
	return
# end main


def evaluate_alprs_on_a_test_set(testDirectory, imageFileExtension, labelFileExtension):
  assert(type(testDirectory) == str and type(imageFileExtension) == str and type(labelFileExtension) == str)
  directory = os.fsencode(testDirectory)
  fileList = os.listdir(directory)
  fileListLength = len(fileList)
  assert(fileListLength % 2 == 0)
  numberOfTestUnits = int(fileListLength)/2


  numberOfCorrectlyDetectedPlates = 0
  characterWiseRecognitionAccuracyNumerator = 0
  characterWiseRecognitionAccuracyDenominator = 0
  numberOfCorrectlyRecognizedPlates = 0


  for i in range(fileListLength):
    print('')
    file = fileList[i]
    fileName = os.fsdecode(file)
    if fileName.endswith(imageFileExtension):
      image = cv2.imread(testDirectory + fileName)
      nextFile = fileList[i + 1]
      nextFileName = os.fsdecode(nextFile)
      if nextFileName.endswith(labelFileExtension):
        with open(testDirectory + nextFileName, 'r') as labelFile:
          label = labelFile.read()
      else:
        print("Error: nextFile should have been the label of previous file!!!")
        os.system("pause")
        continue 

      detectionEvaluation, numberOfCorrectlyRecognizedCharacters, numberOfNecessarilyRecognizedCharacters, recognitionEvaluation = evaluate_alprs_on_an_image(image, label)
      print('detectionEvaluation', detectionEvaluation)
      print('numberOfCorrectlyRecognizedCharacters', numberOfCorrectlyRecognizedCharacters)
      print('numberOfNecessarilyRecognizedCharacters', numberOfNecessarilyRecognizedCharacters)
      print('recognitionEvaluation', recognitionEvaluation)
      if detectionEvaluation == True:
        numberOfCorrectlyDetectedPlates = numberOfCorrectlyDetectedPlates + 1
      else:
        continue
      characterWiseRecognitionAccuracyNumerator = characterWiseRecognitionAccuracyNumerator + numberOfCorrectlyRecognizedCharacters
      characterWiseRecognitionAccuracyDenominator = characterWiseRecognitionAccuracyDenominator + numberOfNecessarilyRecognizedCharacters
      if recognitionEvaluation == True:
        numberOfCorrectlyRecognizedPlates = numberOfCorrectlyRecognizedPlates + 1

      print('numberOfCorrectlyDetectedPlates', numberOfCorrectlyDetectedPlates)
      print('characterWiseRecognitionAccuracyNumerator', characterWiseRecognitionAccuracyNumerator)
      print('characterWiseRecognitionAccuracyDenominator', characterWiseRecognitionAccuracyDenominator)
      print('numberOfCorrectlyRecognizedPlates', numberOfCorrectlyRecognizedPlates)
      #cv2.imshow('image', image)
      #print(label)
      #cv2.waitKey(0)
    else:
      continue


  detectionAccuracy = float(numberOfCorrectlyDetectedPlates)/numberOfTestUnits
  characterWiseRecognitionAccuracy = float(characterWiseRecognitionAccuracyNumerator)/characterWiseRecognitionAccuracyDenominator
  plateWiseRecognitionAccuracy = float(numberOfCorrectlyRecognizedPlates)/numberOfTestUnits


  return detectionAccuracy, characterWiseRecognitionAccuracy, plateWiseRecognitionAccuracy
# end evaluate_alprs_on_a_test_set


def evaluate_alprs_on_an_image(image, label):
  assert(image is not None and label != "")


  numberOfPlates, labelledRectX, labelledRectY, labelledRectWidth, labelledRectHeight, strPlate = label.split()


  numberOfNecessarilyRecognizedCharacters = len(strPlate) - 1 - strPlate.count('*')


  labelledRect = [int(labelledRectX), int(labelledRectY), int(labelledRectWidth), int(labelledRectHeight)]
  imgPlateLocatedRegion, imgSecondStepChosenContour, imgSceneResized, boundingRect = detect_plate_located_region(image)
  if imgPlateLocatedRegion is None or imgSecondStepChosenContour is None:
    print("Notification: couldn't find any possible plate-located region in this image.")
    return False, 0, numberOfNecessarilyRecognizedCharacters, False


  detectionEvaluation = evaluate_detection_of_alprs_on_an_image(image, labelledRect, boundingRect)


  imgPlate = detect_plate(imgPlateLocatedRegion, imgSecondStepChosenContour)
  if imgPlate is None:
    print("Error: the processing process of possible plate-located region yielded an error!!!")
    cv2.destroyAllWindows()
    return True, 0, numberOfNecessarilyRecognizedCharacters, False


  numberOfCorrectlyRecognizedCharacters, recognitionEvaluation = evaluate_recognition_of_alprs_on_an_image(imgPlate, strPlate, numberOfNecessarilyRecognizedCharacters)


  return detectionEvaluation, numberOfCorrectlyRecognizedCharacters, numberOfNecessarilyRecognizedCharacters, recognitionEvaluation
# end evaluate_alprs_on_an_image


def evaluate_detection_of_alprs_on_an_image(image, labelledRect, boundingRect):
  imageHeight, imageWidth, imageChannel = image.shape
  labelledRectImage = np.zeros((imageHeight, imageWidth), np.uint8)
  respondedRectImage = labelledRectImage.copy()


  labelledRectX, labelledRectY, labelledRectWidth, labelledRectHeight = labelledRect
  cv2.rectangle(labelledRectImage, (labelledRectX, labelledRectY), (labelledRectX + labelledRectWidth, labelledRectY + labelledRectHeight), (255, 255, 255), -1)
  #cv2.rectangle(image, (labelledRectX, labelledRectY), (labelledRectX + labelledRectWidth, labelledRectY + labelledRectHeight), (0, 255, 0), 2)
  

  boundingRectX, boundingRectY, boundingRectWidth, boundingRectHeight = boundingRect
  cv2.rectangle(respondedRectImage, (boundingRectX, boundingRectY), (boundingRectX + boundingRectWidth, boundingRectY + boundingRectHeight), (255, 255, 255), -1)
  #cv2.rectangle(image, (boundingRectX, boundingRectY), (boundingRectX + boundingRectWidth, boundingRectY + boundingRectHeight), (0, 0, 255), 2)

  
  intersectionArea = calculate_intersection_area_between_two_rectangle(labelledRectImage, respondedRectImage)
  unionArea = calculate_union_area_between_two_rectangle(labelledRectImage, respondedRectImage)


  detectionCriterion = float(intersectionArea)/unionArea
  return detection_is_correct(detectionCriterion, DETECTION_CRITERION_THRESHOLD)


def evaluate_recognition_of_alprs_on_an_image(imgPlate, strPlate, numberOfNecessarilyRecognizedCharacters):
  numberOfCorrectlyRecognizedCharacters = 0
  listOfTopImgCharacter, listOfBottomImgCharacter = segment_character(imgPlate)
  if listOfTopImgCharacter == [] or listOfBottomImgCharacter == []:
    print("Error: couldn't crop image of plate into list of image of character!!!")
    return 0, False
  listOfImgCharacter = listOfTopImgCharacter + listOfBottomImgCharacter


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
    return 0, False


  print('strOfCharacters', strOfCharacters)
  print('strPlate', strPlate)


  respondedDashIndex = strOfCharacters.index('-')
  labelledDashIndex = strPlate.index('-')
  numberOfCorrectlyRecognizedCharacters = mesure_length_of_same_longest_sub_string_of_two_strings_from_two_ends(strOfCharacters[:respondedDashIndex], strPlate[:labelledDashIndex]) + mesure_length_of_same_longest_sub_string_of_two_strings_from_two_ends(strOfCharacters[respondedDashIndex + 1:], strPlate[labelledDashIndex + 1:])


  '''
  if strOfCharacters == strPlate:
    recognitionEvaluation = True
  else:
    recognitionEvaluation = False
  '''
  recognitionEvaluation = (numberOfNecessarilyRecognizedCharacters == numberOfCorrectlyRecognizedCharacters)


  return numberOfCorrectlyRecognizedCharacters, recognitionEvaluation


def calculate_intersection_area_between_two_rectangle(firstRectangleImage, secondRectangleImage):
  intersectionImage = cv2.bitwise_and(firstRectangleImage, firstRectangleImage, mask = secondRectangleImage)
  return np.sum(intersectionImage == 255)


def calculate_union_area_between_two_rectangle(firstRectangleImage, secondRectangleImage):
  unionImage = cv2.bitwise_or(firstRectangleImage, firstRectangleImage, mask = secondRectangleImage)
  return np.sum(unionImage == 255)


def detection_is_correct(detectionCriterion, threshold):
  return detectionCriterion > threshold


def mesure_length_of_same_longest_sub_string_of_two_strings_from_two_ends(firstStr, secondStr):
  return max(mesure_length_of_possible_same_sub_string_of_two_strings_from_left_to_right(firstStr, secondStr),
    mesure_length_of_possible_same_sub_string_of_two_strings_from_left_to_right(firstStr[::-1], secondStr[::-1]))


'''
def mesure_length_of_possible_same_sub_string_of_two_strings_from_left_to_right(firstStr, secondStr):
  i = 0
  maxIndex = min(len(firstStr), len(secondStr)) - 1
  while firstStr[i] == secondStr[i] and i < maxIndex:
    i = i + 1
  if firstStr[i] == secondStr[i]:
    i = i + 1
  return i
'''


def mesure_length_of_possible_same_sub_string_of_two_strings_from_left_to_right(firstStr, secondStr):
  length = 0
  l = min(len(firstStr), len(secondStr))
  for i in range(l):
    if firstStr[i] == secondStr[i]:
      length = length + 1
  return length


if __name__ == "__main__":
  main()
# end if