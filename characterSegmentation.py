import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


from main import *
from possiblePlateLocatedRegionDetection import *
from plateDetectionAndProcessing import *


# Constants:
PLATE_ASPECT_RATIO = float(19)/14
PLATE_HEIGHT = 200
PLATE_WIDTH = int(PLATE_HEIGHT * PLATE_ASPECT_RATIO)
OX_INTENSITY_SUM_THRESH_AND_PLATE_WIDTH_RATIO = 170
OY_INTENSITY_SUM_THRESH_AND_PLATE_HEIGHT_RATIO = 230
SHIFT_AND_PLATE_HEIGHT_RATIO = 0.125
SHIFT_AND_PLATE_WIDTH_RATIO = 0.092
SHIFT_AWAY_LOCAL_MINIMUM = int(float(30/(200 * PLATE_ASPECT_RATIO)) * PLATE_WIDTH) 
MAX_ALLOWED_NUMBER_OF_CONNECTED_COMPONENT_IN_ONE_IMAGE_OF_CHARACTER = 5
MINIMUM_ALLOWED_AREA_OF_ONE_CONNNECTED_COMPONENT_IN_ONE_IMAGE = int((float(150)/(200 * 200 * float(19)/14)) * PLATE_HEIGHT * PLATE_WIDTH)
MINIMUM_NUMBER_OF_WHITE_PIXEL_IN_ONE_IMAGE_OF_CHARACTER = int((float(1000)/(200 * 200 * float(19)/14)) * PLATE_HEIGHT * PLATE_WIDTH)
MAX_WIDTH_OF_ONE_CHARACTER = int((float(54)/(200 * (float(19)/14))) * PLATE_WIDTH)
MIN_WIDTH_OF_TWO_CHARACTER = int((float(85)/(200 * (float(19)/14))) * PLATE_WIDTH)
STANDARD_MAX_INTENSITY_SUM = int((float(22000)/(200)) * PLATE_HEIGHT)
THRESHOLD_AND_MAX_INTENSITY_SUM_RATIO_IN_CASE_ONE = float(1500)/STANDARD_MAX_INTENSITY_SUM #1500
THRESHOLD_AND_MAX_INTENSITY_SUM_RATIO_IN_CASE_TWO = float(6000)/STANDARD_MAX_INTENSITY_SUM #6000
THRESHOLD_AND_MAX_INTENSITY_SUM_RATIO_IN_CASE_THREE = float(9000)/STANDARD_MAX_INTENSITY_SUM #9000
showSteps = False


def calculate_list_of_intensity_sum(imgBinary, isOXDirection):
	height, width = imgBinary.shape
	if isOXDirection == True:
		listOfHeightIntensity = []
		for j in range(0, height):
			sum = 0
			for i in range(0, width):
				sum = sum + imgBinary[j][i]
			listOfHeightIntensity.append(sum)
		return listOfHeightIntensity
	else:
		listOfWidthIntensity = []
		for i in range(0, width):
			sum = 0
			for j in range(0, height):
				sum = sum + imgBinary[j][i]
			listOfWidthIntensity.append(sum)
		return listOfWidthIntensity
# end calculate_list_of_intensity_sum


def draw_bar_chart_of_binary_image(imgBinary, isOXDirection):
	listOfIntensity = calculate_list_of_intensity_sum(imgBinary, isOXDirection)
	x = range(len(listOfIntensity))
	width = 1
	plt.bar(x, listOfIntensity, width, align = 'center', color = 'blue')
	if isOXDirection:
		plt.xlabel('Ordination y')
		plt.ylabel('Intensity sum of pixels with same ordination y')
	else:
		plt.xlabel('Ordination x')
		plt.ylabel('Intensity sum of pixels with same ordination x')
	plt.show()
# end draw_bar_chart_of_binary_image


def divide_plate(imgBinary):
	height, width = imgBinary.shape
	imgLeftWidth = int(width/2)
	imgLeft = imgBinary[:,0:imgLeftWidth]
	imgRight = imgBinary[:,imgLeftWidth:width]
	begin = int(float(height * 7)/20)
	midle = int(height/2)
	end = int(float(height * 13)/20)

	listOfIntensity = calculate_list_of_intensity_sum(imgLeft, True)
	lOIOne = listOfIntensity[begin:midle]
	maxOne = max(lOIOne)
	maxIndexOne = begin + lOIOne.index(maxOne)
	lOITwo = listOfIntensity[midle:end + 1]
	maxTwo = max(lOITwo)
	maxIndexTwo = midle + lOITwo.index(maxTwo)
	minIndexLeft = int((maxIndexOne + maxIndexTwo)/2)

	listOfIntensity = calculate_list_of_intensity_sum(imgRight, True)
	lOIOne = listOfIntensity[begin:midle]
	maxOne = max(lOIOne)
	maxIndexOne = begin + lOIOne.index(maxOne)
	lOITwo = listOfIntensity[midle:end + 1]
	maxTwo = max(lOITwo)
	maxIndexTwo = midle + lOITwo.index(maxTwo)
	minIndexRight = int((maxIndexOne + maxIndexTwo)/2)

	imgTopHeight = int(height/2)
	pts1 = np.float32([[0, 0], [width - 1, 0], [0, minIndexLeft], [width - 1, minIndexRight]])
	pts2 = np.float32([[0, 0], [width - 1, 0], [0, imgTopHeight - 1], [width - 1, imgTopHeight - 1]])
	M, status = cv2.findHomography(pts1, pts2)
	imgTop = cv2.warpPerspective(imgBinary, M, (width, imgTopHeight))

	imgBottomHeight = imgTopHeight
	pts1 = np.float32([[0, minIndexLeft + 1], [width - 1, minIndexRight + 1], [0, height - 1], [width - 1, height - 1]])
	M, status = cv2.findHomography(pts1, pts2)
	imgBottom = cv2.warpPerspective(imgBinary, M, (width, imgTopHeight))
	return imgTop, imgBottom
# end divide_plate


def preprocess(imgBinary):
	imgPlate = slice_plate_edge(imgBinary)
	return imgPlate
# end preprocess


def slice_plate_edge(imgPlate):
	imgPlate = slice_horizontal_plate_edge(imgPlate)
	imgPlate = slice_vertical_plate_edge(imgPlate)
	return imgPlate
# end slice_plate_edge


def slice_horizontal_plate_edge(imgPlate):
	imgPlateHeight, imgPlateWidth = imgPlate.shape
	beginningY = calculate_slicing_coordination(imgPlate, True, True)
	endingY = calculate_slicing_coordination(imgPlate, True, False)
	if endingY <= beginningY:
		return imgPlate
	imgPlateNewHeight = endingY - beginningY + 1
	if beginningY < 0:
		beginningY = 0
	if beginningY + imgPlateNewHeight > imgPlateHeight:
		imgPlateNewHeight = imgPlateHeight - beginningY
	imgPlate = imgPlate[beginningY:beginningY+imgPlateNewHeight, :]
	imgPlate = cv2.resize(imgPlate, (imgPlateWidth, imgPlateHeight), interpolation = cv2.INTER_AREA)
	return imgPlate
# end slice_horizontal_plate_edge


def slice_vertical_plate_edge(imgPlate):
	if showSteps == True:
		cv2.imshow('imgPlate', imgPlate)
	imgPlateHeight, imgPlateWidth = imgPlate.shape
	beginningX = calculate_slicing_coordination(imgPlate, False, True)
	endingX = calculate_slicing_coordination(imgPlate, False, False)
	if endingX <= beginningX:
		return imgPlate
	imgPlateNewWidth = endingX - beginningX + 1
	if beginningX < 0:
		beginningX = 0
	if beginningX + imgPlateNewWidth > imgPlateWidth:
		imgPlateNewWidth = imgPlateWidth - beginningX
	imgPlate = imgPlate[:, beginningX:beginningX+imgPlateNewWidth]
	imgPlate = cv2.resize(imgPlate, (imgPlateWidth, imgPlateHeight), interpolation = cv2.INTER_AREA)
	return imgPlate
# end slice_vertical_plate_edge


def calculate_slicing_coordination(imgPlate, isHorizontalSlicingMethod, isBackward):
	imgPlateHeight, imgPlateWidth = imgPlate.shape
	centerX = int(imgPlateWidth/2)
	centerY = int(imgPlateHeight/2)
	if isHorizontalSlicingMethod:
		listOfIntensitySum = calculate_list_of_intensity_sum(imgPlate, True)
		intensitySumThresh = OX_INTENSITY_SUM_THRESH_AND_PLATE_WIDTH_RATIO * imgPlateWidth
		shift = int(SHIFT_AND_PLATE_HEIGHT_RATIO * imgPlateHeight)
		if isBackward:
			listOfBackwardIntensitySum = listOfIntensitySum[0:centerY]
			maxBackwardIntensitySum = max(listOfBackwardIntensitySum)
			while (maxBackwardIntensitySum < intensitySumThresh):
				listOfBackwardIntensitySum.remove(maxBackwardIntensitySum)
				if len(listOfBackwardIntensitySum) == 0:
					secondListOfBackwardIntensitySum = listOfIntensitySum[0:shift]
					minSecondBackwardIntensitySum = min(secondListOfBackwardIntensitySum)
					minSecondBackwardIntensitySumIndex = secondListOfBackwardIntensitySum.index(minSecondBackwardIntensitySum)
					return minSecondBackwardIntensitySumIndex
				maxBackwardIntensitySum = max(listOfBackwardIntensitySum)
			maxBackwardIntensitySumIndex = listOfBackwardIntensitySum.index(maxBackwardIntensitySum)
			secondListOfBackwardIntensitySum = listOfIntensitySum[maxBackwardIntensitySumIndex + 1: maxBackwardIntensitySumIndex + 1 + shift]
			minSecondBackwardIntensitySum = min(secondListOfBackwardIntensitySum)
			minSecondBackwardIntensitySumIndex = secondListOfBackwardIntensitySum.index(minSecondBackwardIntensitySum)
			return minSecondBackwardIntensitySumIndex + (maxBackwardIntensitySumIndex + 1)
		else:
			listOfForwardIntensitySum = listOfIntensitySum[centerY + 1:imgPlateHeight]
			maxForwardIntensitySum = max(listOfForwardIntensitySum)
			while (maxForwardIntensitySum < intensitySumThresh):
				listOfForwardIntensitySum.remove(maxForwardIntensitySum)
				if len(listOfForwardIntensitySum) == 0:
					secondListOfForwardIntensitySum = listOfIntensitySum[imgPlateHeight - shift: imgPlateHeight]
					minSecondForwardIntensitySum = min(secondListOfForwardIntensitySum)
					minSecondForwardIntensitySumIndex = secondListOfForwardIntensitySum.index(minSecondForwardIntensitySum)
					return minSecondForwardIntensitySumIndex + (imgPlateHeight - shift)
				maxForwardIntensitySum = max(listOfForwardIntensitySum)
			maxForwardIntensitySumIndex = listOfForwardIntensitySum.index(maxForwardIntensitySum)
			secondListOfForwardIntensitySum = listOfIntensitySum[(centerY + 1) + maxForwardIntensitySumIndex - shift: (centerY + 1) + maxForwardIntensitySumIndex]
			minSecondForwardIntensitySum = min(secondListOfForwardIntensitySum)
			minSecondForwardIntensitySumIndex = secondListOfForwardIntensitySum.index(minSecondForwardIntensitySum)
			return minSecondForwardIntensitySumIndex + (centerY + 1) + maxForwardIntensitySumIndex - shift
	else:
		listOfIntensitySum = calculate_list_of_intensity_sum(imgPlate, False)
		intensitySumThresh = OY_INTENSITY_SUM_THRESH_AND_PLATE_HEIGHT_RATIO * imgPlateHeight
		shift = int(SHIFT_AND_PLATE_WIDTH_RATIO * imgPlateWidth)
		if isBackward:
			listOfBackwardIntensitySum = listOfIntensitySum[0:centerX]  #
			maxBackwardIntensitySum = max(listOfBackwardIntensitySum)
			while (maxBackwardIntensitySum < intensitySumThresh):
				listOfBackwardIntensitySum.remove(maxBackwardIntensitySum)
				if len(listOfBackwardIntensitySum) == 0:
					secondListOfBackwardIntensitySum = listOfIntensitySum[0:shift]
					minSecondBackwardIntensitySum = min(secondListOfBackwardIntensitySum)
					minSecondBackwardIntensitySumIndex = secondListOfBackwardIntensitySum.index(minSecondBackwardIntensitySum)
					return minSecondBackwardIntensitySumIndex
				maxBackwardIntensitySum = max(listOfBackwardIntensitySum)
			maxBackwardIntensitySumIndex = listOfBackwardIntensitySum.index(maxBackwardIntensitySum)
			secondListOfBackwardIntensitySum = listOfIntensitySum[maxBackwardIntensitySumIndex + 1: maxBackwardIntensitySumIndex + 1 + shift]
			minSecondBackwardIntensitySum = min(secondListOfBackwardIntensitySum)
			minSecondBackwardIntensitySumIndex = secondListOfBackwardIntensitySum.index(minSecondBackwardIntensitySum)
			return minSecondBackwardIntensitySumIndex + (maxBackwardIntensitySumIndex + 1)
		else:
			listOfForwardIntensitySum = listOfIntensitySum[centerX + 1:imgPlateWidth] #
			maxForwardIntensitySum = max(listOfForwardIntensitySum)
			while (maxForwardIntensitySum < intensitySumThresh):
				listOfForwardIntensitySum.remove(maxForwardIntensitySum)
				if len(listOfForwardIntensitySum) == 0:
					secondListOfForwardIntensitySum = listOfIntensitySum[imgPlateWidth - shift: imgPlateWidth] #
					minSecondForwardIntensitySum = min(secondListOfForwardIntensitySum)
					minSecondForwardIntensitySumIndex = secondListOfForwardIntensitySum.index(minSecondForwardIntensitySum)
					return minSecondForwardIntensitySumIndex + (imgPlateWidth - shift) #
				maxForwardIntensitySum = max(listOfForwardIntensitySum)
			maxForwardIntensitySumIndex = listOfForwardIntensitySum.index(maxForwardIntensitySum)
			secondListOfForwardIntensitySum = listOfIntensitySum[(centerX + 1) + maxForwardIntensitySumIndex - shift: (centerX + 1) + maxForwardIntensitySumIndex] #
			minSecondForwardIntensitySum = min(secondListOfForwardIntensitySum)
			minSecondForwardIntensitySumIndex = secondListOfForwardIntensitySum.index(minSecondForwardIntensitySum)
			return minSecondForwardIntensitySumIndex + (centerX + 1) + maxForwardIntensitySumIndex - shift #
# end calculate_slicing_coordination


def slice_plate_vertically(imgBinary):
	imgBinaryHeight, imgBinaryWidth = imgBinary.shape
	listOfIntensitySum = calculate_list_of_intensity_sum(imgBinary, isOXDirection = False)
	threshold = calculate_threshold_for_choosing_coordination_to_slice_the_image_of_character(imgBinary, listOfIntensitySum)
	listOfLocalMinimumOfIntensitySumIndex = find_list_of_local_minimum_of_intensity_sum_index(listOfIntensitySum, SHIFT_AWAY_LOCAL_MINIMUM, threshold)
	listOfSlicingIndexCouple = find_list_of_slicing_index_couple(listOfLocalMinimumOfIntensitySumIndex)
	listOfSlicingImage = []
	for i in range(len(listOfSlicingIndexCouple)):
		slicingIndexCouple = listOfSlicingIndexCouple[i]
		beginningSlicingIndex = slicingIndexCouple[0]
		endingSlisingIndex = slicingIndexCouple[1]
		slicingImage = imgBinary[:, beginningSlicingIndex:endingSlisingIndex+1]
		listOfSlicingImage.append(slicingImage)
	return listOfSlicingImage
# end slice_plate_vertically


def calculate_threshold_for_choosing_coordination_to_slice_the_image_of_character(imgBinary, listOfIntensitySum):
	height, width = imgBinary.shape
	maxIntensitySum = max(listOfIntensitySum)
	retVal, labels, stats, centroids = cv2.connectedComponentsWithStats(image = imgBinary, connectivity = 8)
	maxBoundingRectWidth = 0
	for i in range(1, list(stats.shape)[0]):
		stat = stats[i]
		boundingRectWidth = stat[2]
		maxBoundingRectWidth = max(maxBoundingRectWidth, boundingRectWidth)
	if maxBoundingRectWidth <= MAX_WIDTH_OF_ONE_CHARACTER:
		threshold = int(THRESHOLD_AND_MAX_INTENSITY_SUM_RATIO_IN_CASE_ONE * maxIntensitySum)
		return threshold
	elif maxBoundingRectWidth > MAX_WIDTH_OF_ONE_CHARACTER and maxBoundingRectWidth < MIN_WIDTH_OF_TWO_CHARACTER:
		threshold = int(THRESHOLD_AND_MAX_INTENSITY_SUM_RATIO_IN_CASE_TWO * maxIntensitySum)
		return threshold
	else:
		threshold = int(THRESHOLD_AND_MAX_INTENSITY_SUM_RATIO_IN_CASE_THREE * maxIntensitySum)
		return threshold
# end calculate_threshold_for_choosing_coordination_to_slice_the_image_of_character


def find_list_of_local_minimum_of_intensity_sum_index(listOfIntensitySum, shiftAwayLocalMinimum, threshold):
	listOfLocalMinimumOfIntensitySumIndex = []
	listOfIntensitySumLength = len(listOfIntensitySum)
	if listOfIntensitySumLength == 0:
		return listOfLocalMinimumOfIntensitySumIndex
	globalMinimumOfIntensitySum = min(listOfIntensitySum)
	if globalMinimumOfIntensitySum < threshold:
		globalMinimumOfIntensitySumIndex = listOfIntensitySum.index(globalMinimumOfIntensitySum)
		listOfLocalMinimumOfIntensitySumIndex.append(globalMinimumOfIntensitySumIndex)
		if globalMinimumOfIntensitySumIndex - shiftAwayLocalMinimum >= 0 and globalMinimumOfIntensitySumIndex + shiftAwayLocalMinimum <= listOfIntensitySumLength - 1:
			listLeftOfIntensitySum = listOfIntensitySum[0:globalMinimumOfIntensitySumIndex - shiftAwayLocalMinimum]
			listLeftOfLocalMinimumOfIntensitySumIndex = find_list_of_local_minimum_of_intensity_sum_index(listLeftOfIntensitySum, shiftAwayLocalMinimum, threshold)
			listRightOfIntensitySum = listOfIntensitySum[globalMinimumOfIntensitySumIndex + shiftAwayLocalMinimum:listOfIntensitySumLength]
			listRightOfLocalMinimumOfIntensitySumIndex = find_list_of_local_minimum_of_intensity_sum_index(listRightOfIntensitySum, shiftAwayLocalMinimum, threshold)
			listRightOfLocalMinimumOfIntensitySumIndex = [(element + globalMinimumOfIntensitySumIndex + shiftAwayLocalMinimum) for element in listRightOfLocalMinimumOfIntensitySumIndex]
			listOfLocalMinimumOfIntensitySumIndex = listOfLocalMinimumOfIntensitySumIndex + listLeftOfLocalMinimumOfIntensitySumIndex + listRightOfLocalMinimumOfIntensitySumIndex
		elif globalMinimumOfIntensitySumIndex - shiftAwayLocalMinimum >= 0 and globalMinimumOfIntensitySumIndex + shiftAwayLocalMinimum > listOfIntensitySumLength - 1:
			listLeftOfIntensitySum = listOfIntensitySum[0:globalMinimumOfIntensitySumIndex - shiftAwayLocalMinimum]
			listLeftOfLocalMinimumOfIntensitySumIndex = find_list_of_local_minimum_of_intensity_sum_index(listLeftOfIntensitySum, shiftAwayLocalMinimum, threshold)
			listOfLocalMinimumOfIntensitySumIndex = listOfLocalMinimumOfIntensitySumIndex + listLeftOfLocalMinimumOfIntensitySumIndex
		elif globalMinimumOfIntensitySumIndex - shiftAwayLocalMinimum < 0 and globalMinimumOfIntensitySumIndex + shiftAwayLocalMinimum <= listOfIntensitySumLength - 1:
			listRightOfIntensitySum = listOfIntensitySum[globalMinimumOfIntensitySumIndex + shiftAwayLocalMinimum:listOfIntensitySumLength]
			listRightOfLocalMinimumOfIntensitySumIndex = find_list_of_local_minimum_of_intensity_sum_index(listRightOfIntensitySum, shiftAwayLocalMinimum, threshold)
			listRightOfLocalMinimumOfIntensitySumIndex = [(element + globalMinimumOfIntensitySumIndex + shiftAwayLocalMinimum) for element in listRightOfLocalMinimumOfIntensitySumIndex]
			listOfLocalMinimumOfIntensitySumIndex = listOfLocalMinimumOfIntensitySumIndex + listRightOfLocalMinimumOfIntensitySumIndex
		else:
			pass
		listOfLocalMinimumOfIntensitySumIndex.sort()
		return listOfLocalMinimumOfIntensitySumIndex
	else:
		return listOfLocalMinimumOfIntensitySumIndex
# end find_list_of_local_minimum_of_intensity_sum_index


def find_list_of_slicing_index_couple(listOfLocalMinimumOfIntensitySumIndex):
	listOfSlicingIndexCouple = []
	for i in range(len(listOfLocalMinimumOfIntensitySumIndex) - 1):
		coupleIFirstIndex = listOfLocalMinimumOfIntensitySumIndex[i]
		coupleISecondIndex = listOfLocalMinimumOfIntensitySumIndex[i + 1]
		coupleI = [coupleIFirstIndex, coupleISecondIndex]
		listOfSlicingIndexCouple.append(coupleI)
	return listOfSlicingIndexCouple
# end find_list_of_slicing_index_couple


def postprocess(listOfImgPossibleCharacter):
	listOfImgCharacter = []
	listOfChosenConnectedComponentWidth = []
	listOfChosenConnectedComponentHeight = []
	listOfImgPossibleCharacterLength = len(listOfImgPossibleCharacter)
	for imgPossibleCharacter in listOfImgPossibleCharacter:
		imgPossibleCharacter, numberOfChosenConnectedComponent, maxChosenConnectedComponentWidth, maxChosenConnectedComponentHeight, numberOfWhitePixel = remove_small_connected_component_in_image(imgPossibleCharacter)
		listOfChosenConnectedComponentWidth.append(maxChosenConnectedComponentWidth)
		listOfChosenConnectedComponentHeight.append(maxChosenConnectedComponentHeight)
		if do_have_many_connected_component_in_image(numberOfChosenConnectedComponent):
			continue
		else:
			if do_have_small_number_of_white_pixel(numberOfWhitePixel):
				continue
			else:
				listOfImgCharacter.append(imgPossibleCharacter)
	if listOfChosenConnectedComponentWidth == []:
		print("Notification: couldn't find any plate in image.\n")
		return []
	standardWidth = int(max(listOfChosenConnectedComponentWidth))
	standardHeight = int(max(listOfChosenConnectedComponentHeight))
	for i in range(len(listOfImgCharacter)):
		imgCharacter = listOfImgCharacter[i]
		listOfOXIntensitySum = calculate_list_of_intensity_sum(imgCharacter, True)
		imgCharacterHeight, imgCharacterWidth = imgCharacter.shape
		topMargin, bottomMargin = calculate_vertical_margin(imgCharacterHeight, listOfOXIntensitySum)
		y1, y2 = calculate_oy_slicing_coordination_for_character(imgCharacterHeight, listOfOXIntensitySum, topMargin, bottomMargin, standardHeight)
		imgCharacter = imgCharacter[y1:y2-1, :]

		listOfOYIntensitySum = calculate_list_of_intensity_sum(imgCharacter, False)
		imgCharacterHeight, imgCharacterWidth = imgCharacter.shape
		leftMargin, rightMargin = calculate_vertical_margin(imgCharacterWidth, listOfOYIntensitySum)
		x1, x2 = calculate_ox_slicing_coordination_for_character(imgCharacterWidth, listOfOYIntensitySum, leftMargin, rightMargin, standardWidth)
		imgCharacter = imgCharacter[:, x1:x2-1]
		listOfImgCharacter[i] = imgCharacter
	return listOfImgCharacter
# end postprocess


def do_have_many_connected_component_in_image(numberOfChosenConnectedComponent):
	if numberOfChosenConnectedComponent > MAX_ALLOWED_NUMBER_OF_CONNECTED_COMPONENT_IN_ONE_IMAGE_OF_CHARACTER:
		return True
	return False
# end do_have_many_connected_component_in_image


def do_have_small_number_of_white_pixel(numberOfWhitePixel):
	if numberOfWhitePixel < MINIMUM_NUMBER_OF_WHITE_PIXEL_IN_ONE_IMAGE_OF_CHARACTER:
		return True
	return False
# end do_have_small_number_of_white_pixel


def calculate_vertical_margin(height, listOfOXIntensitySum):
	y2 = height - 1
	topMargin, bottomMargin = 0, 0
	while listOfOXIntensitySum[topMargin] == 0:
		topMargin = topMargin + 1
	while listOfOXIntensitySum[y2] == 0:
		y2 = y2 - 1
		bottomMargin = bottomMargin + 1
	return topMargin, bottomMargin
# end calculate_vertical_margin


def calculate_horizontal_margin(width, listOfOYIntensitySum):
	x2 = width - 1
	leftMargin, rightMargin = 0, 0
	while listOfOYIntensitySum[leftMargin] == 0:
		leftMargin = leftMargin + 1
	while listOfOYIntensitySum[x2] == 0:
		x2 = x2 - 1
		rightMargin = rightMargin + 1
	return leftMargin, rightMargin
# end calculate_horizontal_margin


def remove_small_connected_component_in_image(imgPossibleCharacter):
	imgPossibleCharacterHeight, imgPossibleCharacterWidth = imgPossibleCharacter.shape
	maxChosenConnectedComponentWidth = 0
	maxChosenConnectedComponentHeight = 0
	numberOfChosenConnectedComponent = 0
	numberOfWhitePixel = 0
	retVal, labels, stats, centroids = cv2.connectedComponentsWithStats(image = imgPossibleCharacter, connectivity = 8)
	for i in range(1, list(stats.shape)[0]):
		stat = stats[i]
		topLeftX = stat[0]
		topLeftY = stat[1]
		boundingRectWidth = stat[2]
		boundingRectHeight = stat[3]
		connectedComponentArea = stat[4]
		centroid = centroids[i]
		centroidX = centroid[0]
		centroidY = centroid[1]
		if connectedComponentArea < MINIMUM_ALLOWED_AREA_OF_ONE_CONNNECTED_COMPONENT_IN_ONE_IMAGE:
			bottomRightX = topLeftX + boundingRectWidth
			bottomRightY = topLeftY + boundingRectHeight
			# remove small connected component
			for h in range(imgPossibleCharacterWidth):
				for k in range(imgPossibleCharacterHeight):
					if labels[k][h] == i:
						imgPossibleCharacter[k][h] = 0
		else:
			maxChosenConnectedComponentWidth = max(maxChosenConnectedComponentWidth, boundingRectWidth)
			maxChosenConnectedComponentHeight = max(maxChosenConnectedComponentHeight, boundingRectHeight)
			numberOfChosenConnectedComponent = numberOfChosenConnectedComponent + 1
			numberOfWhitePixel = numberOfWhitePixel + connectedComponentArea
	return imgPossibleCharacter, numberOfChosenConnectedComponent, maxChosenConnectedComponentWidth, maxChosenConnectedComponentHeight, numberOfWhitePixel
# end remove_small_connected_component_in_image


def calculate_oy_slicing_coordination_for_character(imgCharacterHeight, listOfOXIntensitySum, topMargin, bottomMargin, standardHeight):
	maxIntensitySum = max(listOfOXIntensitySum)
	maxIntensitySumIndex = listOfOXIntensitySum.index(maxIntensitySum)
	if maxIntensitySumIndex - topMargin > bottomMargin - maxIntensitySumIndex:
		y2 = imgCharacterHeight - bottomMargin
		y1 = y2 - standardHeight + 1
		if y1 < 0:
			y1 = 0
		while listOfOXIntensitySum[y1] > 0 and y1 > 0:
			y1 = y1 - 1
	else:
		y1 = topMargin
		y2 = y1 + standardHeight - 1
		if y2 > imgCharacterHeight - 1:
			y2 = imgCharacterHeight - 1
		while listOfOXIntensitySum[y2] > 0 and y2 < imgCharacterHeight - 1:
			y2 = y2 + 1
	return y1, y2
# end calculate_oy_slicing_coordination_for_character


def calculate_ox_slicing_coordination_for_character(imgCharacterWidth, listOfOYIntensitySum, leftMargin, rightMargin, standardWidth):
	maxIntensitySum = max(listOfOYIntensitySum)
	maxIntensitySumIndex = listOfOYIntensitySum.index(maxIntensitySum)
	if maxIntensitySumIndex - leftMargin > rightMargin - maxIntensitySumIndex:
		x2 = imgCharacterWidth - rightMargin
		x1 = x2 - standardWidth + 1
		if x1 < 0:
			x1 = 0
		while listOfOYIntensitySum[x1] > 0 and x1 > 0:
			x1 = x1 - 1
	else:
		x1 = leftMargin
		x2 = x1 + standardWidth - 1
		if x2 > imgCharacterWidth - 1:
			x2 = imgCharacterWidth - 1
		while listOfOYIntensitySum[x2] > 0 and x2 < imgCharacterWidth - 1:
			x2 = x2 + 1
	return x1, x2
# end calculate_ox_slicing_coordination_for_character


def segment_character(imgBinary):
	# Step: original image
	if imgBinary is None:
		return None
	if showSteps == True:
		cv2.imshow('imgBinary', imgBinary)


	# Step: preprocessing
	imgBinary = preprocess(imgBinary)
	if showSteps == True:
		cv2.imshow('imgBinaryNew', imgBinary)


	# Step: divide plate
	imgTop, imgBottom = divide_plate(imgBinary)
	if showSteps == True:
		cv2.imshow('imgTop', imgTop)
		cv2.imshow('imgBottom', imgBottom)


	# Step: projection
	listOfTopImgPossibleCharacter = slice_plate_vertically(imgTop)
	listOfBottomImgPossibleCharacter = slice_plate_vertically(imgBottom)
	if showSteps == True:
		for imgCharacter in listOfTopImgPossibleCharacter:
			cv2.imshow('imgPossibleCharacter', imgCharacter)
			cv2.waitKey(0)
		for imgCharacter in listOfBottomImgPossibleCharacter:
			cv2.imshow('imgPossibleCharacter', imgCharacter)
			cv2.waitKey(0)


	# Step: postprocessing
	listOfTopImgPossibleCharacter = postprocess(listOfTopImgPossibleCharacter)
	listOfBottomImgPossibleCharacter = postprocess(listOfBottomImgPossibleCharacter)
	if showSteps == True:
		for imgCharacter in listOfTopImgPossibleCharacter:
			cv2.imshow('imgCharacter', imgCharacter)
			cv2.waitKey(0)
		for imgCharacter in listOfBottomImgPossibleCharacter:
			cv2.imshow('imgCharacter', imgCharacter)
			cv2.waitKey(0)


	if showSteps == True:
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	return listOfTopImgPossibleCharacter, listOfBottomImgPossibleCharacter
# end characterSegmentation