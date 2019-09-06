import cv2
import numpy as np
import os


from main import *
from possiblePlateLocatedRegionDetection import *
from characterSegmentation import *


# Constants:
PLATE_ASPECT_RATIO = float(19)/14
PLATE_HEIGHT = 200
PLATE_WIDTH = int(PLATE_HEIGHT * PLATE_ASPECT_RATIO)
SHIFT_AND_IMG_SCENE_HEIGHT_RATIO = float(8)/np.sqrt(150) # 150 la kich thuoc anh cat ra tu anh goc
showSteps = False


def find_extreme_point(contour):
	leftmost = tuple(contour[contour[:,:,0].argmin()][0])
	rightmost = tuple(contour[contour[:,:,0].argmax()][0])
	topmost = tuple(contour[contour[:,:,1].argmin()][0])
	bottommost = tuple(contour[contour[:,:,1].argmax()][0])
	return leftmost, rightmost, topmost, bottommost
# end find_extreme_point


def gamma_correction(img, correction):
    img = img/255.0
    img = cv2.pow(img, correction)
    return np.uint8(img * 255)
# end gamma_correction


def detect_plate(imgScene, imgContour):
	# Step: input image
	if imgScene is None or imgContour is None:
		return None
	if showSteps == True:
		cv2.imshow('imgScene', imgScene)


	# Step: convert to grayscale image
	imgGray = cv2.cvtColor(imgScene, cv2.COLOR_BGR2GRAY)
	if showSteps == True:
		cv2.imshow('imgGray', imgGray)


	# Step: preprocess original image
	intensityMean = cv2.mean(imgGray)
	intensityMean = int(intensityMean[0])
	if intensityMean <= 43:
		clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(30, 30))
		imgHE = clahe.apply(imgGray)
		imgGamma = gamma_correction(imgHE, float(3)/10)
		imgBlurred = cv2.medianBlur(imgGamma, 7)

		if showSteps == True:
			cv2.imshow('imgHE', imgHE)
			cv2.imshow('imgGamma', imgGamma)
			cv2.imshow('imgBlurred', imgBlurred)
	elif intensityMean < 55 and intensityMean > 43:
		clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(15, 15))
		imgHE = clahe.apply(imgGray)
		imgGamma = gamma_correction(imgHE, float(2)/3)
		imgBlurred = cv2.GaussianBlur(imgGamma, (5, 5), 0)

		if showSteps == True:
			cv2.imshow('imgHE', imgHE)
			cv2.imshow('imgGamma', imgGamma)
			cv2.imshow('imgBlurred', imgBlurred)
	elif intensityMean >= 55 and intensityMean < 110:
		clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(11, 11))
		imgHE = clahe.apply(imgGray)
		imgBlurred = cv2.GaussianBlur(imgHE, (5, 5), 0)

		if showSteps == True:
			cv2.imshow('imgHE', imgHE)
			cv2.imshow('imgBlurred', imgBlurred)
	else:
		imgBlurred = cv2.GaussianBlur(imgGray, (5, 5), 0)

		if showSteps == True:
			cv2.imshow('imgBlurred', imgBlurred)


	# Step: convert to binary image
	imgThresh = cv2.adaptiveThreshold(imgBlurred,	                         # input image
	                                   255,                                  # make pixels that pass the threshold full white
	                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # use gaussian rather than mean, seems to give better results
	                                   cv2.THRESH_BINARY_INV,              	 # invert so foreground will be white, background will be black
	                                   35,                                   # size of a pixel neighborhood used to calculate threshold value
	                                   2)
	if showSteps == True:
		cv2.imshow('imgThresh', imgThresh)


	# Step: find all contours
	imgContour, listOfContours, hierarchy = cv2.findContours(imgContour,	# input image
	                                         cv2.RETR_TREE,					# Contour retrieval mode
	                                         cv2.CHAIN_APPROX_NONE)			# Contour approximation method: SIMPLE OR NONE
	if showSteps == True:
		cv2.imshow('imgContour', imgContour)

	
	contour = listOfContours[0]
	imgPlate = None
	if contour == []:
		pass
	else:
		plateEdges = find_plate_edge_lines(contour, imgThresh)
		plateVertices = find_plate_vertices(plateEdges)
		topLeftVertex, topRightVertex, bottomLeftVertex, bottomRightVertex = plateVertices
		pts1 = np.float32([topLeftVertex, topRightVertex, bottomLeftVertex, bottomRightVertex])
		pts2 = np.float32([[0, 0], [PLATE_WIDTH, 0], [0, PLATE_HEIGHT], [PLATE_WIDTH, PLATE_HEIGHT]])
		M = cv2.getPerspectiveTransform(pts1, pts2)
		imgPlate = cv2.warpPerspective(imgThresh, M, (PLATE_WIDTH, PLATE_HEIGHT))
		if showSteps == True:
			cv2.imshow('imgPlate', imgPlate)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return imgPlate
# end detect_plate


def find_plate_vertices(plateEdgeLines):
	[plateLeftEdgeLine, plateRightEdgeLine, plateTopEdgeLine, plateBottomEdgeLine] = plateEdgeLines
	topLeftVertex = find_intersection_of_two_lines(plateLeftEdgeLine, plateTopEdgeLine)
	topRightVertex = find_intersection_of_two_lines(plateRightEdgeLine, plateTopEdgeLine)
	bottomLeftVertex = find_intersection_of_two_lines(plateLeftEdgeLine, plateBottomEdgeLine)
	bottomRightVertex = find_intersection_of_two_lines(plateRightEdgeLine, plateBottomEdgeLine)
	plateVertices = [topLeftVertex, topRightVertex, bottomLeftVertex, bottomRightVertex]
	return plateVertices
# end find_plate_vertices


def find_intersection_of_two_lines(lineOne, lineTwo):
	intersectionPointX = 0
	intersectionPointY = 0
	[pointOfLineOne, alphaOne] = lineOne
	[pointOfLineTwo, alphaTwo] = lineTwo
	x01 = pointOfLineOne[0]
	y01 = pointOfLineOne[1]
	x02 = pointOfLineTwo[0]
	y02 = pointOfLineTwo[1]
	if alphaOne == alphaTwo:
		print('Notification: two lines are parallel.\n')
		return
	else:
		if alphaOne == np.pi/2:
			intersectionPointX = x01
			intersectionPointY = np.tan(alphaTwo) * (intersectionPointX - x02) + y02
		elif alphaTwo == np.pi/2:
			intersectionPointX = x02
			intersectionPointY = np.tan(alphaOne) * (intersectionPointX - x01) + y01
		else:
			intersectionPointX = float(np.tan(alphaOne) * x01 - np.tan(alphaTwo) * x02 - (y01 - y02))/(np.tan(alphaOne) - np.tan(alphaTwo))
			intersectionPointY = np.tan(alphaOne) * (intersectionPointX - x01) + y01
		intersectionPoint = [intersectionPointX, intersectionPointY]
		return intersectionPoint
# end find_intersection_of_two_lines


def find_plate_edge_lines(contour, imgThresh):
	leftmostPointInContour, rightmostPointInContour, topmostPointInContour, bottommostPointInContour = find_extreme_point(contour)

	plateLeftEdgeLinePoint = leftmostPointInContour
	plateRightEdgeLinePoint = rightmostPointInContour
	plateTopEdgeLinePoint = topmostPointInContour
	plateBottomEdgeLinePoint = bottommostPointInContour

	plateLeftEdgeLineAngle = find_plate_edge_angle(plateLeftEdgeLinePoint, 1, imgThresh)
	plateRightEdgeLineAngle = find_plate_edge_angle(plateRightEdgeLinePoint, 2, imgThresh)
	plateTopEdgeLineAngle = find_plate_edge_angle(plateTopEdgeLinePoint, 3, imgThresh)
	plateBottomEdgeLineAngle = find_plate_edge_angle(plateBottomEdgeLinePoint, 4, imgThresh)

	plateLeftEdgeLine = [plateLeftEdgeLinePoint, plateLeftEdgeLineAngle]
	plateRightEdgeLine = [plateRightEdgeLinePoint, plateRightEdgeLineAngle]
	plateTopEdgeLine = [plateTopEdgeLinePoint, plateTopEdgeLineAngle]
	plateBottomEdgeLine = [plateBottomEdgeLinePoint, plateBottomEdgeLineAngle]

	plateEdgeLines = [plateLeftEdgeLine, plateRightEdgeLine, plateTopEdgeLine, plateBottomEdgeLine]
	return plateEdgeLines
# end find_plate_edge_lines


def is_which_line(edgeIndentityNumber):
	isLeftEdge = False
	isRightEdge = False
	isTopEdge = False
	isBottomEdge = False
	if edgeIndentityNumber == 1:
		isLeftEdge = True
	elif edgeIndentityNumber == 2:
		isRightEdge = True
	elif edgeIndentityNumber == 3:
		isTopEdge = True
	else:
		isBottomEdge = True
	return isLeftEdge, isRightEdge, isTopEdge, isBottomEdge
# end is_which_line


def find_plate_edge_angle(pointOfEdge, edgeIndentityNumber, imgBinary):
	imgBinaryHeight, imgBinaryWidth = imgBinary.shape
	isLeftEdge, isRightEdge, isTopEdge, isBottomEdge = is_which_line(edgeIndentityNumber)
	isOXDirection = False
	if isTopEdge or isBottomEdge:
		isOXDirection = True
	pointOfEdgeX = pointOfEdge[0]
	pointOfEdgeY = pointOfEdge[1]
	# ti le voi chieu cao theo ham can, tuc la cham hon tuyen tinh
	backwardShift = int(SHIFT_AND_IMG_SCENE_HEIGHT_RATIO * np.sqrt(imgBinaryHeight))
	forwardShift = backwardShift
	if isOXDirection == False:
		consideredCoordination = pointOfEdgeX
		if consideredCoordination + forwardShift > imgBinaryWidth:
			forwardShift = imgBinaryWidth - consideredCoordination
	else:
		consideredCoordination = pointOfEdge[1]
		if consideredCoordination + forwardShift > imgBinaryHeight:
			forwardShift = imgBinaryHeight - consideredCoordination
	if consideredCoordination - backwardShift < 0:
		backwardShift = consideredCoordination
	listOfDifferencesOfMaxIntensityAndMinIntensityAroundConsideredCoordination = []
	beginningRotatingAngle = -25
	endingRotatingAngle = 26
	for rotatingAngle in range(beginningRotatingAngle, endingRotatingAngle):
		rotationMatrix = cv2.getRotationMatrix2D((pointOfEdgeX, pointOfEdgeY), rotatingAngle, 1)
		rotatedImgBinary = cv2.warpAffine(imgBinary, rotationMatrix, (imgBinaryWidth, imgBinaryHeight))
		listOfIntensitySum = calculate_list_of_intensity_sum(rotatedImgBinary, isOXDirection)
		# consideredListOne: list to find max intensity sum around consideredCoordination
		beginningListOneIndex = consideredCoordination - backwardShift
		endingListOneIndex = consideredCoordination + forwardShift
		consideredListOne = listOfIntensitySum[beginningListOneIndex:endingListOneIndex + 1]
		maxIntensitySum = max(consideredListOne)
		maxIntensitySumIndex = consideredListOne.index(maxIntensitySum)
		# consideredListTwo: list to find min intensity sum around maxIntensitySumIndex
		shift = int(SHIFT_AND_IMG_SCENE_HEIGHT_RATIO * np.sqrt(imgBinaryHeight) / 2)
		if isLeftEdge or isTopEdge:
			consideredListTwo = listOfIntensitySum[maxIntensitySumIndex + beginningListOneIndex: maxIntensitySumIndex + beginningListOneIndex + shift]
		else:
			consideredListTwo = listOfIntensitySum[maxIntensitySumIndex - shift + beginningListOneIndex: maxIntensitySumIndex + beginningListOneIndex]
		minIntensitySum = min(consideredListTwo)
		listOfDifferencesOfMaxIntensityAndMinIntensityAroundConsideredCoordination.append(maxIntensitySum - minIntensitySum)
		#draw_bar_chart_of_binary_image(rotatedImgBinary, isOXDirection)
	maxDifference = max(listOfDifferencesOfMaxIntensityAndMinIntensityAroundConsideredCoordination)
	maxDifferenceIndex = listOfDifferencesOfMaxIntensityAndMinIntensityAroundConsideredCoordination.index(maxDifference)
	# goc xoay so voi truc ngang or doc
	chosenRotatingAngle = maxDifferenceIndex + beginningRotatingAngle
	M = cv2.getRotationMatrix2D((pointOfEdgeX, pointOfEdgeY), chosenRotatingAngle, 1)
	dst = cv2.warpAffine(imgBinary, M, (imgBinaryWidth, imgBinaryHeight))
	if showSteps == True:
		cv2.imshow('dst', dst)
	# goc cua duong thang, cua canh bien so doi voi truc toa do xOy, tuc la goc alpha trong bieu dien duong thang
	edgeAngle = chosenRotatingAngle
	if isOXDirection == False:
		edgeAngle = 90 + chosenRotatingAngle
	else:
		if -90 < chosenRotatingAngle and chosenRotatingAngle < 0:
			edgeAngle = 180 + chosenRotatingAngle
	edgeAngle = float(edgeAngle * np.pi)/180
	cv2.waitKey(0)
	return edgeAngle
# end find_plate_edge_angle