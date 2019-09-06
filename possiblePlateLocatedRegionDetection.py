import cv2
import numpy as np
import os


from main import *
from characterSegmentation import *
from plateDetectionAndProcessing import *


# Constants:
PLATE_ASPECT_RATIO = float(19)/14
MAX_IMAGE_SCENE_RESIZED_WIDTH = 800
MAX_IMAGE_SCENE_RESIZED_HEIGHT = 650
CROPPED_IMAGE_HEIGHT = 150
CROPPED_IMAGE_WIDTH = int(CROPPED_IMAGE_HEIGHT * PLATE_ASPECT_RATIO)
showSteps = True
INDEX_ERROR_INDICATOR = -1
DISTANCE_ERROR_INDICATOR = -1


def choose_second_step_candidate(listOfCandidate):
	if len(listOfCandidate) == 0:
		return None
	elif len(listOfCandidate) == 1:
		chosenCandidate = listOfCandidate[0]
	elif len(listOfCandidate) == 2:
		candidateOne = listOfCandidate[0]
		candidateTwo = listOfCandidate[1]

		# selection criteria
		minAverageOfDistancesOne = candidateOne[1]
		minAverageOfDistancesTwo  = candidateTwo[1]

		if minAverageOfDistancesTwo > minAverageOfDistancesOne:
			chosenCandidate = candidateOne
		else:
			chosenCandidate = candidateTwo
	else:
		newListOfCandidate = []
		lastCandidate = listOfCandidate.pop()
		newListOfCandidate.append(choose_second_step_candidate(listOfCandidate))
		newListOfCandidate.append(lastCandidate)
		chosenCandidate = choose_second_step_candidate(newListOfCandidate)
	return chosenCandidate
# end choose_second_step_candidate


def find_top_left_vertex_index_of_min_rect(minRectVertices):
	if len(minRectVertices) != 4:
		return INDEX_ERROR_INDICATOR
	minDistance = 0
	topLeftIndex = 0
	for i in range(0, len(minRectVertices)):
		minRectVertex = minRectVertices[i]
		vertexDistance = calculate_distance_between_point_and_point(minRectVertex, [0, 0])
		if i == 0:
			minDistance = vertexDistance
			topLeftIndex = 0
		else:
			if vertexDistance < minDistance:
				minDistance = vertexDistance
				topLeftIndex = i
	return topLeftIndex
# end find_top_left_vertex_index_of_min_rect


def calculate_distance_between_point_and_point(A, B):
	if len(A) != 2 or len(B) != 2:
		return DISTANCE_ERROR_INDICATOR
	Ax, Ay = A[0], A[1]
	Bx, By = B[0], B[1]
	return np.sqrt(((Bx - Ax)**2) + ((By - Ay)**2))
# end calculate_distance_between_point_and_point


def calculate_distance_of_point_and_line(point, line):
	if len(point) != 2 or len(line) != 2:
		return DISTANCE_ERROR_INDICATOR
	pointX, pointY = point[0], point[1]
	pointOfLine, alphaOfLine = line
	pointOfLineX, pointOfLineY = pointOfLine[0], pointOfLine[1]
	if alphaOfLine == np.pi/2:
		distanceOfPointAndLine = abs(pointX - pointOfLineX)
		return distanceOfPointAndLine
	else:
		distanceOfPointAndLine = float(abs((np.tan(alphaOfLine) * (pointX - pointOfLineX)) - pointY + pointOfLineY))/(np.sqrt(np.power(np.tan(alphaOfLine), 2) + 1))
		return distanceOfPointAndLine
# end calculate_distance_of_point_and_line


def find_line_of_two_point(A, B):
	if len(A) != 2 or len(B) != 2:
		return None
	Ax, Ay = A[0], A[1]
	Bx, By = B[0], B[1]
	if Ax == Bx and Ay == By:
		alphaOfLine = np.random.randint(179, size = 1)
		alphaOfLine = alphaOfLine[0]
		return A, alphaOfLine
	elif Ax == Bx:
		return A, np.pi/2
	else:
		# alpha from this function go from -90 to +90 degrees
		alphaOfLine = np.arctan(float(By - Ay)/(Bx - Ax))
		if alphaOfLine < 0:
			# modification: alpha go from 0 to 180 degrees
			alphaOfLine = np.pi + alphaOfLine
		return A, alphaOfLine
# end find_line_of_two_point


def resize_large_image(image, resizingCriteria):
	if image is None or len(resizingCriteria) != 2:
		return image
	[resizingCriteriaWidth, resizingCriteriaHeight] = resizingCriteria
	imgHeight, imgWidth = image.shape[:2]
	imgAspectRatio = float(imgWidth)/imgHeight
	if imgWidth > resizingCriteriaWidth:
		image = cv2.resize(image, (resizingCriteriaWidth, int(resizingCriteriaWidth * (imgAspectRatio**(-1)))), interpolation = cv2.INTER_AREA)
		imgHeight, imgWidth = image.shape[:2]
	if imgHeight > resizingCriteriaHeight:
		image = cv2.resize(image, (int(resizingCriteriaHeight * imgAspectRatio), resizingCriteriaHeight), interpolation = cv2.INTER_AREA)
	return image
# end resize_large_image


def sort_min_rect_vertices_according_to_plate_vertices(minRectVertices):
	topLeftIndex = find_top_left_vertex_index_of_min_rect(minRectVertices)
	if topLeftIndex == 1:
		topLeftVertex = minRectVertices[topLeftIndex]
		bottomRightVertex = minRectVertices[3]
		topRightVertex = minRectVertices[2]
		bottomLeftVertex = minRectVertices[0]
	else: # topLeftIndex == 2
		topLeftVertex = minRectVertices[topLeftIndex]
		topRightVertex = minRectVertices[3]
		bottomLeftVertex = minRectVertices[1]
		bottomRightVertex = minRectVertices[0]
	minRectVertices = [topLeftVertex, bottomRightVertex, topRightVertex, bottomLeftVertex]
	return minRectVertices, topLeftIndex
# end sort_min_rect_vertices_according_to_plate_vertices


def min_rect_aspect_ratio_is_suitable(minRectAspectRatio, minDifferenceBetweenMinRectAngleAndDiagonalLines):
	if 43 <= minDifferenceBetweenMinRectAngleAndDiagonalLines and minDifferenceBetweenMinRectAngleAndDiagonalLines <= 45:
		if minRectAspectRatio < 1.154 or minRectAspectRatio > 1.959:
			return False
	if 37.5 <= minDifferenceBetweenMinRectAngleAndDiagonalLines and minDifferenceBetweenMinRectAngleAndDiagonalLines < 43:
		if minRectAspectRatio < 0.988 or minRectAspectRatio > 1.96:
			return False
	elif 30 <= minDifferenceBetweenMinRectAngleAndDiagonalLines and minDifferenceBetweenMinRectAngleAndDiagonalLines < 37.5:
		if minRectAspectRatio < 0.99 or minRectAspectRatio > 2.0:
			return False
	elif 20 <= minDifferenceBetweenMinRectAngleAndDiagonalLines and minDifferenceBetweenMinRectAngleAndDiagonalLines < 30:
		if minRectAspectRatio < 0.65 or minRectAspectRatio > 1.4:
			return False
	elif 10 <= minDifferenceBetweenMinRectAngleAndDiagonalLines and minDifferenceBetweenMinRectAngleAndDiagonalLines < 20:
		if minRectAspectRatio < 0.83 or minRectAspectRatio > 1.9:
			return False
	else:
		if minRectAspectRatio < 1.2 or minRectAspectRatio > 1.93:
			return False
	return True
# end min_rect_aspect_ratio_is_suitable


def find_first_step_three_criteria_for_choosing_contour(minRectVertices, minDifferenceBetweenMinRectAngleAndDiagonalLines, imgSceneResizedWidth, imgSceneResizedHeight):
	[topLeftVertex, bottomRightVertex, topRightVertex, bottomLeftVertex] = minRectVertices
	minRectBottomEdgeLength = calculate_distance_between_point_and_point(bottomLeftVertex, bottomRightVertex)
	minRectLeftEdgeLength = calculate_distance_between_point_and_point(topLeftVertex, bottomLeftVertex)
	minRectAspectRatio = float(minRectBottomEdgeLength)/minRectLeftEdgeLength
	minRectBottomEdgeLengthAndImgSceneResizedWidthRatio = float(minRectBottomEdgeLength)/(imgSceneResizedWidth)
	minRectLeftEdgeLengthAndImgSceneResizedHeightRatio = float(minRectLeftEdgeLength)/(imgSceneResizedHeight)
	minRectAspectRatioIsSuitable = min_rect_aspect_ratio_is_suitable(minRectAspectRatio, minDifferenceBetweenMinRectAngleAndDiagonalLines)
	return minRectBottomEdgeLengthAndImgSceneResizedWidthRatio, minRectLeftEdgeLengthAndImgSceneResizedHeightRatio, minRectAspectRatioIsSuitable
# end find_first_step_three_criteria_for_choosing_contour


def calculate_average_of_distances_of_contour_point_and_nearest_bounding_rect_edge(boundingRect, contour):
	numberOfContourPoints = len(contour)
	[boundingRectX, boundingRectY, boundingRectWidth, boundingRectHeight] = boundingRect
	sumOfDistancesOfContourPointAndNearestBoundingRectEdge = 0
	for point in contour:
		point = point[0]
		pointX = point[0]
		pointY = point[1]

		distanceOfContourPointAndBoundingRectLeftEdge = pointX - boundingRectX
		distanceOfContourPointAndBoundingRectRightEdge = boundingRectX + boundingRectWidth - pointX
		distanceOfContourPointAndBoundingRectTopEdge = pointY - boundingRectY
		distanceOfContourPointAndBoundingRectBottomEdge = boundingRectY + boundingRectHeight - pointY

		distanceOfContourPointAndNearestBoundingRectEdge = min(distanceOfContourPointAndBoundingRectLeftEdge,
															   distanceOfContourPointAndBoundingRectRightEdge,
															   distanceOfContourPointAndBoundingRectTopEdge,
															   distanceOfContourPointAndBoundingRectBottomEdge)

		sumOfDistancesOfContourPointAndNearestBoundingRectEdge = sumOfDistancesOfContourPointAndNearestBoundingRectEdge + distanceOfContourPointAndNearestBoundingRectEdge
	averageOfDistancesOfContourPointAndNearestBoundingRectEdge = float(sumOfDistancesOfContourPointAndNearestBoundingRectEdge)/numberOfContourPoints
	return averageOfDistancesOfContourPointAndNearestBoundingRectEdge
# end calculate_average_of_distances_of_contour_point_and_nearest_bounding_rect_edge


def calculate_average_of_distances_of_contour_point_and_nearest_min_rect_edge(minRectVertices, contour):
	numberOfContourPoints = len(contour)
	[topLeftVertex, bottomRightVertex, topRightVertex, bottomLeftVertex] = minRectVertices
	sumOfDistancesOfContourPointAndNearestMinRectEdge = 0
	for point in contour:
		point = point[0]
		
		minRectLeftLine = find_line_of_two_point(topLeftVertex, bottomLeftVertex)
		minRectRightLine = find_line_of_two_point(topRightVertex, bottomRightVertex)
		minRectTopLine = find_line_of_two_point(topLeftVertex, topRightVertex)
		minRectBottomLine = find_line_of_two_point(bottomLeftVertex, bottomRightVertex)

		distanceOfContourPointAndMinRectLeftEdge = calculate_distance_of_point_and_line(point, minRectLeftLine)
		distanceOfContourPointAndMinRectRightEdge = calculate_distance_of_point_and_line(point, minRectRightLine)
		distanceOfContourPointAndMinRectTopEdge = calculate_distance_of_point_and_line(point, minRectTopLine)
		distanceOfContourPointAndMinRectBottomEdge = calculate_distance_of_point_and_line(point, minRectBottomLine)

		distanceOfContourPointAndNearestMinRectEdge = min(distanceOfContourPointAndMinRectLeftEdge,
														distanceOfContourPointAndMinRectRightEdge,
														distanceOfContourPointAndMinRectTopEdge,
														distanceOfContourPointAndMinRectBottomEdge)

		sumOfDistancesOfContourPointAndNearestMinRectEdge = sumOfDistancesOfContourPointAndNearestMinRectEdge + distanceOfContourPointAndNearestMinRectEdge
	averageOfDistancesOfContourPointAndNearestMinRectEdge = float(sumOfDistancesOfContourPointAndNearestMinRectEdge)/numberOfContourPoints
	return averageOfDistancesOfContourPointAndNearestMinRectEdge
# end calculate_average_of_distances_of_contour_point_and_nearest_min_rect_edge


def find_second_step_criterion_for_choosing_contour(firstAverage, secondAverage):
	return min(firstAverage, secondAverage)
# end find_second_step_criterion_for_choosing_contour


def plate_was_taken_from_front_view(firstAverage, secondAverage):
	return (firstAverage < secondAverage)
# end plate_was_taken_from_front_view


def detect_plate_located_region(imgScene):
	# Step: input image
	if imgScene is None:
		return None
	if showSteps == True:
		cv2.imshow('imgScene', imgScene)


	# Step: resize large image
	resizingCriteria = [MAX_IMAGE_SCENE_RESIZED_WIDTH, MAX_IMAGE_SCENE_RESIZED_HEIGHT]
	imgSceneResized = resize_large_image(imgScene, resizingCriteria)
	if imgSceneResized is None:
		return None
	imgSceneResizedHeight, imgSceneResizedWidth = imgSceneResized.shape[:2]
	if showSteps == True:
		cv2.imshow('imgSceneResized', imgSceneResized)


	# Step: convert input image to grayscale image
	imgGray = cv2.cvtColor(imgSceneResized, cv2.COLOR_BGR2GRAY)
	if showSteps == True:
		cv2.imshow('imgGray', imgGray)


	# Step: blur, smooth the grayscale image
	imgBlurred = cv2.GaussianBlur(imgGray, (5, 5), 0)						# (5,5): Gaussian Kernel Size, 0: caculate auto the deviation of GaussianBlur
	if showSteps == True:
		cv2.imshow('imgBlurred', imgBlurred)


	# Step: edge detection using Canny
	imgCanny = cv2.Canny(imgBlurred, threshold1 = 30, threshold2 = 180, edges = 0, apertureSize = 3, L2gradient = True)
	if showSteps == True:
		cv2.imshow('imgCanny', imgCanny)


	# Step: find all contours
	imgCanny, listOfContours, hierarchy = cv2.findContours(imgCanny,		# input image
	                                         cv2.RETR_EXTERNAL,				# Contour retrieval mode
	                                         cv2.CHAIN_APPROX_SIMPLE)		# Contour approximation method: SIMPLE OR NONE


	# Step: list all contours
	imgSceneResizedWithFirstStepChosenContours = imgSceneResized.copy()
	imgContours = np.zeros(imgSceneResized.shape, np.uint8)
	imgNormalRect = imgContours.copy()
	imgRotatedRect = imgContours.copy()
	listOfFirstStepChosenCandidate = []
	for i in range(0, len(listOfContours)):
		contour = listOfContours[i]


		if showSteps == True:
			cv2.drawContours(imgContours, [contour], 0, (255, 255, 255), 1)
			cv2.imshow('imgContours', imgContours)


		boundingRectX, boundingRectY, boundingRectWidth, boundingRectHeight = cv2.boundingRect(contour)
		if showSteps == True:
			cv2.rectangle(imgNormalRect, (boundingRectX, boundingRectY), (boundingRectX + boundingRectWidth, boundingRectY + boundingRectHeight), (255, 255, 255), 1)
			cv2.imshow('imgNormalRect', imgNormalRect)


		minRect = cv2.minAreaRect(contour)
		((minRectXCenter, minRectYCenter), (minRectWidth, minRectHeight), minRectAngle) = minRect
		minRectVertices = cv2.boxPoints(minRect)
		minRectVertices = np.int0(minRectVertices)
		'''
		minRectVertices
		[[  0 372]
		 [  0 371]
		 [  9 368]
		 [ 10 370]]
		'''
		if showSteps == True:
			cv2.drawContours(imgRotatedRect, [minRectVertices], 0, (255, 255, 255), 1)
			cv2.imshow('imgRotatedRect', imgRotatedRect)


		minRectVertices, topLeftIndex = sort_min_rect_vertices_according_to_plate_vertices(minRectVertices)
		minDifferenceBetweenMinRectAngleAndDiagonalLines = abs(abs(minRectAngle) - 45)
		[topLeftVertex, bottomRightVertex, topRightVertex, bottomLeftVertex] = minRectVertices
		firstCriterion,	secondCriterion, thirdCriterion = find_first_step_three_criteria_for_choosing_contour(minRectVertices,
																											minDifferenceBetweenMinRectAngleAndDiagonalLines,
																											imgSceneResizedWidth,
																											imgSceneResizedHeight)


		# Step: first step to choose contour
		if firstCriterion > 0.1 and secondCriterion > 0.1 and thirdCriterion:
			boundingRectOfFirstStepChosenContour = [boundingRectX, boundingRectY, boundingRectWidth, boundingRectHeight]

			firstAverage = calculate_average_of_distances_of_contour_point_and_nearest_bounding_rect_edge(boundingRectOfFirstStepChosenContour, contour)
			secondAverage = calculate_average_of_distances_of_contour_point_and_nearest_min_rect_edge(minRectVertices, contour)
			criterion = find_second_step_criterion_for_choosing_contour(firstAverage, secondAverage)
			plateTakenFromFrontView = plate_was_taken_from_front_view(firstAverage, secondAverage)

			firstStepChosenCandidate = [minRectVertices,
										criterion,
										boundingRectOfFirstStepChosenContour,
										plateTakenFromFrontView,
										topLeftIndex,
										minDifferenceBetweenMinRectAngleAndDiagonalLines,
										contour,
										minRect]

			listOfFirstStepChosenCandidate.append(firstStepChosenCandidate)

			if showSteps == True:
				cv2.drawContours(imgSceneResizedWithFirstStepChosenContours, [contour], 0, (0, 0, 255), 2)
				cv2.imshow('imgSceneResizedWithFirstStepChosenContours', imgSceneResizedWithFirstStepChosenContours)
				cv2.waitKey(0)
	# end for loop


	# Step: second step to choose contour
	secondStepChosenCandidate = choose_second_step_candidate(listOfFirstStepChosenCandidate)


	# Step: crop image based on license plate located region
	imgPlateLocatedRegion, imgSecondStepChosenContour = obtain_plate_located_region_by_cropping_image_scene_resized(secondStepChosenCandidate, imgSceneResized)
	if imgPlateLocatedRegion is None or imgSecondStepChosenContour is None:
		print("Notification: couldn't crop image scene.")
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		return None, None, imgSceneResized, []
	if showSteps == True:
		cv2.imshow('imgPlateLocatedRegion', imgPlateLocatedRegion)
		cv2.imshow('imgSecondStepChosenContourAfter', imgSecondStepChosenContour)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return imgPlateLocatedRegion, imgSecondStepChosenContour, imgSceneResized, secondStepChosenCandidate[2]
# end detect_plate_located_region


def obtain_plate_located_region_by_cropping_image_scene_resized(secondStepChosenCandidate, imgSceneResized):
	imgPlateLocatedRegion = None
	imgSecondStepChosenContour = None
	imgSceneResizedHeight, imgSceneResizedWidth = imgSceneResized.shape[:2]
	if secondStepChosenCandidate is not None:
		imgSecondStepChosenContour = np.zeros(imgSceneResized.shape, np.uint8)
		plateTakenFromFrontView = secondStepChosenCandidate[3]
		[boundingRectX, boundingRectY, boundingRectWidth, boundingRectHeight] = secondStepChosenCandidate[2]
		contour = secondStepChosenCandidate[6]
		cv2.drawContours(imgSecondStepChosenContour, [contour], 0, (255, 255, 255), 1)
		imgSecondStepChosenContour = cv2.cvtColor(imgSecondStepChosenContour, cv2.COLOR_BGR2GRAY)
		if plateTakenFromFrontView:
			print('Notification: crop without transformation.')
			shiftXAndBoundingRectWidthRatio = float(3/100)					# 3%
			shiftYAndBoundingRectHeightRatio = float(3/100)					# 3%
			shiftX = shiftXAndBoundingRectWidthRatio * boundingRectWidth
			shiftY = shiftYAndBoundingRectHeightRatio * boundingRectHeight
			croppedImageTopLeftX = int(boundingRectX - shiftX)
			croppedImageTopLeftY = int(boundingRectY - shiftY)
			croppedImageWidth = int(boundingRectWidth + (2 * shiftX))
			croppedImageHeight = int(boundingRectHeight + (2 * shiftY))
			if croppedImageTopLeftX < 0:
				croppedImageTopLeftX = 0
			if croppedImageTopLeftY < 0:
				croppedImageTopLeftY = 0
			if croppedImageTopLeftX + croppedImageWidth > imgSceneResizedWidth:
				croppedImageWidth = imgSceneResizedWidth - croppedImageTopLeftX
			if croppedImageTopLeftY + croppedImageHeight > imgSceneResizedHeight:
				croppedImageHeight = imgSceneResizedHeight - croppedImageTopLeftY
			imgPlateLocatedRegion = imgSceneResized[croppedImageTopLeftY:croppedImageTopLeftY+croppedImageHeight, croppedImageTopLeftX:croppedImageTopLeftX+croppedImageWidth]
			imgPlateLocatedRegion = cv2.resize(imgPlateLocatedRegion, (CROPPED_IMAGE_WIDTH, CROPPED_IMAGE_HEIGHT), interpolation = cv2.INTER_AREA)	
			imgSecondStepChosenContour = imgSecondStepChosenContour[croppedImageTopLeftY:croppedImageTopLeftY+croppedImageHeight, croppedImageTopLeftX:croppedImageTopLeftX+croppedImageWidth]
			imgSecondStepChosenContour = cv2.resize(imgSecondStepChosenContour, (CROPPED_IMAGE_WIDTH, CROPPED_IMAGE_HEIGHT), interpolation = cv2.INTER_AREA)
		else:
			print('Notification: crop with transformation.')
			((minRectXCenter, minRectYCenter), (minRectWidth, minRectHeight), minRectAngle) = secondStepChosenCandidate[7]
			minDifferenceBetweenMinRectAngleAndDiagonalLines = secondStepChosenCandidate[5]
			minRectVertices = secondStepChosenCandidate[0]
			[topLeftVertex, bottomRightVertex, topRightVertex, bottomLeftVertex] = minRectVertices
			topLeftIndex = secondStepChosenCandidate[4]
			leftInclinationRatio = 0
			rightInclinationRatio = 0
			if topLeftIndex == 1:
				print('Notification: left perspective or license plate lean to the left.')
				if 30 <= minDifferenceBetweenMinRectAngleAndDiagonalLines and minDifferenceBetweenMinRectAngleAndDiagonalLines < 35:
					leftInclinationRatio = 0.05
					rightInclinationRatio = 0.06
				elif 20 <= minDifferenceBetweenMinRectAngleAndDiagonalLines and minDifferenceBetweenMinRectAngleAndDiagonalLines < 30:
					leftInclinationRatio = 0.06
					rightInclinationRatio = 0.1
				elif 17 <= minDifferenceBetweenMinRectAngleAndDiagonalLines and minDifferenceBetweenMinRectAngleAndDiagonalLines < 20:
					leftInclinationRatio = 0.1
					rightInclinationRatio = 0.2
				elif minDifferenceBetweenMinRectAngleAndDiagonalLines < 17:
					leftInclinationRatio = 0.2
					rightInclinationRatio = 0.3
				else:
					leftInclinationRatio = 0.04
					rightInclinationRatio = 0.05
			else: # topLeftIndex == 2
				print('Notification: right perspective or license plate lean to the right.')
				if 30 <= minDifferenceBetweenMinRectAngleAndDiagonalLines and minDifferenceBetweenMinRectAngleAndDiagonalLines < 35:
					leftInclinationRatio = 0.06
					rightInclinationRatio = 0.05
				elif 20 <= minDifferenceBetweenMinRectAngleAndDiagonalLines and minDifferenceBetweenMinRectAngleAndDiagonalLines < 30:
					leftInclinationRatio = 0.1
					rightInclinationRatio = 0.06
				elif 17 <= minDifferenceBetweenMinRectAngleAndDiagonalLines and minDifferenceBetweenMinRectAngleAndDiagonalLines < 20:
					leftInclinationRatio = 0.2
					rightInclinationRatio = 0.1
				elif minDifferenceBetweenMinRectAngleAndDiagonalLines < 17:
					leftInclinationRatio = 0.3
					rightInclinationRatio = 0.2
				else:
					leftInclinationRatio = 0.05
					rightInclinationRatio = 0.04
			modifiedTopLeftVertex = topLeftVertex - leftInclinationRatio * (bottomRightVertex - topLeftVertex)
			modifiedBottomRightVertex = bottomRightVertex - leftInclinationRatio * (topLeftVertex - bottomRightVertex)
			modifiedTopRightVertex = topRightVertex - rightInclinationRatio * (bottomLeftVertex - topRightVertex)
			modifiedBottomLeftVertex = bottomLeftVertex - rightInclinationRatio * (topRightVertex - bottomLeftVertex)

			modifiedTopLeftVertex = [int(i) for i in modifiedTopLeftVertex]
			modifiedBottomRightVertex = [int(i) for i in modifiedBottomRightVertex]
			modifiedTopRightVertex = [int(i) for i in modifiedTopRightVertex]
			modifiedBottomLeftVertex = [int(i) for i in modifiedBottomLeftVertex]

			croppedImageHeight = int(minRectHeight * (1 + 2.0 * max(leftInclinationRatio, rightInclinationRatio)))
			croppedImageWidth = int(croppedImageHeight * PLATE_ASPECT_RATIO)
			pts1 = np.float32([modifiedTopLeftVertex, modifiedTopRightVertex, modifiedBottomLeftVertex, modifiedBottomRightVertex])
			pts2 = np.float32([[0, 0], [croppedImageWidth, 0], [0, croppedImageHeight], [croppedImageWidth, croppedImageHeight]])
			M = cv2.getPerspectiveTransform(pts1, pts2)
			imgPlateLocatedRegion = cv2.warpPerspective(imgSceneResized, M, (croppedImageWidth, croppedImageHeight))
			imgSecondStepChosenContour = cv2.warpPerspective(imgSecondStepChosenContour, M, (croppedImageWidth, croppedImageHeight))

			# xoay roi resize chu khong gop resize vao chung voi phan xoay, ly do vi trong mot so truong hop khi contour qua to va manh
			# thi viec xoay co the lam hong contour
			croppedImageHeight = CROPPED_IMAGE_HEIGHT
			croppedImageWidth = int(croppedImageHeight * PLATE_ASPECT_RATIO)
			imgPlateLocatedRegion = cv2.resize(imgPlateLocatedRegion, (croppedImageWidth, croppedImageHeight), interpolation = cv2.INTER_AREA)
			imgSecondStepChosenContour = cv2.resize(imgSecondStepChosenContour, (croppedImageWidth, croppedImageHeight), interpolation = cv2.INTER_AREA)
	return imgPlateLocatedRegion, imgSecondStepChosenContour
# end obtain_plate_located_region_by_cropping_image_scene_resized