import cv2
import numpy as np
import os




# Constants:
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 650


def image_point_distance(A, B):
	Ax, Ay = A[0], A[1]
	Bx, By = B[0], B[1]
	return np.sqrt(((Bx - Ax)**2) + ((By - Ay)**2))


def find_point_with_min_distance(point, listOfPoints):
	min = 1000000000
	returnedPoint = []
	for pointTwo in listOfPoints:
		distance = image_point_distance(point, pointTwo[0])
		if distance < min:
			min = distance
			returnedPoint.append(pointTwo[0])
	return returnedPoint[-1]


'''
	minX = 10000
		maxX = 0
		minY = 10000
		maxY = 0
	for i in range(0, len(listOfPoints)):
		x, y = listOfPoints[i]
		if x < minX:
			minX = x
		if x > maxX:
			maxX = x
		if y < minY:
			minY = y
		if y > maxY:
			maxY = y
	for point in listOfPoints:
		x, y = point
		if minX == x:
			leftmost = point
		if maxX == x:				# chu y truc toa do cua anh, Oy huong xuong
			rightmost = point
		if maxY == y:
			bottommost = point
		if minY == y:
			topmost = point


def find_extreme_point(listOfPoints):
	A = listOfPoints.copy()
	for i in range(0, len(listOfPoints)):
		listOfPoints[i] = list(A[i])
	leftmost = listOfPoints[listOfPoints[:,0].argmin()]
	rightmost = listOfPoints[listOfPoints[:,0].argmax()]
	topmost = listOfPoints[listOfPoints[:,1].argmin()]
	bottommost = listOfPoints[listOfPoints[:,1].argmax()]
	return (leftmost, rightmost, topmost, bottommost)
'''


def is_collinear(pointA, pointB, pointC):
	(x1, y1) = pointA
	(x2, y2) = pointB
	(x3, y3) = pointC
	temp = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
	if temp == 0:
		return True
	else:
		return False


def gamma_correction(img, correction):
    img = img/255.0
    img = cv2.pow(img, correction)
    return np.uint8(img*255)


def detect_plates(imgScene):
	# Step 1: Load original image
	# Can sua: 5, 6, 7, 11, 17(kiem tra lai angle), 21
	# trinh bay: 25: chong loa
	# 70.5, 71.3, 71.9, 78.435, 83.24, 86.36, 111.46 (hong),  chuan nhat la 74
	# min = 80
	cv2.imshow('imgScene', imgScene)
	print('imgScene')
	print(imgScene)


	# Step 2: Resize large images
	w, h, channel = imgScene.shape
	aspectRatio = float(w)/h
	if h > w:
		imgScene = cv2.resize(imgScene, (IMAGE_WIDTH, int(aspectRatio * IMAGE_WIDTH)), interpolation = cv2.INTER_AREA)
	else:
		imgScene = cv2.resize(imgScene, (int(IMAGE_HEIGHT * (aspectRatio)**(-1)), IMAGE_HEIGHT), interpolation = cv2.INTER_AREA)
	cv2.imshow('imgSceneResized', imgScene)
	w, h, channel = imgScene.shape


	# Step 2: Convert the original image to grayscale image
	imgGray = cv2.cvtColor(imgScene, cv2.COLOR_BGR2GRAY)
	cv2.imshow('imgGray', imgGray)


	clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(30,30))
	imgHE = clahe.apply(imgGray)
	cv2.imshow('imgHE', imgHE)


	imgGamma = gamma_correction(imgHE, float(2)/3)
	cv2.imshow('imgGamma', imgGamma)


	# Step 3: Blur, smooth the grayscale image
	imgBlurred = cv2.GaussianBlur(imgGamma, (3, 3), 0) 						 # (5,5): Gaussian Kernel Size, 0: caculate auto the deviation of GaussianBlur
	cv2.imshow('imgBlurred', imgBlurred)


	# Step 5: Convert to binary image
	imgThresh = cv2.adaptiveThreshold(imgBlurred,                            # input image
	                                   255,                                  # make pixels that pass the threshold full white
	                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # use gaussian rather than mean, seems to give better results
	                                   cv2.THRESH_BINARY_INV,                	 # invert so foreground will be white, background will be black
	                                   35,                                   # size of a pixel neighborhood used to calculate threshold value
	                                   2)
	cv2.imshow('imgThresh', imgThresh)


	# Step 4: Gradient filter (high-pass filter): Laplacian
	#imgHighPass = cv2.Laplacian(imgThresh, cv2.CV_8UC1)
	#cv2.imshow('imgHighPass', imgHighPass)


	# Step 5: 
	kernel = np.ones((3,3), np.uint8)
	imgDilation = cv2.dilate(imgThresh, kernel, iterations = 1)
	cv2.imshow('imgDilation', imgDilation)
	#imgOpening = cv2.morphologyEx(imgThresh, cv2.MORPH_OPEN, kernel)
	#cv2.imshow('imgOpening', imgOpening)
	#imgClosing = cv2.morphologyEx(imgOpening, cv2.MORPH_CLOSE, kernel)
	#cv2.imshow('imgClosing', imgClosing)
	#imgOpeningTwo = cv2.morphologyEx(imgClosing, cv2.MORPH_OPEN, kernel)
	#cv2.imshow('imgOpeningTwo', imgOpeningTwo)
	#imgClosingTwo = cv2.morphologyEx(imgOpeningTwo, cv2.MORPH_CLOSE, kernel)
	#cv2.imshow('imgOpening', imgOpening)


	# Step 6: Edge detection using Canny
	imgCanny = cv2.Canny(imgDilation, threshold1 = 30, threshold2 = 40, edges = 0, apertureSize = 5, L2gradient = True)
	cv2.imshow('imgCanny', imgCanny)
	print(type(imgCanny))


	print('imgCannyimgCannyimgCannyimgCannyimgCannyimgCannyimgCannyimgCannyimgCannyimgCannyimgCanny')
	print(imgCanny)
	print('edges')
	#print(edges)

	'''
	minLineLength = 100
	maxLineGap = 10
	lines = cv2.HoughLinesP(imgCanny,1,np.pi/180,100,minLineLength,maxLineGap)
	for line in lines:
		x1,y1,x2,y2 = line[0]
		cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
	cv2.imshow('houghlines5.jpg',img)
	'''


	# Step 7: Find all contours
	imgCanny, listOfContours, hierarchy = cv2.findContours(imgCanny,      	 # input image
	                                         cv2.RETR_EXTERNAL,              # Contour retrieval mode
	                                         cv2.CHAIN_APPROX_SIMPLE)		 # Contour approximation method: SIMPLE OR NONE
	'''
	redundantImgCany, redundantListOfContours, usedHierarchy = cv2.findContours(imgCanny,
																				cv2.RETR_TREE,
																				cv2.CHAIN_APPROX_SIMPLE)
	for i in range(0, usedHierarchy.shape[1]):
		print(usedHierarchy[0][i][3])
	'''


	# Step 8: List all contours
	imgContours = np.zeros(imgScene.shape, np.uint8)
	imgContoursAfterSeveralCondition = np.zeros(imgScene.shape, dtype = np.uint8)
	imgApproxContours = imgContours.copy()
	imgConvexContours = imgContours.copy()
	imgNormalRect = imgContours.copy()
	imgRotatedRect = imgContours.copy()
	imgCircle = imgContours.copy()
	#imgChildContours = imgContours.copy()
	aaaa = imgScene.copy()
	aaaaa = imgScene.copy()
	numberOfPossiblePlate = 0
	imgPossiblePlate = imgContours.copy()
	img = imgScene.copy()
	for i in range(0, len(listOfContours)):
		contour = listOfContours[i]


		cv2.drawContours(imgContours, [contour], 0, (255, 255, 255), 1)
		moment = cv2.moments(contour)
		#cx = int(moment['m10']/moment['m00'])
		#cy = int(moment['m01']/moment['m00'])										 # area = moment['m00']
		perimeter = cv2.arcLength(contour, True)							 		 # perimeter of a closed contour
		cv2.imshow('imgContours', imgContours)
		contourArea = cv2.contourArea(contour)


		'''
		epsilon = 0.1 * perimeter
		approx = cv2.approxPolyDP(contour, epsilon, True)
		cv2.drawContours(imgApproxContours, [approx], 0, (255, 255, 255), 1)
		cv2.imshow('imgApproxContours', imgApproxContours)

		((x, y), radius) = cv2.minEnclosingCircle(contour)
		center = (int(x), int(y))
		radius = int(radius)
		cv2.circle(imgCircle, center, radius, (255, 255, 255), 1)
		cv2.imshow('imgCircle', imgCircle)\

		if cv2.isContourConvex(contour):
			print('True')
		else:
			print('False')
		'''


		rect = cv2.minAreaRect(contour)
		((xCenter, yCenter), (minRectWidth, minRectHeight), angle) = rect
		box = cv2.boxPoints(rect)
		box = np.int0(box)
		print('angle: ' + str(angle))
		'''
		box
		[[  0 372]
		 [  0 371]
		 [  9 368]
		 [ 10 370]]
		'''
		cv2.drawContours(imgRotatedRect, [box], 0, (255, 255, 255), 1)
		cv2.imshow('imgRotatedRect', imgRotatedRect)
		minRectArea = minRectWidth * minRectHeight


		hull = cv2.convexHull(contour, True)
		print('hull')
		print(hull)
		cv2.drawContours(imgConvexContours, [hull], 0, (255, 255, 255), 1)
		cv2.imshow('imgConvexContours', imgConvexContours)
		convexContourArea = cv2.contourArea(hull)
		leftmost = tuple(contour[contour[:,:,0].argmin()][0])			# Point A
		rightmost = tuple(contour[contour[:,:,0].argmax()][0])		# Point B
		topmost = tuple(contour[contour[:,:,1].argmin()][0])			# Point C
		bottommost = tuple(contour[contour[:,:,1].argmax()][0])		# Point D
		print('contour')
		print(contour)
		print(contour.shape)
		print(type(contour))


		# Perspective transformation
		#AD = abs(leftmost - )



		x, y, w, h = cv2.boundingRect(contour)
		cv2.rectangle(imgNormalRect, (x, y), (x + w, y + h), (255, 255, 255), 1)
		cv2.imshow('imgNormalRect', imgNormalRect)
		rectArea = w * h


		#zero = np.zeros(imgContours.shape, np.uint8)
		#rOI = imgThresh[y:y+h, x:x+w]
		#mean = cv2.mean(rOI)
		#meanVal = cv2.mean(contour)
		#meanApprox = int(mean[0])
		#aspectRatio = float(w)/h
		#if rectArea != 0:
		#	extent = float(contourArea * 1000)/rectArea
		#print('Mean is ' + str(mean))
		#print('MeanVal is ' + str(meanVal))
		#print('MeanApprox is ' + str(meanApprox))
		#print('Width is ' + str(int(w)))
		#print('Height is ' + str(int(h)))
		#print('Aspect Ratio is ' + str(aspectRatio))
		print('Angle: ' + str(angle))
		print('MinRectWidth: ' + str(minRectWidth))
		print('MinRectHeight: ' + str(minRectHeight))
		print('Perimeter: ' + str(perimeter))
		print('ContourArea: ' + str(contourArea))					# areaOne
		print('ConvexContourArea: ' + str(convexContourArea))		# areaTwo
		print('RectArea: ' + str(rectArea))							# areaThree
		print('MinRectArea: ' + str(minRectArea))					# areaFour
		if minRectArea == 0:
			continue
		twoFourRatio = float(convexContourArea)/minRectArea
		print('TwoFourRatio: ' + str(twoFourRatio))


		#trial = np.zeros((imgContours.shape), np.uint8)
		#cv2.rectangle(trial, (20, 20), (20 + 75, 20 + 50), (255, 255, 255), 2)
		#cv2.imshow('trial', trial)


		#   and w >= 97 and h >= 52 and w <= 325 and h <= 230
		#if meanApprox >= 161 and meanApprox <= 187 and aspectRatio >= 1.1 and w >= 97 and h >= 52 and w <= 325 and h <= 230:
			#for j in range(0, usedHierarchy[])
		minOne = min(minRectWidth, minRectHeight)
		maxOne = max(minRectWidth, minRectHeight)
		ratio = float(maxOne)/minOne
		print('ratio...............: ' + str(ratio))
		if twoFourRatio > 0.67 and min(minRectWidth, minRectHeight) > 50 and max(minRectWidth, minRectHeight) > 75 and minRectArea > 2800 and ratio > 0.8 and ratio < 1.8:
			numberOfPossiblePlate = numberOfPossiblePlate + 1
			print('Possible license plate number %d' % (numberOfPossiblePlate))
			cv2.rectangle(imgScene, (x, y), (x + w, y + h), (0, 0, 255), 2)
			cv2.imshow('imgSceneAfterDetection', imgScene)


			point = []
			for i in range(0, 4):
				p = find_point_with_min_distance(box[i], hull)
				point.append(p)
			print('point')
			print(point)
			cv2.line(aaaa, tuple(point[0]), tuple(point[1]), (0, 0, 255), 3)
			cv2.line(aaaa, tuple(point[1]), tuple(point[2]), (0, 0, 255), 3)
			cv2.line(aaaa, tuple(point[2]), tuple(point[3]), (0, 0, 255), 3)
			cv2.line(aaaa, tuple(point[3]), tuple(point[0]), (0, 0, 255), 3)
			cv2.imshow('aaaa', aaaa)
			#ha = np.empty((0, 1, 2), dtype = int)
			#for i in range(0, 4):
			#	ha = np.append(ha, ([[point[i]]]), axis = 0)


			#leftmost = tuple(ha[ha[:,:,0].argmin()][0])			# Point A
			#rightmost = tuple(ha[ha[:,:,0].argmax()][0])		# Point B
			#topmost = tuple(ha[ha[:,:,1].argmin()][0])			# Point C
			#bottommost = tuple(ha[ha[:,:,1].argmax()][0])		# Point D
			
			# Khi nhin bien so tu ben: 				trai                        phai
			# A 	                               tren,trai                 duoi,trai
			# B   								   duoi,phai				tren,phai
			# C 									tren,phai				tren,trai
			# D          							duoi,trai 				duoi,phai

			print('ABCD')
			#print(leftmost)
			#print(rightmost)
			#print(topmost)
			#print(bottommost)
			print('box')
			print(box)
			# A, B, C, D respectively
			#AC = image_point_distance(leftmost, topmost) # width
			#print('AC: ' + str(AC))
			#AD = image_point_distance(leftmost, bottommost) # height
			#print('AD: ' + str(AD))
			ratio = float(19)/14 # ti le thuc cua bien so xe may viet nam
			A = point[1]
			B = point[3]
			C = point[2]
			D = point[0]
			print('A: ' + str(A[0]) + ' ' + str(A[1]))
			if angle >= -45 and angle < 0:
				w = int(image_point_distance(A, C))
				h = int(image_point_distance(A, D))
				pts1 = np.float32([A,C,D,B])
				pts2 = np.float32([[0,0],[200 * ratio,0],[0, 200],[200 * ratio, 200]])
				M = cv2.getPerspectiveTransform(pts1,pts2)
				dst = cv2.warpPerspective(imgThresh,M,(int(ratio * 200),200))
				print('Chup tu ben trai hoac khung nghieng sang trai')
				print('aspectRatio: ' + str(float(w)/h))
				dk = (float(w)/h > 1.0 and float(w)/h < 2.0)
				print('dk: ' + str(dk))
			elif angle > -90 and angle < -45:
				w = int(image_point_distance(C, B))
				h = int(image_point_distance(C, A))
				pts1 = np.float32([A,C,D,B])
				pts2 = np.float32([[0,200],[0,0],[200 * ratio, 200],[200 * ratio, 0]])
				M = cv2.getPerspectiveTransform(pts1,pts2)
				dst = cv2.warpPerspective(imgThresh,M,(int(ratio * 200),200))
				print('Chup tu ben phai hoac khung nghieng sang phai')
				dk = (float(w)/h > 0.5 and float(w)/h < 1.0)
				print('dk: ' + str(dk))
				print('aspectRatio: ' + str(float(w)/h))
			elif angle == 0: # angle = 0 or angle = -90 -> anh da nam ngang, chi viec cat ra, khong can bien doi gi
				x = A[0]
				y = A[1]
				w = int(image_point_distance(A, C))
				h = int(image_point_distance(A, D))
				dst = imgThresh[y:y+h, x:x+w]
				print('Chup chinh dien')
				dk = (float(w)/h > 1.0 and float(w)/h < 2.0)
				print('dk: ' + str(dk))
				print('aspectRatio: ' + str(float(w)/h))
			else:
				x = C[0]
				y = C[1]
				w = int(image_point_distance(C, B))
				h = int(image_point_distance(C, A))
				dst = imgThresh[y:y+h, x:x+w]
				print('Chup chinh dien')
				dk = (float(w)/h > 0.5 and float(w)/h < 1.0)
				print('dk: ' + str(dk))
				print('aspectRatio: ' + str(float(w)/h))
			intensityMeanOfPossiblePlate = (cv2.mean(dst))[0]
			print('intensityMeanOfPossiblePlate: ' + str(intensityMeanOfPossiblePlate))

			# Dieu kien: 2 cap canh tren-duoi, trai-phai phai song song voi nhau
			dk2 = False
			dk3 = False
			if D[0] - B[0] == 0 and D[1] - B[1] == 0:
				dk2 = False
				print('Vao tren')
			elif D[0] - B[0] == 0:
				parallelOne = A[0] - C[0]
			elif D[1] - B[1] == 0:
				parallelOne = A[1] - C[1]
			else:
				parallelOne = float(A[0] - C[0])/(D[0] - B[0]) - float(A[1] - C[1])/(D[1] - B[1])
			if C[0] - B[0] == 0 and C[1] - B[1] == 0:
				dk3 = False
				print('Vao duoi')
			elif C[0] - B[0] == 0:
				parallelOne = A[0] - D[0]
			elif C[1] - B[1] == 0:
				parallelOne = A[1] - D[1]
			else:
				parallelTwo = float(A[0] - D[0])/(C[0] - B[0]) - float(A[1] - D[1])/(C[1] - B[1])
			print('parallelOne: ' + str(parallelOne))
			#print('parallelTwo: ' + str(parallelTwo))
			if intensityMeanOfPossiblePlate > 66 and intensityMeanOfPossiblePlate < 115:
				cv2.drawContours(imgContoursAfterSeveralCondition, [contour], 0, (255, 255, 255), 1)
				cv2.imshow('imgContoursAfterSeveralCondition', imgContoursAfterSeveralCondition)
				cv2.imshow('dst', dst)
			cv2.waitKey(0)
		print('\n')
		#cv2.waitKey(0)
	#abc = cv2.Canny(imgContoursAfterSeveralCondition, 30, 40, 0, 3, True)
	#cv2.imshow('abc', abc)
	abc = cv2.cvtColor(imgContoursAfterSeveralCondition, cv2.COLOR_BGR2GRAY)
	cv2.imshow('abc', abc)
	lines = cv2.HoughLines(abc,1,np.pi/180,50)
	for line in lines:
		rho,theta = line[0]
		print('rho: ' + str(rho))
		print('theta: ' + str((float(theta)*180)/np.pi))
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a*rho
		y0 = b*rho
		x1 = int(x0 + 1000*(-b))
		y1 = int(y0 + 1000*(a))
		x2 = int(x0 - 1000*(-b))
		y2 = int(y0 - 1000*(a))
		cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)
		cv2.imshow('houghlines3.jpg',img)
		cv2.waitKey(0)
	if numberOfPossiblePlate == 1:
		print('There are 1 possible license plate')
	else:
		print('There are %d possible license plates.\n' % numberOfPossiblePlate)
	return img


	# Voi moi buc anh, duyet anh va xet cac vung ROI, voi moi vung ROI thi dung perspective transformation de chuyen ve huong cua no roi phan tich tiep


	# Mau trung binh (doi voi VN la black-white, blue-white, red-white), contour == rectangle?, child contours co phai la ky tu khong (de sau)
	# Mot so special situations:
	# 1: co nhieu va mo (da xong)
	# 2: thieu sang (chinh minVal, maxVal cua Canny -> da xong)
	# 3: cac mau khac (tam bo qua vi day la bien xe oto -> kien nghi, doi voi xe may thi khong co 2 cap mau con lai)
	# 4: nhin cac huong khac nhau -> cung lam anh huong den cuong do mau trung binh cua bien so xe
		# 1: nhin truc dien or nhin tu tren xuong -> bien la hinh vuong va dang xu ly 90 %
		# 2: xe dang nga de re or xe dang chong chan -> xe la hinh vuong nhung duoc quay mot goc alpha
		# 3: xe duoc nhin o hai ben -> bien xe thanh hinh binh hanh, chieu dai bi bien dang, thu nho lai
	# 5: khong phu thuoc kich thuoc


	# Situation 4:
	# Ket qua cua Canny la mot tap rat nhieu contour, do ta phai tinh den truong hop thieu sang va nhieu
	# Duyet qua tap nay, giu lai nhung contour co S cua convex contour xap xi (90 %) = S cua minRect va contourArea du lon
	# Voi moi contour, Dung perspectiveTransform quay phan anh tren imgScene ung voi contour ve hinh anh nhin truc dien va cat ra lam anh moi
	# Voi moi anh moi nay, xet cac dac tinh nhu ti le dai rong, cuong do mau trung binh
	# con o imgScene xet mot so dac tinh khac cua anh nhu o duoi contour co phải lốp xe không, cấu trúc contour con của nó ntn
	# tu hai yeu to tren de suy ra contour co phai la bien số xe không?


	# Aspect Ratio
	# 1.18
	# 1.343
	# 1.398
	# 1.243
	# 0.786: sai
	# 1.135: dung
	# 1.885: dung @@
	# 0.65: sai
	# 1.28: sai @@
	# 1.336: dung


	# 310 226
	# 98 52


	# imgScene
	# (194.11816072778458, 195.19844641972733, 196.78210776008132, 0.0)
	# (162.9175654853621, 153.60806368772472, 153.29419619928095, 0.0): 2: chuan nhat
	# (123.48777348777348, 118.20499517374517, 182.99404761904762, 0.0): sai nhat
	# (76.03080847723706, 69.45937990580848, 143.99136577708006, 0.0) sai nhat nhat
	# (189.89720416468475, 168.18397547698325, 172.83811637862692, 0.0): 5: chuan
	# (173.02357844967028, 172.87781510513875, 171.34160756501183, 0.0): 6: sai
	# (146.5176812104152, 156.45575299085152, 175.11418015482056, 0.0): 7: sai
	# (136.81451475353916, 123.01031894934334, 139.64045710387174, 0.0): 8: cuc sai
	# (146.13604746317512, 134.547599563557, 133.79405346426623, 0.0): 9 dung
	# (170.71617933723198, 155.45492202729045, 171.23901072124755, 0.0)


	# imgThresh
	# 1: 175
	# 2: 182
	# 3: 166
	# 4: 161
	# 5: 169
	# 6: 165
	# 7: 165
	# 8: 172
	# 9: 176
	# 10: 182
	# 11: 171
	# 12: 165


	# mean
	# color: (119.3983328335832, 114.1322548725637, 105.5795172413793, 0.0)
	# Canny: (3.5202998500749625, 0.0, 0.0, 0.0)
	# Gray: (112.17445877061469, 0.0, 0.0, 0.0)