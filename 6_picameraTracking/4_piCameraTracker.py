##################################################
###           Pi Opencv Ball Locator           ###
##################################################

#--- IMPORTS ---#
# piCamera imports
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

# Opencv imports
import cv2
import numpy as np
import imutils
from scipy.spatial import distance as dist # calculate pixel distance

# Raspberry Pi Servo imports
import RPi.GPIO as IO

#--- Definitions ---# 
minCircle = 10
NumRows = 3
NumCols = 3
# ImgW = 640
# ImgH = 480
ImgW = 320
ImgH = 240

fps = 60
block = [0,0] # image segment dimensions
Dist = [50,50] # basic number to check the range of distances

# calculate the centre points(correponding to actuators)
centrePoints = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]] 
# Store distance from tracked ball to segment centres
distance = [[],[],[],[], [], [], [], [], []]


#---- INITIALISATIONS ---#
### initialise the camera ###
camera = PiCamera()
camera.resolution = (ImgW, ImgH)
camera.framerate = fps
rawCapture = PiRGBArray(camera, size=(ImgW, ImgH))
time.sleep(0.1) # allow the camera to warmup

### Servo Initialisation ###
keyPin = 18
IO.setmode(IO.BCM)
IO.setup(keyPin, IO.OUT)
IO.setwarnings(False)

p = IO.PWM(keyPin, 50) # set pin 18 as a PWM pin with a frequency of 50 Hz
p.start(7.5) # start PWM

###  Target Color definition ### 
# target color hsv range
ColorLower = (0,0,0)
ColorUpper = (0,0,0)

# set upper and lower target color hsv values
def defineTargetColor(h, s, v, flag):
	# set top hsv value
	if flag == True:
		ColorUpper = (h,s,v)		
	# set bottom hsv value	
	else:
		ColorLower = (h,s,v)

## set hue limits for tracked object
# blueLower = (110,50,50)
# blueUpper = (130,255,255)

## tennis ball greenLower
# greeen lower
defineTargetColor(20,100,100, True)
# green upper
defineTargetColor(50,255,255, False)


#---- FUNCTIONS ---#
# calculate the centre points(x,y) of the image segments, return as array
# for example an image with 9 segments(3x3) has the central (x,y) of each 
# segment returned in the array
def populateArray(block):
	tmp = 0
	Arr = [0,0,0]

	for a in range(3):
		if tmp == 0:
			Arr[a] = tmp + int(block/2)
		else:	
			Arr[a] = tmp + block
		tmp = Arr[a]
	return Arr

# Segment the initial frame and calculate the segment centrepoints, store these in 
# centrePoints Array
def prepSegments(frame):
	# confirm dimensions of test frame
	ImgH = frame.shape[0]

	# calculate the segment dimensions based on image size and number of segments save these in block variable
	block = [int(ImgH/NumRows), int(ImgW/NumCols)]
	#print(frame.shape)


	cntW = populateArray(block[1])
	cntH = populateArray(block[0])

	# Insert x,y segment centre values into array
	xCnt = 0
	yCnt = 0
	Counter = 0
	for x in range(len(centrePoints)):
		if(xCnt == 3):
			yCnt += 1
			xCnt = 0

		centrePoints[Counter] = cntW[xCnt], cntH[yCnt]

		xCnt += 1
		Counter += 1

	#print(centrePoints)

# Calculate and rtn the distance(pixels) between ball centre point and segment centre
def calDistance(distance, ballLoc):
	for a in range(len(distance)):
		distance[a] = int(dist.euclidean(centrePoints[a], ballLoc))
	print(distance)

# Development function to find the distance range for specific image segments	
# used to tune the distance to PWM transformation function(transformDist)
def trackCameraDistanceBoundaries(distance, Dist):
	# print("This is the Distance {} and this is DistHigh {}\n".format(distance, Dist[1]))
	if distance > Dist[1]:
		print("distance {} is higher than DistHigh {}".format(distance, Dist[1]))
		Dist[1] = distance
	elif distance < Dist[0]:
		print("distance {} is higher than DistLow {}".format(distance, Dist[0]))
		Dist[0] = distance	
	return Dist

# Transforms the distance values into corresponding PWM values
def transformDist(distance, ImgW):
	OldMin = 0
	OldMax = 180 
	NewMin = 2.5 # 0 degrees
	NewMax = 7.5 # 90 degrees(so servo arm is vertically in the air)
	invertedDist = OldMax - distance[4] # Invert total distance (OldMax - distance)

	# OldRange = (OldMax-OldMin)
	OldRange = (OldMax - OldMin)  
	# NewRange = (NewMax - NewMin)  
	NewRange = (NewMax - NewMin)  
	# NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
	
	#print("before doing the operation\n, distance: {}, OldMin: {}, NewRange: {}, OldRange: {}, NewMin:{}". format(distance[4], OldMin, NewRange, OldRange, NewMin))

	# Transform inverted distance into PWM range(2.5-12.5)
	CurrentVal = ((((invertedDist - OldMin) * NewRange) / OldRange) + NewMin)

	#print("This is the current val {} \n".format(CurrentVal))
	return CurrentVal

# update servo with new PWM position
def updateServo(position):
	print("updating servo to position: {}\n".format(position))
	if(float(position) <12.6 and float(position) > 2.4):	
		p.ChangeDutyCycle(position)
	else:
		return False		

#---- MAIN ---#

firstLoop = True # bool to run setup function on first run
startTime = 0
finishTime = 0
iterations = 0
# capture frames from the camera
for rawFrame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	if iterations == 0:
		startTime = time.time()
		print("starting time: {}\n".format(startTime))
	
	# grab the raw NumPy Array
	frame = rawFrame.array

	# resize(confirm it is correct size) frame and convert to HSV
	frame = imutils.resize(frame, width=ImgW)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	# calculate segments and centre points only on first frame
	if firstLoop:
		print("first loop")
		firstLoop = False
		prepSegments(frame)
	
	# Visually show the segment centres
	for b in range(len(centrePoints)):
		cv2.circle(frame, centrePoints[b], 3, (0, 255, 0), -1)

	# generate mask and dilate and erode to remove minor points
	mask = cv2.inRange(hsv, ColorLower, ColorUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

	# find contours
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2]
	# initialise (x,y) ball centre
	center = None

	# continue if a contour was found
	if len(cnts) > 0:
		# find the largest contour, 
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)

		# calculate the contour centre
		M = cv2.moments(c)
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
		center = (cX, cY)

		# only continue if radius is above the threshold
		if radius > minCircle:
			# draw circle to frame
			cv2.circle(frame, (int(x), int(y)), int(radius),(0, 100, 255), 2)
			# draw centre point to frame
			cv2.circle(frame, center, 5, (255, 0, 255), -1)
			
			# calculate distance from ball to single segment centre
			calDistance(distance, center)
			# DEV USE:
			Dist = trackCameraDistanceBoundaries(distance[4], Dist) # track the highest and lowest recorded values for centre pin

			# Transform distance(pixels) to PWM based servo position, update servo
			servoPos = transformDist(distance, ImgW) # transform the distance to servo PWM value
			updateServo(servoPos) # set Servo to new position

	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# clear the stream in preparation for the next frame
	rawCapture.truncate(0)
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break 
	
	# increment iteration counter
	iterations += 1
	print("iteration: {}\n".format(iterations))

# calculate FPS
finishTime = time.time()
timeEleapsed = finishTime - startTime
print("Time taken : {} seconds".format(timeEleapsed))

# Calculate frames per second
trueFPS  = iterations / timeEleapsed
print("Estimated frames per second : {}".format(trueFPS))

# cleanup the camera and close any open windows
print("Cleaning up")
camera.release()
cv2.destroyAllWindows()