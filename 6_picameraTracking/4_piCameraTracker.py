##################################################
###           Pi Opencv Ball Locator           ###
##################################################

#--- IMPORTS ---#
# piCamera imports
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

# python imports
from scipy.spatial import distance as dist # calculate pixel distance
import math # to calculate the maximum distance a certain actuator(actuators active area dimensions)

# Opencv imports
import cv2
import numpy as np
import imutils

# Raspberry Pi Actuator imports
import RPi.GPIO as GPIO

#--- Definitions ---# 
# minimum valid circle
minCircle = 10

# number of rows and cols in image segment
NumRows = 3
NumCols = 3

# image dimensions
# ImgW = 640
# ImgH = 480
ImgW = 320
ImgH = 240

# calculate the diagonal distance from the four actuators to the center point(max distance they must deal with)
maxDistance = (math.sqrt(math.pow(ImgH/NumRows, 2)+math.pow(ImgW/NumCols, 2)))

# camera settings
fps = 60

# placeholders for image segment details
block = [0,0] # image segment dimensions


## print Options
p_Iterations = False # current iteration number
p_AccVals = False # actuator values
p_AccPositions = False # translated actuator PWM signal
p_Distances = False # distance values

# Flags
ACTUATORSON = True # route signals to actuators
TUNEHSVRANGE = False # tune the target HSV range, with trackbar and target color area of image
DISPLAY = False # true if working from gui

# calculate the centre points of a 9x9 grid layed on top of the image
centrePoints = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]] 
# Store distance from tracked ball to segment centres
distance = [[],[],[],[], [], [], [], [], []]
# PWM value
pwmVals = [[],[],[],[], [], [], [], [], []]

#---- INITIALISATIONS ---#
### initialise the camera ###
camera = PiCamera()
camera.resolution = (ImgW, ImgH)
camera.framerate = fps
rawCapture = PiRGBArray(camera, size=(ImgW, ImgH))
time.sleep(0.1) # allow the camera to warmup

### Actuator Initialisation ###
if ACTUATORSON:
	# pin numbers
	pin1 = 3
	pin2 = 5
	pin3 = 7
	pin4 = 8

	# motor config details
	ACCfreq = 200
	ACCtopLimit = 100
	ACCbottomLimit = 0

	# used to stop motors if no circle found
	pwmValsZERO = [[0],[0],[0],[0]]

	# choose pin numbering system
	GPIO.setmode(GPIO.BOARD)

	# prevent Pi warning on startup
	GPIO.setwarnings(False)

	# set pins as output
	GPIO.setup(pin1, GPIO.OUT)
	GPIO.setup(pin2, GPIO.OUT)
	GPIO.setup(pin3, GPIO.OUT)
	GPIO.setup(pin4, GPIO.OUT)

	# set PWM pins and frequency
	ACCa = GPIO.PWM(pin1, ACCfreq)
	ACCb = GPIO.PWM(pin2, ACCfreq)
	ACCc = GPIO.PWM(pin3, ACCfreq)
	ACCd = GPIO.PWM(pin4, ACCfreq)

	# initialise PWM values
	ACCa.start(0)
	ACCb.start(0)
	ACCc.start(0)
	ACCd.start(0)


###  Target Color definition ### 

## set hue limits for tracked object
# blue nivea bottle top
# blueLower = (110,50,50)
# blueUpper = (130,255,255)

# green tennis ball
greenLower = (20,100,100)
greenUpper = (50,255,255)

# plastic Green Ball
# greenLower = (29,86,6)
# greenUpper = (64,255,255)

# target color hsv range, set to current target color range
ColorLower = greenLower
ColorUpper = greenUpper

# function to assign the new track
def nothing(x):
	pass

if TUNEHSVRANGE:
	## adjust target color
	cv2.namedWindow('upperLimit')
	cv2.namedWindow('bottomLimit')
	cv2.namedWindow('mask')

	# create trackbars for bottom of target color range
	cv2.createTrackbar('H','bottomLimit',ColorLower[0],255,nothing)
	cv2.createTrackbar('S','bottomLimit',ColorLower[1],255,nothing)
	cv2.createTrackbar('V','bottomLimit',ColorLower[2],255,nothing)

	# create trackbars for top of target color range
	cv2.createTrackbar('H','upperLimit',ColorUpper[0],255,nothing)
	cv2.createTrackbar('S','upperLimit',ColorUpper[1],255,nothing)
	cv2.createTrackbar('V','upperLimit',ColorUpper[2],255,nothing)


#---- FUNCTIONS ---#

### Color Range Functions

# get and update hsv range values from trackbar
def getHSVRange():
	# get upper hsv range values
	h1 = cv2.getTrackbarPos('H', 'upperLimit')
	s1 = cv2.getTrackbarPos('S', 'upperLimit')
	v1 = cv2.getTrackbarPos('V', 'upperLimit')
	
	# get lower hsv range values
	h2 = cv2.getTrackbarPos('H', 'bottomLimit')
	s2 = cv2.getTrackbarPos('S', 'bottomLimit')
	v2 = cv2.getTrackbarPos('V', 'bottomLimit')

	# update stored hsv range values
	global ColorUpper 
	ColorUpper = (h1,s1,v1)
	global ColorLower 
	ColorLower = (h2,s2,v2)


### Image segmenting functions

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


### Ball Location Functions

# Calculate and rtn the distance(pixels) between ball centre point and segment centre
def calDistance(distance, ballLoc):
	for a in range(len(distance)):
		tmpDist = int(dist.euclidean(centrePoints[a], ballLoc))
		if tmpDist > maxDistance:
			distance[a] = maxDistance
		elif tmpDist < 0:
			distance[a] = 0
		else:
			distance[a] = tmpDist
	if p_AccVals:
		print("The current distances to the centre points are: {}".format(distance))

# Transforms the distance values into corresponding PWM values
def transformDist(distance, pwmVals):
	# old range
	OldRange = (maxDistance - 0)
	# new range
	NewRange = (ACCtopLimit - ACCbottomLimit)  

	for i in range(len(pwmVals)):
		# Invert total distance (OldMax - distance) to have a high value for closer
		invertedDist = maxDistance - distance[i] 


		# Transform inverted distance into PWM range(0-100)
			# NewValue = ((((OldValue - OldMin) * NewRange) / OldRange) + NewMin)
		pwmVals[i] = (((invertedDist) * NewRange) / OldRange)

		# print("before doing the operation\n, distance: {}, invertedDist {}, NewRange: {}, OldRange: {}". format(distance[i], invertedDist, NewRange, OldRange))
		# print("This is the current val {} \n".format(pwmVals[i]))
	
# update actuator with new PWM value
def confirmPWMVals(value):
	# print PWM position if flagged
	if p_AccPositions:
		print("\n\nThese are the values INSIDE: {}\n".format(value))

	# only send signals to positions which have actuators
	activePins = (0,2,6,8)
	pwmValues = [0,0,0,0]

	# confirm all values are in range and save to array if true
	for i in range(len(activePins)):
		if(float(value[activePins[i]]) > ACCtopLimit or float(value[activePins[i]]) < ACCbottomLimit):	
			print("ERR: Adjusted PWM value not within safe range, value: {}.".format(value[activePins[i]]))
			return False		
		else:
			# if value is within range save to array
			pwmValues[i] = float(value[activePins[i]])
			# pwmValues[i] = float(0.0)

	# update the actuators with new pwm values
	updateActVals(pwmValues)

def updateActVals(values):
	if (len(values) != 4):
		print("ERR: Incorrect number of pin values entered, value: {}.".format(len(values)))
		return False

	# convert list to numpy float array
	value1 = np.array(values) + 0.

	# print the values and type if debug flag is true
	if p_AccPositions:
		print("These are the type: {} and values: {}".format(type(value1), value1))

	# assign pins to new PWM values
	ACCa.ChangeDutyCycle(value1[0])
	ACCb.ChangeDutyCycle(value1[1])
	ACCc.ChangeDutyCycle(value1[2])
	ACCd.ChangeDutyCycle(value1[3])


#---- MAIN ---#

firstLoop = True # bool to run setup function on first run

# variables to track the FPS 
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

	# update hsv range if adjusting flag is true
	if TUNEHSVRANGE:
		# update hsv values from trackbar
		getHSVRange()

	# visually show the segment centres
	for b in range(len(centrePoints)):
		cv2.circle(frame, centrePoints[b], 3, (0, 255, 0), -1)

	# generate mask and dilate and erode to remove minor points
	mask = cv2.inRange(hsv, ColorLower, ColorUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

	# if tuning hsv range show identified area of color in image
	if TUNEHSVRANGE:
		cv2.imshow("mask", mask)
		# print hsv limits
		print("Bottom Limits: {} Upper limits: {}".format(ColorLower, ColorUpper))

	# find contours
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2]
	# initialise (x,y) ball centre
	center = None

	# continue if a contour was found
	if len(cnts) > 0:
		# find the largest contour, 
		contour = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(contour)

		# calculate the contour centre
		M = cv2.moments(contour)
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
		center = (cX, cY)

		# only continue if radius is above the threshold
		if radius > minCircle:
			# draw circle to frame
			cv2.circle(frame, (int(x), int(y)), int(radius),(0, 100, 255), 2)
			# draw centre point to frame
			cv2.circle(frame, center, 5, (255, 0, 255), -1)
			
			if ACTUATORSON:
				# calculate distance from ball to single segment centre
				calDistance(distance, center)
	
				if p_Distances:
					print("This is the distance: {}".format(distance))
			
				# Transform distance(pixels) to PWM based actuator control value
				transformDist(distance, pwmVals)

				# send actuator new control value
				confirmPWMVals(pwmVals) 
	
		elif ACTUATORSON:
			# set all actuators to minimum PWM value if no valid circle found
			updateActVals(pwmValsZERO)

	elif ACTUATORSON:
		# set all actuators to minimum PWM value if no valid circle found
		updateActVals(pwmValsZERO)

	# show the frame if DISPLAY flag == true
	if(DISPLAY):
		cv2.imshow("Frame", frame)	
	key = cv2.waitKey(1) & 0xFF
 
	# clear the stream in preparation for the next frame
	rawCapture.truncate(0)
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break 
	
	# increment iteration counter
	iterations += 1
	
	# print iteration number if flagged
	if p_Iterations:
		print("iteration: {}\n".format(iterations))

# calculate FPS
finishTime = time.time()
timeEleapsed = finishTime - startTime
print("\nTime taken : {} seconds".format(timeEleapsed))

# Calculate frames per second
trueFPS  = iterations / timeEleapsed
print("Estimated frames per second : {}\n".format(trueFPS))

# cleanup the camera and close any open windows
print("\nCleaning up\n")

# cleanup windows if DISPLAY == true
if(DISPLAY):
	cv2.destroyAllWindows()