# piCamera imports
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

# opencv imports
import cv2
import numpy as np
import imutils
from scipy.spatial import distance as dist # calculate pixel distance


# Definitions
minCircle = 10
NumRows = 3
NumCols = 3
ImgW = 320
ImgH = 240
fps = 32
block = [0,0] # image segment dimensions

# set hue limits for tracked object
blueLower = (110,50,50)
blueUpper = (130,255,255)

# calculate the centre points(correponding to actuators)
centrePoints = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]] 
# Store distance from tracked ball to segment centres
distance = [[],[],[],[], [], [], [], [], []]

# initialise the camera
camera = PiCamera()
camera.resolution = (ImgW, ImgH)
camera.framerate = fps
rawCapture = PiRGBArray(camera, size=(ImgW, ImgH))

# allow the camera to warmup
time.sleep(0.1)

# populate x and y arrays
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

def calDistance(distance, ballLoc):
	for a in range(len(distance)):
		distance[a] = int(dist.euclidean(centrePoints[a], ballLoc))
	print(distance)

def prepSegments(frame):
	# get dimensions of test frame
	ImgH = frame.shape[0]

	# calculate the segment dimensions based on image size and num. segments
	block = [int(ImgH/NumRows), int(ImgW/NumCols)]

	print(frame.shape)

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

	print(centrePoints)

# bool to runsetup setup function on first run
firstLoop = True

# capture frames from the camera
for rawFrame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
	frame = rawFrame.array

	# resize frame and convert to HSV
	frame = imutils.resize(frame, width=ImgW)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	if firstLoop:
		print("first loop")
		firstLoop = False
		prepSegments(frame)
	
	# Visually show the segment centres
	for b in range(len(centrePoints)):
		cv2.circle(frame, centrePoints[b], 3, (0, 255, 0), -1)

	# generate mask and dilate and erode to remove minor points
	mask = cv2.inRange(hsv, blueLower, blueUpper)
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
			calDistance(distance, center)



	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# clear the stream in preparation for the next frame
	rawCapture.truncate(0)
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break 

# cleanup the camera and close any open windows
print("Cleaning up")
camera.release()
cv2.destroyAllWindows()