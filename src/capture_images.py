import cv2

i = 21
# Camera IO; 0 -> default web cam, change idx for accessing other cams
cap = cv2.VideoCapture(0)

aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

# Error handling
if not cap.isOpened():
	raise IOError("Cannot open camera! Check connections again.")

# Display FPS
print("Frames per second:- ", cap.get(cv2.CAP_PROP_FPS))

while True:
	ret, frame = cap.read()
	
	frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)
	frame_markers = cv2.aruco.drawDetectedMarkers(frame.copy(), corners, ids)

	# Display live stream
	cv2.imshow("Live stream", frame_markers)
	
	k = cv2.waitKey(1)
	if k == 27:
		break
	elif k == 13:
		cv2.imwrite("./calibrationImages/chArUco_" + str(i) + ".png", frame)
		print("Saved ./calibrationImages/chArUco_" + str(i) + ".png")
		i += 1