import cv2

cap = cv2.VideoCapture(0)

# Error handling
if not cap.isOpened():
	raise IOError("Cannot open camera! Check the connection again.")

# Display FPS
print("Frames per second:- ", cap.get(cv2.CAP_PROP_FPS))

while True:
	ret, frame = cap.read()
	
	frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)
	# BGR to GRAY conversion
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# ArUco Dictionary initialisation
	aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
	# ArUco Detector parameters
	parameters =  cv2.aruco.DetectorParameters_create()
	# ArUco Marker detection 
	corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
	# Marking live stream
	frame_markers = cv2.aruco.drawDetectedMarkers(frame.copy(), corners, ids)
	# Display live stream
	cv2.imshow("Live stream", frame_markers)
	
	# Esc key for exiting stream
	c = cv2.waitKey(1)
	if c==27:
		break

# Release frame windows
cap.release()
cv2.destroyAllWindows()
