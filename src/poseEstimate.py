import numpy as np
import cv2
import cv2.aruco as aruco
import glob
from calibrate_camera import loadInrensicparameters

# 0 -> default camera,...
cap = cv2.VideoCapture(0)

# arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# set dictionary size depending on the aruco marker selected
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

# Load intrensic parameter of camera
mtx, dist = loadInrensicparameters("./camera_intrensics.yml")

# ArUco marker length
marker_length = 0.01

# Reference Id
refId = 1

# Returns pose of 2 w.r.t 1
def relativePose(rvec1, tvec1, rvec2, tvec2):
    
    RT_ref, _ = cv2.Rodrigues(rvec1)
    rel_rvec, _ = cv2.Rodrigues(np.matmul(cv2.Rodrigues(rvec2)[0], RT_ref))
    rel_tvec = tvec2 - np.dot(RT_ref, tvec1.reshape((3, 1))).reshape((1, 3))

    return rel_rvec, rel_tvec

###------------------ ARUCO TRACKER ---------------------------
while (True):
    ret, frame = cap.read()

    # BGR2GRAY
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # This specify parameters for ArUco detector, we will have to change after arena setup based on results
    parameters = aruco.DetectorParameters_create()
    parameters.adaptiveThreshConstant = 10

    # lists of ids and the corners belonging to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # font for displaying text (below)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Checking if there is detection
    if np.all(ids != None):
        # Rotational and translational vectors
        rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(corners, marker_length, mtx, dist)
        (rvec-tvec).any() # To get rid of that nasty numpy value array error
        
        ref_idx = 0
        
        for i in range(0, ids.size):
            #print((relativePose(rvec[ref_idx], tvec[ref_idx], rvec[i], tvec[i])[1]))
            # draw axis for the aruco markers
            aruco.drawAxis(frame, mtx, dist, rvec[i], tvec[i], 2*marker_length)

        # draw a square around the markers
        frame = aruco.drawDetectedMarkers(frame.copy(), corners, ids)

        # Print markers found
        strg = ''
        for i in range(0, ids.size):
            strg += str(ids[i][0])+', '

        #cv2.putText(frame, "Id: " + strg, (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)
        print("Id: " + strg)

    else:
        #cv2.putText(frame, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)
        print("No Ids")

    # Display Live stream
    cv2.imshow('Live stream',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close all windows
cap.release()
cv2.destroyAllWindows()