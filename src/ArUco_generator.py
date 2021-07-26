import cv2
import numpy as np


def gen(number_of_ids=1, op_dir='./markers'):
	'''Available dictonaries:-
		DICT_4X4_50 
		Python: cv.aruco.DICT_4X4_50
		DICT_4X4_100 
		Python: cv.aruco.DICT_4X4_100
		DICT_4X4_250 
		Python: cv.aruco.DICT_4X4_250
		DICT_4X4_1000 
		Python: cv.aruco.DICT_4X4_1000
		DICT_5X5_50 
		Python: cv.aruco.DICT_5X5_50
		DICT_5X5_100 
		Python: cv.aruco.DICT_5X5_100
		DICT_5X5_250 
		Python: cv.aruco.DICT_5X5_250
		DICT_5X5_1000 
		Python: cv.aruco.DICT_5X5_1000
		DICT_6X6_50 
		Python: cv.aruco.DICT_6X6_50
		DICT_6X6_100 
		Python: cv.aruco.DICT_6X6_100
		DICT_6X6_250 
		Python: cv.aruco.DICT_6X6_250
		DICT_6X6_1000 
		Python: cv.aruco.DICT_6X6_1000
		DICT_7X7_50 
		Python: cv.aruco.DICT_7X7_50
		DICT_7X7_100 
		Python: cv.aruco.DICT_7X7_100
		DICT_7X7_250 
		Python: cv.aruco.DICT_7X7_250
		DICT_7X7_1000 
		Python: cv.aruco.DICT_7X7_1000
		DICT_ARUCO_ORIGINAL 
		Python: cv.aruco.DICT_ARUCO_ORIGINAL
		DICT_APRILTAG_16h5 
		Python: cv.aruco.DICT_APRILTAG_16h5
		4x4 bits, minimum hamming distance between any two codes = 5, 30 codes
	'''
	dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

	for i in range(1, number_of_ids+1):
		markerImage = np.zeros((200, 200), dtype=np.uint8)
		markerImage = cv2.aruco.drawMarker(dictionary, i**2, 200, markerImage, 1)
		cv2.imwrite(op_dir + "/marker_id_" + str(i**2) + ".png", markerImage)

	print("Generated {} markers and saved in {}".format(number_of_ids, op_dir))

if __name__ == "__main__":
	gen(number_of_ids=2)