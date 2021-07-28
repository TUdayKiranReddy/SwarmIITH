import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl

aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
board = cv2.aruco.CharucoBoard_create(7, 8, 1, .8, aruco_dict)

imboard = board.draw((2000, 2000))
cv2.imwrite("chessboard.tiff", imboard)
plt.imshow(imboard, cmap = mpl.cm.gray, interpolation = "nearest")
plt.show()