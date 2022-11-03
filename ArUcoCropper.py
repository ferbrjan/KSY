import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

srcFolder = "Dataset"
trgFolder = "DatasetCropped"

print("Loading files")
files = []
os.makedirs(trgFolder, exist_ok=True)
for subdir in os.scandir(srcFolder):
	if not subdir.is_dir():
		continue
	winpath = (trgFolder + subdir.path[len(srcFolder):]).replace(os.sep, '/')
	os.makedirs(winpath, exist_ok=True)
	for file in os.scandir(subdir.path):
		files.append(file)
print("Files loaded successfully")

arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_50)
arucoParams = cv2.aruco.DetectorParameters_create()
for file in files:
	print("Working on: ", file.path)
	img = cv2.cvtColor(cv2.imread(file.path), cv2.COLOR_BGR2RGB)

	(corners, ids, rejected) = cv2.aruco.detectMarkers(img, arucoDict, parameters=arucoParams)
	if(ids is None or len(ids) != 4):
		print("Failed to detect ArUco: ", file.path)
		continue
	#print("DETECTED ARUCOS:", ids)

	bounding_corners = [None, None, None, None]
	for i in range(len(ids)):
		id = ids[i]
		corner = corners[i]
		#Pick the corner closest to the clovece board
		bounding_corners[id[0]-1]=(corner[0][2])

	# The image needs to be zoomed in to compensate the 1cm gap between the baord and the ArUco's
	zoom = 0.06
	mid1 = (bounding_corners[0] + bounding_corners[2]) / 2  # The midpoint between TL, BR
	mid2 = (bounding_corners[1] + bounding_corners[3]) / 2  # The midpoint between TR, BL
	bounding_corners[0] = (bounding_corners[0] + mid1*zoom)/(1+zoom)
	bounding_corners[2] = (bounding_corners[2] + mid1*zoom)/(1+zoom)
	bounding_corners[1] = (bounding_corners[1] + mid2*zoom)/(1+zoom)
	bounding_corners[3] = (bounding_corners[3] + mid2*zoom)/(1+zoom)

	# Swap order from TL,TR,BR,BL to TL,TR,BL,BR (T = Top, B = Bottom, L = Left, R = Right)
	bounding_corners[2], bounding_corners[3] = bounding_corners[3], bounding_corners[2]

	# Output image resolution
	out_size = 640
	M = cv2.getPerspectiveTransform(np.float32(bounding_corners), np.float32([[0,0],[out_size,0],[0,out_size],[out_size,out_size]]))
	img = cv2.warpPerspective(img, M, (out_size, out_size), flags=cv2.INTER_NEAREST)

	outpath = trgFolder + file.path[len(srcFolder):]
	outpath = outpath.replace(os.sep, '/')
	print("Writing result to: ", outpath)
	cv2.imwrite(outpath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

	#plt.imshow(img)
	#plt.show()