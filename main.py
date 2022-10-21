# import the necessary packages
from pyimagesearch.shapedetector import ShapeDetector
import argparse
import imutils
import cv2
import numpy as np
import random as rng


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="path to the input image")
    args = vars(ap.parse_args())
    # load the image and resize it to a smaller factor so that
    # the shapes can be approximated better
    image = cv2.imread(args["image"])
    resized = imutils.resize(image, width=300)
    ratio = image.shape[0] / float(resized.shape[0])
    # convert the resized image to grayscale, blur it slightly,
    # and threshold it
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Image", gray)
    cv2.waitKey(0)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imshow("Image", blurred)
    cv2.waitKey(0)
    laplacian = cv2.Laplacian(blurred,cv2.CV_16S,3)
    cv2.imshow("Image", laplacian)
    cv2.waitKey(0)
    thresh = cv2.threshold(blurred, 50, 350, cv2.THRESH_BINARY)[1]
    cv2.imshow("Image", thresh  )
    cv2.waitKey(0)
    canny = cv2.Canny(image, 100,150,3)
    cv2.imshow("Image", canny )
    cv2.waitKey(0)
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    cv2.imshow("Image", image)
    cv2.waitKey(0)

    contour_list = []
    hierarchy_list =[]
    contour_ids=[]
    print(hierarchy.shape)
    for i in range(len(contours)):
        approx = cv2.approxPolyDP(contours[i], 0.01 * cv2.arcLength(contours[i], True), True)
        area = cv2.contourArea(contours[i])
        k = cv2.isContourConvex(approx)
        if ((len(approx) > 8) & (len(approx) < 23) & (area > 1000) & k):
            contour_list.append(contours[i])
            hierarchy_list.append(hierarchy[0][i])
            contour_ids.append(i)
    print(contour_ids)
    contour_list2=[]
    for i in range(len(contour_list)):
        if hierarchy_list[i][3] in contour_ids: #Check if the parent contour is present
            continue
        if hierarchy_list[i][3]+1 in contour_ids: #parent parent
            continue
        else:
            print(hierarchy_list[i])
            contour_list2.append(contour_list[i])

    print(image.shape )
    blank_image = np.zeros(image.shape, np.uint8)
    cv2.drawContours(blank_image, contour_list2 , -1, (255, 0, 0), 2)
    cv2.imshow('Objects Detected', blank_image)
    cv2.waitKey(0)




    # loop over the contours
