"""
Change path on line 18

"""





import argparse
import imutils
import cv2
import numpy as np
from align import align_images
import random as rng

def find_fields(args):
    image = cv2.imread("/Users/ferbrjan/Documents/KSY/DatasetCropped/0/3.jpg") #change path pls DEFAULT IMAGE USED FOR TEMPLATE
    #cv2.imshow("Image", image)
    #cv2.waitKey(0)
    canny = cv2.Canny(image, 40, 400, 3)
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_list = []
    hierarchy_list = []
    contour_ids = []
    for i in range(len(contours)):
        approx = cv2.approxPolyDP(contours[i], 0.01 * cv2.arcLength(contours[i], True), True)
        area = cv2.contourArea(contours[i])
        k = cv2.isContourConvex(approx)
        if ((len(approx) > 4) & (len(approx) < 30) & (area > 100) & k):
            contour_list.append(contours[i])
            hierarchy_list.append(hierarchy[0][i])
            contour_ids.append(i)

    blank_image = np.zeros(image.shape, np.uint8)
    cx_copy = 0
    cy_copy = 0
    for i in contour_list:
        M = cv2.moments(i)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            if (int(cx_copy) == int(cx) and int(cy_copy) == int(cy)):
                cx_copy = cx
                cy_copy = cy
            else:
                cv2.drawContours(blank_image, [i], -1, (255, 255, 255), 2)
                cx_copy = cx
                cy_copy = cy

    # Missing playing fields
    cv2.circle(blank_image, (270, 562), 18, (255, 255, 255), 2)
    cv2.circle(blank_image, (369, 72), 18, (255, 255, 255), 2)
    cv2.circle(blank_image, (73, 318), 18, (255, 255, 255), 2)
    cv2.circle(blank_image, (73, 269), 18, (255, 255, 255), 2)
    cv2.circle(blank_image, (564, 367), 18, (255, 255, 255), 2)

    # Missing homefields
    cv2.circle(blank_image, (59, 36), 14, (255, 255, 255), 2)
    cv2.circle(blank_image, (603, 59), 14, (255, 255, 255), 2)

    blank_image = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)

    board = cv2.imread(args["image"])  # , cv.COLOR_BGR2RGBA)
    #cv2.imshow("IMAGEL", board)
    #cv2.waitKey(0)

    output, transform_matrix = align_images(board, image)
    #cv2.imshow("IMAGEL", output)
    #cv2.waitKey(0)
    circles = cv2.HoughCircles(blank_image, cv2.HOUGH_GRADIENT, 1, 15, param1=10, param2=20, minRadius=10, maxRadius=30)

    circles = np.append(circles, [[[320, 368, 14]]], axis=1)
    circles = np.append(circles, [[[320, 270, 14]]], axis=1)

    map = [9, 17, 6, 23, 22, 7, 34, 42, 13, 32, 2, 5, 15, 45, 47, 44, 40, 29, 48, 52, 12, 24, 16, 26, 57, 25, 28, 3, 35,
           53, 11, 18, 14, 33, 39, 58, 19, 8, 20, 10, 62, 30, 49, 31, 0, 46, 41, 59, 69, 50, 54, 51, 68, 65, 60, 55, 36,
           1, 56, 38, 63, 64, 43, 71, 61, 37, 67, 21, 66, 4, 27, 70]

    circles1 = np.zeros([72, 3])

    cnt = 0
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        cnt = 0
        for i in range(len(circles)):
            circles1[i] = circles[map[i]]
            cnt = cnt + 1

    if circles1 is not None:
        cnt = 0
        for (x, y, r) in circles1:
            cv2.circle(output, (int(x), int(y)), int(r), (0, 255, 0), 4)
            cnt = cnt + 1;
            # print(x,y,r)
            #cv2.imshow("output", np.hstack([image, output]))
            #cv2.waitKey(0)
        # show the output image
        #cv2.imshow("output", np.hstack([image, output]))
        #cv2.waitKey(0)

    P = circles1[0:39]

    blue_boundary_lower = [86, 31, 4]
    blue_boundary_upper = [220, 88, 50]

    left_check = image[int(circles1[40][0])][int(circles1[40][1])]
    top_check = image[int(circles1[44][0])][int(circles1[44][1])]
    right_check = image[int(circles1[48][0])][int(circles1[48][1])]
    bottom_check = image[int(circles1[52][0])][int(circles1[52][1])]

    if left_check[0] > blue_boundary_lower[0] and left_check[0] < blue_boundary_upper[0] and left_check[1] > \
            blue_boundary_lower[1] and left_check[1] < blue_boundary_upper[1] and left_check[2] > blue_boundary_lower[
        2] and left_check[2] < blue_boundary_upper[2]:
        #print("blue found in left!")
        B = [circles1[40], circles1[41], circles1[42], circles1[43], circles1[56], circles1[57], circles1[58],
             circles1[59]]
        R = [circles1[44], circles1[45], circles1[46], circles1[47], circles1[60], circles1[61], circles1[62],
             circles1[63]]
        G = [circles1[48], circles1[49], circles1[50], circles1[51], circles1[64], circles1[65], circles1[66],
             circles1[67]]
        Y = [circles1[52], circles1[53], circles1[54], circles1[55], circles1[68], circles1[69], circles1[70],
             circles1[71]]

    if top_check[0] > blue_boundary_lower[0] and top_check[0] < blue_boundary_upper[0] and top_check[1] > \
            blue_boundary_lower[1] and top_check[1] < blue_boundary_upper[1] and top_check[2] > blue_boundary_lower[
        2] and top_check[2] < blue_boundary_upper[2]:
        #print("blue found in top!")
        Y = [circles1[40], circles1[41], circles1[42], circles1[43], circles1[56], circles1[57], circles1[58],
             circles1[59]]
        B = [circles1[44], circles1[45], circles1[46], circles1[47], circles1[60], circles1[61], circles1[62],
             circles1[63]]
        R = [circles1[48], circles1[49], circles1[50], circles1[51], circles1[64], circles1[65], circles1[66],
             circles1[67]]
        G = [circles1[52], circles1[53], circles1[54], circles1[55], circles1[68], circles1[69], circles1[70],
             circles1[71]]

    if right_check[0] > blue_boundary_lower[0] and right_check[0] < blue_boundary_upper[0] and right_check[1] > \
            blue_boundary_lower[1] and right_check[1] < blue_boundary_upper[1] and right_check[2] > blue_boundary_lower[
        2] and right_check[2] < blue_boundary_upper[2]:
        #print("blue found in right!")
        G = [circles1[40], circles1[41], circles1[42], circles1[43], circles1[56], circles1[57], circles1[58],
             circles1[59]]
        Y = [circles1[44], circles1[45], circles1[46], circles1[47], circles1[60], circles1[61], circles1[62],
             circles1[63]]
        B = [circles1[48], circles1[49], circles1[50], circles1[51], circles1[64], circles1[65], circles1[66],
             circles1[67]]
        R = [circles1[52], circles1[53], circles1[54], circles1[55], circles1[68], circles1[69], circles1[70],
             circles1[71]]

    if bottom_check[0] > blue_boundary_lower[0] and bottom_check[0] < blue_boundary_upper[0] and bottom_check[1] > \
            blue_boundary_lower[1] and bottom_check[1] < blue_boundary_upper[1] and bottom_check[2] > \
            blue_boundary_lower[2] and bottom_check[2] < blue_boundary_upper[2]:
        #print("blue found in bottom!")
        R = [circles1[40], circles1[41], circles1[42], circles1[43], circles1[56], circles1[57], circles1[58],
             circles1[59]]
        G = [circles1[44], circles1[45], circles1[46], circles1[47], circles1[60], circles1[61], circles1[62],
             circles1[63]]
        Y = [circles1[48], circles1[49], circles1[50], circles1[51], circles1[64], circles1[65], circles1[66],
             circles1[67]]
        B = [circles1[52], circles1[53], circles1[54], circles1[55], circles1[68], circles1[69], circles1[70],
             circles1[71]]
    #print(P)
    #print(B)
    #print(R)
    #print(G)
    #print(Y)
    #print(transform_matrix)
    return P, B, R, G, Y, transform_matrix

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="path to the input image")
    args = vars(ap.parse_args())
    P,B,R,G,Y,transform=find_fields(args)