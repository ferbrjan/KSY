import cv2 as cv
import numpy as np
from aux import FeatureExtraction, feature_matching

def align_images(img0,img1):
    features0 = FeatureExtraction(img0)
    features1 = FeatureExtraction(img1)

    matches = feature_matching(features0, features1)
    matched_image = cv.drawMatches(img0, features0.kps, img1, features1.kps, matches, None, flags=2)

    #cv.imshow("IMAGE",matched_image)
    #cv.waitKey(0)

    H, _ = cv.findHomography( features0.matched_pts, features1.matched_pts, cv.RANSAC, 5.0)
    #print("H is:",H)

    h, w, c = img1.shape
    warped = cv.warpPerspective(img0, H, (w, h), borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    #cv.imshow("IMAGE",img0)
    #cv.waitKey(0)
    #cv.imshow("IMAGE",warped) #warped is now matching the image from align_fields.py
    #cv.waitKey(0)

    output = np.zeros((h, w, 3), np.uint8)
    alpha = warped[:, :, 2] / 255.0
    output[:, :, 0] = (1. - alpha) * img1[:, :, 0] + alpha * warped[:, :, 0]
    output[:, :, 1] = (1. - alpha) * img1[:, :, 1] + alpha * warped[:, :, 1]
    output[:, :, 2] = (1. - alpha) * img1[:, :, 2] + alpha * warped[:, :, 2]

    #cv.imshow("IMAGE",output)
    #cv.waitKey(0)
    return warped,H