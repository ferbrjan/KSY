from align_fields import find_fields
import argparse
import imutils
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())
P,B,R,G,Y,transform=find_fields(args)
print(len(P),len(B),len(R),len(G),len(Y))
print(transform)

#This will be the input from NN
fig_pos_inp = [['Y', 73, 599], ['G', 563, 372], ['G', 588, 553], ['R', 370, 80], ['R', 599, 83], ['Y', 86, 598], ['G', 173, 364], ['R', 280, 367], ['B', 76, 274], ['B', 39, 87], ['Y', 274, 424], ['B', 267, 521]]
image = cv2.imread(args["image"])
dictionary = {}
for i in range (len(fig_pos_inp)):
    x = fig_pos_inp[i][1]
    y = fig_pos_inp[i][2]
    color = fig_pos_inp[i][0]

    vector = [x,y,1]
    transformed_vector = np.matmul(transform,vector)
    x_t = transformed_vector[0]
    y_t = transformed_vector[1]

    union = np.concatenate((P,B,R,G,Y))
    dist_min = np.inf
    idx=-1
    for j in range (len(union)):
        dist = np.sqrt((x_t-union[j][0])**2 + (y_t-union[j][1])**2)
        #print(dist)
        if (dist<dist_min):
            dist_min=dist
            idx = j

    field = ""
    if idx<40:
        field = "P" + str(idx)
    elif idx<48:
        field = "B" + str(idx-40)
    elif idx<56:
        field = "R" + str(idx-48)
    elif idx<64:
        field = "G" + str(idx-56)
    else:
        field = "Y" + str(idx-64)
    #print(idx)
    #print(field)
    #cv2.circle(image, (int(union[idx][0]), int(union[idx][1])), int(union[idx][2]), (0, 255, 0), 4)
    dictionary[field] = color

#print(dictionary)
#cv2.circle(image, (int(union[39][0]), int(union[39][1])), int(union[39][2]), (255, 255, 0), 4)
#cv2.imshow("IMAGE",image)
#cv2.waitKey(0)




#SAVE DICTIONARY TO JSON!!!!