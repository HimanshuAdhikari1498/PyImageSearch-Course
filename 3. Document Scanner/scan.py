# import the necessary packages
from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="path to the image to be scanned")
args=vars(ap.parse_args())

# load the image then compute the ratio of the old height to the new height and then clone it, and then resize it
image=cv2.imread(args["image"])
ratio=image.shape[0]/500.0
orig=image.copy()
image=imutils.resize(image,height=500)

# convert the image to grayscale, blur it, and find edges in the image
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray=cv2.GaussianBlur(gray,(5,5),0)
edged=cv2.Canny(gray,75,200)

# show the original image and edge detected image
print("EDGE DETECTION")
cv2.imshow("IMAGE",image)
cv2.imshow("EDGED",edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# find the contours in the edged image, sorted it and keeping only the largest ones, and initialize the screen contour
cnts=cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cnts=imutils.grab_contours(cnts)
cnts=sorted(cnts,key=cv2.contourArea,reverse=True)[:5]

for c in cnts:
    # approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    
    # if our approximated contour has four points, then we can assume that we have found our screen
	if len(approx) == 4:
		screenCnt = approx
		break

# show the contour (outline) of the piece of paper      
print("FINDING CONTOURS")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("OUTLINE", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# apply the four point transform to obtain a top-down view of the original image
wraped=four_point_transform(orig,screenCnt.reshape(4,2)*ratio)

# convert the warped image to grayscale, then threshold it to give it that 'black and white' paper effect
wraped=cv2.cvtColor(wraped,cv2.COLOR_BGR2GRAY)
T = threshold_local(wraped, 11, offset = 10, method = "gaussian")
wraped = (wraped > T).astype("uint8") * 255

# Show the original and scanned images
print("APPLY PERSPECTIVE TRANSFORM")
cv2.imshow("Original", imutils.resize(orig, height = 650))
cv2.imshow("Scanned", imutils.resize(wraped, height = 650))
cv2.waitKey(0)

