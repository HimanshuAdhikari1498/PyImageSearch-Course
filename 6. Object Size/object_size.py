# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

# midpoint function that wil return midpoint between top-left,top-right,bottom-right and bottom-left
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# construct the argument parse and parse the arguments    
ap=argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="path to the input image")
ap.add_argument("-w","--width",type=float,required=True,help="width of the left-most object in the image (in inches)")
args=vars(ap.parse_args())

# load the image, convert it to grayscale, and blur it
image=cv2.imread(args["image"])
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray=cv2.GaussianBlur(gray,(7,7),0)

# perform edge detection, then perform a dilation and erosion to close gaps in between object edges
edged=cv2.Canny(gray,50,100)
edged=cv2.dilate(edged,None,iterations=1)
edged=cv2.erode(edged,None,iterations=1)

# find contours in the edged image
cnts=cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts=imutils.grab_contours(cnts)

# sort the contours from left-to-right and initialize the 'pixels per metric' calibration variable
(cnts,_)=contours.sort_contours(cnts)
pixelsPerMetric = None

# loop over the contours individually
for c in cnts:
    # if the contour is not sufficiently large, ignore it
    if cv2.contourArea(c)<100:
        continue
    
    # compute the rotated bounding box of the contour
    orig=image.copy()
    box=cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    
    # order the points in the contour such that they appear in top-left, top-right, bottom-right, and bottom-left order, then draw the outline of the rotated bounding box
    box=perspective.order_points(box)
    cv2.drawContours(orig,[box.astype("int")],-1,(0,255,0),2)
    
    #cv2.imshow("Orig",orig)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    # loop over the original points and draw them
    for (x, y) in box:
        cv2.circle(orig,(int(x),int(y)),5,(0,0,255),-1)
        
        #cv2.imshow("Orig",orig)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    
    # unpack the ordered bounding box,
    (tl,tr,br,bl)=box
    #compute the midpoint between the top-left and top-right coordinates, and then the midpoint between bottom-left and bottom-right coordinates
    (tltrx,tltry)=midpoint(tl,tr)
    (blbrx,blbry)=midpoint(bl,br)
    
    # compute the midpoint between the top-left and bottom-left points, and then the midpoint between the top-right and bottom-right
    (tlblx,tlbly)=midpoint(tl,bl)
    (trbrx,trbry)=midpoint(tr,br)
    
    # draw the midpoints on the image
    cv2.circle(orig,(int(tltrx),int(tltry)),5,(255,0,0),-1)
    cv2.circle(orig,(int(blbrx),int(blbry)),5,(255,0,0),-1)
    cv2.circle(orig,(int(tlblx),int(tlbly)),5,(255,0,0),-1)
    cv2.circle(orig,(int(trbrx),int(trbry)),5,(255,0,0),-1)
    
    #cv2.imshow("Orig",orig)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    # draw lines between the midpoints
    cv2.line(orig, (int(tltrx), int(tltry)), (int(blbrx), int(blbry)),(255, 0, 255), 2)
    cv2.line(orig, (int(tlblx), int(tlbly)), (int(trbrx), int(trbry)),(255, 0, 255), 2)
    
    #cv2.imshow("Orig",orig)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    # compute the Euclidean distance between the midpoints
    dA = dist.euclidean((tltrx, tltry), (blbrx, blbrx))
    dB = dist.euclidean((tlblx, tlbly), (trbrx, trbry))
    
    # if the pixels per metric has not been initialized, then compute it as the ratio of pixels to supplied metric(in this case, inches) that is euclidean distance between width and given width
    if pixelsPerMetric is None:
        pixelsPerMetric=dB/args["width"]
    
    # compute the size of the object
    dimA=dA/pixelsPerMetric
    dimB=dB/pixelsPerMetric
    
    # draw the object sizes on the image
    cv2.putText(orig,"{:.1f}in".format(dimA),(int(tltrx-15),int(tltry-10)),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,255,255),2)
    cv2.putText(orig,"{:.1f}in".format(dimA),(int(trbrx+10),int(trbry)),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,255,255),2)
    
    # show the output image
    cv2.imshow("Image",orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
