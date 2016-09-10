# -*- coding: utf-8 -*-
"""
Created on Tue Sep 06 20:54:36 2016

@authors: JONATHAN DOMINGUEZ, ANTONIO ARREOLA, JAVIER FLORES AND HECTOR SANTOYO
"""
# THIS CODE IS BASED IN ANOTHER CODE IMPLEMENTED FOR THE DETECTION OF SHAPES
# IN IMAGES. THEREFORE THIS IS AN IMPROVEMENT OF THE CODE AND IT IS ADAPTED TO
# MAKE THE DETECTION FOR VIDEO CAPTURE

# AUTHOR: ADRIAN ROSERBROCK

# TO DOWNLOAD THE ORIGINAL CODE YOU CAN VISIT THE FOLLOWING WEBSITE:
# http://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/ 

# import the necessary packages
import numpy as np # library for drawing and scientific calculations
import argparse # library for efficient conversion of data types
import imutils # library made by a computer scientist to improve standard functions such as resize
import cv2 # library that has all the computer vision algorithms
from clases.shapedetector import ShapeDetector # ShapeDetector class for simplifying the code
from clases.detect_color import ColorDetector # ColorDetector class for improving the performance of the algorithm

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser() # create an object of the type ArgumentParser

ap.add_argument("-v", "--video",
    help="path to the (optional) video file") # pass the data to convert or process, in this case the video

ap.add_argument("-b", "--buffer", type=int, default=8, 
    help="max buffer size") # pass the buffer to generate space for taking more frames per seconds 
    
args = vars(ap.parse_args()) # store the attribute of the video

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
    camera = cv2.VideoCapture(0)

# otherwise, grab a reference to the video file
else:
    camera = cv2.VideoCapture(args["video"])
 
# generate a yellow range for detection, this was important because the algorithm
# doesnt detect the yellow objects   
yellowLower = (15, 100, 100)
yellowUpper = (45, 255, 255)  

# keep looping for generating frames per second and process them
while True:
    # grab the current frame
    (grabbed, image) = camera.read()

    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if args.get("video") and not grabbed:
        break      
    
    # load the image and resize it to a smaller factor so that
    # the shapes can be approximated better, this is done for 
    # lowering the processing power needed
    resized = imutils.resize(image, width=300)
    ratio = image.shape[0] / float(resized.shape[0]) 
    
    # the size of the filters or mask that you are going to apply to the frame
    kernel = np.ones((5, 5), np.uint8) 
    
    # The image is resized and transformed to gray scale to simplify the CPU processing
    # A series of filtering are done, including a smoothing for eliminating noise
    # A basic thresholding with an improvement using the Otsu Method
    # Finally a morphological transformation is done with OPENING, which removes small objects
    # and its very helpful for avoiding that m00 (moment of mass) is null
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    mask1 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel) 
    
    # find contours in the thresholded image and initialize the
    # shape detector
    cnts = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    sd = ShapeDetector()  
    
    # You find the maximum contour area, this will be useful for avoiding 
    # contour detection at all time. Remember that you have to scale the area
    # to the ratio for which the image was resized. Then you find the minimum
    # closing circle and its radius, and this is useful for having a value  that 
    # will allow you to detect only the contours that reach this radius
    cmax = max(cnts, key=cv2.contourArea)      
    cmax = cmax.astype("float")
    cmax *= ratio
    cmax = cmax.astype("int")
    ((x, y), maxradius) = cv2.minEnclosingCircle(cmax)
    
    # You find the minimum contour area, this will be useful for avoiding 
    # contour detection at all time. Remember that you have to scale the area
    # to the ratio for which the image was resized. Then you find the minimum
    # closing circle and its radius, and this is useful for having a value  that 
    # will allow you to detect only the contours that reach this radius    
    cmin = min(cnts, key=cv2.contourArea)      
    cmin = cmin.astype("float")
    cmin *= ratio
    cmin = cmin.astype("int")
    ((x, y), minradius) = cv2.minEnclosingCircle(cmin)
    
   # only proceed if at least one contour was found
    if len(cnts) > 0:   
        # only proceed if the contour detected is of a considerable size
        if maxradius <= 250 and minradius>=20: 
            # iterate over ALL the contours that were detected
            for c in cnts:               
                # compute the center of the contour, then detect the name of the
                # shape using only the contour, for this we need to calculate the 
                # mass center with the function moments(), then rescale the center
                # coordinates with the ratio of the image that was resized
                M = cv2.moments(c)
                cX = int((M["m10"] / M["m00"]) * ratio)
                cY = int((M["m01"] / M["m00"]) * ratio)
                
                # For more details of this functio, open the ShapeDetector class file
                # this will return an array with the shape that was detected and the
                # number of sides of the figure
                shape = sd.detect(c)[0]
                sides = sd.detect(c)[1]
                
                # multiply the contour (x, y)-coordinates by the resize ratio,
                # then draw the contours and the name of the shape on the image
                c = c.astype("float")
                c *= ratio
                c = c.astype("int")      
                cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
                cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.4, (255, 255, 255), 2)
    
    # This will create an object of the type ColorDetector(). This is done 
    # for improving the algorithm as the yellow color was not detected with the
    # previous filtering, so an specific filter is done for detecting shapes with
    # the color yellow
    color_detector = ColorDetector()
    color_detector.detect(image, resized, yellowLower, yellowUpper, ratio)           
    
    # show the frame to our screen
    cv2.imshow("Shapes detected", image)
    
    # wait for a key to close the program
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break
    
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()