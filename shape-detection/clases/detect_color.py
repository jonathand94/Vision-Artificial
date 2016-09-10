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

# Import the computer vision library and the class created to detect shapes
import cv2
from clases.shapedetector import ShapeDetector

# THIS CLASS IS AN ADD-ON, IT IS NOT FROM THE CODE WE DOWNLOAD FROM INTERNET!!
class ColorDetector:
    # Constructor of the class
    def __init__(self):
        pass
    
    # Method to detect a range of colors
    def detect(self, image, resized, lower, upper, ratio):
        sd = ShapeDetector()
        
        # Make the same filtering using a GaussianBlur, an erosion and dilation.
        # The difference is in the filter used to detect the specific color
        # range that you want
        blurred = cv2.GaussianBlur(resized, (11, 11), 0)
        mask = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(mask, lower, upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)  
        
        # You find again the contours
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        # only proceed if at least one contour was found
        if len(cnts) > 0:   
            #print(len(cnts))
            for c in cnts:               
                    # compute the center of the contour, then detect the name of the
                    # shape using only the contour
                    M = cv2.moments(c)
                    cX = int((M["m10"] / M["m00"]) * ratio)
                    cY = int((M["m01"] / M["m00"]) * ratio)
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
                                