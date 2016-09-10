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
import cv2
import math

# THIS CLASS IS FROM THE CODE WE DOWNLOAD FROM INTERNET!!!
# The only add-ons were the ones corresponding to circularities and changes
# done in the range of shapes we wanted to detect
class ShapeDetector:
    
    def __init__(self):
        pass

    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        #Calculates a contour perimeter or a curve length.
        peri = cv2.arcLength(c, True) 
        # Contour approximation is predicated on the assumption 
        # that a curve can be approximated by a series of short 
        # line segments. This leads to a resulting approximated 
        # curve that consists of a subset of points that were defined 
        # by the original cruve.|
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        
        # Calculate area of contour
        area = cv2.contourArea(c)
        
        #Calculate Circularity
        circ = Circularity()
        check_circ = circ.calculate(area, peri, len(approx))
        
        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3 and check_circ:
            shape = "triangle"

        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        elif len(approx) == 4 and check_circ:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)

            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "parallelogram"

        # if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5 and check_circ:
            shape = "pentagon"
        
        # otherwise, we assume the shape is a circle
        elif (len(approx) >= 6) and check_circ == "circle":
            shape = "nearly circular"
            
        # in the latter case you dont know which shape it is
        else:
            shape = "???"
        
        result = [shape, len(approx)]
        # return the name of the shape
        return result

# THIS CLASS IS AN ADD-ON, IT IS NOT FROM THE CODE WE DOWNLOAD FROM INTERNET!!!      
class Circularity:
    def __init__(self):
        pass
    
    # Method used to calculate the circularity of the detected contour
    def calculate(self, area, perimeter, sides):
        # calculate the real circularity of the detected contour 
        circ_real = (4 * math.pi * area) / pow(perimeter, 2) 
        
        # initialize two arrays for the theoretical circularity
        # and the midpoints between circularities
        circs_th1 = []
        circs_th2 = []
        
        # calculate the theoretical circularities up to an octagon
        for s in range(3, 10):
            peri_th = s
            area_th = s/(4*math.tan(math.pi/s))
            circ_th = (4 * math.pi * area_th) / pow(peri_th, 2) 
            circs_th1.append(circ_th)
        
        # calculate the midpoints between circularities up to an octagon
        for i in range(0, len(circs_th1)):
            if i >= len(circs_th1)-1:
                break
            else:
                median_value = (circs_th1[i+1]-circs_th1[i])/2
                circs_th2.append(circs_th1[i] + median_value)
        
        #verify that the shape is an hexagon or less
        if sides <= 5:  
            # take care if the shape is a triangle because there is no other
            # closed shape with less sides
            if sides == 3:
                if circ_real<=0.69:
                    circularity = True
                else:
                    circularity = False
                # this range of circularities was omitted as it was more reliable
                # to work with empirical measurements for a triangle
                #   this_circ_th0 = circs_th1[0]
                #   this_circ_th1 = circs_th2[0]
                    
            # improved range for the circularity of the square/parallelogram
            # using empirical measurements
            elif sides == 4:
                if circ_real<=0.8 and circ_real>0.50:
                    circularity = True
                else:
                    circularity = False                
            # define the range of circularities for the pentagon
            else:
                this_circ_th0 = circs_th1[sides-4]
                this_circ_th1 = circs_th2[sides-3]
                # check if the circularity of the contour that is being analyzed,
                # is in the range between the midpoint circularity of the pentagon
                # and hexagon, and the theoretical circularity from the previous shape
                if (circ_real<=this_circ_th1 and circ_real>=this_circ_th0):
                    circularity = True     
                else:
                    circularity = False
        # consider any shape with more than 5 sides, as nearly circular         
        else:
            if circ_real >= 0.82:
                circularity = "circle"
            else:
                circularity = False
                
        return circularity     
