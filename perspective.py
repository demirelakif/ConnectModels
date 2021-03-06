import os
import cv2
import numpy as np
import helpers
import math
#
def perspective(img,image):
    #fileName = f.split(".")[0]
    #img = cv2.imread(f+"/" + fileName,0)
    #image = cv2.imread("" + fileName)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    padded = np.zeros(shape=(h + 2,w + 2),dtype=np.uint8)
    padded[1:h+1,1:w+1] = img
    imgCanny = cv2.Canny(padded,22,25)
    contours, _ = cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    applyPerspective = True
    
    corners = []
    CornersX, CornersY = [], []
    for contour in contours:
        peri = cv2.arcLength(contour,True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        for i in approx:
            x, y = i[0]
            corners.append([x,y])
            CornersX.append(x)
            CornersY.append(y)
            
    threshDistance = int((max(CornersY) - min(CornersY)) * 0.3)
    
    TopCorners = []
    for corner in corners:
        if corner[1] < min(CornersY) + threshDistance and TopCorners.count(corner) == 0:
            TopCorners.append(corner)
    
    BottomCorners = []
    leftTopCorner, rightTopCorner, leftBottomCorner, rightBottomCorner = 0, 0, 0, 0
    
    for corner in corners:
        if corner[1] > max(CornersY) - threshDistance and BottomCorners.count(corner) == 0:
            BottomCorners.append(corner)
            
    if len(TopCorners) > 1:
        TopCornersX = [corner[0] for corner in TopCorners]
        middleOfTopCorners = int(min(TopCornersX) + ((max(TopCornersX) - min(TopCornersX))/2))
        leftTopCorner = [corner for corner in TopCorners if corner[0] < middleOfTopCorners]
        rightTopCorner = [corner for corner in TopCorners if corner[0] > middleOfTopCorners]
        if len(leftTopCorner) == 1:
            leftTopCorner = leftTopCorner[0]
        elif len(leftTopCorner) > 1:
            x = [corner[0] for corner in leftTopCorner]
            y = [corner[1] for corner in leftTopCorner]
            leftTopCorner = [min(x),min(y)]
        else:
            leftTopCorner = [min(CornersX),min(CornersY)]
        if len(rightTopCorner) == 1:
            rightTopCorner = rightTopCorner[0]
        elif len(rightTopCorner) > 1:
            x = [corner[0] for corner in rightTopCorner]
            y = [corner[1] for corner in rightTopCorner]
            rightTopCorner = [max(x),min(y)]
        else:
            rightTopCorner = [max(CornersX),min(CornersY)]
    else:
        # print("??st k????e bulamad??")
        applyPerspective = False
    if len(BottomCorners) > 1:
        BottomCornersX = [corner[0] for corner in BottomCorners]
        middleOfBottomCorners = int(min(BottomCornersX) + ((max(BottomCornersX) - min(BottomCornersX))/2))
        leftBottomCorner = [corner for corner in BottomCorners if corner[0] < middleOfBottomCorners]
        rightBottomCorner = [corner for corner in BottomCorners if corner[0] > middleOfBottomCorners]
        if len(leftBottomCorner) == 1:
            leftBottomCorner = leftBottomCorner[0]
        elif len(leftBottomCorner) > 1:
            x = [corner[0] for corner in leftBottomCorner]
            y = [corner[1] for corner in leftBottomCorner]
            leftBottomCorner = [min(x),max(y)]
        else:
            leftBottomCorner = [min(CornersX),max(CornersY)]
        if len(rightBottomCorner) == 1:
            rightBottomCorner = rightBottomCorner[0]
        elif len(rightBottomCorner) > 1:
            x = [corner[0] for corner in rightBottomCorner]
            y = [corner[1] for corner in rightBottomCorner]
            rightBottomCorner = [max(x),max(y)]
        else:
            rightBottomCorner = [max(CornersX),max(CornersY)]
    else:
        # print("Alt k????e bulamad??")
        applyPerspective = False
        
    if applyPerspective:
        widthX = abs(leftTopCorner[0] - rightTopCorner[0])
        widthY = abs(leftTopCorner[1] - rightTopCorner[1])
        PerspectiveWidth = int(math.sqrt((widthX * widthX) + (widthY * widthY)))
        heightX = abs(leftTopCorner[0] - leftBottomCorner[0])
        heightY = abs(leftTopCorner[1] - leftBottomCorner[1])
        PerspectiveHeight = int(math.sqrt((heightX * heightX) + (heightY * heightY)))
        pts1 = np.float32([leftTopCorner,rightTopCorner,leftBottomCorner,rightBottomCorner])

        pts2 = np.float32([[0,0],[PerspectiveWidth,0],[0,PerspectiveHeight],[PerspectiveWidth,PerspectiveHeight]])
        PerspectiveTransform = cv2.getPerspectiveTransform(pts1,pts2)
        PerspectiveImage = cv2.warpPerspective(image,PerspectiveTransform,(PerspectiveWidth,PerspectiveHeight))

        #cv2.imwrite("perspective/" + fileName,PerspectiveImage)
        return PerspectiveImage
    else:
        image = helpers.get_segment_crop(image.squeeze(),0,img.squeeze())
        #cv2.imwrite("perspective/" + fileName,image)
        return image
