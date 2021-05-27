import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('baboon.png',0)
img2 = img.copy()
template = cv.imread('Righteye.png',0)
template2 = cv.imread('Lefteye.png',0)
w, h = template.shape[::-1]
v, g = template2.shape[::-1]

methods = ['cv.TM_CCOEFF']
for meth in methods:
    img = img2.copy()
    method = eval(meth)
    # Apply template Matching
    res = cv.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
   
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv.rectangle(img,top_left, bottom_right, 255, 2)
    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle('RIGHT EYE')
    plt.show()
    
for meth in methods:
    img = img2.copy()
    method = eval(meth)
    # Apply template Matching
    res1 = cv.matchTemplate(img,template2,method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res1)
   
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv.rectangle(img,top_left, bottom_right, 255, 2)
    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle('LEFT EYE')
    plt.show()
