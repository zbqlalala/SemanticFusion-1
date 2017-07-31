import cv2
import numpy as np
from matplotlib import pyplot as plt

# img = cv2.imread('Test.jpg',1)
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cv2.imshow('image', img)
# cv2.imshow('img-gray', img_gray)
# cv2.waitKey()
# cv2.destroyAllWindows()

# Create a black image
# img = np.full((512,512,3),255, np.uint8)
# plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()

img = np.zeros((512,512,3),np.uint8);

# Draw a diagonal blue line with thickness of 5 px
cv2.line(img,(0,0),(511,511),(255,0,0),5)

cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)

cv2.ellipse(img,(256,256),(100,50),0,0,180,255,-1)

pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
pts = pts.reshape((-1,1,2))
cv2.polylines(img,[pts],True,(0,255,255))

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)

cv2.imshow('---img---', img)
cv2.waitKey()