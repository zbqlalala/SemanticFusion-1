import os
import cv2

datasetsDirs = '/home/hoangqc/Desktop/Datasets/TuSimpleCVPR/train_set/clips'

list = open("output.txt","r").read().splitlines()
for file in list:
    # print file
    img = cv2.imread(str(file),1)
    cv2.imshow('test',img)
    quitKey=cv2.waitKey(5)
    if quitKey == 27:
        break
cv2.destroyAllWindows()