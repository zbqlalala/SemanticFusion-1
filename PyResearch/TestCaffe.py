from glob import glob

import os
import cv2

def listFiles(dataDir):
    pathStr = ''
    command =r''+dataDir
    for path, subdirs, files in os.walk(command):
        for filename in sorted(files,key=lambda f: int(os.path.splitext(f)[0])):
            file = os.path.join(path, filename)
            pathStr +=  (str(file) + os.linesep)
    return  pathStr



datasetsDirs = '/home/hoangqc/Desktop/Datasets/TuSimpleCVPR/train_set/clips'
# myCommand = r'' + datasetsDirs
# f = []
# a = open("outputx.txt", "w")
# for path, subdirs, files in os.walk(myCommand):
#    for filename in files:
#      f = os.path.join(path, filename)
#      a.write(str(f) + os.linesep)

# a = open("output.txt", "w")
# fileList = listFiles(datasetsDirs)
# a.write(str(fileList) + os.linesep)

list = open("output.txt","r").read().splitlines()
for file in list:
    # print file
    img = cv2.imread(str(file),1)
    cv2.imshow('test',img)
    quitKey=cv2.waitKey(5)
    if quitKey == 27:
        break

cv2.destroyAllWindows()