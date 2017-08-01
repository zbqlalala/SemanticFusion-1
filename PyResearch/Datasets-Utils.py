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
a = open("output.txt", "w")
fileList = listFiles(datasetsDirs)
a.write(str(fileList) + os.linesep)
a.close()