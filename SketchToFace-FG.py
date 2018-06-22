import cv2
import numpy as np
import os

def readPictures(path):
    files = sorted(os.listdir(path))
    picList = []

    for imagefile in files:
        print(path + imagefile)
        img = cv2.imread(path + imagefile, 0)
        picList.append(img)

    return picList

def generateSiftDescriptors(imgList):
    print("DEsc")

def main():
    print("start")
    sketches = readPictures("./sketches/")
    photos = readPictures("./photos/")



if __name__ == "__main__":
    main()