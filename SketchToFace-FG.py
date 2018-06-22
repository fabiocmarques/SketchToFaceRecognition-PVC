import cv2
import numpy as np
import os

S = 32
displacement = 16


def readPictures(path):
    files = sorted(os.listdir(path))
    picList = []

    for imagefile in files:
        if imagefile[-4:] != ".jpg":
            continue

        img = cv2.imread(path + imagefile, 0)
        picList.append(img)

    return picList

def generateSiftDescriptors(imgList):
    sift = cv2.xfeatures2d.SIFT_create()
    descList = []

    for img in imgList:
        descList.append([])
        h, w = img.shape

        for x in range(0, w, displacement):
            for y in range(0, h, displacement):
                print("X: " + str(x) + " Y: " + str(y))
                if x + S >= w or y + S >= h:
                    continue

                desc = sift.compute(img, cv2.KeyPoint(x, y, 32))
                descList[-1].append(desc)
            
        print(descList)
        exit(0)

    return descList

def main():
    print("start")
    sketches = readPictures("./sketches/")
    photos = readPictures("./photos/")

    sketchesDescs = generateSiftDescriptors(sketches)


if __name__ == "__main__":
    main()