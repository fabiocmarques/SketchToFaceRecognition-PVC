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
               # print("X: " + str(x) + " Y: " + str(y))
                if x + S >= w or y + S >= h:
                    continue

                kp = cv2.KeyPoint(x, y, 32)
                _, desc = sift.compute(img, [kp])
                descList[-1].append(desc[0])

    return np.asarray(descList)

def findPhoto(sketchDesc, photosDescs):
    shortestDistance = float("inf")
    shortestDistanceIndex = 0

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    
    for index, photoDesc in enumerate(photosDescs):
        matches = flann.knnMatch(sketchDesc, photoDesc, 2)
        distance = sum([m[0].distance + m[1].distance for m in matches])

        if distance < shortestDistance:
            shortestDistance = distance
            shortestDistanceIndex = index

    return shortestDistanceIndex


def main():
    sketches = readPictures("./sketches/")
    photos = readPictures("./photos/")

    sketchesDescs = generateSiftDescriptors(sketches)
    photosDescs = generateSiftDescriptors(photos)

    acertos = 0

    for sketchIndex, sketch in enumerate(sketchesDescs):
        photoIndex = findPhoto(sketch, photosDescs)
        print("\n\n" + str(sketchIndex) + " : "+ str(photoIndex))

        if sketchIndex == photoIndex:
            acertos += 1

    print("Acertos: " + str((acertos / len(sketchesDescs)) * 100) + " %")

if __name__ == "__main__":
    main()