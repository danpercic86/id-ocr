import os

import cv2
import numpy as np
import pytesseract

per = 25

roi = [
    [(548, 396), (599, 420), 'text', 'Serie'],
    [(626, 396), (744, 427), 'text', 'Număr'],
    [(353, 421), (559, 453), 'text', 'CNP'],
    [(320, 472), (857, 499), 'text', 'Nume'],
    [(319, 515), (857, 541), 'text', 'Prenume'],
    [(319, 562), (679, 588), 'text', 'Naţionalitate'],
    [(794, 560), (847, 590), 'text', 'Sex'],
    [(319, 606), (857, 632), 'text', 'Loc naştere'],
    [(319, 648), (857, 704), 'text', 'Domiciliu'],
    [(319, 719), (623, 755), 'text', 'Emisă de'],
    [(628, 722), (740, 753), 'text', 'Valabil de la'],
    [(750, 721), (880, 754), 'text', 'Valabil până la'],

]

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

imgTemplate = cv2.imread('template.jpg')
h, w, c = imgTemplate.shape
# imgT = cv2.resize(imgTemplate, (w // 1, h // 1))
orb = cv2.ORB_create(1000)

keyPoints1, descriptors1 = orb.detectAndCompute(imgTemplate, None)
imgKeyPoints = cv2.drawKeypoints(imgTemplate, keyPoints1, None)
# cv2.imshow("Key Points Img", imgKeyPoints)
# cv2.imshow("Output", imgT)
path = 'buletine'
myPicList = os.listdir(path)

for j, y in enumerate(myPicList):
    img = cv2.imread(path + "/" + y)
    keyPoints2, descriptors2 = orb.detectAndCompute(img, None)
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    # match descriptors
    matches = bf.match(descriptors2, descriptors1)
    # sort them in order of their distance
    matches.sort(key=lambda x: x.distance)
    # draw first "per" percentage matches. per is set to 25
    good = matches[:int(len(matches) * (per / 100))]
    imgMatch = cv2.drawMatches(img, keyPoints2, imgTemplate, keyPoints1, good[:100], None, flags=2)
    # imgMatch = cv2.resize(imgMatch, (w // 3, h // 3))
    # cv2.imshow(y, imgMatch)
    # we take all the points in our good matches list and we use them
    # to align the corners of our input image
    srcPoints = np.float32([keyPoints2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstPoints = np.float32([keyPoints1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
    imgScan = cv2.warpPerspective(img, M, (w, h))
    # cv2.imshow(y, imgScan)

    imgShow = imgScan.copy()
    imgMask = np.zeros_like(imgShow)

    for x, r in enumerate(roi):
        cv2.rectangle(imgMask, (r[0][0], r[0][1]), (r[1][0], r[1][1]), (0, 255, 0), 2)
        imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1, 0)
        imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
        if r[2] == 'text':
            print(f"{r[3]} : {pytesseract.image_to_string(imgCrop, lang='eng+ron', config='--psm 11  --oem 3')}")
    cv2.imshow(y, imgShow)

cv2.waitKey(0)
