import os

import cv2
import numpy as np
import pytesseract

per = 25

roi = [
    [(548, 396), (599, 420), "text", "Serie"],
    [(626, 396), (744, 427), "text", "Număr"],
    [(353, 421), (559, 453), "text", "CNP"],
    [(320, 472), (857, 499), "text", "Nume"],
    [(319, 515), (857, 541), "text", "Prenume"],
    [(319, 562), (679, 588), "text", "Naţionalitate"],
    [(794, 560), (847, 590), "text", "Sex"],
    [(319, 606), (857, 632), "text", "Loc naştere"],
    [(319, 648), (857, 704), "text", "Domiciliu"],
    [(319, 719), (623, 755), "text", "Emisă de"],
    [(628, 722), (740, 753), "text", "Valabil de la"],
    [(750, 721), (880, 754), "text", "Valabil până la"],
]

pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"

imgT = cv2.imread("template.jpg")
h, w, c = imgT.shape
# imgT = cv2.resize(imgT, (w // 3, h // 3))

# we use orb because it's free to use
orb = cv2.ORB_create(1000)

# keypoints are the unique elements to our image
# descriptors are the representation of these keypoints that would be easier
# for the computer to understand and differentiate between

kp1, des1 = orb.detectAndCompute(imgT, None)
# impKp1 = cv2.drawKeypoints(imgT, kp1, None)
# none as the above parameter is if we don't want to store the image result
path = "buletine"
myPicList = os.listdir(path)

for j, y in enumerate(myPicList):
    img = cv2.imread(path + "/" + y)
    # img = cv2.resize(img, (w // 3, h // 3))
    # cv2.imshow(y, img)
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des2, des1)
    matches.sort(key=lambda x: x.distance)
    good = matches[: int(len(matches) * (per / 100))]
    imgMatch = cv2.drawMatches(img, kp2, imgT, kp1, good[:100], None, flags=2)
    # cv2.imshow(y, imgMatch)

    srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
    imgScan = cv2.warpPerspective(img, M, (w, h))
    # imgScan = cv2.resize(imgScan, (w // 3, h // 3))
    # cv2.imshow(y, imgScan)
    imgShow = imgScan.copy()
    imgMask = np.zeros_like(imgShow)

    myData = []
    print(f"########Extracting Data from CI{j}###########")

    for x, r in enumerate(roi):
        cv2.rectangle(imgMask, (r[0][0], r[0][1]), (r[1][0], r[1][1]), (0, 255, 0), 2)
        imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1, 0)

        imgCrop = imgScan[r[0][1] : r[1][1], r[0][0] : r[1][0]]
        hsv = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 0])
        upper = np.array([100, 175, 110])
        mask = cv2.inRange(hsv, lower, upper)

        # Invert image and OCR
        invert = 255 - mask
        data = pytesseract.image_to_string(
            invert, lang="eng+ron", config="--psm 6 --oem 3"
        )

        if r[2] == "text":
            print(f"{r[3]} : {data}")
            myData.append(data)
            cv2.imshow(str(x), mask)
            # cv2.imshow(str(x), invert)

    # imgShow = cv2.resize(imgShow, (w // 1, h // 1))
    # cv2.imshow(y + "2", imgShow)

# cv2.imshow("KeyPointsQuery", impKp1)
# cv2.imshow("Output", imgT)
cv2.waitKey(0)
