import numpy as np
from cv2 import cv2
from numpy.typing import NDArray

orb = cv2.ORB_create(5000)

percentage = 25


def match(img: NDArray, template: NDArray):
    key_points1, descriptors1 = orb.detectAndCompute(template, None)
    # img_key_points = cv2.drawKeypoints(template, key_points1, None)
    key_points2, descriptors2 = orb.detectAndCompute(img, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # match descriptors
    matches = list(bf.match(descriptors2, descriptors1))

    # sort them in order of their distance
    matches.sort(key=lambda x: x.distance)

    # draw first "percentage" matches. percentage is set to 25
    good = matches[:int(len(matches) * (percentage / 100))]
    img_match = cv2.drawMatches(img, key_points2, template, key_points1, good[:100], None, flags=2)

    # we take all the points in our good matches list and we use them
    # to align the corners of our input image
    src_points = np.float32([key_points2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_points = np.float32([key_points1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

    return M, img_match
