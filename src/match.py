import numpy as np
from cv2 import cv2
from numpy.typing import NDArray

orb = cv2.ORB_create(1000)

percentage = 25


def match(img: NDArray, template: NDArray):
    template_key_points, template_descriptors = orb.detectAndCompute(
        cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), None
    )
    # img_key_points = cv2.drawKeypoints(template, template_key_points, None)
    img_key_points, img_descriptors = orb.detectAndCompute(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None
    )

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # match descriptors
    matches = list(bf.match(img_descriptors, template_descriptors))

    # sort them in order of their distance
    matches.sort(key=lambda x: x.distance)

    # draw first "percentage" matches. percentage is set to 25
    good = matches[: int(len(matches) * (percentage / 100))]
    img_match = cv2.drawMatches(
        img, img_key_points, template, template_key_points, good[:100], None, flags=2
    )

    # we take all the points in our good matches list and we use them
    # to align the corners of our input image
    src_points = np.float32([img_key_points[m.queryIdx].pt for m in good]).reshape(
        -1, 1, 2
    )
    dst_points = np.float32([template_key_points[m.trainIdx].pt for m in good]).reshape(
        -1, 1, 2
    )

    M, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

    return M, img_match
