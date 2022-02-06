import json

import cv2
import numpy as np
from pytesseract import pytesseract

per = 25


def fix_date(value: str):
    value = value.split('.')
    value.reverse()
    value[0] = f'20{value[0]}' if len(value[0]) == 2 else value[0]
    return "-".join(value)


roi = [
    [(548, 396), (599, 420), 'serial'],
    [(626, 396), (744, 427), 'serial_no'],
    [(353, 421), (559, 453), 'PNC'],
    [(320, 472), (857, 499), 'name'],
    [(319, 515), (857, 541), 'surname'],
    [(319, 562), (679, 588), 'nationality'],
    [(794, 560), (847, 590), 'sex'],
    [(319, 606), (857, 632), 'birth_place'],
    [(319, 648), (857, 704), 'address'],
    [(319, 719), (623, 755), 'issuer'],
    [(628, 722), (740, 753), 'start_validity', fix_date],
    [(750, 721), (880, 754), 'end_validity', fix_date],
]

pytesseract.tesseract_cmd = '/usr/bin/tesseract'
orb = cv2.ORB_create(1000)
base_path = '/src/'


def process(filepath: str):
    img_template = cv2.imread(base_path + 'ocr/CI_template.jpg')
    h, w, c = img_template.shape
    kp1, des1 = orb.detectAndCompute(img_template, None)

    img = cv2.imread(base_path + '/media/to_process/' + filepath)
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des2, des1)
    matches.sort(key=lambda x: x.distance)
    good = matches[:int(len(matches) * (per / 100))]

    src_points = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_points = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    img_scan = cv2.warpPerspective(img, M, (w, h))
    img_show = img_scan.copy()
    img_mask = np.zeros_like(img_show)

    student_profile = {}

    for index, r in enumerate(roi):
        cv2.rectangle(img_mask, (r[0][0], r[0][1]), (r[1][0], r[1][1]), (0, 255, 0), 2)
        img_show = cv2.addWeighted(img_show, 0.99, img_mask, 0.1, 0)

        img_crop = img_scan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
        gray_image = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
        threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # cv2.imshow(str(index), threshold_image)

        result: str = pytesseract.image_to_string(threshold_image, lang='eng+ron', config='--psm 6 --oem 3')
        result = str(result.replace('\n', '').replace('\f', ''))
        if len(r) > 3 and hasattr(r[3], '__call__'):
            result = r[3](result)
        student_profile.update({r[2]: result})

    print(json.dumps(student_profile, indent=4))
    return student_profile
