#!/usr/bin/env python
# Sources partially taken from
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html

import cv2
import numpy as np
import glob
import sys
import json

# path to directory containing calibration images
calibration_directory = sys.argv[1]

# Dimensions of checkerboard
CHECKERBOARD = (6, 9)
# Criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 3D and 2D points storage
obj_points = []
img_points = []


# Setup checkerboard 3D points on z=0
obj_p = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
obj_p[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Grab all jpegs from given folder
images = glob.glob(calibration_directory + '/*.jpg')
for img in images:
    img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    
    # If desired number of corners detected
    if ret:
        # 3D Obj points same for all images
        obj_points.append(obj_p)
        # Refine pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # Extracted 2D refined points
        img_points.append(corners2)

cv2.destroyAllWindows()

gray = cv2.imread(images[0], cv2.IMREAD_GRAYSCALE)

# Performing camera calibration
_, mtx, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

# Write calibration output to json
mat = mtx.tolist()
dist = dist.tolist()
with open("calibration.json", "w") as f:
    json.dump({"K": mat, "dist": dist}, f)
