import cv2
import sys
import numpy as np
from lane import Lane

test_video = cv2.VideoCapture(sys.argv[1])

ret, frame = test_video.read()
h, w = frame.shape[:2]

# Define intrinsics and distortion coefficient to rectify images
intrinsic = np.float32(
    [
        [1.15694034e+03, 0.00000000e+00, 6.65948601e+02],
        [0.00000000e+00, 1.15213869e+03, 3.88785179e+02],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ])
dist = np.float32([-2.37636627e-01, -8.54128604e-02, -7.90956194e-04, -1.15908550e-04, 1.05741158e-01])

poly = np.array([
    [681, 446],
    [1118, 720],
    [192, 720],
    [600, 446]
])
poly = np.float32(poly)
offset = 350

undistorted = cv2.undistort(frame, intrinsic, dist)
lane = Lane(undistorted, poly, offset)
delay = 0

fps = test_video.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter('output_images/outpy.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (w, h))
while ret:

    undistorted = cv2.undistort(frame, intrinsic, dist)
    lane.process(undistorted)
    mask = lane.mask_frame(undistorted)

    cv2.imshow("vid", mask)
    out.write(mask)
    key = cv2.waitKey(delay)
    if key == 27:
        break
    elif key == 32:
        delay = 1 if delay == 0 else 0

    ret, frame = test_video.read()

test_video.release()
out.release()
cv2.destroyAllWindows()
