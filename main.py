import cv2
import sys
import numpy as np
import json
from lane import Lane

lim = cv2.imread("resources/lane.jpg")
car = cv2.imread("resources/car.png", cv2.IMREAD_UNCHANGED)


def draw_car_and_lane(im, car_location_meters=0):
    """
    Create a small lane indicator on the top right corner of image for better visualization
    :param im: The image frame to draw
    :param car_location_meters: Current location of car in lane in meters should be zero if perfectly centered
    otherwise positive for right and negative for left i.e. [-3.7/2, +3.7/2] where 3.7 is width of lane
    :return: Image with car location drawn
    """

    h_, w_ = im.shape[:2]
    lim_c = np.copy(lim)

    # Set offsets and parameters
    car_width_pixels = 75
    lane_offset = 50
    lane_width_pixels = 100
    lane_width_meters = 3.7

    # Calculate car pixel positions
    car_x = car_location_meters/lane_width_meters + 0.5
    car_x *= lane_width_pixels
    car_x = int(car_x - car_width_pixels/2) + lane_offset

    # Draw car on lane image
    alpha = car[:, :, 3]/255
    # Taking image transparency into account
    alpha = alpha[:, :, None] * np.ones(3, dtype=float)[None, None, :]
    roi = lim_c[50:165, car_x:car_x+car_width_pixels]
    roi = car[:, :, :3] * alpha + roi * (1 - alpha)
    lim_c[50:165, car_x:car_x + car_width_pixels] = roi

    # Show on top right corner
    im[20:220, -220: -20] = lim_c

    return im


test_video = cv2.VideoCapture(sys.argv[1])
# Set True to export video
export = False

ret, frame = test_video.read()
h, w = frame.shape[:2]

# Read intrinsics and distortion coefficient to rectify images
intrinsic = None
dist = None

with open("calibration.json", "r") as f:
    cal_obj = json.load(f)
    intrinsic = np.array(cal_obj["K"])
    dist = np.array(cal_obj["dist"])

# Pre calculated poly of interest for perspective transform
poly = np.array([
    [681, 446],
    [1118, 720],
    [192, 720],
    [600, 446]
])
poly = np.float32(poly)

# transform width offset
offset = 350

# Initialize Lane class with undistorted image, poly of interest and perspective transform offset
undistorted = cv2.undistort(frame, intrinsic, dist)
lane = Lane(undistorted, poly, offset)
delay = 0

if export:
    # Fps of output video should be same as input
    fps = test_video.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter('output_images/output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (w, h))

while ret:

    # Undistort image
    undistorted = cv2.undistort(frame, intrinsic, dist)

    # Feed it to the lane
    lane.process(undistorted)

    # Mask the undistorted frame with lane highlight
    mask = lane.mask_frame(undistorted)

    # Draw vehicle position indicator
    mask = draw_car_and_lane(mask, lane.vehicle_center)

    # Visualize and write to video as video writing is enabled
    cv2.imshow("vid", mask)
    if export:
        out.write(mask)
    key = cv2.waitKey(delay)

    # Check for interrupts
    if key == 27:
        break
    elif key == 32:
        delay = 1 if delay == 0 else 0

    # Read next frame
    ret, frame = test_video.read()

# Release video handles and destroy created windows
test_video.release()
if export:
    out.release()
cv2.destroyAllWindows()
