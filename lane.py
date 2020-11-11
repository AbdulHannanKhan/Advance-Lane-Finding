import numpy as np
import cv2


def eval_poly(vec, poly):
    """
    Evaluates value of a polynomial at a given point
    :param vec: The given point :float
    :param poly: The polynomial :ndarray[3,]
    :return: value of polynomial at given point :float
    """
    return vec ** 2 * poly[0] + vec * poly[1] + poly[2]


def curvature_of_poly(poly, y):
    """
    Given a polynomial and a point y calculate its curvature
    :param poly: The polynomial :ndarray[3,]
    :param y: The point to calculate curvature at :float
    :return: The curvature of Polynomial at y
    """
    a, b, c = poly
    return ((1 + (2 * a * y + b) ** 2) ** (3 / 2)) / np.abs(2 * a)


class Lane:
    left_anchor = None
    right_anchor = None

    left_poly = None
    left_poly_m = None
    right_poly = None
    right_poly_m = None

    win_count = None
    search_window_margin = None
    min_votes = None

    image_size = None
    mean_dist = 0
    dist_count = 0

    data_min = 0

    curvature = 0
    vehicle_center = 0

    lane_width = 3.7  # average lane width
    vehicle_width = 2.5  # average vehicle width
    xm_per_pix = lane_width / 580  # 1280 - 350(offset)*2 = 580px
    ym_per_pix = 30 / 720  # 30 meters actual lane length in ROI perspective projected on 720px

    M_ = None
    M_inv_ = None

    @classmethod
    def threshold(cls, frame):
        """
        Combine Saturation and Sobel thresholds to extract possible lane indication pixels
        :param frame: The given image to extract lane pixels
        :return: Grayscale image with highlighted possible lane pixels
        """
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS)  # Convert to HLS
        s_ = image[:, :, 2]  # extract S channel from HLS
        _h, _w = s_.shape
        image = cv2.GaussianBlur(s_, (5, 5), 1)  # Blur before thresholding to reduce noise
        highly_saturated = np.uint8(image > 200)
        sobel_strong_edges = np.uint8(cv2.Sobel(image, cv2.CV_64F, 1, 0) > 20)

        # Highlight where highly saturated or strong sobel edge pixels are found
        image = highly_saturated * 50 + sobel_strong_edges * 50
        return image

    def find_hist_based_anchors(self, frame):
        """
        Using histograms find left and right lane polynomial starting points
        :param frame: Input frame
        :return: None
        """
        # define bounds
        frame_height = frame.shape[0]
        win_height = int(frame_height / self.win_count)
        mid_point = int(frame.shape[1] / 2)

        # calculate histogram of last 1/8th row patch
        hist = np.sum(frame[-win_height:, :] > 0, 0)

        # extract max values one from left half of image and one from right half as left and right anchors
        # respectively
        self.left_anchor = np.argmax(hist[:mid_point])
        self.right_anchor = np.argmax(hist[mid_point:]) + mid_point

    def extract_poly(self, frame):

        """
        Use left and right anchors as starting point and apply sliding window approach to find points of interest
        for lane polynomial
        :param frame: Input frame
        :return: None
        """

        debug = np.copy(frame)  # for debug draw sliding window rects

        # Define current left and right x positions
        cur_left = self.left_anchor
        cur_right = self.right_anchor

        # Search parameters setup
        height, width = frame.shape[:2]
        win_height = int(height / self.win_count)
        margin = self.search_window_margin

        # Storage for left and right points of interest for polynomial
        nonzero_indices_left = []
        nonzero_indices_right = []

        # Extract all nonzero x and y locations from frame
        nonzero_y, nonzero_x = np.nonzero(frame)

        # For all sliding windows
        for i in range(self.win_count):

            # Define window start and end
            win_set = height - (i + 1) * win_height
            win_end = height - i * win_height

            # Find left and right polynomial candidates by checking if they lie inside the sliding window
            left_candidates = (
                    (nonzero_y >= win_set) &
                    (nonzero_y < win_end) &
                    (nonzero_x >= max(cur_left - margin, 0)) &
                    (nonzero_x < min(cur_left + margin, width))
            ).nonzero()[0]

            right_candidates = (
                    (nonzero_y >= win_set) &
                    (nonzero_y < win_end) &
                    (nonzero_x >= max(cur_right - margin, 0)) &
                    (nonzero_x < min(cur_right + margin, width))
            ).nonzero()[0]

            # Add found candidates to their respective storages
            nonzero_indices_left += left_candidates.tolist()
            nonzero_indices_right += right_candidates.tolist()

            # If there are more candidates than minimum votes shift the current x positions to mean of current window
            if np.sum(left_candidates) > self.min_votes:
                cur_left = np.mean(nonzero_x[left_candidates])

            if np.sum(right_candidates) > self.min_votes:
                cur_right = np.mean(nonzero_x[right_candidates])

            # Draw rects for debugging
            cv2.rectangle(debug, (int(cur_left - margin), win_set), (int(cur_left + margin), win_end), 255)
            cv2.rectangle(debug, (int(cur_right - margin), win_set), (int(cur_right + margin), win_end), 255)

        # Extract x and y indices of candidates for both left and right polynomial
        left_y = nonzero_y[nonzero_indices_left]
        left_x = nonzero_x[nonzero_indices_left]

        right_y = nonzero_y[nonzero_indices_right]
        right_x = nonzero_x[nonzero_indices_right]

        # if total candidate points of polynomial are greater than a threshold fit polynomial to the points
        # Also find metric polynomials to use for curvature and vehicle position detection
        if np.sum(nonzero_indices_left) > 100:
            self.left_poly = np.polyfit(left_y, left_x, 2)
            # Find a metric polynomial by converting points to read world points
            left_y_metric = left_y * self.ym_per_pix
            left_x_metric = (left_x - self.warp_offset) * self.xm_per_pix  # Consider perspective transform offset
            self.left_poly_m = np.polyfit(left_y_metric, left_x_metric, 2)
        if np.sum(nonzero_indices_right) > 100:
            self.right_poly = np.polyfit(right_y, right_x, 2)
            right_y_metric = right_y * self.ym_per_pix
            right_x_metric = (right_x - self.warp_offset) * self.xm_per_pix
            self.right_poly_m = np.polyfit(right_y_metric, right_x_metric, 2)

        # keep track of overall mean pixel distances between left and right polynomials
        self.mean_dist += self.right_anchor - self.left_anchor
        self.dist_count += 1

        # estimate curvature and vehicle position using found lane polynomials
        self.estimate_curvature_and_position()

    def estimate_curvature_and_position(self):
        """
        Estimates curvature of lane and position of vehicle
        :return: None
        """
        height = self.image_size[0]
        eval_point = (height - 1) * self.ym_per_pix  # point closest to vehicle to estimate curvature at

        # Find curvature of both polynomials and take mean
        left_curvature = curvature_of_poly(self.left_poly_m, eval_point)
        right_curvature = curvature_of_poly(self.right_poly_m, eval_point)
        self.curvature = (left_curvature + right_curvature) / 2

        # Find vehicle position
        absolute_vehicle_center = (eval_poly(eval_point, self.right_poly_m) +
                                   eval_poly(eval_point, self.left_poly_m)) / 2

        # Estimate vehicle position relative to lane center
        self.vehicle_center = self.lane_width / 2 - absolute_vehicle_center

    def create_image_mask(self):
        """
        Create image mask based on lane polynomials to highlight frame
        :return: Mask image
        """
        h, w = self.image_size
        im = np.zeros((h, w, 3), dtype=np.uint8)

        # Sample y points starting from top confidence location to bottom of image
        plot_y = np.linspace(self.data_min, h - 1, h - self.data_min)

        # Calculate values of polynomials at y sample points
        left_plt_x = self.left_poly[0] * plot_y ** 2 + self.left_poly[1] * plot_y + self.left_poly[2]
        right_plt_x = self.right_poly[0] * plot_y ** 2 + self.right_poly[1] * plot_y + self.right_poly[2]

        # Update mean dist using intercepts of polynomials
        self.mean_dist += right_plt_x[-1] - left_plt_x[-1]
        self.dist_count += 1

        # For each sampled y
        for i in range(h - self.data_min):

            # Find start and end lane pixel
            start = int(max(0, left_plt_x[i]))
            end = int(min(w, right_plt_x[i]))

            # Color lane pixels for current row to be green
            im[i + self.data_min, start:end, 1] = 255
            # Add Red spectrum based on how much away vehicle is from lane center
            im[i + self.data_min, start:end, 2] = \
                abs(self.vehicle_center) / ((self.lane_width - self.vehicle_width) / 2) * 255

        return im

    def mask_frame(self, frame):
        """
        Mask/Highlight given frame with currently estimated lane area
        :param frame: Current frame
        :return: Masked frame
        """

        # Get mask, un wrap the perspective and overlay on frame
        mask = self.create_image_mask()
        lane_mask = self.perspective_unwrap(mask)
        frame = cv2.addWeighted(frame, 1, lane_mask, 0.5, 0)

        # Check where vehicle is relative to lane center
        direction = "left" if self.vehicle_center < 0 else "right"

        # Show current curvature and vehicle position on image
        cv2.putText(
            frame,
            f"Curvature: {int(self.curvature)} m",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (255, 255, 255), 2,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            f"{direction}: {int(abs(self.vehicle_center) * 100) / 100} m",
            (10, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (255, 255, 255), 2,
            cv2.LINE_AA
        )
        return frame

    def process(self, frame):
        """
        Update polynomials using previously estimated lane polynomials and given frame
        :param frame: Current undistorted video frame
        :return:
        """
        # Perspective wrap and threshold the frame
        frame = self.preprocess_frame(frame)

        # Set search window margin
        margin = self.search_window_margin

        # Extract nonzero x and y locations from current frame
        nonzero_y, nonzero_x = np.nonzero(frame)

        # Given polynomials and search window margin check which nonzero pixels are polynomial candidates
        nonzero_left = eval_poly(nonzero_y, self.left_poly)
        nonzero_right = eval_poly(nonzero_y, self.right_poly)

        left_candidates = (
                (nonzero_x >= nonzero_left - margin) &
                (nonzero_x < nonzero_left + margin)
        ).nonzero()[0]

        right_candidates = (
                (nonzero_x >= nonzero_right - margin) &
                (nonzero_x < nonzero_right + margin)
        ).nonzero()[0]

        # Extract x and y indices of polynomial candidates for both left and right
        left_y = nonzero_y[left_candidates]
        left_x = nonzero_x[left_candidates]

        right_y = nonzero_y[right_candidates]
        right_x = nonzero_x[right_candidates]

        # Find confidence point i.e. the y point from top where we have data from both left and right polynomial
        # we don't want to highlight area of lane where we are not confident
        if np.sum(left_y) > 0 and np.sum(right_y) > 0:
            self.data_min = max(left_y.min(), right_y.min())

        # If polynomial candidates are greater than a threshold update polynomials both pixel and metric
        if np.sum(left_candidates) > 50:
            self.left_poly *= 0.7
            self.left_poly_m *= 0.7
            self.left_poly += 0.3 * np.polyfit(left_y, left_x, 2)
            self.left_poly_m += 0.3 * \
                                np.polyfit(left_y * self.ym_per_pix, (left_x - self.warp_offset) * self.xm_per_pix, 2)
        if np.sum(right_candidates > 50):
            self.right_poly *= 0.7
            self.right_poly_m *= 0.7
            self.right_poly += 0.3 * np.polyfit(right_y, right_x, 2)
            self.right_poly_m += 0.3 * \
                                 np.polyfit(
                                     right_y * self.ym_per_pix, (right_x - self.warp_offset) * self.xm_per_pix, 2
                                 )

        # Check if the found polynomials intercepts are correct if not reinitialize using sliding window method
        if not self.are_intercepts_correct():
            self.init(frame)

        # Estimate lane curvature and vehicle position
        self.estimate_curvature_and_position()

    def are_intercepts_correct(self):
        """
        Check if polynomial are correct by checking if their intercepts are at least 200 pixels apart
        :return: None
        """
        return self.right_poly[2] - self.left_poly[2] > 200

    def __init__(self, frame, roi, warp_offset, win_count=8, search_window_margin=30, min_votes=50):

        # Initialize internal parameters
        self.win_count = win_count
        self.image_size = frame.shape[:2]
        self.search_window_margin = search_window_margin
        self.min_votes = min_votes
        self.roi_source = roi
        self.warp_offset = warp_offset

        # Estimate perspective transform matrices
        self.estimate_perspective_transform()

        # Initialize polynomials with sliding window method
        preprocessed = self.preprocess_frame(frame)
        self.init(preprocessed)

    def preprocess_frame(self, frame):
        """
        Perspective wrap and threshold frames to make the ready for processing
        :param frame: Image
        :return: None
        """
        wrap = self.perspective_wrap(frame)
        return self.threshold(wrap)

    def init(self, frame):
        """
        Initialize using sliding window method
        :param frame: Image
        :return: None
        """
        self.find_hist_based_anchors(frame)
        self.extract_poly(frame)

    def estimate_perspective_transform(self):
        """
        Calculate perspective transform matrices
        :return: None
        """
        h, w = self.image_size
        offset = self.warp_offset

        # Create destination polygon based on offset and image dimensions
        roi_dest = np.float32([[w - offset, 0], [w - offset, h], [offset, h], [offset, 0]])

        self.M_ = cv2.getPerspectiveTransform(np.float32(self.roi_source), roi_dest)
        self.M_inv_ = cv2.getPerspectiveTransform(roi_dest, np.float32(self.roi_source))

    def perspective_wrap(self, frame):
        """
        Perspective Transform to obtain bird eye view
        :param frame: Image
        :return: None
        """
        h, w = self.image_size
        return cv2.warpPerspective(frame, self.M_, (w, h))

    def perspective_unwrap(self, frame):
        """
        Perspective Transform inverse to obtain original frame from bird eye view
        :param frame: Image
        :return: None
        """
        h, w = self.image_size
        return cv2.warpPerspective(frame, self.M_inv_, (w, h))
