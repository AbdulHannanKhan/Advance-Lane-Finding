import numpy as np
import cv2


def eval_poly(vec, poly):
    return vec ** 2 * poly[0] + vec * poly[1] + poly[2]


def curvature_of_poly(poly, y):
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

    xm_per_pix = 3.7 / 580
    ym_per_pix = 30 / 720

    M_ = None
    M_inv_ = None

    @classmethod
    def threshold(cls, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS)
        s = image[:, :, 2]
        _h, _w = s.shape
        image = cv2.GaussianBlur(s, (5, 5), 1)
        t = np.uint8(image > 200)
        image = t * 50 + np.uint8(cv2.Sobel(image, cv2.CV_64F, 1, 0) > 20) * 50
        return image

    def find_hist_based_anchors(self, frame):
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

        debug = np.copy(frame)
        cur_left = self.left_anchor
        cur_right = self.right_anchor

        height, width = frame.shape[:2]
        win_height = int(height / self.win_count)
        margin = self.search_window_margin

        nonzero_indices_left = []
        nonzero_indices_right = []

        nonzero_y, nonzero_x = np.nonzero(frame)

        for i in range(self.win_count):
            win_set = height - (i + 1) * win_height
            win_end = height - i * win_height

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

            nonzero_indices_left += left_candidates.tolist()
            nonzero_indices_right += right_candidates.tolist()

            if np.sum(left_candidates) > self.min_votes:
                cur_left = np.mean(nonzero_x[left_candidates])

            if np.sum(right_candidates) > self.min_votes:
                cur_right = np.mean(nonzero_x[right_candidates])

            cv2.rectangle(debug, (int(cur_left - margin), win_set), (int(cur_left + margin), win_end), 255)
            cv2.rectangle(debug, (int(cur_right - margin), win_set), (int(cur_right + margin), win_end), 255)

        left_y = nonzero_y[nonzero_indices_left]
        left_x = nonzero_x[nonzero_indices_left]

        right_y = nonzero_y[nonzero_indices_right]
        right_x = nonzero_x[nonzero_indices_right]

        if np.sum(nonzero_indices_left) > 100:
            self.left_poly = np.polyfit(left_y, left_x, 2)
            self.left_poly_m = np.polyfit(left_y * self.ym_per_pix, (left_x - 350) * self.xm_per_pix, 2)
        if np.sum(nonzero_indices_right) > 100:
            self.right_poly = np.polyfit(right_y, right_x, 2)
            self.right_poly_m = np.polyfit(right_y * self.ym_per_pix, (right_x - 350) * self.xm_per_pix, 2)
        self.mean_dist += self.right_anchor - self.left_anchor
        self.dist_count += 1
        self.estimate_curvature_and_position()

    def estimate_curvature_and_position(self):

        height = self.image_size[0]
        eval_point = (height - 1) * self.ym_per_pix
        left_curvature = curvature_of_poly(self.left_poly_m, eval_point)
        right_curvature = curvature_of_poly(self.right_poly_m, eval_point)
        self.curvature = (left_curvature + right_curvature) / 2
        self.vehicle_center = 3.7 / 2 - \
                              (
                                      eval_poly(eval_point, self.right_poly_m)
                                      + eval_poly(eval_point, self.left_poly_m)
                              ) / 2

    def create_image_mask(self):

        h, w = self.image_size
        im = np.zeros((h, w, 3), dtype=np.uint8)

        plot_y = np.linspace(self.data_min, h - 1, h - self.data_min)
        left_plt_x = self.left_poly[0] * plot_y ** 2 + self.left_poly[1] * plot_y + self.left_poly[2]
        right_plt_x = self.right_poly[0] * plot_y ** 2 + self.right_poly[1] * plot_y + self.right_poly[2]

        self.mean_dist += right_plt_x[-1] - left_plt_x[-1]
        self.dist_count += 1

        for i in range(h - self.data_min):
            start = int(max(0, left_plt_x[i]))
            end = int(min(w, right_plt_x[i]))
            im[i + self.data_min, start:end, 1] = 255
            im[i + self.data_min, start:end, 2] = abs(self.vehicle_center) / ((3.7 - 2.5) / 2) * 255

        return im

    def mask_frame(self, frame):

        mask = self.create_image_mask()
        lane_mask = self.perspective_unwrap(mask)
        frame = cv2.addWeighted(frame, 1, lane_mask, 0.5, 0)
        direction = "left" if self.vehicle_center < 0 else "right"
        cv2.putText(
            frame,
            f"curve: {int(self.curvature)} m",
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

        frame = self.preprocess_frame(frame)
        margin = self.search_window_margin

        nonzero_y, nonzero_x = np.nonzero(frame)
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

        left_y = nonzero_y[left_candidates]
        left_x = nonzero_x[left_candidates]

        right_y = nonzero_y[right_candidates]
        right_x = nonzero_x[right_candidates]

        if np.sum(left_y) > 0 and np.sum(right_y) > 0:
            self.data_min = max(left_y.min(), right_y.min())

        if np.sum(left_candidates) > 50:
            self.left_poly *= 0.7
            self.left_poly_m *= 0.7
            self.left_poly += 0.3 * np.polyfit(left_y, left_x, 2)
            self.left_poly_m += 0.3 * np.polyfit(left_y * self.ym_per_pix, (left_x - 350) * self.xm_per_pix, 2)
        if np.sum(right_candidates > 50):
            self.right_poly *= 0.7
            self.right_poly_m *= 0.7
            self.right_poly += 0.3 * np.polyfit(right_y, right_x, 2)
            self.right_poly_m += 0.3 * np.polyfit(right_y * self.ym_per_pix, (right_x - 350) * self.xm_per_pix, 2)
        if not self.are_intercepts_correct():
            self.init(frame)

        self.estimate_curvature_and_position()

    def are_intercepts_correct(self):
        return self.right_poly[2] - self.left_poly[2] > 200

    def __init__(self, frame, roi, warp_offset, win_count=8, search_window_margin=30, min_votes=50):
        self.win_count = win_count
        self.image_size = frame.shape[:2]
        self.search_window_margin = search_window_margin
        self.min_votes = min_votes
        self.roi_source = roi
        self.warp_offset = warp_offset
        self.estimate_perspective_transform()

        preprocessed = self.preprocess_frame(frame)
        self.init(preprocessed)

    def preprocess_frame(self, frame):
        wrap = self.perspective_wrap(frame)
        return self.threshold(wrap)

    def init(self, frame):
        self.find_hist_based_anchors(frame)
        self.extract_poly(frame)

    def estimate_perspective_transform(self):
        h, w = self.image_size
        offset = self.warp_offset
        roi_dest = np.float32([[w - offset, 0], [w - offset, h], [offset, h], [offset, 0]])

        self.M_ = cv2.getPerspectiveTransform(np.float32(self.roi_source), roi_dest)
        self.M_inv_ = cv2.getPerspectiveTransform(roi_dest, np.float32(self.roi_source))

    def perspective_wrap(self, frame):

        h, w = self.image_size
        return cv2.warpPerspective(frame, self.M_, (w, h))

    def perspective_unwrap(self, frame):

        h, w = self.image_size
        return cv2.warpPerspective(frame, self.M_inv_, (w, h))
