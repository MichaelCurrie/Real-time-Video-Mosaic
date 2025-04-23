"""
Note: to play the resulting mjpeg, please use `vlc --demux ffmpeg ttk.mjpeg`
"""

import cv2
from pathlib import Path
import numpy as np
import subprocess
import cv2.aruco
import os

from utils import (
    is_homography_distorted,
    # detect_new_aruco,
    mp4_to_mjpeg,
    detect_aruco_markers,
)

CONVERT_VIDEO = False
MOSAIC_PATH = "Mosaics"
# SRC_PATH = Path("Data") / "my-own.mp4"
SRC_PATH = Path("Data") / "ttk.mp4"
VIDEO_WIDTH = 1280
NUM_FRAMES = 1000
SKIP_FRAMES = 3
START_FRAME = 200

MATCH_ARUCO_IF_POSSIBLE = True


class VideoMosaic:
    def __init__(
        self,
        first_image,
        output_height_times=2,
        output_width_times=4,
        detector_type="sift",
    ):
        """
        Initializes the mosaic creation. The first frame is used to set up
        the panorama output image size and the feature detector.

        This class processes every frame and generates the panorama

        Args:
            first_image (image for the first frame): first image to initialize the output size
            output_height_times (int, optional): determines the output height based on input image height. Defaults to 2.
            output_width_times (int, optional): determines the output width based on input image width. Defaults to 4.
            detector_type (str, optional): the detector for feature detection. It can be "sift" or "orb". Defaults to "sift".

        """
        self.common_tag_ids = {}
        self.prev_aruco_markers = {}

        self.detector_type = detector_type
        if detector_type == "sift":
            self.detector = cv2.SIFT_create(700)
            self.bf = cv2.BFMatcher()
        elif detector_type == "orb":
            self.detector = cv2.ORB_create(700)
            self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Enable visualization for debugging.
        self.visualize = True

        self.process_first_frame(first_image)

        self.output_img = np.zeros(
            shape=(
                int(output_height_times * first_image.shape[0]),
                int(output_width_times * first_image.shape[1]),
                first_image.shape[2],
            ),
            dtype=first_image.dtype,
        )

        # offset: center the initial frame.
        self.w_offset = int(self.output_img.shape[0] / 2 - first_image.shape[0] / 2)
        self.h_offset = int(self.output_img.shape[1] / 2 - first_image.shape[1] / 2)

        self.output_img[
            self.w_offset : self.w_offset + first_image.shape[0],
            self.h_offset : self.h_offset + first_image.shape[1],
            :,
        ] = first_image

        self.H_old = np.eye(3)
        self.H_old[0, 2] = self.h_offset
        self.H_old[1, 2] = self.w_offset

        # Make the match window resizable if visualization is on.
        if self.visualize:
            cv2.namedWindow("matches", cv2.WINDOW_NORMAL)
            # Optionally, resize the matches window as desired.
            cv2.resizeWindow("matches", 640, 480)

    def process_first_frame(self, first_image):
        self.frame_prev = first_image
        frame_gray_prev = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
        self.kp_prev, self.des_prev = self.detector.detectAndCompute(
            frame_gray_prev, None
        )
        self.prev_aruco_markers = detect_aruco_markers(first_image)

    def match(self, des_cur, des_prev):
        """Matches the descriptors between the current and previous frames.

        Args:
            des_cur (np array): current frame descriptor
            des_prev (np arrau): previous frame descriptor

        Returns:
            array: and array of matches between descriptors
        """

        if self.detector_type == "sift":
            pair_matches = self.bf.knnMatch(des_cur, des_prev, k=2)
            matches = [m for m, n in pair_matches if m.distance < 0.7 * n.distance]
        elif self.detector_type == "orb":
            matches = self.bf.match(des_cur, des_prev)

        matches = sorted(matches, key=lambda x: x.distance)
        matches = matches[: min(len(matches), 20)]
        if self.visualize:
            match_img = cv2.drawMatches(
                self.frame_cur,
                self.kp_cur,
                self.frame_prev,
                self.kp_prev,
                matches,
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            cv2.imshow("matches", match_img)
        return matches

    def process_frame(self, frame_cur):
        self.frame_cur = frame_cur
        frame_gray_cur = cv2.cvtColor(frame_cur, cv2.COLOR_BGR2GRAY)
        # TODO: compute this only if no arcuo was detected (but we
        # need it for the previous frame, so have to go back and calculate it..)
        self.kp_cur, self.des_cur = self.detector.detectAndCompute(frame_gray_cur, None)

        # Detect markers in the current frame
        cur_aruco_markers = detect_aruco_markers(self.frame_cur)
        # Check if previous frame had markers and if there is a common marker
        if self.prev_aruco_markers:
            self.common_tag_ids = set(self.prev_aruco_markers.keys()).intersection(
                cur_aruco_markers.keys()
            )

        if self.common_tag_ids:
            # Take the first common marker to estimate the transformation
            marker_id = list(self.common_tag_ids)[0]
            # Use the corners from the previous frame (source) and current frame (destination)
            src_pts = self.prev_aruco_markers[marker_id]  # from previous frame
            dst_pts = cur_aruco_markers[marker_id]  # from current frame
            # TODO: I think these points need to be in the same order (maybe clockwise from upper left?) for findHomography to work
            pass

            H_aruco, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC)
        else:
            H_aruco = None

        # Fallback: use feature matching-based homography.
        self.matches = self.match(self.des_cur, self.des_prev)
        if len(self.matches) < 4:
            print(f"Feature matches {len(self.matches)} < 4 so cannot find homography")
            H_feature = None
        else:
            H_feature = self.findHomography(self.kp_cur, self.kp_prev, self.matches)

        if H_aruco is not None and MATCH_ARUCO_IF_POSSIBLE:
            H_conviction = H_aruco
        elif H_feature is not None:
            H_conviction = H_feature
        else:
            print("No homography could be used for this frame")
            return

        # Compose the new homography with the previous accumulated transform.
        self.H = np.matmul(self.H_old, H_conviction)

        if False and is_homography_distorted(self.frame_cur, self.H):
            print("Distortion too high; restarting mosaic with current frame.")
            # Optionally save the current mosaic segment before resetting
            # e.g., cv2.imwrite(f"mosaic_segment_reset.jpg", self.output_img)
            # Reinitialize with the current frame:
            self.__init__(self.frame_cur, detector_type=self.detector_type)
            return

        # Warp the current frame into the mosaic using the determined homography.
        self.warp(self.frame_cur, self.H)

        # Update the previous markers for the next iteration.
        self.prev_aruco_markers = cur_aruco_markers
        self.H_old = self.H
        self.kp_prev = self.kp_cur
        self.des_prev = self.des_cur
        self.frame_prev = self.frame_cur

    @staticmethod
    def findHomography(image_1_kp, image_2_kp, matches):
        """Calculates the homography between two frames given matching keypoints.

        Args:
            image_1_kp (np array): keypoints of image 1
            image_2_kp (np_array): keypoints of image 2
            matches (np array): matches between keypoints in image 1 and image 2

        Returns:
            np arrat of shape [3,3]: Homography matrix

        Note: taken from https://github.com/cmcguinness/focusstack/blob/master/FocusStack.py

        """
        image_1_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
        image_2_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
        for i, match in enumerate(matches):
            image_1_points[i] = image_1_kp[match.queryIdx].pt
            image_2_points[i] = image_2_kp[match.trainIdx].pt
        homography, mask = cv2.findHomography(
            image_1_points, image_2_points, cv2.RANSAC, ransacReprojThreshold=2.0
        )
        return homography

    def warp(self, frame_cur, H):
        """Warps the current frame based on the homography H and updates the mosaic.

        Args:
            frame_cur (np array): current frame
            H (np array of shape [3,3]): homography matrix

        Returns:
            np array: image output of mosaicing
        """
        warped_img = cv2.warpPerspective(
            frame_cur,
            H,
            (self.output_img.shape[1], self.output_img.shape[0]),
            flags=cv2.INTER_LINEAR,
        )

        transformed_corners = self.get_transformed_corners(frame_cur, H)
        warped_img = self.draw_border(warped_img, transformed_corners)

        self.output_img[warped_img > 0] = warped_img[warped_img > 0]
        output_temp = np.copy(self.output_img)
        output_temp = self.draw_border(
            output_temp, transformed_corners, color=(0, 0, 255)
        )

        # Show the output mosaic in a resizable window.
        cv2.imshow("output", output_temp / 255.0)
        return self.output_img

    @staticmethod
    def get_transformed_corners(frame_cur, H):
        """Finds the corners of the input frame after applying the homography.
        Args:
            frame_cur (np array): current frame
            H (np array of shape [3,3]): Homography matrix

        Returns:
            [np array]: a list of 4 corner points after warping
        """
        corners = np.array(
            [
                [
                    [0, 0],
                    [frame_cur.shape[1], 0],
                    [frame_cur.shape[1], frame_cur.shape[0]],
                    [0, frame_cur.shape[0]],
                ]
            ],
            dtype=np.float32,
        )
        transformed_corners = cv2.perspectiveTransform(corners, H)
        return np.array(transformed_corners, dtype=np.int32)

    def draw_border(self, image, corners, color=(0, 0, 0)):
        """Draws a rectangular border on the image using the given corner points.

        Args:
            image ([type]): current mosaiced output
            corners (np array): list of corner points
            color (tuple, optional): color of the border lines. Defaults to (0, 0, 0).

        Returns:
            np array: the output image with border
        """
        for i in range(corners.shape[1] - 1, -1, -1):
            cv2.line(
                image,
                tuple(corners[0, i, :]),
                tuple(corners[0, i - 1, :]),
                thickness=5,
                color=color,
            )
        return image


def create_mosaics(video_path, mosaic_path, display_size=(640, 480)):
    assert Path(mosaic_path).is_dir()

    print("Creating mosaic... press 'q' to quit.")
    cap = cv2.VideoCapture(video_path)

    # Create a resizable window for mosaic output.
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("output", display_size[0], display_size[1])

    # Variables to track new marker IDs and segment count
    detected_ids = set()
    segment_count = 0

    is_success, first_image = cap.read()
    if not is_success:
        print("BAD FRAME")
        return

    video_mosaic = VideoMosaic(first_image=first_image, detector_type="sift")

    while cap.isOpened():
        is_success, frame_cur = cap.read()
        if not is_success:
            print("BAD FRAME")
            break

        is_done = video_mosaic.process_frame(frame_cur)

        # After processing the frame, check for new ArUco markers.
        # We use video_mosaic.H_old (which holds the current accumulated transformation)
        # new_marker_found, mosaic_position = detect_new_aruco(
        #    frame_cur, video_mosaic.H_old, detected_ids
        # )
        if len(video_mosaic.common_tag_ids) > 0:
            # import code
            # code.interact(banner="tag1 figure out mosaic position", local=locals())
            # Mark the location in the mosaic image (green circle)
            # cv2.circle(video_mosaic.output_img, mosaic_position, 10, (0, 255, 0), -1)
            # Save the current mosaic segment
            segment_filename = Path(mosaic_path) / f"mosaic_segment_{segment_count}.jpg"
            cv2.imwrite(segment_filename, video_mosaic.output_img)
            print(f"Segment saved: {segment_filename}")
            segment_count += 1

            # Restart the mosaic using the current frame so the new segment begins here.
            video_mosaic = VideoMosaic(frame_cur, detector_type="sift")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Final mosaic saved to {mosaic_path}")


if __name__ == "__main__":
    dst_path = Path(SRC_PATH).parent / (Path(SRC_PATH).stem + ".mjpeg")

    print("Converting video")
    if CONVERT_VIDEO:
        mp4_to_mjpeg(
            src_path=str(SRC_PATH),
            dst_path=str(dst_path),
            video_width=VIDEO_WIDTH,
            num_frames=NUM_FRAMES,
            skip_frames=SKIP_FRAMES,
            start_frame=START_FRAME,
        )

    print("Creating mosaics")
    os.makedirs(MOSAIC_PATH, exist_ok=True)
    create_mosaics(
        video_path=str(dst_path), mosaic_path=MOSAIC_PATH, display_size=(640, 480)
    )
