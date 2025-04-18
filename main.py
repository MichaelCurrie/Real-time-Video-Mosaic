"""
Note: to play the resulting mjpeg, please use `vlc --demux ffmpeg ttk.mjpeg`
"""

import cv2
from pathlib import Path
import numpy as np
import subprocess
import cv2.aruco


CONVERT_VIDEO = False
MOSAIC_PATH = ""
# SRC_PATH = Path("Data") / "my-own.mp4"
SRC_PATH = Path("Data") / "ttk.mp4"
VIDEO_WIDTH = 1280
NUM_FRAMES = 1000
SKIP_FRAMES = 3
START_FRAME = 200

MATCH_ARUCO_IF_POSSIBLE = False


def detect_new_aruco(frame, current_transform, detected_ids):
    """
        Detects ArUco markers in the current frame using the 4x4_1000 dictionary.
        If a marker (with a new ID) is found, computes its center in the mosaic coordinate
        system using the provided transformation matrix (current_transform), and returns
        a tuple (True, (x, y)). Otherwise returns (False, None).

        Parameters:
          frame: the current frame (BGR image)
          current_transform: the homography matrix (e.g. video_mosaic.H_old) at the current frame;
             use the one that best represents the mapping of this frame into the mosaic coordinates.
          detected_ids: a set containing marker IDs that have been seen in previous frames.
    def is_homography_distorted(frame, H, threshold_area_ratio=2.0, threshold_angle=30):
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    parameters = cv2.aruco.DetectorParameters()
    parameters.minDistanceToBorder = 5
    parameters.adaptiveThreshWinSizeMax = 15
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    if ids is not None:
        # Loop over each detected marker
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id not in detected_ids:
                detected_ids.add(marker_id)
                # Use the marker corners (shape: [1,4,2]) to compute a center.
                marker_corners = corners[i][0]  # now shape (4, 2)
                center = marker_corners.mean(axis=0)
                # Convert center to homogeneous coordinates.
                center_homog = np.array([center[0], center[1], 1.0])
                # Transform it into the mosaic coordinate space.
                mosaic_center = np.dot(current_transform, center_homog)
                mosaic_center /= mosaic_center[2]  # normalize homogeneous coordinate
                return True, (int(mosaic_center[0]), int(mosaic_center[1]))
    return False, None


def is_homography_distorted(frame, H, threshold_area_ratio=2.0, threshold_angle=30):
    """
    Returns True if the homography causes a warped quadrilateral
    whose area differs too much from the original, or its corner angles
    deviate significantly from 90°.

    Args:
      frame: the current frame (assumed shape [h, w, 3])
      H: the 3x3 homography matrix.
      threshold_area_ratio: if the warped area is more than threshold_area_ratio times
         or less than the original area, consider it distorted.
      threshold_angle: if the average deviation from 90° (in degrees) is above this threshold.
    """
    h, w = frame.shape[:2]
    # Define frame corners in homogeneous coordinates
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(
        -1, 1, 2
    )
    warped_corners = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

    # Compute bounding box area for the warped corners
    min_xy = warped_corners.min(axis=0)
    max_xy = warped_corners.max(axis=0)
    warped_area = (max_xy[0] - min_xy[0]) * (max_xy[1] - min_xy[1])
    orig_area = w * h
    area_ratio = warped_area / orig_area

    # Helper to compute angle at ptB given three points A, B, C.
    def angle(ptA, ptB, ptC):
        v1 = ptA - ptB
        v2 = ptC - ptB
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    angles = []
    for i in range(4):
        a = warped_corners[i - 1]
        b = warped_corners[i]
        c = warped_corners[(i + 1) % 4]
        angles.append(angle(a, b, c))
    # Average absolute deviation from 90 degrees.
    angle_deviation = np.mean([abs(a - 90) for a in angles])

    # If the warped area deviates too much or the angle distortion is high, flag it.
    if (
        area_ratio > threshold_area_ratio
        or area_ratio < 1.0 / threshold_area_ratio
        or angle_deviation > threshold_angle
    ):
        return True
    return False


def mp4_to_mjpeg(
    src_path: str,
    dst_path: str,
    video_width: int,
    num_frames: int,
    skip_frames: int = 1,
    start_frame: int = 0,
):
    """
    Converts an MP4 video to an MJPEG video by scaling, selecting frames,
    and re-timing the output. Frames are selected only if the frame number
    is at least start_frame and satisfies the skip_frames condition.

    e.g.
    ffmpeg -i Data\DJI_0001.MP4 -vf "scale=640:-2,select='not(mod(n,10))',setpts=N/(FRAME_RATE*TB)" -frames:v 200 -c:v mjpeg dest.mjpeg

    Parameters:
        src_path (str): Path to the source MP4 file.
        dst_path (str): Destination path for the MJPEG output.
        video_width (int): Width to scale the video (height is computed to keep aspect ratio).
        num_frames (int): Number of output frames.
        skip_frames (int): Process every nth frame (i.e. selects frames where mod(n, skip_frames) is 0).
        start_frame (int): The frame number to start processing from.
    """
    assert Path(src_path).suffix == ".mp4"
    assert Path(dst_path).suffix == ".mjpeg"

    filter_chain = (
        f"scale={video_width}:-2,"
        f"select='gte(n,{start_frame})*not(mod(n,{skip_frames}))',"
        "setpts=N/(FRAME_RATE*TB)"
    )

    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-i",
        str(src_path),
        "-vf",
        filter_chain,
        "-frames:v",
        f"{num_frames}",
        "-c:v",
        "mjpeg",
        str(dst_path),
    ]

    result = subprocess.run(ffmpeg_command, capture_output=True, text=True)

    if result.returncode != 0:
        print(
            f"An error occurred while converting {src_path} mp4 -> mjpeg:\n{result.stderr}"
        )
    else:
        print(
            f"Success converting {src_path} mp4 -> mjpeg. Output saved to {dst_path}.\n{result.stdout}"
        )


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
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
        self.aruco_parameters = cv2.aruco.DetectorParameters()
        self.aruco_parameters.minDistanceToBorder = 5
        self.aruco_parameters.adaptiveThreshWinSizeMax = 15
        self.aruco_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

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

    def detect_aruco_markers(self, frame):
        # detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_parameters)
        corners, ids, _ = cv2.aruco.detectMarkers(
            frame, self.aruco_dict, parameters=self.aruco_parameters
        )
        markers = {}
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                # Each marker returns corners with shape (1,4,2); reshape to (4,2)
                markers[int(marker_id)] = corners[i].reshape(4, 2)
        return markers

    def process_first_frame(self, first_image):
        self.frame_prev = first_image
        frame_gray_prev = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
        self.kp_prev, self.des_prev = self.detector.detectAndCompute(
            frame_gray_prev, None
        )
        self.prev_aruco_markers = self.detect_aruco_markers(first_image)

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
        self.kp_cur, self.des_cur = self.detector.detectAndCompute(frame_gray_cur, None)

        common_ids = False

        # Detect markers in the current frame
        cur_aruco_markers = self.detect_aruco_markers(self.frame_cur)
        # Check if previous frame had markers and if there is a common marker
        if self.prev_aruco_markers:
            common_ids = set(self.prev_aruco_markers.keys()).intersection(
                cur_aruco_markers.keys()
            )
        # Update the previous markers for the next iteration.
        self.prev_aruco_markers = cur_aruco_markers

        if common_ids and MATCH_ARUCO_IF_POSSIBLE:
            # Take the first common marker to estimate the transformation
            marker_id = list(common_ids)[0]
            # Use the corners from the previous frame (source) and current frame (destination)
            src_pts = self.prev_aruco_markers[marker_id]  # from previous frame
            dst_pts = cur_aruco_markers[marker_id]  # from current frame
            H_aruco, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC)
            # Compose the new homography with the previous accumulated transform.
            self.H = np.matmul(self.H_old, H_aruco)
        else:
            # Fallback: use feature matching-based homography.
            self.matches = self.match(self.des_cur, self.des_prev)
            if len(self.matches) < 4:
                return
            self.H = self.findHomography(self.kp_cur, self.kp_prev, self.matches)
            if is_homography_distorted(self.frame_cur, self.H):
                print("Distortion too high; restarting mosaic with current frame.")
                # Optionally save the current mosaic segment before resetting
                # e.g., cv2.imwrite(f"mosaic_segment_reset.jpg", self.output_img)
                # Reinitialize with the current frame:
                self.__init__(self.frame_cur, detector_type=self.detector_type)
                return

            self.H = np.matmul(self.H_old, self.H)

        # Warp the current frame into the mosaic using the determined homography.
        self.warp(self.frame_cur, self.H)

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
        new_marker_found, mosaic_position = detect_new_aruco(
            frame_cur, video_mosaic.H_old, detected_ids
        )
        if new_marker_found:
            # Mark the location in the mosaic image (green circle)
            cv2.circle(video_mosaic.output_img, mosaic_position, 10, (0, 255, 0), -1)
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
    create_mosaics(
        video_path=str(dst_path), mosaic_path=MOSAIC_PATH, display_size=(640, 480)
    )
