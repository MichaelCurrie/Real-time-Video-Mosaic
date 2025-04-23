import os
import cv2
import cv2.aruco as aruco
import numpy as np
import time
import cv2
from pathlib import Path
import numpy as np
import subprocess
import cv2.aruco
import os


def get_aruco_dict_and_params():
    # Create dictionary and detector parameters.
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    # Create the detector instance. (Note: In later OpenCV versions you might use ArucoDetector.)
    parameters = cv2.aruco.DetectorParameters()
    parameters.minDistanceToBorder = 5
    parameters.adaptiveThreshWinSizeMax = 15
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return aruco_dict, parameters


def detect_aruco_markers(image):
    """Detect with subpixel and convert to a nice dict structure"""
    detector = get_detector()
    corners, tag_ids, _ = detector.detectMarkers(image)
    if tag_ids is None:
        return {}

    corners = refine_corners(tag_ids=tag_ids, corners=corners, image=image)

    markers = {tag_ids[i][0]: corners[i, ...] for i in range(len(tag_ids))}
    assert all(v.shape == (4, 2) for k, v in markers.items())

    return markers


def get_detector():
    aruco_dict, parameters = get_aruco_dict_and_params()

    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    return detector


def refine_corners(tag_ids, corners, image):
    if tag_ids is None or len(tag_ids) == 0:
        return None
    num_ids = len(tag_ids)
    tag_ids = list(tag_ids.reshape(num_ids))
    # Generate subpixel information for the corners
    stop_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners0 = np.vstack(corners).squeeze()
    if corners0.shape == (4, 2):
        # Single detections (n=1) are shape (4, 2).
        # We need shape (1, 4, 2) to be consistent
        corners0 = corners0.reshape(1, 4, 2)

    # Now convert from shape (n, 4, 2) to (n*4, 1, 2)
    corners00 = corners0.reshape(corners0.shape[0] * 4, 1, 2)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners2 = cv2.cornerSubPix(gray, corners00, (5, 5), (-1, -1), stop_criteria)
    corners2 = corners2.reshape(corners0.shape)

    return corners2


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
