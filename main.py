import cv2
from pathlib import Path
import numpy as np
import subprocess

MOSAIC_PATH = "mosaic.jpg"
SRC_PATH = Path("Data") / "my-own.mp4"
VIDEO_WIDTH = 640
NUM_FRAMES = 100
SKIP_FRAMES = 5
START_FRAME = 0


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
        """This class processes every frame and generates the panorama

        Args:
            first_image (image for the first frame): first image to initialize the output size
            output_height_times (int, optional): determines the output height based on input image height. Defaults to 2.
            output_width_times (int, optional): determines the output width based on input image width. Defaults to 4.
            detector_type (str, optional): the detector for feature detection. It can be "sift" or "orb". Defaults to "sift".
        """
        self.detector_type = detector_type
        if detector_type == "sift":
            self.detector = cv2.SIFT_create(700)
            self.bf = cv2.BFMatcher()
        elif detector_type == "orb":
            self.detector = cv2.ORB_create(700)
            self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.visualize = True

        self.process_first_frame(first_image)

        self.output_img = np.zeros(
            shape=(
                int(output_height_times * first_image.shape[0]),
                int(output_width_times * first_image.shape[1]),
                first_image.shape[2],
            )
        )

        # offset
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

    def process_first_frame(self, first_image):
        """processes the first frame for feature detection and description

        Args:
            first_image (cv2 image/np array): first image for feature detection
        """
        self.frame_prev = first_image
        frame_gray_prev = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
        self.kp_prev, self.des_prev = self.detector.detectAndCompute(
            frame_gray_prev, None
        )

    def match(self, des_cur, des_prev):
        """matches the descriptors

        Args:
            des_cur (np array): current frame descriptor
            des_prev (np arrau): previous frame descriptor

        Returns:
            array: and array of matches between descriptors
        """
        # matching
        if self.detector_type == "sift":
            pair_matches = self.bf.knnMatch(des_cur, des_prev, k=2)
            matches = []
            for m, n in pair_matches:
                if m.distance < 0.7 * n.distance:
                    matches.append(m)

        elif self.detector_type == "orb":
            matches = self.bf.match(des_cur, des_prev)

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

        # get the maximum of 20  best matches
        matches = matches[: min(len(matches), 20)]
        # Draw first 10 matches.
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
        """gets an image and processes that image for mosaicing

        Args:
            frame_cur (np array): input of current frame for the mosaicing
        """
        self.frame_cur = frame_cur
        frame_gray_cur = cv2.cvtColor(frame_cur, cv2.COLOR_BGR2GRAY)
        self.kp_cur, self.des_cur = self.detector.detectAndCompute(frame_gray_cur, None)

        self.matches = self.match(self.des_cur, self.des_prev)

        if len(self.matches) < 4:
            return

        self.H = self.findHomography(self.kp_cur, self.kp_prev, self.matches)
        self.H = np.matmul(self.H_old, self.H)
        # TODO: check for bad Homography

        self.warp(self.frame_cur, self.H)

        # loop preparation
        self.H_old = self.H
        self.kp_prev = self.kp_cur
        self.des_prev = self.des_cur
        self.frame_prev = self.frame_cur

    @staticmethod
    def findHomography(image_1_kp, image_2_kp, matches):
        """gets two matches and calculate the homography between two images

        Args:
            image_1_kp (np array): keypoints of image 1
            image_2_kp (np_array): keypoints of image 2
            matches (np array): matches between keypoints in image 1 and image 2

        Returns:
            np arrat of shape [3,3]: Homography matrix
        """
        # taken from https://github.com/cmcguinness/focusstack/blob/master/FocusStack.py

        image_1_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
        image_2_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
        for i in range(0, len(matches)):
            image_1_points[i] = image_1_kp[matches[i].queryIdx].pt
            image_2_points[i] = image_2_kp[matches[i].trainIdx].pt

        homography, mask = cv2.findHomography(
            image_1_points, image_2_points, cv2.RANSAC, ransacReprojThreshold=2.0
        )

        return homography

    def warp(self, frame_cur, H):
        """warps the current frame based of calculated homography H

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

        cv2.imshow("output", output_temp / 255.0)

        return self.output_img

    @staticmethod
    def get_transformed_corners(frame_cur, H):
        """finds the corner of the current frame after warp

        Args:
            frame_cur (np array): current frame
            H (np array of shape [3,3]): Homography matrix

        Returns:
            [np array]: a list of 4 corner points after warping
        """
        corner_0 = np.array([0, 0])
        corner_1 = np.array([frame_cur.shape[1], 0])
        corner_2 = np.array([frame_cur.shape[1], frame_cur.shape[0]])
        corner_3 = np.array([0, frame_cur.shape[0]])

        corners = np.array([[corner_0, corner_1, corner_2, corner_3]], dtype=np.float32)
        transformed_corners = cv2.perspectiveTransform(corners, H)

        transformed_corners = np.array(transformed_corners, dtype=np.int32)
        # mask = np.zeros(shape=(output.shape[0], output.shape[1], 1))
        # cv2.fillPoly(mask, transformed_corners, color=(1, 0, 0))
        # cv2.imshow('mask', mask)

        return transformed_corners

    def draw_border(self, image, corners, color=(0, 0, 0)):
        """This functions draw rectancle border

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


def create_mosaic(video_path, mosaic_path):
    print("Creating mosaic... press 0 to quit and save.")
    cap = cv2.VideoCapture(video_path)
    is_first_frame = True
    cap.read()
    while cap.isOpened():
        ret, frame_cur = cap.read()
        if not ret:
            if is_first_frame:
                continue
            break

        if is_first_frame:
            video_mosaic = VideoMosaic(frame_cur, detector_type="sift")
            is_first_frame = False
            continue

        # process each frame
        video_mosaic.process_frame(frame_cur)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()
    cv2.imwrite(mosaic_path, video_mosaic.output_img)
    print(f"Mosaic saved to {mosaic_path}")


if __name__ == "__main__":
    dst_path = Path(SRC_PATH).parent / (Path(SRC_PATH).stem + ".mjpeg")

    mp4_to_mjpeg(
        src_path=SRC_PATH,
        dst_path=dst_path,
        video_width=VIDEO_WIDTH,
        num_frames=NUM_FRAMES,
        skip_frames=SKIP_FRAMES,
    )

    create_mosaic(video_path=dst_path, mosaic_path=MOSAIC_PATH)
