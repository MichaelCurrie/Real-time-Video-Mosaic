import os
from functools import partial
from pathlib import Path
from time import time

import cv2
import cv2.aruco
import numpy as np
from utils import detect_aruco_markers, mp4_to_mjpeg

# INPUT PATH
# SRC_PATH = Path("Data") / "zoom.mjpeg"
SRC_PATH = Path("Data") / "rotate.mjpeg"
# DST_PATH = SRC_PATH

# VIDEO CONVERSION OPTIONS
CONVERT_VIDEO = False
VIDEO_WIDTH = 1280
NUM_FRAMES = 150
SKIP_FRAMES = 2
START_FRAME = 0

# MATCHING OPTIONS
MATCH_ARUCO_IF_POSSIBLE = False
USE_ARUCO = False
DETECTOR_TYPE = "sift"

# OUTPUT PATH
MOSAIC_PATH = "Mosaics"

USE_CUDA_IF_POSSIBLE = False

if USE_CUDA_IF_POSSIBLE:
    assert (
        DETECTOR_TYPE != "sift"
    ), "DETECTOR_TYPE = 'sift' not implemented for CUDA sadly; change to 'orb' please"
    # CUDA / CPU setup
    print("CUDA devices:", cv2.cuda.getCudaEnabledDeviceCount())
    USE_CUDA = cv2.cuda.getCudaEnabledDeviceCount() > 0
else:
    USE_CUDA = False

assert DETECTOR_TYPE in ["orb", "sift"]

if USE_CUDA:
    if False:
        cv2.cuda.printCudaDeviceInfo(0)
    # no cuda-sift in prebuilt wheels
    SIFT_create = cv2.SIFT_create
    # ORB_create = cv2.cuda.ORB_create
    ORB_create = partial(
        cv2.cuda.ORB_create,
        nfeatures=2000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31,
        fastThreshold=20,
    )
    BFMatcher = partial(cv2.cuda.DescriptorMatcher_createBFMatcher, cv2.NORM_HAMMING)
else:
    print("No CUDA devices requested or found — falling back to CPU.")
    SIFT_create = cv2.SIFT_create
    ORB_create = cv2.ORB_create
    if DETECTOR_TYPE == "sift":
        # float descriptors → L2 distance
        BFMatcher = partial(cv2.BFMatcher, cv2.NORM_L2)
    else:
        # binary descriptors → Hamming + crossCheck
        BFMatcher = partial(cv2.BFMatcher, cv2.NORM_HAMMING, crossCheck=True)

print(f"Using method: {DETECTOR_TYPE} with CUDA: {USE_CUDA}")


class VideoMosaic:
    def __init__(
        self,
        first_image,
        output_height_times=2,
        output_width_times=6,
        detector_type: str = "sift",
    ):
        self.detector_type = detector_type.lower()
        self.use_cuda = USE_CUDA and self.detector_type == "orb"
        self.all_aruco_markers = set()
        self.common_aruco_markers = set()
        self.prev_aruco_markers = {}
        self.cur_aruco_markers = {}
        self.mosaic_frame_index = 0

        # initialize detector & matcher
        if self.detector_type == "sift":
            self.detector = SIFT_create(1400)
            self.bf = BFMatcher()
        elif self.detector_type == "orb":
            self.detector = ORB_create(nfeatures=700)
            self.bf = BFMatcher()
        else:
            raise ValueError(f"Unknown detector: {detector_type}")

        # first-frame feature detection
        self.frame_prev = first_image
        gray_prev = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
        if self.use_cuda:
            gpu_prev = cv2.cuda_GpuMat()
            gpu_prev.upload(gray_prev)
            kp_gpu, des_gpu = self.detector.detectAndComputeAsync(gpu_prev, None)
            self.kp_prev = self.detector.convert(kp_gpu)
            self.des_prev = des_gpu.download()
        else:
            self.kp_prev, self.des_prev = self.detector.detectAndCompute(
                gray_prev, None
            )

        if USE_ARUCO:
            self.prev_aruco_markers = detect_aruco_markers(first_image)

        # setup mosaic canvas
        h, w = first_image.shape[:2]
        self.output_img = np.zeros(
            (output_height_times * h, output_width_times * w, 3),
            dtype=first_image.dtype,
        )
        self.w_offset = (self.output_img.shape[0] - h) // 2
        self.h_offset = (self.output_img.shape[1] - w) // 2
        self.output_img[
            self.w_offset : self.w_offset + h,
            self.h_offset : self.h_offset + w,
        ] = first_image

        self.H_old = np.eye(3)
        self.H_old[0, 2] = self.h_offset
        self.H_old[1, 2] = self.w_offset

        # visualization
        self.visualize = True
        if self.visualize:
            cv2.namedWindow("matches", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("matches", 640, 480)
            cv2.namedWindow("output", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("output", 640, 480)

    def process_frame(self, frame_cur):
        start = time()

        self.frame_cur = frame_cur

        # ArUco detection
        if USE_ARUCO:
            self.cur_aruco_markers = detect_aruco_markers(frame_cur)
            self.all_aruco_markers |= set(self.cur_aruco_markers)
            if self.prev_aruco_markers:
                self.common_aruco_markers = set(self.prev_aruco_markers.keys()) & set(
                    self.cur_aruco_markers.keys()
                )
            else:
                self.common_aruco_markers = set()
            if self.common_aruco_markers:
                mid = next(iter(self.common_aruco_markers))
                src = self.prev_aruco_markers[mid]
                dst = self.cur_aruco_markers[mid]
                H_aruco, _ = cv2.findHomography(dst, src, cv2.RANSAC)
            else:
                H_aruco = None
        else:
            H_aruco = None

        # Feature detect & match
        gray_cur = cv2.cvtColor(frame_cur, cv2.COLOR_BGR2GRAY)
        if self.use_cuda:
            gpu_cur = cv2.cuda_GpuMat()
            gpu_cur.upload(gray_cur)
            kp_gpu, des_gpu = self.detector.detectAndComputeAsync(gpu_cur, None)
            self.kp_cur = self.detector.convert(kp_gpu)
            self.des_cur = des_gpu.download()

            cur_gm = cv2.cuda_GpuMat()
            cur_gm.upload(self.des_cur)
            prev_gm = cv2.cuda_GpuMat()
            prev_gm.upload(self.des_prev)
            knn_gpu = self.bf.knnMatchAsync(cur_gm, prev_gm, 2)
            raw = self.bf.knnMatchConvert(knn_gpu)
            matches = [m for m, n in raw if m.distance < 0.7 * n.distance]

        elif self.detector_type == "sift":
            self.kp_cur, self.des_cur = self.detector.detectAndCompute(gray_cur, None)
            raw = self.bf.knnMatch(self.des_cur, self.des_prev, k=2)
            matches = [m for m, n in raw if m.distance < 0.7 * n.distance]

        else:
            self.kp_cur, self.des_cur = self.detector.detectAndCompute(gray_cur, None)
            matches = self.bf.match(self.des_cur, self.des_prev)

        matches = sorted(matches, key=lambda x: x.distance)[:20]
        if self.visualize:
            vis = cv2.drawMatches(
                frame_cur,
                self.kp_cur,
                self.frame_prev,
                self.kp_prev,
                matches,
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            cv2.imshow("matches", vis)

        # Homography via features
        if len(matches) < 4:
            print(f"Feature matches {len(matches)} < 4 so cannot find homography")
            H_feature = None
        else:
            H_feature = self.findHomography(self.kp_cur, self.kp_prev, matches)

        # Choose which homography
        if H_aruco is not None and MATCH_ARUCO_IF_POSSIBLE:
            H_map, method = H_aruco, "aruco"
        elif H_feature is not None:
            H_map, method = H_feature, "feature"
        else:
            print("No homography could be used for this frame")
            return

        # Accumulate + warp
        self.H_old = self.H_old @ H_map
        warped = cv2.warpPerspective(
            frame_cur,
            self.H_old,
            (self.output_img.shape[1], self.output_img.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_TRANSPARENT,
        )
        mask = warped > 0
        self.output_img[mask] = warped[mask]

        if self.visualize:
            cv2.imshow("output", self.output_img / 255.0)

        # Update state
        self.kp_prev, self.des_prev = self.kp_cur, self.des_cur
        self.frame_prev = frame_cur
        self.prev_aruco_markers = self.cur_aruco_markers.copy()
        self.mosaic_frame_index += 1
        print(
            f"Processed #{self.mosaic_frame_index} via {method}; tags={sorted(self.all_aruco_markers)} in {(time() - start)*1000:.2f} ms"
        )

    @staticmethod
    def findHomography(kp1, kp2, matches):
        pts1 = np.array(
            [kp1[m.queryIdx].pt for m in matches], dtype=np.float32
        ).reshape(-1, 1, 2)
        pts2 = np.array(
            [kp2[m.trainIdx].pt for m in matches], dtype=np.float32
        ).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 2.0)
        return H


def create_mosaics(video_path, mosaic_path, detector_type, display_size=(640, 480)):
    assert Path(mosaic_path).is_dir()
    print("Creating mosaic... press 'q' to quit.")
    cap = cv2.VideoCapture(video_path)
    segment = 0
    while True:
        ok, first = cap.read()
        if not ok:
            break
        vm = VideoMosaic(first, detector_type=detector_type)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            vm.process_frame(frame)
            if USE_ARUCO and vm.cur_aruco_markers and not vm.common_aruco_markers:
                break
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        out_file = Path(mosaic_path) / f"mosaic_{segment}.jpg"
        cv2.imwrite(str(out_file), vm.output_img)
        print(f"Saved {out_file}")
        segment += 1
    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    if CONVERT_VIDEO:
        mp4_to_mjpeg(
            str(SRC_PATH),
            str(DST_PATH),
            VIDEO_WIDTH,
            NUM_FRAMES,
            SKIP_FRAMES,
            START_FRAME,
        )
    os.makedirs(MOSAIC_PATH, exist_ok=True)

    create_mosaics(str(DST_PATH), MOSAIC_PATH, DETECTOR_TYPE)
