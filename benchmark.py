import os
import cv2
import cv2.aruco as aruco
import numpy as np
import time

from utils import get_detector, refine_corners

NUM_TRIALS = 100

TEST_IMAGE = os.path.join("Data", "test_aruco_img.png")


if __name__ == "__main__":
    test_image = cv2.imread(TEST_IMAGE)

    # ---------------------------
    # Benchmark detector creation
    # ---------------------------
    start_time = time.time()

    for i in range(NUM_TRIALS):
        detector = get_detector()

    detector_creation_time = time.time() - start_time

    intervals1 = []
    intervals2 = []

    # ---------------------------
    # Benchmark marker detection ("prediction")
    # ---------------------------

    # You can either use the detector instance or the convenience function.
    # Here we use the function for detection using the same dictionary and parameters.
    for i in range(NUM_TRIALS):
        st = time.time()
        corners, tag_ids, rejected = detector.detectMarkers(test_image)
        # corners, tag_ids, rejected = cv2.aruco.detectMarkers(
        #    test_image, *get_aruco_dict_and_params()
        # )
        interval1 = time.time() - st
        intervals1.append(interval1)

        st = time.time()
        corners2 = refine_corners(tag_ids=tag_ids, corners=corners, image=test_image)
        # This will be np.array of float of shape (n, 2)
        centers = np.mean(corners2, axis=1)

        interval2 = time.time() - st
        intervals2.append(interval2)

        print(
            f"Trial {i:2d}: {interval1 * 1000:.1f} ms detect; {interval2*1000:.1f} ms subpixel"
        )

    # ---------------------------
    # Display benchmark results
    # ---------------------------
    detect_avg = (sum(intervals1) / NUM_TRIALS) * 1000
    subpix_avg = (sum(intervals2) / NUM_TRIALS) * 1000
    print("Detected marker IDs:", tag_ids)
    print(f"Time to create the detector: {detector_creation_time*1000:.1f} ms")
    print(f"Average time to detect markers: {detect_avg:.1f} ms")
    print(f"Average time to do subpixel: {subpix_avg:.1f} ms")

    print("TODO: display diagnostic info about cv2 version (is it GPU/CPU etc.)")

    # Optionally, display the test image with detected markers drawn
    if False and ids is not None:
        output_image = cv2.aruco.drawDetectedMarkers(
            test_image.copy(), corners, tag_ids
        )
        cv2.imshow("Detected Markers", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
