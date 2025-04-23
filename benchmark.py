import os
import cv2
import cv2.aruco as aruco
import numpy as np
import time

NUM_TRIALS = 100

TEST_IMAGE = os.path.join("Data", "test_aruco_img.png")


def get_aruco_dict_and_params():
    # Create dictionary and detector parameters.
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    # Create the detector instance. (Note: In later OpenCV versions you might use ArucoDetector.)
    parameters = cv2.aruco.DetectorParameters()
    parameters.minDistanceToBorder = 5
    parameters.adaptiveThreshWinSizeMax = 15
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return aruco_dict, parameters


def get_detector():
    aruco_dict, parameters = get_aruco_dict_and_params()

    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    return detector


def get_centers(tag_ids, corners, image):
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

    # This will be np.array of float of shape (n, 2)
    centers = np.mean(corners2, axis=1)

    return centers


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
        centers = get_centers(tag_ids=tag_ids, corners=corners, image=test_image)
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
